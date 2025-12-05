import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
)

######################################################################
# Shared: Attention pooling + DNABERT encoder mixin
######################################################################

class AttentionPooling(nn.Module):
    """Learned attention pooling over sequence dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        returns: [B, D]
        """
        scores = self.att(x)                    # [B, L, 1]
        weights = torch.softmax(scores, dim=1)  # [B, L, 1]
        pooled = (x * weights).sum(dim=1)       # [B, D]
        return pooled


class DNABERTEncoderMixin:
    """
    Mixin that provides DNABERT-based DNA encoding with chunking + pooling.
    Returns a vector in a specified hidden dimension.
    """
    def _init_dnabert_encoder(
        self,
        gene_model_name: str = "zhihan1996/DNA_bert_6",
        gene_chunk_nt: int = 512,
        gene_chunk_overlap: int = 0,
        freeze_gene_encoder: bool = True,
        target_hidden_dim: int = 768,
    ):
        # DNABERT encoder
        self.gene_tokenizer = AutoTokenizer.from_pretrained(
            gene_model_name,
            trust_remote_code=True,
        )
        self.gene_encoder = AutoModel.from_pretrained(
            gene_model_name,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.gene_hidden_dim = self.gene_encoder.config.hidden_size

        self.gene_token_pool = AttentionPooling(self.gene_hidden_dim)

        self.gene_chunk_nt = gene_chunk_nt
        self.gene_chunk_overlap = gene_chunk_overlap

        if freeze_gene_encoder:
            for p in self.gene_encoder.parameters():
                p.requires_grad = False

        # Map DNABERT hidden dim -> target decoder hidden dim
        self.gene_adaptor = nn.Sequential(
            nn.Linear(self.gene_hidden_dim, target_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(target_hidden_dim),
        )

    def _chunk_dna(self, dna_seq: str):
        seq = dna_seq.strip().upper()
        L = len(seq)
        if L <= self.gene_chunk_nt:
            return [seq]

        if self.gene_chunk_overlap >= self.gene_chunk_nt:
            step = self.gene_chunk_nt
        else:
            step = self.gene_chunk_nt - self.gene_chunk_overlap

        chunks = []
        i = 0
        while i < L:
            chunk = seq[i : i + self.gene_chunk_nt]
            chunks.append(chunk)
            if i + self.gene_chunk_nt >= L:
                break
            i += step
        return chunks

    def encode_gene_single(self, dna_seq: str, device: str) -> torch.Tensor:
        """
        Encode DNA into a vector in the target decoder hidden space: [hidden_dim].
        """
        chunks = self._chunk_dna(dna_seq)

        toks = self.gene_tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            if device == "cuda":
                with autocast("cuda", dtype=torch.float16):
                    out = self.gene_encoder(**toks).last_hidden_state  # [B,L,H]
            else:
                out = self.gene_encoder(**toks).last_hidden_state

        pooled = self.gene_token_pool(out)      # [B, gene_hidden_dim]
        gene_vec = pooled.mean(dim=0)           # [gene_hidden_dim]
        gene_vec = self.gene_adaptor(gene_vec)  # [hidden_dim]
        return gene_vec


######################################################################
# 0. GeneChatModel: DNABERT + GPT-2 decoder
######################################################################

class GeneChatModel(nn.Module, DNABERTEncoderMixin):
    """
    DNABERT encoder + GPT-2 decoder.
    DNA embedding is prepended to the prompt as a soft prefix.
    """
    def __init__(
        self,
        gene_model_name: str = "zhihan1996/DNA_bert_6",
        lm_name: str = "gpt2",
        gene_chunk_nt: int = 512,
        gene_chunk_overlap: int = 0,
        freeze_gene_encoder: bool = True,
    ):
        super().__init__()

        # GPT-2 side
        from transformers import AutoModelForCausalLM
        self.txt_tok = AutoTokenizer.from_pretrained(lm_name)
        if self.txt_tok.pad_token is None:
            self.txt_tok.pad_token = self.txt_tok.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        hidden_dim = self.lm.config.n_embd

        # DNABERT encoder init
        self._init_dnabert_encoder(
            gene_model_name=gene_model_name,
            gene_chunk_nt=gene_chunk_nt,
            gene_chunk_overlap=gene_chunk_overlap,
            freeze_gene_encoder=freeze_gene_encoder,
            target_hidden_dim=hidden_dim,
        )

        # Prompt text
        self.prompt_text = (
            "You are an expert genomic annotation assistant.\n"
            "Gene: [DNA]\n"
            "Summary:"
        )

    def forward_single(self, dna: str, target: str, device: str):
        """
        Causal LM training:
          input = gene embedding (soft prefix) + prompt + target
          labels = mask prompt, only train on target tokens
        """
        # Encode gene
        gene_vec = self.encode_gene_single(dna, device=device)  # [hidden_dim]
        gene_emb = gene_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        # Tokenize prompt + target
        full_text = self.prompt_text + " " + target
        enc = self.txt_tok(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.lm.config.n_positions - 1,  # leave room for gene token
        )
        input_ids = enc.input_ids.to(device)  # [1, L]

        # Get text embeddings
        txt_emb = self.lm.transformer.wte(input_ids)  # [1, L, hidden_dim]

        # Concatenate gene embedding + text embeddings
        inputs_embeds = torch.cat([gene_emb, txt_emb], dim=1)  # [1, 1+L, hidden_dim]

        # Build labels: -100 for gene + prompt, real tokens for target
        prompt_len = len(self.txt_tok(self.prompt_text, add_special_tokens=False).input_ids)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        # Shift labels and prepend -100 for gene token
        labels = torch.cat([
            torch.full((1, 1), -100, dtype=labels.dtype, device=device),
            labels
        ], dim=1)  # [1, 1+L]

        # Forward pass
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        return outputs.loss

    def generate(
        self,
        dna: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
    ) -> str:
        self.eval()
        # Encode gene
        gene_vec = self.encode_gene_single(dna, device=device)
        gene_emb = gene_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        # Tokenize prompt only
        enc = self.txt_tok(
            self.prompt_text,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)
        txt_emb = self.lm.transformer.wte(input_ids)

        # Concatenate gene + prompt embeddings
        inputs_embeds = torch.cat([gene_emb, txt_emb], dim=1)

        # Generate
        gen_ids = self.lm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.txt_tok.pad_token_id,
        )

        # Decode (skip the prompt part)
        full_text = self.txt_tok.decode(gen_ids[0], skip_special_tokens=True)
        # Try to extract just the summary part after "Summary:"
        if "Summary:" in full_text:
            summary = full_text.split("Summary:")[-1].strip()
        else:
            summary = full_text.strip()
        return summary


######################################################################
# 1. DNABERT + BART decoder
######################################################################

class DNABERTBartDecoder(nn.Module, DNABERTEncoderMixin):
    """
    DNABERT encoder + BART seq2seq decoder.
    DNA embedding is injected as a special <|gene|> token embedding
    into BART's encoder side.
    """
    def __init__(
        self,
        gene_model_name: str = "zhihan1996/DNA_bert_6",
        bart_name: str = "facebook/bart-base",
        gene_chunk_nt: int = 512,
        gene_chunk_overlap: int = 0,
        freeze_gene_encoder: bool = True,
    ):
        super().__init__()

        # BART side
        self.txt_tok = AutoTokenizer.from_pretrained(bart_name)
        if self.txt_tok.pad_token is None:
            # BART usually has a pad token, but just in case
            self.txt_tok.pad_token = self.txt_tok.eos_token

        self.gene_token = "<|gene|>"
        special_tokens = {"additional_special_tokens": []}
        if self.gene_token not in self.txt_tok.get_vocab():
            special_tokens["additional_special_tokens"].append(self.gene_token)

        if special_tokens["additional_special_tokens"]:
            self.txt_tok.add_special_tokens(special_tokens)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(bart_name)
        self.model.resize_token_embeddings(len(self.txt_tok))

        hidden_dim = self.model.config.d_model

        # DNABERT encoder init
        self._init_dnabert_encoder(
            gene_model_name=gene_model_name,
            gene_chunk_nt=gene_chunk_nt,
            gene_chunk_overlap=gene_chunk_overlap,
            freeze_gene_encoder=freeze_gene_encoder,
            target_hidden_dim=hidden_dim,
        )

        self.gene_token_id = self.txt_tok.convert_tokens_to_ids(self.gene_token)

        # Prompt lives on encoder side
        self.prompt_text = (
            "You are an expert genomic annotation assistant.\n"
            f"Gene: {self.gene_token}\n"
            "Summary:"
        )

    def _build_encoder_inputs(self, dna: str, device: str):
        enc = self.txt_tok(
            self.prompt_text,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = enc.input_ids[0].to(device)         # [L]
        attention_mask = enc.attention_mask[0].to(device)

        # get base token embeddings
        embeds = self.model.get_input_embeddings()(input_ids)  # [L, d_model]

        # inject DNABERT embedding at <|gene|> positions
        gene_positions = (input_ids == self.gene_token_id).nonzero(as_tuple=False)
        if gene_positions.numel() > 0:
            gene_vec = self.encode_gene_single(dna, device=device)  # [d_model]
            for pos in gene_positions:
                idx = pos[0].item()
                embeds[idx] = gene_vec

        return embeds.unsqueeze(0), attention_mask.unsqueeze(0)

    def forward_single(self, dna: str, target: str, device: str):
        """
        Standard seq2seq training:
          encoder input = prompt with gene token (embedding replaced by DNABERT)
          decoder target = summary text
        """
        encoder_embeds, encoder_mask = self._build_encoder_inputs(dna, device)

        # target/labels
        tgt = self.txt_tok(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).input_ids[0].to(device)

        labels = tgt.clone()
        labels[labels == self.txt_tok.pad_token_id] = -100
        labels = labels.unsqueeze(0)  # [1, L_y]

        outputs = self.model(
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_mask,
            labels=labels,
        )
        return outputs.loss

    def generate(
        self,
        dna: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
        num_beams: int = 4,
    ) -> str:
        self.eval()
        encoder_embeds, encoder_mask = self._build_encoder_inputs(dna, device)

        gen_ids = self.model.generate(
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.txt_tok.decode(gen_ids[0], skip_special_tokens=True).strip()


######################################################################
# 2. DNABERT + T5 decoder
######################################################################

class DNABERTT5Decoder(nn.Module, DNABERTEncoderMixin):
    """
    DNABERT encoder + T5 seq2seq decoder.
    DNA embedding is injected as <|gene|> token embedding on encoder side.
    """
    def __init__(
        self,
        gene_model_name: str = "zhihan1996/DNA_bert_6",
        t5_name: str = "t5-small",
        gene_chunk_nt: int = 512,
        gene_chunk_overlap: int = 0,
        freeze_gene_encoder: bool = True,
    ):
        super().__init__()

        self.txt_tok = AutoTokenizer.from_pretrained(t5_name)
        if self.txt_tok.pad_token is None:
            self.txt_tok.pad_token = self.txt_tok.eos_token

        self.gene_token = "<|gene|>"
        special_tokens = {"additional_special_tokens": []}
        if self.gene_token not in self.txt_tok.get_vocab():
            special_tokens["additional_special_tokens"].append(self.gene_token)

        if special_tokens["additional_special_tokens"]:
            self.txt_tok.add_special_tokens(special_tokens)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
        self.model.resize_token_embeddings(len(self.txt_tok))

        hidden_dim = self.model.config.d_model

        # DNABERT encoder init
        self._init_dnabert_encoder(
            gene_model_name=gene_model_name,
            gene_chunk_nt=gene_chunk_nt,
            gene_chunk_overlap=gene_chunk_overlap,
            freeze_gene_encoder=freeze_gene_encoder,
            target_hidden_dim=hidden_dim,
        )

        self.gene_token_id = self.txt_tok.convert_tokens_to_ids(self.gene_token)

        self.prompt_text = (
            "summarize gene function: "
            f"{self.gene_token}"
        )

    def _build_encoder_inputs(self, dna: str, device: str):
        enc = self.txt_tok(
            self.prompt_text,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = enc.input_ids[0].to(device)
        attention_mask = enc.attention_mask[0].to(device)

        embeds = self.model.get_input_embeddings()(input_ids)  # [L, d_model]

        gene_positions = (input_ids == self.gene_token_id).nonzero(as_tuple=False)
        if gene_positions.numel() > 0:
            gene_vec = self.encode_gene_single(dna, device=device)
            for pos in gene_positions:
                idx = pos[0].item()
                embeds[idx] = gene_vec

        return embeds.unsqueeze(0), attention_mask.unsqueeze(0)

    def forward_single(self, dna: str, target: str, device: str):
        encoder_embeds, encoder_mask = self._build_encoder_inputs(dna, device)

        tgt = self.txt_tok(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.n_positions
            if hasattr(self.model.config, "n_positions")
            else 256,
        ).input_ids[0].to(device)

        labels = tgt.clone()
        labels[labels == self.txt_tok.pad_token_id] = -100
        labels = labels.unsqueeze(0)

        outputs = self.model(
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_mask,
            labels=labels,
        )
        return outputs.loss

    def generate(
        self,
        dna: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
        num_beams: int = 4,
    ) -> str:
        self.eval()
        encoder_embeds, encoder_mask = self._build_encoder_inputs(dna, device)

        gen_ids = self.model.generate(
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.txt_tok.decode(gen_ids[0], skip_special_tokens=True).strip()


######################################################################
# 3. DNABERT + GRU decoder (custom LM)
######################################################################

class DNABERTGRUDecoder(nn.Module, DNABERTEncoderMixin):
    """
    DNABERT encoder -> GRU decoder LM over a GPT-2-like tokenizer.
    Fully trainable from scratch on summaries.
    """
    def __init__(
        self,
        gene_model_name: str = "zhihan1996/DNA_bert_6",
        gene_chunk_nt: int = 512,
        gene_chunk_overlap: int = 0,
        freeze_gene_encoder: bool = True,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        txt_model_name: str = "gpt2",
    ):
        super().__init__()

        # text tokenizer
        self.txt_tok = AutoTokenizer.from_pretrained(txt_model_name)
        if self.txt_tok.pad_token is None:
            self.txt_tok.add_special_tokens({"pad_token": "<|pad|>"})

        vocab_size = len(self.txt_tok)

        # GRU LM
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # DNABERT encoder
        self._init_dnabert_encoder(
            gene_model_name=gene_model_name,
            gene_chunk_nt=gene_chunk_nt,
            gene_chunk_overlap=gene_chunk_overlap,
            freeze_gene_encoder=freeze_gene_encoder,
            target_hidden_dim=hidden_dim,
        )

    def forward_single(self, dna: str, target: str, device: str):
        """
        Condition GRU on DNABERT gene_vec as initial hidden state.
        Train with next-token prediction on the target summary.
        """
        # encode DNA
        gene_vec = self.encode_gene_single(dna, device=device)  # [hidden_dim]
        h0 = gene_vec.unsqueeze(0).unsqueeze(0)  # [1,1,H] -> [num_layers, batch, H]

        # encode target summary
        enc = self.txt_tok(
            target,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        ).to(device)

        input_ids = enc.input_ids  # [1, L]
        # teacher forcing: x = tokens[:-1], y = tokens[1:]
        x_ids = input_ids[:, :-1]
        y_ids = input_ids[:, 1:]

        emb = self.embed(x_ids)  # [1, L-1, E]

        # GRU forward
        out, _ = self.gru(emb, h0)  # [1, L-1, H]
        logits = self.head(out)     # [1, L-1, V]

        # loss
        vocab_size = logits.size(-1)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            y_ids.reshape(-1),
            ignore_index=self.txt_tok.pad_token_id,
        )
        return loss

    def generate(
        self,
        dna: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
    ) -> str:
        self.eval()
        gene_vec = self.encode_gene_single(dna, device=device)
        h = gene_vec.unsqueeze(0).unsqueeze(0)  # [1,1,H]

        # start from eos token as a "start" token
        start_id = (
            self.txt_tok.bos_token_id
            if self.txt_tok.bos_token_id is not None
            else self.txt_tok.eos_token_id
        )
        prev_id = torch.tensor([[start_id]], device=device, dtype=torch.long)

        generated = []

        for _ in range(max_new_tokens):
            emb = self.embed(prev_id)  # [1,1,E]
            out, h = self.gru(emb, h)  # [1,1,H]
            logits = self.head(out[:, -1])  # [1,V]
            next_id = torch.argmax(logits, dim=-1)  # [1]

            if next_id.item() == self.txt_tok.eos_token_id:
                break

            generated.append(next_id.item())
            prev_id = next_id.unsqueeze(0)

        if not generated:
            return ""
        return self.txt_tok.decode(generated, skip_special_tokens=True).strip()


######################################################################
# 4. Random summary baseline
######################################################################

class RandomSummaryBaseline(nn.Module):
    """
    Baseline: ignore DNA; randomly sample a summary from the training set.
    """
    def __init__(self, train_data):
        """
        train_data: list of dicts with key 'target', like in dataset.read_data()
        """
        super().__init__()
        self.summaries = [item["target"] for item in train_data]

    def forward_single(self, dna: str, target: str, device: str):
        # no training signal; just return zero
        return torch.tensor(0.0, device=device)

    def generate(
        self,
        dna: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
    ) -> str:
        return random.choice(self.summaries)


