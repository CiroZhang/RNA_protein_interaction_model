# genechat_model.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


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


class GeneChatModel(nn.Module):
    """
    DNABERT-6 encoder (frozen) + deeper adapter + GPT-2 decoder.

    - DNABERT-6 encodes DNA into a vector.
    - A 2-layer adapter maps DNA embedding to GPT-2 hidden dim.
    - GPT-2 takes a prompt like:

        "You are an expert genomic annotation assistant.\n"
        "Gene: <|gene|><|gene|><|gene|><|gene|>\n"
        "Summary:"

      and we replace ALL <|gene|> token embeddings with the adapted DNA embedding.
    """
    def __init__(
        self,
        gene_model_name: str = "zhihan1996/DNA_bert_6",
        gpt2_name: str = "gpt2",
        gene_chunk_nt: int = 512,
        gene_chunk_overlap: int = 0,
        freeze_gene_encoder: bool = True,
    ):
        super().__init__()

        # ---------- DNA encoder (DNABERT) ----------
        self.gene_tokenizer = AutoTokenizer.from_pretrained(
            gene_model_name,
            trust_remote_code=True,
        )
        self.gene_encoder = AutoModel.from_pretrained(
            gene_model_name,
            trust_remote_code=True,
        )
        self.gene_hidden_dim = self.gene_encoder.config.hidden_size  # usually 768

        self.gene_token_pool = AttentionPooling(self.gene_hidden_dim)

        self.gene_chunk_nt = gene_chunk_nt
        self.gene_chunk_overlap = gene_chunk_overlap

        if freeze_gene_encoder:
            for p in self.gene_encoder.parameters():
                p.requires_grad = False

        # ---------- GPT-2 text side ----------
        self.txt_tok = AutoTokenizer.from_pretrained(gpt2_name)

        # Make sure we have pad & a special <|gene|> token
        special_tokens = {}
        if self.txt_tok.pad_token is None:
            special_tokens["pad_token"] = "<|pad|>"

        self.gene_token = "<|gene|>"
        special_tokens.setdefault("additional_special_tokens", [])
        if self.gene_token not in special_tokens["additional_special_tokens"]:
            special_tokens["additional_special_tokens"].append(self.gene_token)

        if special_tokens:
            self.txt_tok.add_special_tokens(special_tokens)

        self.lm = AutoModelForCausalLM.from_pretrained(gpt2_name)
        # Resize embeddings to account for new tokens
        self.lm.resize_token_embeddings(len(self.txt_tok))

        self.hidden_dim = self.lm.config.n_embd

        # 2-layer adapter: DNABERT hidden -> GPT-2 hidden
        self.gene_adaptor = nn.Sequential(
            nn.Linear(self.gene_hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # Cache gene token id
        self.gene_token_id = self.txt_tok.convert_tokens_to_ids(self.gene_token)

        # Stronger, more structured prompt with 4 gene tokens
        self.prompt_text = (
            "You are an expert genomic annotation assistant.\n"
            f"Gene: {self.gene_token}{self.gene_token}{self.gene_token}{self.gene_token}\n"
            "Summary:"
        )

    # ------------------------------------------------------------------
    # DNA utilities
    # ------------------------------------------------------------------
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
        """Encode DNA into a vector in GPT-2 hidden space: [hidden_dim]."""
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
                    out = self.gene_encoder(**toks).last_hidden_state  # [B,L,hidden]
            else:
                out = self.gene_encoder(**toks).last_hidden_state

        pooled = self.gene_token_pool(out)           # [B, gene_hidden]
        gene_vec = pooled.mean(dim=0)                # [gene_hidden] average across chunks
        gene_vec = self.gene_adaptor(gene_vec)       # [hidden_dim]
        return gene_vec

    # ------------------------------------------------------------------
    # Input building for training
    # ------------------------------------------------------------------
    def build_inputs_for_sample(self, dna: str, target: str, device: str):
        """
        Build GPT-2 inputs_embeds, labels, and attention_mask for one sample.

        Full text:
          prompt_text + " " + target

        Loss is only applied on the target part (prompt tokens are -100).
        """
        # 1) Encode prompt alone to know its length
        prompt_ids = self.txt_tok(
            self.prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]  # [P]
        prompt_len = prompt_ids.size(0)

        # 2) Encode full sequence: prompt + space + target
        full_text = self.prompt_text + " " + target
        enc = self.txt_tok(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.lm.config.n_positions,
        )
        input_ids = enc.input_ids[0].to(device)         # [L]
        attention_mask = enc.attention_mask[0].to(device)

        # 3) Build labels: same as input_ids but mask out prompt part
        labels = input_ids.clone()
        labels[:prompt_len] = -100                      # ignore prompt tokens in loss

        # 4) Convert to embeddings
        inputs_embeds = self.lm.transformer.wte(input_ids)   # [L, hidden_dim]

        # 5) Inject gene embedding at ALL <|gene|> positions
        gene_positions = (input_ids == self.gene_token_id).nonzero(as_tuple=False)
        if gene_positions.numel() > 0:
            gene_vec = self.encode_gene_single(dna, device=device)   # [hidden_dim]
            for pos in gene_positions:
                idx = pos[0].item()
                inputs_embeds[idx] = gene_vec

        return (
            inputs_embeds.unsqueeze(0),   # [1, L, hidden_dim]
            labels.unsqueeze(0),          # [1, L]
            attention_mask.unsqueeze(0),  # [1, L]
        )

    # ------------------------------------------------------------------
    # One-sample training forward
    # ------------------------------------------------------------------
    def forward_single(self, dna: str, target: str, device: str):
        """Returns loss for a single (dna, summary) pair."""
        inputs_embeds, labels, attention_mask = self.build_inputs_for_sample(
            dna, target, device
        )

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        dna: str,
        max_new_tokens: int = 80,
        device: str = "cpu",
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate a gene summary given DNA."""
        self.eval()

        # 1) Encode prompt
        enc = self.txt_tok(
            self.prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc.input_ids[0].to(device)         # [L]
        attention_mask = enc.attention_mask[0].to(device)
        inputs_embeds = self.lm.transformer.wte(input_ids)  # [L, hidden_dim]

        # 2) Inject gene embedding at ALL <|gene|> tokens
        gene_positions = (input_ids == self.gene_token_id).nonzero(as_tuple=False)
        if gene_positions.numel() > 0:
            gene_vec = self.encode_gene_single(dna, device=device)   # [hidden_dim]
            for pos in gene_positions:
                idx = pos[0].item()
                inputs_embeds[idx] = gene_vec

        # 3) Use GPT-2 generate with inputs_embeds
        gen_ids = self.lm.generate(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.txt_tok.eos_token_id,
        )

        # We only want the generated continuation after the prompt
        out_ids = gen_ids[0][input_ids.size(0):]  # strip prompt part
        text = self.txt_tok.decode(out_ids, skip_special_tokens=True)
        return text.strip()

