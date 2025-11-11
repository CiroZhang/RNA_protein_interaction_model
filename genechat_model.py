from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class GeneChatModel(nn.Module):
    def __init__(
        self,
        gene_model_name="zhihan1996/DNABERT-2-117M",
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        gene_out_dim=512,
        text_out_dim=512,
        hidden_dim=1024,
        freeze_encoders=True
    ):
        super().__init__()
        # --- Gene encoder ---
        self.gene_tokenizer = AutoTokenizer.from_pretrained(gene_model_name, trust_remote_code=True)
        self.gene_encoder = AutoModel.from_pretrained(gene_model_name, trust_remote_code=True)
        self.gene_proj = nn.Linear(768, gene_out_dim)

        # --- Text encoder ---
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, text_out_dim)

        # --- Decoder ---
        vocab_size = self.text_tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, text_out_dim)
        self.gru = nn.GRU(text_out_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # --- Freeze encoders if desired ---
        if freeze_encoders:
            for p in self.gene_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    # Encode gene and text
    def encode_gene(self, dna_seq, device):
        inputs = self.gene_tokenizer(dna_seq, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.gene_encoder(**inputs)
        pooled = outputs[0].mean(dim=1)
        return self.gene_proj(pooled)

    def encode_text(self, text, device):
        tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.text_proj(pooled)

    # Forward (for training or inference)
    def forward(self, dna_seq, text_prompt, decoder_input_ids, device="cpu"):
        gene_emb = self.encode_gene(dna_seq, device)
        text_emb = self.encode_text(text_prompt, device)

        # Combine context embeddings
        context = torch.cat([gene_emb, text_emb], dim=-1)
        h0 = context.unsqueeze(0)

        x = self.embedding(decoder_input_ids)
        output, _ = self.gru(x, h0)
        logits = self.fc_out(output)
        return logits

    # Generate text (greedy decoding)
    def generate(self, dna_seq, text_prompt, max_len=20, device="cpu"):
        self.eval()
        gene_emb = self.encode_gene(dna_seq, device)
        text_emb = self.encode_text(text_prompt, device)
        context = torch.cat([gene_emb, text_emb], dim=-1)
        h = context.unsqueeze(0)

        input_id = torch.tensor([[self.text_tokenizer.cls_token_id]]).to(device)
        generated = []

        for _ in range(max_len):
            x = self.embedding(input_id)
            output, h = self.gru(x, h)
            logits = self.fc_out(output[:, -1])
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token.item())
            input_id = next_token.unsqueeze(0)

        return self.text_tokenizer.decode(generated, skip_special_tokens=True)
