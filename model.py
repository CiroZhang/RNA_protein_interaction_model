import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RNAProteinInteractionModel(nn.Module):
    """
    Deep learning model for predicting RNA-protein interactions.

    Takes RNA and protein sequences as input and predicts whether they interact.
    Uses CNN-based feature extraction followed by fusion and classification.
    """

    def __init__(
        self,
        rna_vocab_size: int = 5,  # A, C, G, U, padding
        protein_vocab_size: int = 21,  # 20 amino acids + padding
        embedding_dim: int = 128,
        num_filters: int = 256,
        kernel_sizes: list = [3, 5, 7, 9],
        dropout: float = 0.3,
        hidden_dim: int = 512
    ):
        """
        Args:
            rna_vocab_size: Size of RNA vocabulary (default 5 for A,C,G,U + padding)
            protein_vocab_size: Size of protein vocabulary (default 21 for 20 AA + padding)
            embedding_dim: Dimension of embedding layers
            num_filters: Number of filters per kernel size in CNN
            kernel_sizes: List of kernel sizes for multi-scale feature extraction
            dropout: Dropout rate for regularization
            hidden_dim: Dimension of hidden fully connected layers
        """
        super(RNAProteinInteractionModel, self).__init__()

        # Embedding layers
        self.rna_embedding = nn.Embedding(rna_vocab_size, embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim, padding_idx=0)

        # RNA feature extractor (multi-scale CNN)
        self.rna_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.rna_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        # Protein feature extractor (multi-scale CNN)
        self.protein_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.protein_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        # Feature fusion dimension
        fusion_dim = num_filters * len(kernel_sizes) * 2  # *2 for RNA + protein

        # Fully connected layers for classification
        self.fc1 = nn.Linear(fusion_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Binary classification

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def extract_sequence_features(
        self,
        x: torch.Tensor,
        embedding: nn.Embedding,
        convs: nn.ModuleList,
        batch_norms: nn.ModuleList
    ) -> torch.Tensor:
        """
        Extract features from a sequence using embedding + multi-scale CNN.

        Args:
            x: Input sequence tensor [batch_size, seq_len]
            embedding: Embedding layer
            convs: List of convolutional layers
            batch_norms: List of batch normalization layers

        Returns:
            Feature tensor [batch_size, num_filters * len(kernel_sizes)]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        x = embedding(x)

        # Transpose for conv1d: [batch_size, embedding_dim, seq_len]
        x = x.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv, bn in zip(convs, batch_norms):
            # Conv + BatchNorm + ReLU: [batch_size, num_filters, seq_len]
            conv_out = F.relu(bn(conv(x)))
            # Global max pooling: [batch_size, num_filters]
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            conv_outputs.append(pooled)

        # Concatenate all scales: [batch_size, num_filters * len(kernel_sizes)]
        features = torch.cat(conv_outputs, dim=1)

        return features

    def forward(
        self,
        rna_seq: torch.Tensor,
        protein_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            rna_seq: RNA sequence tensor [batch_size, rna_seq_len]
            protein_seq: Protein sequence tensor [batch_size, protein_seq_len]

        Returns:
            Interaction prediction logits [batch_size, 1]
        """
        # Extract RNA features
        rna_features = self.extract_sequence_features(
            rna_seq,
            self.rna_embedding,
            self.rna_convs,
            self.rna_batch_norms
        )

        # Extract protein features
        protein_features = self.extract_sequence_features(
            protein_seq,
            self.protein_embedding,
            self.protein_convs,
            self.protein_batch_norms
        )

        # Concatenate features from both sequences
        combined = torch.cat([rna_features, protein_features], dim=1)

        # Fully connected layers with dropout
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output logits
        logits = self.fc3(x)

        return logits

    def predict_proba(
        self,
        rna_seq: torch.Tensor,
        protein_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Get interaction probability predictions.

        Args:
            rna_seq: RNA sequence tensor [batch_size, rna_seq_len]
            protein_seq: Protein sequence tensor [batch_size, protein_seq_len]

        Returns:
            Interaction probabilities [batch_size, 1]
        """
        logits = self.forward(rna_seq, protein_seq)
        return torch.sigmoid(logits)

    def predict(
        self,
        rna_seq: torch.Tensor,
        protein_seq: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get binary interaction predictions.

        Args:
            rna_seq: RNA sequence tensor [batch_size, rna_seq_len]
            protein_seq: Protein sequence tensor [batch_size, protein_seq_len]
            threshold: Classification threshold (default 0.5)

        Returns:
            Binary predictions [batch_size, 1]
        """
        proba = self.predict_proba(rna_seq, protein_seq)
        return (proba >= threshold).long()


def create_model(
    rna_vocab_size: int = 5,
    protein_vocab_size: int = 21,
    **kwargs
) -> RNAProteinInteractionModel:
    """
    Factory function to create a model instance.

    Args:
        rna_vocab_size: Size of RNA vocabulary
        protein_vocab_size: Size of protein vocabulary
        **kwargs: Additional arguments passed to the model

    Returns:
        Initialized model
    """
    return RNAProteinInteractionModel(
        rna_vocab_size=rna_vocab_size,
        protein_vocab_size=protein_vocab_size,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Example input (batch_size=2, sequence lengths can vary with padding)
    batch_size = 2
    rna_seq_len = 100
    protein_seq_len = 200

    # Random sequences (in practice, these would be encoded from actual sequences)
    rna_seq = torch.randint(1, 5, (batch_size, rna_seq_len))  # 1-4 for A,C,G,U
    protein_seq = torch.randint(1, 21, (batch_size, protein_seq_len))  # 1-20 for amino acids

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(rna_seq, protein_seq)
        probabilities = model.predict_proba(rna_seq, protein_seq)
        predictions = model.predict(rna_seq, protein_seq)

    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits: {logits.squeeze()}")
    print(f"\nProbabilities: {probabilities.squeeze()}")
    print(f"Predictions: {predictions.squeeze()}")
