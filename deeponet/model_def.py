import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """
    Simple DeepONet-style architecture:
      - Branch: t (design param) -> latent (B, H)
      - Trunk:  x (coords)        -> latent (B, P, H)
      - Interaction: elementwise product
      - Final: linear projection to 3 displacement components

    Notes:
      - Expect inputs already normalized / scaled.
      - x should be (B, P, 3) and t should be (B, 1).
    """
    def __init__(self, branch_dim: int = 1, trunk_dim: int = 3, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()

        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.final = nn.Linear(hidden_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: (B, 1)
        x: (B, P, 3)
        returns: (B, P, 3)
        """
        B = self.branch(t)          # (B, H)
        T = self.trunk(x)           # (B, P, H)
        interaction = T * B.unsqueeze(1)  # (B, P, H)
        out = self.final(interaction)     # (B, P, 3)
        return out + self.bias
