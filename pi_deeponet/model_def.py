from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: int, act: str = "silu"):
        super().__init__()
        if act.lower() == "tanh":
            activation = nn.Tanh()
        elif act.lower() == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.SiLU()

        layers = [nn.Linear(in_dim, hidden_dim), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet(nn.Module):
    """DeepONet operator network.

    - Branch: parameter/features (B, branch_dim) -> (B, hidden)
    - Trunk: coordinates (B, N, 3) -> (B, N, hidden)
    - Interaction: elementwise product, then linear projection -> (B, N, 3)

    NOTE: Hard-BC는 이 모델 내부가 아니라, 훈련 루프에서 out에 게이트(g)를 곱해 적용합니다.
    """

    def __init__(
        self,
        branch_dim: int = 4,
        trunk_dim: int = 3,
        hidden_dim: int = 128,
        depth: int = 4,
        output_dim: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        self.branch = MLP(branch_dim, hidden_dim, depth, hidden_dim, act=act)
        # trunk는 (B,N,3)이므로 reshape 후 MLP 적용
        self.trunk = MLP(trunk_dim, hidden_dim, depth, hidden_dim, act=act)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, bfeat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            bfeat: (B, branch_dim)
            x: (B, N, 3) normalized coords in [-1,1]

        Returns:
            (B, N, 3) scaled displacement (model space)
        """
        B = self.branch(bfeat)              # (B, H)
        T = self.trunk(x)                   # (B, N, H) via broadcasting in MLP (supports last dim)
        # torch sequential will treat x as (..., in_dim) and keep leading dims
        # so shape stays (B,N,H)
        interaction = T * B[:, None, :]     # (B, N, H)
        out = self.proj(interaction) + self.bias
        return out
