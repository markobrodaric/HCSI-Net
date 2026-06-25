from __future__ import annotations

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a feature map from (B, C, H, W) to node descriptors (B, H*W, C).
    """
    return x.flatten(2).transpose(1, 2).contiguous()

def infer_token_grid(num_tokens: int) -> tuple[int, int]:
    """
    Infer a square token grid (H, W) from the number of tokens.
    """
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected a square token grid, got {num_tokens} tokens.")
    return side, side

def make_position_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create normalized coordinates in [-1, 1] x [-1, 1] with shape (1, H*W, 2).
    """
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class HCSIStage(nn.Module):
    """
    One stage of Hierarchical Cross-Stream Interaction (HCSI).
    """

    def __init__(
        self,
        cnn_channels: int,
        vit_channels: int,
        descriptor_dim: int = 64,
        beta_pos: float = 0.2,
        tau: float = 1.0,
    ) -> None:
        super().__init__()

        self.cnn_channels = cnn_channels
        self.vit_channels = vit_channels
        self.descriptor_dim = descriptor_dim
        self.beta_pos = beta_pos
        self.tau = tau

        self.cnn_proj = nn.Conv2d(cnn_channels, descriptor_dim, kernel_size=1, bias=False)
        self.vit_proj = nn.Linear(vit_channels, descriptor_dim, bias=False)

        self.cnn_pos_proj = nn.Linear(2, descriptor_dim, bias=False)
        self.vit_pos_proj = nn.Linear(2, descriptor_dim, bias=False)

        self.context_norm = nn.LayerNorm(descriptor_dim, elementwise_affine=False)

    def forward(
        self,
        cnn_features: torch.Tensor,
        vit_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c_cnn, h_cnn, w_cnn = cnn_features.shape
        b_vit, n_vit, c_vit = vit_tokens.shape

        if b != b_vit:
            raise ValueError("CNN and ViT batch sizes must match.")
        if c_cnn != self.cnn_channels:
            raise ValueError(f"Expected {self.cnn_channels} CNN channels, got {c_cnn}.")
        if c_vit != self.vit_channels:
            raise ValueError(f"Expected {self.vit_channels} ViT channels, got {c_vit}.")

        h_vit, w_vit = infer_token_grid(n_vit)
        device = cnn_features.device
        dtype = cnn_features.dtype

        # ------------------------------------------------------------------
        # 1) Shared Descriptor Projection (SDP)
        # ------------------------------------------------------------------
        cnn_desc = self.cnn_proj(cnn_features)              # (B, d, Hc, Wc)
        cnn_desc = flatten_spatial(cnn_desc)                # (B, Nc, d)
        cnn_desc = l2_normalize(cnn_desc)

        vit_desc = self.vit_proj(vit_tokens)                # (B, Nt, d)
        vit_desc = l2_normalize(vit_desc)

        cnn_pos = make_position_grid(h_cnn, w_cnn, device, dtype).expand(b, -1, -1)
        vit_pos = make_position_grid(h_vit, w_vit, device, dtype).expand(b, -1, -1)

        cnn_pos = l2_normalize(self.cnn_pos_proj(cnn_pos))
        vit_pos = l2_normalize(self.vit_pos_proj(vit_pos))

        q_cnn = l2_normalize(cnn_desc + self.beta_pos * cnn_pos)
        q_vit = l2_normalize(vit_desc + self.beta_pos * vit_pos)

        # ------------------------------------------------------------------
        # 2) Bi-directional Spatial Cross-Gating (BSCG)
        # ------------------------------------------------------------------
        scale = 1.0 / (max(self.tau, 1e-6) * math.sqrt(self.descriptor_dim))
        similarity = torch.matmul(q_cnn, q_vit.transpose(1, 2)) * scale  # (B, Nc, Nt)

        attn_cnn_to_vit = torch.softmax(similarity, dim=-1)               # (B, Nc, Nt)
        attn_vit_to_cnn = torch.softmax(similarity.transpose(1, 2), dim=-1)  # (B, Nt, Nc)

        ctx_cnn = torch.matmul(attn_cnn_to_vit, q_vit)                    # (B, Nc, d)
        ctx_vit = torch.matmul(attn_vit_to_cnn, q_cnn)                    # (B, Nt, d)

        ctx_cnn = self.context_norm(ctx_cnn)
        ctx_vit = self.context_norm(ctx_vit)

        score_cnn = (q_cnn * ctx_cnn).sum(dim=-1, keepdim=True)           # (B, Nc, 1)
        score_vit = (q_vit * ctx_vit).sum(dim=-1, keepdim=True)           # (B, Nt, 1)

        gate_cnn = torch.sigmoid(score_cnn).transpose(1, 2).reshape(b, 1, h_cnn, w_cnn)
        gate_vit = torch.sigmoid(score_vit)                                # (B, Nt, 1)

        gated_cnn_features = cnn_features * gate_cnn
        gated_vit_tokens = vit_tokens * gate_vit

        return gated_cnn_features, gated_vit_tokens

class HCSINet(nn.Module):
    
    def __init__(
        self,
        cnn_backbone: nn.Module,
        vit_backbone: nn.Module,
        num_classes: int = 2,
        cnn_channels: Sequence[int] = (32, 56, 160, 272, 1792),
        vit_channels: Sequence[int] = (128, 256, 512, 1024, 1024),
        descriptor_dim: int = 64,
        beta_pos: float = 0.2,
        tau: float = 1.0,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if len(cnn_channels) != len(vit_channels):
            raise ValueError("cnn_channels and vit_channels must have the same length.")

        self.cnn_backbone = cnn_backbone
        self.vit_backbone = vit_backbone
        self.num_stages = len(cnn_channels)

        self.hcsi_stages = nn.ModuleList(
            [
                HCSIStage(
                    cnn_channels=cnn_channels[k],
                    vit_channels=vit_channels[k],
                    descriptor_dim=descriptor_dim,
                    beta_pos=beta_pos,
                    tau=tau,
                )
                for k in range(self.num_stages)
            ]
        )

        fused_dim = cnn_channels[-1] + vit_channels[-1]

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the two-stream backbone with HCSI interaction and return the final
        stage outputs before global pooling.

        """
        cnn_state = x
        vit_state = x

        for stage_idx, hcsi_stage in enumerate(self.hcsi_stages):
            cnn_features = self.cnn_backbone.extract_features(cnn_state, stage=stage_idx)
            vit_tokens = self.vit_backbone.forward_features(vit_state, stage=stage_idx)

            cnn_state, vit_state = hcsi_stage(cnn_features, vit_tokens)

        return cnn_state, vit_state

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the pooled and concatenated feature vector:
          z = [z_c; z_t]
        """
        cnn_features, vit_tokens = self.forward_backbone(x)

        cnn_vector = cnn_features.mean(dim=(2, 3))  # global average pooling
        vit_vector = vit_tokens.mean(dim=1)         # token average pooling

        return torch.cat([cnn_vector, vit_vector], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return classification logits.
        """
        fused_features = self.forward_features(x)
        return self.head(fused_features)
    

