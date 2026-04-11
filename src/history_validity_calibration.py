import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostHocHistoryValidityCalibrator(nn.Module):
    """
    Trainable post-hoc calibrator used only for the prototype/comparison row.

    Modes:
      - exact
      - near
      - full
    """

    def __init__(
        self,
        num_relations: int,
        mode: str = "full",
        rel_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        init_gamma_exact: float = 0.02,
        init_gamma_near: float = 0.10,
        stale_init: float = 0.40,
        init_base_scale: float = 1.0,
        max_bias: float = 2.5,
    ):
        super().__init__()
        assert mode in {"exact", "near", "full"}

        self.mode = mode
        self.max_bias = float(max_bias)
        self.dropout = nn.Dropout(dropout)

        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)

        feat_dim = 3 + rel_emb_dim  # [seen, rec, freq] + relation embedding

        self.exact_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.near_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.stale_weight = nn.Parameter(torch.tensor(float(stale_init), dtype=torch.float32))
        self.gamma_exact = nn.Parameter(torch.tensor(float(init_gamma_exact), dtype=torch.float32))
        self.gamma_near = nn.Parameter(torch.tensor(float(init_gamma_near), dtype=torch.float32))
        self.base_scale = nn.Parameter(torch.tensor(float(init_base_scale), dtype=torch.float32))

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        freq_feat = freq_feat / (freq_feat.max(dim=1, keepdim=True).values + 1e-8)
        return freq_feat * seen

    def _compose_features(self, rel_ids, seen, dt, freq):
        rel_vec = self.rel_emb(rel_ids).unsqueeze(1).expand(-1, seen.size(1), -1)
        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)
        feat = torch.cat(
            [
                seen.unsqueeze(-1),
                rec.unsqueeze(-1),
                freq_feat.unsqueeze(-1),
                rel_vec,
            ],
            dim=-1,
        )
        return feat, rec, freq_feat

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        feat_sr, rec_sr, _ = self._compose_features(rel_ids, seen_sr, dt_sr, freq_sr)
        exact_score = self.exact_mlp(feat_sr).squeeze(-1)
        stale = (1.0 - rec_sr) * seen_sr
        exact_score = exact_score - torch.clamp(self.stale_weight, min=0.0, max=5.0) * stale
        exact_score = torch.tanh(exact_score) * seen_sr

        if self.mode == "exact":
            hist_bias = torch.clamp(self.gamma_exact, min=0.0, max=1.0) * exact_score
        else:
            feat_so, _, _ = self._compose_features(rel_ids, seen_so, dt_so, freq_so)
            feat_ro, _, _ = self._compose_features(rel_ids, seen_ro, dt_ro, freq_ro)

            near_so = torch.tanh(self.near_mlp(feat_so).squeeze(-1)) * seen_so
            near_ro = torch.tanh(self.near_mlp(feat_ro).squeeze(-1)) * seen_ro
            near_score = 0.5 * (near_so + near_ro)

            if self.mode == "near":
                hist_bias = torch.clamp(self.gamma_near, min=0.0, max=1.0) * near_score
            else:
                hist_bias = (
                    torch.clamp(self.gamma_exact, min=0.0, max=1.0) * exact_score
                    + torch.clamp(self.gamma_near, min=0.0, max=1.0) * near_score
                )

        hist_bias = self.dropout(hist_bias)
        hist_bias = self.max_bias * torch.tanh(hist_bias / max(self.max_bias, 1e-6))
        scaled_base = torch.clamp(self.base_scale, min=0.25, max=4.0) * base_scores
        adjusted = scaled_base + hist_bias
        return adjusted, hist_bias