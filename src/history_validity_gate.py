import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Triple = Tuple[int, int, int, int]


def triples_array_to_list(array_like) -> List[Triple]:
    arr = np.asarray(array_like)
    return [tuple(map(int, row[:4])) for row in arr]


def augment_with_inverse(triples: List[Triple], num_rels: int) -> List[Triple]:
    aug = []
    for s, r, o, t in triples:
        aug.append((s, r, o, t))
        aug.append((o, r + num_rels, s, t))
    return aug


def build_sr_history(triples: List[Triple]) -> Dict[Tuple[int, int], Dict[int, List[int]]]:
    sr_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        sr_hist[(s, r)][o].append(t)
    for sr_key in sr_hist:
        for o in sr_hist[sr_key]:
            sr_hist[sr_key][o].sort()
    return sr_hist


def build_so_history(triples: List[Triple]):
    so_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        so_hist[s][o].append(t)
    for s in so_hist:
        for o in so_hist[s]:
            so_hist[s][o].sort()
    return so_hist


def build_ro_history(triples: List[Triple]):
    ro_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        ro_hist[r][o].append(t)
    for r in ro_hist:
        for o in ro_hist[r]:
            ro_hist[r][o].sort()
    return ro_hist


def last_time_before(times: List[int], t: int):
    idx = bisect.bisect_left(times, t) - 1
    if idx < 0:
        return None
    return times[idx]


def freq_before(times: List[int], t: int) -> int:
    return bisect.bisect_left(times, t)


def build_topk_candidate_ids(base_scores: torch.Tensor, gold_ids: torch.Tensor, topk_cands: int) -> torch.Tensor:
    k = min(topk_cands, base_scores.size(1))
    topk_ids = torch.topk(base_scores, k=k, dim=1).indices

    rows = []
    for i in range(topk_ids.size(0)):
        row = topk_ids[i].tolist()
        g = int(gold_ids[i].item())
        if g not in row:
            row[-1] = g
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long, device=base_scores.device)


def scatter_topk_back(full_scores: torch.Tensor, candidate_ids: torch.Tensor, adjusted_topk_scores: torch.Tensor):
    out = full_scores.clone()
    out.scatter_(1, candidate_ids, adjusted_topk_scores)
    return out


def build_topk_history_features_dual(
    query_triples,
    candidate_ids,
    sr_hist,
    so_hist,
    ro_hist,
    device,
):
    if torch.is_tensor(candidate_ids):
        cand_np = candidate_ids.detach().cpu().numpy()
    else:
        cand_np = np.asarray(candidate_ids)

    if torch.is_tensor(query_triples):
        query_np = query_triples.detach().cpu().numpy()
    else:
        query_np = np.asarray(query_triples)

    batch_size, k = cand_np.shape

    seen_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    seen_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    seen_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    for i in range(batch_size):
        s, r, _, t = map(int, query_np[i][:4])

        cand_map_sr = sr_hist.get((s, r), {})
        cand_map_so = so_hist.get(s, {})
        cand_map_ro = ro_hist.get(r, {})

        for j, cand_o in enumerate(cand_np[i]):
            cand_o = int(cand_o)

            times_sr = cand_map_sr.get(cand_o, [])
            if times_sr:
                lt = last_time_before(times_sr, t)
                if lt is not None:
                    seen_sr[i, j] = 1.0
                    dt_sr[i, j] = float(t - lt)
                    freq_sr[i, j] = float(freq_before(times_sr, t))

            times_so = cand_map_so.get(cand_o, [])
            if times_so:
                lt = last_time_before(times_so, t)
                if lt is not None:
                    seen_so[i, j] = 1.0
                    dt_so[i, j] = float(t - lt)
                    freq_so[i, j] = float(freq_before(times_so, t))

            times_ro = cand_map_ro.get(cand_o, [])
            if times_ro:
                lt = last_time_before(times_ro, t)
                if lt is not None:
                    seen_ro[i, j] = 1.0
                    dt_ro[i, j] = float(t - lt)
                    freq_ro[i, j] = float(freq_before(times_ro, t))

    return seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro


def novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist):
    times_sr = sr_hist.get((s, r), {}).get(o, [])
    lt_sr = last_time_before(times_sr, t)
    if lt_sr is not None:
        return "repeat"

    times_so = so_hist.get(s, {}).get(o, [])
    lt_so = last_time_before(times_so, t)

    times_ro = ro_hist.get(r, {}).get(o, [])
    lt_ro = last_time_before(times_ro, t)

    if lt_so is not None or lt_ro is not None:
        return "near_repeat"

    return "novel"


def stale_exact_bucket(s, r, o, t, sr_hist):
    times = sr_hist.get((s, r), {}).get(o, [])
    lt = last_time_before(times, t)
    if lt is None:
        return "novel"
    gap = t - lt
    if gap <= 1:
        return "recent"
    if gap <= 10:
        return "mid"
    return "stale"


class HistoryValidityAdapter(nn.Module):
    """
    Reusable end-to-end candidate-level history-validity adapter.

    Modes:
      - exact_only
      - dual_branch
    """

    def __init__(
        self,
        num_relations: int,
        mode: str = "dual_branch",
        gamma_exact: float = 0.005,
        gamma_near: float = 0.08,
        stale_init: float = 0.2,
    ):
        super().__init__()
        assert mode in {"exact_only", "dual_branch"}

        self.mode = mode
        self.register_buffer("gamma_exact", torch.tensor(float(gamma_exact), dtype=torch.float32))
        self.register_buffer("gamma_near", torch.tensor(float(gamma_near), dtype=torch.float32))

        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)
        self.rel_bias_sr = nn.Embedding(num_relations, 1)

        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so = nn.Embedding(num_relations, 1)

        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro = nn.Embedding(num_relations, 1)

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        nn.init.constant_(self.rel_w_stale_sr.weight, float(stale_init))

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        freq_feat = freq_feat / (freq_feat.max(dim=1, keepdim=True).values + 1e-8)
        freq_feat = freq_feat * seen
        return freq_feat

    def _branch_exact(self, rel_ids, seen, dt, freq):
        lam = F.softplus(self.rel_lambda_sr(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = self.rel_w_rec_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = self.rel_w_freq_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wstale = self.rel_w_stale_sr(rel_ids).squeeze(-1).unsqueeze(1)
        b = self.rel_bias_sr(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        stale = (1.0 - rec) * seen
        freq_feat = self._normalize_freq(freq, seen)

        score = wrec * rec + wfreq * freq_feat - wstale * stale + b
        return torch.tanh(score) * seen

    def _branch_near(self, rel_ids, seen, dt, freq, emb_lambda, emb_wrec, emb_wfreq, emb_bias):
        lam = F.softplus(emb_lambda(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = emb_wrec(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = emb_wfreq(rel_ids).squeeze(-1).unsqueeze(1)
        b = emb_bias(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)

        score = wrec * rec + wfreq * freq_feat + b
        return torch.tanh(score) * seen

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        g_sr = self._branch_exact(rel_ids, seen_sr, dt_sr, freq_sr)

        if self.mode == "exact_only":
            hist_bias = self.gamma_exact * g_sr
        else:
            g_so = self._branch_near(
                rel_ids, seen_so, dt_so, freq_so,
                self.rel_lambda_so, self.rel_w_rec_so, self.rel_w_freq_so, self.rel_bias_so
            )
            g_ro = self._branch_near(
                rel_ids, seen_ro, dt_ro, freq_ro,
                self.rel_lambda_ro, self.rel_w_rec_ro, self.rel_w_freq_ro, self.rel_bias_ro
            )
            hist_bias = self.gamma_exact * g_sr + self.gamma_near * 0.5 * (g_so + g_ro)

        logits = base_scores + hist_bias
        return logits, hist_bias