import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append("..")

from rgcn import utils
from src.history_validity_gate import (
    triples_array_to_list,
    augment_with_inverse,
    build_sr_history,
    build_so_history,
    build_ro_history,
    build_topk_candidate_ids,
    build_topk_history_features_dual,
    scatter_topk_back,
    novelty_bucket_from_history,
    stale_exact_bucket,
)
from src.history_validity_calibration import PostHocHistoryValidityCalibrator


def save_json(obj, path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_dump(path):
    obj = np.load(path)
    return obj["scores"].astype(np.float32), obj["triples"].astype(np.int64)


def build_histories(data, num_rels):
    train_triples = triples_array_to_list(data.train)
    valid_triples = triples_array_to_list(data.valid)

    train_aug = augment_with_inverse(train_triples, num_rels)
    train_valid_aug = augment_with_inverse(train_triples + valid_triples, num_rels)

    train_hist = {
        "sr": build_sr_history(train_aug),
        "so": build_so_history(train_aug),
        "ro": build_ro_history(train_aug),
    }
    train_valid_hist = {
        "sr": build_sr_history(train_valid_aug),
        "so": build_so_history(train_valid_aug),
        "ro": build_ro_history(train_valid_aug),
    }
    return train_hist, train_valid_hist


def ensure_tensor(x, dtype, device):
    if torch.is_tensor(x):
        if x.device != device or x.dtype != dtype:
            return x.to(device=device, dtype=dtype)
        return x
    return torch.as_tensor(x, dtype=dtype, device=device)


def apply_calibrator_full(
    calibrator,
    base_scores,
    triples_4col,
    histories,
    topk_cands,
    device,
):
    """Apply RHVC using score-only candidates.

    The hidden target entity is never used to choose which candidates receive
    calibration. The object column is masked before history feature creation.
    """
    triples_4col_t = ensure_tensor(
        triples_4col,
        torch.long,
        device,
    )
    base_scores_t = ensure_tensor(
        base_scores,
        torch.float32,
        device,
    )

    rel_ids = triples_4col_t[:, 1]

    # Leakage-safe: candidate membership depends only on model scores.
    candidate_ids = build_topk_candidate_ids(
        base_scores_t,
        topk_cands,
    )

    base_topk = torch.gather(
        base_scores_t,
        dim=1,
        index=candidate_ids,
    )

    # Defence in depth: history feature construction must not read the target.
    feature_queries = triples_4col_t.clone()
    feature_queries[:, 2] = -1

    (
        seen_sr,
        dt_sr,
        freq_sr,
        seen_so,
        dt_so,
        freq_so,
        seen_ro,
        dt_ro,
        freq_ro,
    ) = build_topk_history_features_dual(
        query_triples=feature_queries,
        candidate_ids=candidate_ids,
        sr_hist=histories["sr"],
        so_hist=histories["so"],
        ro_hist=histories["ro"],
        device=device,
        mode=calibrator.mode,
    )

    adjusted_topk, hist_bias = calibrator(
        base_topk,
        rel_ids,
        seen_sr,
        dt_sr,
        freq_sr,
        seen_so,
        dt_so,
        freq_so,
        seen_ro,
        dt_ro,
        freq_ro,
    )

    adjusted_full = scatter_topk_back(
        base_scores_t,
        candidate_ids,
        adjusted_topk,
    )

    return (
        adjusted_full,
        adjusted_topk,
        candidate_ids,
        hist_bias,
        (seen_sr, dt_sr),
    )


def compute_batch_loss(
    calibrator,
    base_scores_batch,
    triples_batch,
    histories,
    args,
    device,
):
    """Train RHVC without forcing the gold entity into top-k."""
    (
        adjusted_full,
        adjusted_topk,
        candidate_ids,
        hist_bias,
        aux,
    ) = apply_calibrator_full(
        calibrator,
        base_scores_batch,
        triples_batch,
        histories,
        args.topk_cands,
        device,
    )

    gold_ids = triples_batch[:, 2].to(
        device=device,
        dtype=torch.long,
    )

    # Full-entity supervision preserves the target label without exposing it to
    # top-k candidate selection.
    ce_loss = F.cross_entropy(
        adjusted_full,
        gold_ids,
    )

    seen_sr, dt_sr = aux
    pairwise_loss = torch.tensor(
        0.0,
        device=device,
    )

    if args.pairwise_weight > 0:
        stale_mask = (
            (seen_sr > 0)
            & (dt_sr > 10)
            & (candidate_ids != gold_ids.unsqueeze(1))
        )

        if stale_mask.any():
            stale_scores = adjusted_topk.masked_fill(
                ~stale_mask,
                -1e9,
            )
            stale_idx = stale_scores.argmax(dim=1)
            valid_rows = stale_scores.max(dim=1).values > -1e8

            if valid_rows.any():
                row_ids = torch.arange(
                    adjusted_full.size(0),
                    device=device,
                )[valid_rows]

                gold_logits = adjusted_full[
                    row_ids,
                    gold_ids[valid_rows],
                ]
                stale_logits = adjusted_topk[
                    row_ids,
                    stale_idx[valid_rows],
                ]

                pairwise_loss = F.relu(
                    args.margin
                    - (gold_logits - stale_logits)
                ).mean()

    bias_reg = hist_bias.pow(2).mean()

    total = (
        ce_loss
        + args.pairwise_weight * pairwise_loss
        + args.bias_reg * bias_reg
    )

    return (
        total,
        ce_loss.detach(),
        pairwise_loss.detach(),
        bias_reg.detach(),
    )


def get_unique_times(array_like):
    arr = np.asarray(array_like)
    return [int(x) for x in sorted(np.unique(arr[:, 3]).tolist())]


def safe_div(x, y):
    return 0.0 if y == 0 else x / y


def finalize_stats(stats):
    out = {}
    for k, v in stats.items():
        c = max(v["count"], 1)
        out[k] = {
            "count": int(v["count"]),
            "MRR": safe_div(v["MRR"], c),
            "Hits@1": safe_div(v["Hits@1"], c),
            "Hits@3": safe_div(v["Hits@3"], c),
            "Hits@10": safe_div(v["Hits@10"], c),
        }
    return out


@torch.no_grad()
def analyze_adjusted_dump(
    calibrator,
    scores_np,
    triples_np,
    time_list,
    all_ans_list,
    histories,
    device,
    topk_cands,
):
    """Evaluate calibrated scores without contaminating raw predictions."""
    overall = {
        "MRR": 0.0,
        "Hits@1": 0.0,
        "Hits@3": 0.0,
        "Hits@10": 0.0,
        "count": 0,
    }
    bucket_stats = {
        "repeat": {
            "MRR": 0.0,
            "Hits@1": 0.0,
            "Hits@3": 0.0,
            "Hits@10": 0.0,
            "count": 0,
        },
        "near_repeat": {
            "MRR": 0.0,
            "Hits@1": 0.0,
            "Hits@3": 0.0,
            "Hits@10": 0.0,
            "count": 0,
        },
        "novel": {
            "MRR": 0.0,
            "Hits@1": 0.0,
            "Hits@3": 0.0,
            "Hits@10": 0.0,
            "count": 0,
        },
    }
    stale_total = 0
    stale_count = 0

    unique_dump_times = [
        int(x)
        for x in sorted(
            np.unique(triples_np[:, 3]).tolist()
        )
    ]
    assert unique_dump_times == [int(x) for x in time_list], (
        "Dump times do not match expected split times.\n"
        f"dump={unique_dump_times[:5]}... "
        f"expected={time_list[:5]}..."
    )

    for time_idx, t in enumerate(time_list):
        mask = triples_np[:, 3] == int(t)
        snap_scores_np = scores_np[mask]
        snap_triples_np = triples_np[mask]

        (
            adjusted_full,
            _,
            _,
            _,
            _,
        ) = apply_calibrator_full(
            calibrator,
            snap_scores_np,
            snap_triples_np,
            histories,
            topk_cands,
            device,
        )

        snap_triples_3 = torch.tensor(
            snap_triples_np[:, :3],
            dtype=torch.long,
            device=device,
        )

        # Top-1 diagnostics must use unfiltered predictions.
        top1 = adjusted_full.argmax(dim=1)

        # Filtered ranking is metric-only. Pass a clone so a filtering utility
        # cannot modify adjusted_full in place.
        _, _, _, rank_filter = utils.get_total_rank(
            snap_triples_3,
            adjusted_full.clone(),
            all_ans_list[time_idx],
            eval_bz=1000,
            rel_predict=0,
        )

        for i in range(adjusted_full.size(0)):
            s, r, o, cur_t = map(
                int,
                snap_triples_np[i],
            )
            rank = int(rank_filter[i].item())
            pred_o = int(top1[i].item())

            bucket = novelty_bucket_from_history(
                s,
                r,
                o,
                cur_t,
                histories["sr"],
                histories["so"],
                histories["ro"],
            )
            pred_stale_bucket = stale_exact_bucket(
                s,
                r,
                pred_o,
                cur_t,
                histories["sr"],
            )

            mrr = 1.0 / rank
            h1 = 1.0 if rank <= 1 else 0.0
            h3 = 1.0 if rank <= 3 else 0.0
            h10 = 1.0 if rank <= 10 else 0.0

            overall["MRR"] += mrr
            overall["Hits@1"] += h1
            overall["Hits@3"] += h3
            overall["Hits@10"] += h10
            overall["count"] += 1

            bucket_stats[bucket]["MRR"] += mrr
            bucket_stats[bucket]["Hits@1"] += h1
            bucket_stats[bucket]["Hits@3"] += h3
            bucket_stats[bucket]["Hits@10"] += h10
            bucket_stats[bucket]["count"] += 1

            if bucket in {"near_repeat", "novel"}:
                stale_total += 1
                if pred_stale_bucket == "stale":
                    stale_count += 1

    overall_out = {
        "count": int(overall["count"]),
        "MRR": safe_div(
            overall["MRR"],
            overall["count"],
        ),
        "Hits@1": safe_div(
            overall["Hits@1"],
            overall["count"],
        ),
        "Hits@3": safe_div(
            overall["Hits@3"],
            overall["count"],
        ),
        "Hits@10": safe_div(
            overall["Hits@10"],
            overall["count"],
        ),
    }

    return {
        "overall_filtered": overall_out,
        "bucket_metrics_filtered": finalize_stats(
            bucket_stats
        ),
        "stale_top1_interference": {
            "count": int(stale_total),
            "stale_top1_count": int(stale_count),
            "stale_top1_rate": safe_div(
                stale_count,
                stale_total,
            ),
        },
    }


def evaluate_dev_mrr(
    calibrator,
    scores_np,
    triples_np,
    time_list,
    all_ans_list,
    histories,
    device,
    topk_cands,
):
    summary = analyze_adjusted_dump(
        calibrator,
        scores_np,
        triples_np,
        time_list,
        all_ans_list,
        histories,
        device,
        topk_cands,
    )
    return summary["overall_filtered"]["MRR"], summary


def split_valid_dump_by_tail_time(
    valid_scores,
    valid_triples,
    valid_times,
    dev_frac,
):
    """Create disjoint temporal calibration-train and validation-dev splits."""
    if len(valid_times) < 2:
        raise ValueError(
            "At least two validation timestamps are required "
            "for disjoint calibration-train and validation-dev splits."
        )

    if not 0.0 < dev_frac < 1.0:
        raise ValueError(
            f"dev_frac must be between 0 and 1, got {dev_frac}"
        )

    dev_num_times = int(
        round(len(valid_times) * dev_frac)
    )
    dev_num_times = max(
        1,
        min(
            dev_num_times,
            len(valid_times) - 1,
        ),
    )

    train_times = valid_times[:-dev_num_times]
    dev_times = valid_times[-dev_num_times:]

    train_mask = np.isin(
        valid_triples[:, 3],
        np.asarray(train_times, dtype=np.int64),
    )
    dev_mask = np.isin(
        valid_triples[:, 3],
        np.asarray(dev_times, dtype=np.int64),
    )

    if np.any(train_mask & dev_mask):
        raise RuntimeError(
            "Calibration-train and validation-dev rows overlap."
        )

    train_scores_np = valid_scores[train_mask]
    train_triples_np = valid_triples[train_mask]
    dev_scores_np = valid_scores[dev_mask]
    dev_triples_np = valid_triples[dev_mask]

    return (
        train_scores_np,
        train_triples_np,
        train_times,
        dev_scores_np,
        dev_triples_np,
        dev_times,
    )


def main():
    parser = argparse.ArgumentParser(description="Post-hoc RHVC calibrator for RE-GCN dumps")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--valid-dump", type=str, required=True)
    parser.add_argument("--test-dump", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-rels", type=int, default=-1)

    parser.add_argument("--mode", type=str, default="full", choices=["exact", "near", "full"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--topk-cands", type=int, default=256)
    parser.add_argument("--eval-topk-cands", type=int, default=256)
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=3)

    parser.add_argument("--pairwise-weight", type=float, default=0.30)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--bias-reg", type=float, default=1e-4)

    parser.add_argument("--rel-emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--init-gamma-exact", type=float, default=0.02)
    parser.add_argument("--init-gamma-near", type=float, default=0.10)
    parser.add_argument("--stale-init", type=float, default=0.40)
    parser.add_argument("--init-base-scale", type=float, default=1.0)
    parser.add_argument("--max-bias", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = utils.load_data(args.dataset)
    num_rels = data.num_rels if args.num_rels < 0 else args.num_rels
    train_hist, train_valid_hist = build_histories(data, num_rels)

    valid_scores, valid_triples = load_dump(args.valid_dump)
    test_scores, test_triples = load_dump(args.test_dump)

    valid_times = get_unique_times(data.valid)
    test_times = get_unique_times(data.test)

    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, data.num_rels, data.num_nodes, False)
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, data.num_rels, data.num_nodes, False)

    train_scores_np, train_triples_np, train_times, dev_scores_np, dev_triples_np, dev_times = split_valid_dump_by_tail_time(
        valid_scores, valid_triples, valid_times, args.dev_frac
    )

    calibrator = PostHocHistoryValidityCalibrator(
        num_relations=num_rels * 2,
        mode=args.mode,
        rel_emb_dim=args.rel_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        init_gamma_exact=args.init_gamma_exact,
        init_gamma_near=args.init_gamma_near,
        stale_init=args.stale_init,
        init_base_scale=args.init_base_scale,
        max_bias=args.max_bias,
    ).to(device)

    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorDataset(
        torch.tensor(train_scores_np, dtype=torch.float32),
        torch.tensor(train_triples_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_dev_mrr = -1.0
    best_epoch = -1
    patience_left = args.patience
    history_rows = []

    best_state_path = os.path.join(args.out_dir, "rhvc_full.pt")

    for epoch in range(args.epochs):
        calibrator.train()
        loss_vals = []
        ce_vals = []
        pair_vals = []
        reg_vals = []

        for batch_scores, batch_triples in loader:
            batch_scores = batch_scores.to(device)
            batch_triples = batch_triples.to(device)

            optimizer.zero_grad()
            loss, ce_loss, pair_loss, reg_loss = compute_batch_loss(
                calibrator,
                batch_scores,
                batch_triples,
                train_hist,
                args,
                device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
            optimizer.step()

            loss_vals.append(float(loss.item()))
            ce_vals.append(float(ce_loss.item()))
            pair_vals.append(float(pair_loss.item()))
            reg_vals.append(float(reg_loss.item()))

        calibrator.eval()
        dev_mrr, _ = evaluate_dev_mrr(
            calibrator,
            dev_scores_np,
            dev_triples_np,
            dev_times,
            all_ans_list_valid[-len(dev_times):],
            train_hist,
            device,
            args.eval_topk_cands,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(loss_vals)) if loss_vals else None,
            "ce_loss": float(np.mean(ce_vals)) if ce_vals else None,
            "pairwise_loss": float(np.mean(pair_vals)) if pair_vals else None,
            "bias_reg": float(np.mean(reg_vals)) if reg_vals else None,
            "dev_mrr": float(dev_mrr),
            "is_best": False,
        }

        if dev_mrr > best_dev_mrr:
            best_dev_mrr = dev_mrr
            best_epoch = epoch
            patience_left = args.patience
            row["is_best"] = True
            torch.save(
                {
                    "state_dict": calibrator.state_dict(),
                    "epoch": epoch,
                    "best_dev_mrr": best_dev_mrr,
                },
                best_state_path,
            )
        else:
            if epoch + 1 >= args.min_epochs:
                patience_left -= 1
                if patience_left <= 0:
                    history_rows.append(row)
                    break

        history_rows.append(row)

    checkpoint = torch.load(best_state_path, map_location=device)
    calibrator.load_state_dict(checkpoint["state_dict"])
    calibrator.eval()

    dev_full = analyze_adjusted_dump(
        calibrator,
        dev_scores_np,
        dev_triples_np,
        dev_times,
        all_ans_list_valid[-len(dev_times):],
        train_hist,
        device,
        args.eval_topk_cands,
    )

    valid_full = analyze_adjusted_dump(
        calibrator,
        valid_scores,
        valid_triples,
        valid_times,
        all_ans_list_valid,
        train_hist,
        device,
        args.eval_topk_cands,
    )

    test_full = analyze_adjusted_dump(
        calibrator,
        test_scores,
        test_triples,
        test_times,
        all_ans_list_test,
        train_valid_hist,
        device,
        args.eval_topk_cands,
    )

    results = {
        "config": vars(args),
        "training": {
            "best_epoch": int(best_epoch),
            "best_dev_mrr": float(best_dev_mrr),
            "history": history_rows,
            "checkpoint_path": best_state_path,
            "dev_split": {
                "mode": "tail_time",
                "train_times": train_times,
                "dev_times": dev_times,
            },
        },
        "validation_protocol": {
            "selection_split": "held-out validation-dev",
            "full_validation_contains_calibrator_training_rows": True,
            "full_validation_metrics_are_diagnostic_only": True,
        },
        "dev_full": dev_full,
        "valid_full": valid_full,
        "test_full": test_full,
        "overall_filtered": test_full["overall_filtered"],
        "bucket_metrics_filtered": test_full["bucket_metrics_filtered"],
        "stale_top1_interference": test_full["stale_top1_interference"],
    }

    save_json(results, os.path.join(args.out_dir, "results_full.json"))
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_dev_mrr": best_dev_mrr,
                "test_mrr": results["overall_filtered"]["MRR"],
                "test_stale_top1_rate": results["stale_top1_interference"]["stale_top1_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
