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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def candidate_gold_positions(candidate_ids, gold_ids):
    pos = []
    candidate_ids_cpu = candidate_ids.detach().cpu().tolist()
    gold_ids_cpu = gold_ids.detach().cpu().tolist()
    for row, g in zip(candidate_ids_cpu, gold_ids_cpu):
        pos.append(row.index(int(g)))
    return torch.tensor(pos, dtype=torch.long, device=candidate_ids.device)


def apply_calibrator_full(
    calibrator,
    base_scores,
    triples_4col,
    histories,
    topk_cands,
    device,
):
    triples_4col_t = torch.tensor(triples_4col, dtype=torch.long, device=device)
    base_scores_t = torch.tensor(base_scores, dtype=torch.float32, device=device)

    triples_3col = triples_4col_t[:, :3]
    rel_ids = triples_3col[:, 1]
    gold_ids = triples_3col[:, 2]

    candidate_ids = build_topk_candidate_ids(base_scores_t, gold_ids, topk_cands)
    base_topk = torch.gather(base_scores_t, 1, candidate_ids)

    seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro = build_topk_history_features_dual(
        query_triples=triples_4col_t,
        candidate_ids=candidate_ids,
        sr_hist=histories["sr"],
        so_hist=histories["so"],
        ro_hist=histories["ro"],
        device=device,
    )

    adjusted_topk, hist_bias = calibrator(
        base_topk,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    )

    adjusted_full = scatter_topk_back(base_scores_t, candidate_ids, adjusted_topk)
    target_pos = candidate_gold_positions(candidate_ids, gold_ids)
    return adjusted_full, adjusted_topk, candidate_ids, target_pos, hist_bias, (seen_sr, dt_sr)


def compute_batch_loss(
    calibrator,
    base_scores_batch,
    triples_batch,
    histories,
    args,
    device,
):
    adjusted_full, adjusted_topk, candidate_ids, target_pos, hist_bias, aux = apply_calibrator_full(
        calibrator,
        base_scores_batch,
        triples_batch,
        histories,
        args.topk_cands,
        device,
    )

    ce_loss = F.cross_entropy(adjusted_topk, target_pos)

    seen_sr, dt_sr = aux
    pairwise_loss = torch.tensor(0.0, device=device)
    if args.pairwise_weight > 0:
        stale_mask = (seen_sr > 0) & (dt_sr > 10)
        stale_mask = stale_mask & (candidate_ids != triples_batch[:, 2:3])
        if stale_mask.any():
            stale_scores = adjusted_topk.masked_fill(~stale_mask, -1e9)
            stale_idx = stale_scores.argmax(dim=1)
            valid_rows = stale_scores.max(dim=1).values > -1e8
            if valid_rows.any():
                row_ids = torch.arange(adjusted_topk.size(0), device=device)[valid_rows]
                gold_logits = adjusted_topk[row_ids, target_pos[valid_rows]]
                stale_logits = adjusted_topk[row_ids, stale_idx[valid_rows]]
                pairwise_loss = F.relu(args.margin - (gold_logits - stale_logits)).mean()

    bias_reg = hist_bias.pow(2).mean()
    total = ce_loss + args.pairwise_weight * pairwise_loss + args.bias_reg * bias_reg
    return total, ce_loss.detach(), pairwise_loss.detach(), bias_reg.detach()


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
    overall = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0}
    bucket_stats = {
        "repeat": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
        "near_repeat": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
        "novel": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
    }
    stale_total = 0
    stale_count = 0

    unique_dump_times = [int(x) for x in sorted(np.unique(triples_np[:, 3]).tolist())]
    assert unique_dump_times == [int(x) for x in time_list], (
        f"Dump times do not match expected split times.\n"
        f"dump={unique_dump_times[:5]}... expected={time_list[:5]}..."
    )

    for time_idx, t in enumerate(time_list):
        mask = triples_np[:, 3] == int(t)
        snap_scores_np = scores_np[mask]
        snap_triples_np = triples_np[mask]

        adjusted_full, _, _, _, _, _ = apply_calibrator_full(
            calibrator,
            snap_scores_np,
            snap_triples_np,
            histories,
            topk_cands,
            device,
        )

        snap_triples_3 = torch.tensor(snap_triples_np[:, :3], dtype=torch.long, device=device)
        _, _, _, rank_filter = utils.get_total_rank(
            snap_triples_3,
            adjusted_full,
            all_ans_list[time_idx],
            eval_bz=1000,
            rel_predict=0,
        )

        top1 = adjusted_full.argmax(dim=1)

        for i in range(adjusted_full.size(0)):
            s, r, o, cur_t = map(int, snap_triples_np[i])
            rank = int(rank_filter[i].item())
            pred_o = int(top1[i].item())

            bucket = novelty_bucket_from_history(s, r, o, cur_t, histories["sr"], histories["so"], histories["ro"])
            pred_stale_bucket = stale_exact_bucket(s, r, pred_o, cur_t, histories["sr"])

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
        "MRR": safe_div(overall["MRR"], overall["count"]),
        "Hits@1": safe_div(overall["Hits@1"], overall["count"]),
        "Hits@3": safe_div(overall["Hits@3"], overall["count"]),
        "Hits@10": safe_div(overall["Hits@10"], overall["count"]),
    }

    return {
        "overall_filtered": overall_out,
        "bucket_metrics_filtered": finalize_stats(bucket_stats),
        "stale_top1_interference": {
            "count": int(stale_total),
            "stale_top1_count": int(stale_count),
            "stale_top1_rate": safe_div(stale_count, stale_total),
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

    num_rows = len(valid_triples)
    indices = np.arange(num_rows)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    dev_size = max(1, int(round(num_rows * args.dev_frac)))
    dev_idx = np.sort(indices[:dev_size])
    train_idx = np.sort(indices[dev_size:])

    train_scores_np = valid_scores[train_idx]
    train_triples_np = valid_triples[train_idx]
    dev_scores_np = valid_scores[dev_idx]
    dev_triples_np = valid_triples[dev_idx]

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
        dev_mrr, dev_summary = evaluate_dev_mrr(
            calibrator,
            dev_scores_np,
            dev_triples_np,
            valid_times,
            all_ans_list_valid,
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
            torch.save({"state_dict": calibrator.state_dict(), "epoch": epoch, "best_dev_mrr": best_dev_mrr}, best_state_path)
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
        },
        "valid_full": valid_full,
        "test_full": test_full,
        "overall_filtered": test_full["overall_filtered"],
        "bucket_metrics_filtered": test_full["bucket_metrics_filtered"],
        "stale_top1_interference": test_full["stale_top1_interference"],
    }

    save_json(results, os.path.join(args.out_dir, "results_full.json"))
    print(json.dumps({
        "best_epoch": best_epoch,
        "best_dev_mrr": best_dev_mrr,
        "test_mrr": results["overall_filtered"]["MRR"],
        "test_stale_top1_rate": results["stale_top1_interference"]["stale_top1_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()