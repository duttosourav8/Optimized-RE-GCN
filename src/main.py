import argparse
import itertools
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")

from rgcn import utils
from rgcn.utils import build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
from src.history_validity_gate import (
    triples_array_to_list,
    augment_with_inverse,
    build_sr_history,
    build_so_history,
    build_ro_history,
)


def save_json(obj, path):
    if path == "":
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_unique_times(array_like):
    arr = np.asarray(array_like)
    return [int(x) for x in sorted(np.unique(arr[:, 3]).tolist())]


def split_by_time_with_times(array_like):
    snaps = utils.split_by_time(array_like)
    times = get_unique_times(array_like)
    assert len(snaps) == len(times), (len(snaps), len(times))
    return snaps, times


def build_hva_histories(data, num_rels):
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


def build_graph_cache(snapshot_list, num_nodes, num_rels, gpu):
    """
    Cache temporal graphs once on CPU.
    This preserves baseline semantics while removing repeated graph construction.
    """
    return [build_sub_graph(num_nodes, num_rels, snap, False, gpu) for snap in snapshot_list]


def test(
    model,
    history_list,
    history_graph_list,
    test_list,
    test_graph_list,
    time_list,
    num_rels,
    num_nodes,
    use_cuda,
    all_ans_list,
    all_ans_r_list,
    ckpt_path,
    static_graph,
    mode,
    args,
    hva_histories=None,
):
    ranks_raw, ranks_filter = [], []
    ranks_raw_r, ranks_filter_r = [], []

    dump_scores = []
    dump_triples = []

    if mode == "test":
        if use_cuda:
            checkpoint = torch.load(ckpt_path, map_location=torch.device(f"cuda:{args.gpu}"))
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        print("Load checkpoint:", ckpt_path, "epoch:", checkpoint["epoch"])
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    input_snap_list = [snap for snap in history_list[-args.test_history_len:]]
    input_graphs = [g for g in history_graph_list[-args.test_history_len:]]

    for time_idx, (test_snap, test_graph, current_time) in enumerate(
        tqdm(list(zip(test_list, test_graph_list, time_list)), total=len(test_list))
    ):
        test_triples_input = torch.LongTensor(test_snap)
        if use_cuda:
            test_triples_input = test_triples_input.cuda(args.gpu)

        test_triples, final_score, final_r_score = model.predict(
            input_graphs,
            num_rels,
            static_graph,
            test_triples_input,
            use_cuda,
            current_time=int(current_time),
            hva_histories=hva_histories,
        )

        if args.dump_full_scores:
            time_col = torch.full(
                (test_triples.size(0), 1),
                int(current_time),
                dtype=torch.long,
                device=test_triples.device,
            )
            triples_with_time = torch.cat([test_triples, time_col], dim=1)
            dump_scores.append(final_score.detach().cpu().numpy().astype(np.float32))
            dump_triples.append(triples_with_time.detach().cpu().numpy().astype(np.int64))

        _, _, rank_raw_r, rank_filter_r = utils.get_total_rank(
            test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1
        )
        _, _, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)

        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)

            if len(predicted_snap):
                predicted_graph = build_sub_graph(num_nodes, num_rels, predicted_snap, False, args.gpu)
                input_snap_list.pop(0)
                input_snap_list.append(predicted_snap)
                input_graphs.pop(0)
                input_graphs.append(predicted_graph)
        else:
            input_snap_list.pop(0)
            input_snap_list.append(test_snap)
            input_graphs.pop(0)
            input_graphs.append(test_graph)

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    if args.dump_full_scores:
        if args.full_score_path == "":
            raise ValueError("--dump-full-scores requires --full-score-path")
        save_dir = os.path.dirname(args.full_score_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        all_scores = np.concatenate(dump_scores, axis=0)
        all_triples = np.concatenate(dump_triples, axis=0)
        np.savez_compressed(args.full_score_path, scores=all_scores, triples=all_triples)
        print("Saved full scores to:", args.full_score_path)
        print("Scores shape:", all_scores.shape, "Triples shape:", all_triples.shape)

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = split_by_time_with_times(data.train)
    valid_list, valid_times = split_by_time_with_times(data.valid)
    test_list, test_times = split_by_time_with_times(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight{}-discount{}-angle{}-dp{}_{}_{}_{}-gpu{}-{}".format(
        args.dataset,
        args.encoder,
        args.decoder,
        args.n_layers,
        args.dilate_len,
        args.train_history_len,
        args.weight,
        args.discount,
        args.angle,
        args.dropout,
        args.input_dropout,
        args.hidden_dropout,
        args.feat_dropout,
        args.gpu,
        args.save,
    )

    os.makedirs("../models", exist_ok=True)
    model_state_file = os.path.join("../models", model_name)

    load_ckpt_path = model_state_file
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        load_ckpt_path = args.resume_ckpt

    print("Checkpoint used:", load_ckpt_path)
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    print("building cached temporal graphs on CPU ...")
    train_graphs = build_graph_cache(train_list, num_nodes, num_rels, args.gpu)
    valid_graphs = build_graph_cache(valid_list, num_nodes, num_rels, args.gpu)
    test_graphs = build_graph_cache(test_list, num_nodes, num_rels, args.gpu)

    static_graph = None
    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False)
        )
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
    else:
        num_static_rels, num_words = 0, 0

    model = RecurrentRGCN(
        args.decoder,
        args.encoder,
        num_nodes,
        num_rels,
        num_static_rels,
        num_words,
        args.n_hidden,
        args.opn,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_basis=args.n_basis,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        aggregation=args.aggregation,
        weight=args.weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis,
        use_history_gate=args.use_history_gate,
        hva_topk=args.hva_topk,
        hva_mode=args.hva_mode,
        hva_gamma_exact=args.hva_gamma_exact,
        hva_gamma_near=args.hva_gamma_near,
        hva_stale_init=args.hva_stale_init,
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    hva_hist_train = None
    hva_hist_train_valid = None
    if args.use_history_gate:
        hva_hist_train, hva_hist_train_valid = build_hva_histories(data, num_rels)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_log = {
        "config": vars(args),
        "epochs": [],
        "best_mrr": 0.0,
        "best_epoch": None,
    }
    best_mrr = 0.0
    start_epoch = 0

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mrr = ckpt.get("best_mrr", 0.0)
        train_log["best_mrr"] = best_mrr
        print(f"Resumed from epoch {start_epoch}, best_mrr={best_mrr:.6f}")

    def resolve_train_log_path():
        if args.train_log_path:
            return args.train_log_path
        if args.ckpt_dir:
            return os.path.join(args.ckpt_dir, "training_log.json")
        return ""

    train_log_path = resolve_train_log_path()

    if args.eval_mode == "dump_valid":
        return test(
            model,
            train_list,
            train_graphs,
            valid_list,
            valid_graphs,
            valid_times,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_valid,
            all_ans_list_r_valid,
            load_ckpt_path,
            static_graph,
            mode="test",
            args=args,
            hva_histories=hva_hist_train,
        )

    if args.eval_mode == "dump_test":
        return test(
            model,
            train_list + valid_list,
            train_graphs + valid_graphs,
            test_list,
            test_graphs,
            test_times,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            load_ckpt_path,
            static_graph,
            mode="test",
            args=args,
            hva_histories=hva_hist_train_valid,
        )

    if args.test:
        return test(
            model,
            train_list + valid_list,
            train_graphs + valid_graphs,
            test_list,
            test_graphs,
            test_times,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            load_ckpt_path,
            static_graph,
            mode="test",
            args=args,
            hva_histories=hva_hist_train_valid,
        )

    print("---------------------------------------- start training ----------------------------------------")
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        losses, losses_e, losses_r, losses_static = [], [], [], []

        idx = list(range(len(train_list)))
        random.shuffle(idx)

        for train_sample_num in tqdm(idx):
            if train_sample_num == 0:
                continue

            output = train_list[train_sample_num: train_sample_num + 1]
            current_time = int(train_times[train_sample_num])

            if train_sample_num - args.train_history_len < 0:
                start_idx = 0
            else:
                start_idx = train_sample_num - args.train_history_len

            history_glist = train_graphs[start_idx:train_sample_num]

            if use_cuda:
                output = [torch.from_numpy(_).long().cuda(args.gpu) for _ in output]
            else:
                output = [torch.from_numpy(_).long() for _ in output]

            loss_e, loss_r, loss_static = model.get_loss(
                history_glist,
                output[0],
                static_graph,
                use_cuda,
                current_time=current_time,
                hva_histories=hva_hist_train,
            )
            loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static

            losses.append(loss.item())
            losses_e.append(loss_e.item())
            losses_r.append(loss_r.item())
            losses_static.append(loss_static.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_loss_entity": float(np.mean(losses_e)),
            "train_loss_relation": float(np.mean(losses_r)),
            "train_loss_static": float(np.mean(losses_static)),
            "val_mrr_filter": None,
            "is_best": False,
        }

        if args.ckpt_dir:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            latest_path = os.path.join(args.ckpt_dir, "latest.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mrr": best_mrr,
                },
                latest_path,
            )

        if epoch and ((epoch + 1) % args.evaluate_every == 0):
            _, mrr_filter, _, mrr_filter_r = test(
                model,
                train_list,
                train_graphs,
                valid_list,
                valid_graphs,
                valid_times,
                num_rels,
                num_nodes,
                use_cuda,
                all_ans_list_valid,
                all_ans_list_r_valid,
                model_state_file,
                static_graph,
                mode="train",
                args=args,
                hva_histories=hva_hist_train,
            )

            current_mrr = mrr_filter if not args.relation_evaluation else mrr_filter_r
            epoch_row["val_mrr_filter"] = float(current_mrr)

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                train_log["best_mrr"] = float(best_mrr)
                train_log["best_epoch"] = int(epoch)
                epoch_row["is_best"] = True

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_mrr": best_mrr,
                    },
                    model_state_file,
                )

                if args.ckpt_dir:
                    best_path = os.path.join(args.ckpt_dir, "best.pt")
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_mrr": best_mrr,
                        },
                        best_path,
                    )
                    print(f"Saved best checkpoint: {best_path}")

        train_log["epochs"].append(epoch_row)
        save_json(train_log, train_log_path)

        print(
            "Epoch {:04d} | Ave Loss {:.4f} | ent-rel-static {:.4f}-{:.4f}-{:.4f} | Best Filtered MRR {:.6f}".format(
                epoch,
                np.mean(losses),
                np.mean(losses_e),
                np.mean(losses_r),
                np.mean(losses_static),
                best_mrr,
            )
        )

    final_eval_ckpt = model_state_file
    if args.ckpt_dir:
        best_path = os.path.join(args.ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            final_eval_ckpt = best_path

    return test(
        model,
        train_list + valid_list,
        train_graphs + valid_graphs,
        test_list,
        test_graphs,
        test_times,
        num_rels,
        num_nodes,
        use_cuda,
        all_ans_list_test,
        all_ans_list_r_test,
        final_eval_ckpt,
        static_graph,
        mode="test",
        args=args,
        hva_histories=hva_hist_train_valid,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RE-GCN with optional end-to-end HVA")

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--run-analysis", action="store_true", default=False)
    parser.add_argument("--run-statistic", action="store_true", default=False)

    parser.add_argument("--dump-full-scores", action="store_true", default=False)
    parser.add_argument("--full-score-path", type=str, default="")
    parser.add_argument("--eval-mode", type=str, default="normal", choices=["normal", "dump_valid", "dump_test"])

    parser.add_argument("--multi-step", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--add-static-graph", action="store_true", default=False)
    parser.add_argument("--add-rel-word", action="store_true", default=False)
    parser.add_argument("--relation-evaluation", action="store_true", default=False)

    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--angle", type=int, default=10)

    parser.add_argument("--encoder", type=str, default="uvrgcn")
    parser.add_argument("--aggregation", type=str, default="none")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action="store_true", default=False)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--opn", type=str, default="sub")

    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--self-loop", action="store_true", default=True)
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--relation-prediction", action="store_true", default=False)
    parser.add_argument("--entity-prediction", action="store_true", default=False)
    parser.add_argument("--split_by_relation", action="store_true", default=False)

    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    parser.add_argument("--evaluate-every", type=int, default=20)

    parser.add_argument("--decoder", type=str, default="convtranse")
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)

    parser.add_argument("--train-history-len", type=int, default=10)
    parser.add_argument("--test-history-len", type=int, default=20)
    parser.add_argument("--dilate-len", type=int, default=1)

    parser.add_argument("--grid-search", action="store_true", default=False)
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases")
    parser.add_argument("--num-k", type=int, default=500)

    parser.add_argument("--save", type=str, default="one")
    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--train-log-path", type=str, default="")

    parser.add_argument("--use-history-gate", action="store_true", default=False)
    parser.add_argument("--hva-topk", type=int, default=256)
    parser.add_argument("--hva-mode", type=str, default="dual_branch", choices=["exact_only", "dual_branch"])
    parser.add_argument("--hva-gamma-exact", type=float, default=0.005)
    parser.add_argument("--hva-gamma-near", type=float, default=0.08)
    parser.add_argument("--hva-stale-init", type=float, default=0.2)

    args = parser.parse_args()
    print(args)

    if args.grid_search:
        out_log = "{}.{}.gs".format(args.dataset, args.encoder + "-" + args.decoder + "-" + args.save)
        o_f = open(out_log, "w")
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(",")

        if args.tune == "" or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)

        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        grid = list(grid)

        print("* {} hyperparameter combinations to try".format(len(grid)))
        o_f.write("* {} hyperparameter combinations to try\n".format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):
            o_f = open(out_log, "a")

            if not (type(grid_entry) is list or type(grid_entry) is tuple):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)

            print("* Hyperparameter Set {}:".format(i))
            o_f.write("* Hyperparameter Set {}:\n".format(i))
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")

            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = run_experiment(
                args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3]
            )
            print("MRR raw/filter: {:.6f} / {:.6f}".format(mrr_raw, mrr_filter))
            o_f.write("MRR raw/filter: {:.6f} / {:.6f}\n".format(mrr_raw, mrr_filter))
            o_f.close()
    else:
        run_experiment(args)

    sys.exit()
