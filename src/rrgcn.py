import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from src.history_validity_gate import (
    HistoryValidityAdapter,
    build_topk_candidate_ids,
    build_topk_history_features_dual,
    scatter_topk_back,
)


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(
                self.h_dim,
                self.h_dim,
                self.num_rels,
                self.num_bases,
                activation=act,
                dropout=self.dropout,
                self_loop=self.self_loop,
                skip_connect=sc,
                rel_emb=self.rel_emb,
            )
        raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata["id"].squeeze()
            g.ndata["h"] = init_ent_emb[node_id]
            _, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop("h")

        if self.features is not None:
            print("----------------Feature is not None, Attention ------------")
            g.ndata["id"] = self.features
        node_id = g.ndata["id"].squeeze()
        g.ndata["h"] = init_ent_emb[node_id]
        if self.skip_connect:
            prev_h = []
            for layer in self.layers:
                prev_h = layer(g, prev_h)
        else:
            for layer in self.layers:
                layer(g, [])
        return g.ndata.pop("h")


class RecurrentRGCN(nn.Module):
    def __init__(
        self,
        decoder_name,
        encoder_name,
        num_ents,
        num_rels,
        num_static_rels,
        num_words,
        h_dim,
        opn,
        sequence_len,
        num_bases=-1,
        num_basis=-1,
        num_hidden_layers=1,
        dropout=0,
        self_loop=False,
        skip_connect=False,
        layer_norm=False,
        input_dropout=0,
        hidden_dropout=0,
        feat_dropout=0,
        aggregation="cat",
        weight=1,
        discount=0,
        angle=0,
        use_static=False,
        entity_prediction=False,
        relation_prediction=False,
        use_cuda=False,
        gpu=0,
        analysis=False,
        use_history_gate=False,
        hva_topk=256,
        hva_mode="dual_branch",
        hva_gamma_exact=0.005,
        hva_gamma_near=0.08,
        hva_stale_init=0.2,
    ):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.use_history_gate = use_history_gate
        self.hva_topk = hva_topk
        self.hva_mode = hva_mode
        self.hva_gamma_exact = hva_gamma_exact
        self.hva_gamma_near = hva_gamma_near
        self.hva_stale_init = hva_stale_init

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(
                self.h_dim,
                self.h_dim,
                self.num_static_rels * 2,
                num_bases,
                activation=F.rrelu,
                dropout=dropout,
                self_loop=False,
                skip_connect=False,
            )
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(
            num_ents,
            h_dim,
            h_dim,
            num_rels * 2,
            num_bases,
            num_basis,
            num_hidden_layers,
            dropout,
            self_loop,
            skip_connect,
            encoder_name,
            self.opn,
            self.emb_rel,
            use_cuda,
            analysis,
        )

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain("relu"))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

        if self.use_history_gate:
            self.history_validity_adapter = HistoryValidityAdapter(
                num_relations=self.num_rels * 2,
                mode=self.hva_mode,
                gamma_exact=self.hva_gamma_exact,
                gamma_near=self.hva_gamma_near,
                stale_init=self.hva_stale_init,
            )
            print(
                f"[HVA] enabled | mode={self.hva_mode} | topk={self.hva_topk} "
                f"| gamma_exact={self.hva_gamma_exact} | gamma_near={self.hva_gamma_near} "
                f"| stale_init={self.hva_stale_init}"
            )
        else:
            self.history_validity_adapter = None

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata["h"] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop("h")[: self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = self.h.new_zeros((self.num_rels * 2, self.h_dim))

            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]: span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean

            x_input = torch.cat((self.emb_rel, x_input), dim=1)
            if i == 0:
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
            else:
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1 - time_weight) * self.h
            history_embs.append(self.h)

        return history_embs, static_emb, self.h_0, gate_list, degree_list

    def _apply_history_validity_adapter(
        self,
        entity_logits,
        all_triples,
        current_time,
        hva_histories,
    ):
        """Apply HVA without using the hidden target entity."""
        if (
            (not self.use_history_gate)
            or (self.history_validity_adapter is None)
            or (hva_histories is None)
            or (current_time is None)
        ):
            return entity_logits

        rel_ids = all_triples[:, 1]

        # Candidate selection and history lookup are discrete operations.
        # Keep only these operations under no_grad. The learnable adapter
        # remains outside so it receives gradients during training.
        with torch.no_grad():
            candidate_ids = build_topk_candidate_ids(
                entity_logits,
                self.hva_topk,
            )

            time_col = torch.full(
                (all_triples.size(0), 1),
                int(current_time),
                dtype=torch.long,
                device=all_triples.device,
            )
            query_triples = torch.cat(
                [all_triples, time_col],
                dim=1,
            )

            # Defence in depth: the true target object must not be available
            # to history feature construction.
            query_triples = query_triples.clone()
            query_triples[:, 2] = -1

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
                query_triples=query_triples,
                candidate_ids=candidate_ids,
                sr_hist=hva_histories["sr"],
                so_hist=hva_histories["so"],
                ro_hist=hva_histories["ro"],
                device=entity_logits.device,
                mode=self.hva_mode,
            )

        base_scores_topk = torch.gather(
            entity_logits,
            dim=1,
            index=candidate_ids,
        )

        adjusted_topk_scores, _ = self.history_validity_adapter(
            base_scores_topk,
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

        return scatter_topk_back(
            entity_logits,
            candidate_ids,
            adjusted_topk_scores,
        )


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda, current_time=None, hva_histories=None):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score = self._apply_history_validity_adapter(score, all_triples, current_time, hva_histories)

            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel

    def get_loss(self, glist, triples, static_graph, use_cuda, current_time=None, hva_histories=None):
        device = torch.device(f"cuda:{self.gpu}") if use_cuda else triples.device
        loss_ent = torch.zeros(1, device=device)
        loss_rel = torch.zeros(1, device=device)
        loss_static = torch.zeros(1, device=device)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        if use_cuda:
            all_triples = all_triples.cuda(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            scores_ob = self._apply_history_validity_adapter(scores_ob, all_triples, current_time, hva_histories)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for _, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_static