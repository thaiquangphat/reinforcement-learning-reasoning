import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from src.model.amr_graph import *
from src.model.amr_gnn import AMRGNN


class HierarchicalQueryPathRLV2(nn.Module):
    """
    Hierarchical RL model:
    - Manager: low-frequency policy outputs subgoal prototype index in [0..L-1].
    - Worker: high-frequency policy selects neighbor actions conditioned on subgoal embedding.
    - Path encoder: GRU that consumes visited node embeddings and provides path context p_t.
    """

    def __init__(
        self,
        encoder: str = "sbert",
        in_dim: Optional[int] = None,
        num_hops: int = 100,
        manager_horizon: int = 4,
        num_prototypes: int = 128,
        path_hidden_dim: Optional[int] = None,
        reg_lambda: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reward_config: Optional[Dict[str, float]] = None,
    ):
        super(HierarchicalQueryPathRLV2, self).__init__()
        self.device = device
        self.encoder_name = encoder
        self.num_hops = num_hops
        self.manager_horizon = manager_horizon  # K steps per manager decision
        self.num_prototypes = num_prototypes
        self.in_dim = in_dim or self._get_encoder_dim(encoder)
        self.path_hidden_dim = path_hidden_dim or self.in_dim
        self.reg_lambda = reg_lambda

        # small 2-layer GCN: maps AMR concept embeddings -> same dim as in_dim
        self.amr_gnn = AMRGNN(self.in_dim, self.in_dim, self.in_dim)

        # ---------- Path encoder (GRU) ----------
        # consumes node embeddings sequentially and yields path context p_t
        self.path_gru = nn.GRUCell(self.in_dim, self.path_hidden_dim)

        # ---------- Query updater LSTM (keeps query hidden like old model) ----------
        self.lstm = nn.LSTMCell(self.in_dim, self.in_dim)

        # ---------- Prototype subgoals (learned embeddings) ----------
        self.prototypes = nn.Embedding(self.num_prototypes, self.in_dim)

        # ---------- Manager policy (over prototypes) + value ----------
        # input: [q_t, h_t, p_t, v_target] -> logits over prototypes
        mgr_in_dim = 4 * self.in_dim
        self.manager_policy = nn.Sequential(
            nn.Linear(mgr_in_dim, self.in_dim),
            nn.ReLU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.num_prototypes),
        )
        self.manager_value = nn.Sequential(
            nn.Linear(mgr_in_dim, self.in_dim),
            nn.ReLU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, 1),
        )

        # ---------- Worker policy (scores neighbors conditioned on goal) + value ----------
        # For candidate j we compute score = MLP([w_t ; v_j ; w_t * v_j]) where w_t = [q_t; p_t; h_t; g_m]
        worker_ctx_dim = 4 * self.in_dim
        self.worker_inner = nn.Sequential(
            nn.Linear(3 * self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, 1),
        )
        # worker value head: input [q_t; p_t; h_t; g_m]
        self.worker_value = nn.Sequential(
            nn.Linear(worker_ctx_dim, self.in_dim),
            nn.ReLU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, 1),
        )

        self.worker_ctx_proj = nn.Linear(4 * self.in_dim, self.in_dim)

        # ---------- Auxiliary heads (optional) ----------
        # distance regressor: predict cosine-distance-to-target
        self.aux_dist = nn.Sequential(
            nn.Linear(3 * self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, 1),
        )

        # Reward config
        default_rc = dict(
            R_success=4.0,
            R_fail=-0.5,
            alpha_sim=0.5,
            beta_len=0.01,
            gamma_cycle=0.2,
            alpha_int=1.0,  # intrinsic scaling for worker
            R_reach_subgoal=1.0,  # bonus when subgoal reached
        )
        self.reward_config = {**default_rc, **(reward_config or {})}

        # Textual encoders (lazily loaded)
        self._sbert = None
        self._bert = None
        self._lazy_init_encoders()

        self.to(self.device)

    # ----------------------------- Encoder helpers -----------------------------
    def _get_encoder_dim(self, encoder: str) -> int:
        if encoder == "sbert":
            return 768
        if encoder == "bert":
            return 768
        if isinstance(self.in_dim, int) and self.in_dim:
            return self.in_dim
        return 768

    def _lazy_init_encoders(self):
        if self.encoder_name == "sbert" and SentenceTransformer is not None and self._sbert is None:
            self._sbert = SentenceTransformer("all-mpnet-base-v2")
        if self.encoder_name == "bert" and BertModel is not None and self._bert is None:
            self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self._bert = BertModel.from_pretrained("bert-base-uncased")

    def get_embedding(self, item: Any) -> torch.Tensor:
        """Return (1, D) tensor on device representing question embedding or accept precomputed tensor."""
        if isinstance(item, torch.Tensor):
            return item.detach().to(self.device).unsqueeze(0) if item.dim() == 1 else item.to(self.device)
        if self.encoder_name == "sbert":
            emb = self._sbert.encode(item, convert_to_tensor=True)
            return emb.clone().detach().to(self.device)
        elif self.encoder_name == "bert":
            inputs = self._bert_tokenizer(item, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].to(self.device)
        raise NotImplementedError(f"Encoder {self.encoder_name} not implemented")

    # ----------------------------- Utilities -----------------------------
    def _score_worker_candidate(self, ctx: torch.Tensor, cand_h: torch.Tensor) -> torch.Tensor:
        """
        ctx: (C,) worker context vector (w_t flattened)
        cand_h: (num_cand, D)
        returns logits: (num_cand,)
        """
        num_cand = cand_h.size(0)
        # Project context to same dim as candidate (D)
        ctx_proj = self.worker_ctx_proj(ctx)   # (D,)
        ctx_expand = ctx_proj.unsqueeze(0).expand(num_cand, -1)  # (num_cand, D)
        prod = ctx_expand * cand_h
        inp = torch.cat([ctx_expand, cand_h, prod], dim=1)  # (num_cand, 3*D)
        logits = self.worker_inner(inp).squeeze(-1)
        return logits

    def _select_action_from_logits(self, logits: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, float, torch.Tensor]:
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            chosen = int(torch.argmax(probs).item())
        else:
            chosen = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(chosen, device=logits.device))
        entropy = float(dist.entropy().mean().item())
        return chosen, logp, entropy, probs.detach()

    def cosine_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # returns scalar (1 - cosine_similarity) for vectors a,b with shape (D,)
        return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)

    # ----------------------------- Episode runner -----------------------------
    def run_episode(
        self,
        start_idx: int,
        question: Any,
        adj: torch.Tensor,
        rgat_nodes: torch.Tensor,
        num_hops: Optional[int] = None,
        target_idx: Optional[int] = None,
        deterministic: bool = False,
        mask_visited: bool = True,
        amr: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Hierarchical episode:
        - Manager picks prototype id every manager_horizon steps; worker acts up to manager_horizon steps.
        Returns trajectory for PPO:
            idx_list (visited indices),
            q_list (query hidden states),
            logps_worker, logps_manager,
            values_worker, values_manager,
            entropies_worker, entropies_manager,
            rewards (combined), rewards_worker_intrinsic,
            successes flags, etc.
        """
        num_hops = num_hops or self.num_hops
        device = self.device
        self._amr_manager_state = None

        # ----------------- AMR encoding (optional) -----------------
        amr_graph = None
        amr_node_emb = None
        amr_var2idx = None
        if amr is not None:
            # Accept raw AMR string, or (nodes, root) tuple, or AMRGraph-like dict
            amr_graph = AMRGraph(amr) if not isinstance(amr, AMRGraph) else amr
            amr_var2idx = amr_graph.var2idx
            # build input features for AMR nodes by encoding concept strings
            # self.get_embedding returns (1,D) for a concept; stack them
            feats = []
            for c in amr_graph.concepts:
                # strip quotes around literals if any
                cc = c.strip('"')
                emb = self.get_embedding(cc).to(device)  # shape (1, D)
                feats.append(emb)
            feats = torch.cat(feats, dim=0)  # (N, D)
            # compute AMR node embeddings via small GNN
            amr_node_emb = self.amr_gnn(feats, amr_graph.adj.to(device))  # (N, D)
            # keep for manager decisions below
            # convert adjacency to python-friendly neighbors list
            amr_neighbors = {i: torch.nonzero(amr_graph.adj[i], as_tuple=False).squeeze(-1).tolist()
                             for i in range(len(amr_graph.var_list))}
        else:
            amr_neighbors = None

        # embeddings
        q_t = self.get_embedding(question).to(device)[0]  # (D,)
        h_t = rgat_nodes[start_idx].to(device)
        # initialize LSTM and path hidden
        hx = torch.zeros(self.in_dim, device=device)
        cx = torch.zeros(self.in_dim, device=device)
        path_h = torch.zeros(self.path_hidden_dim, device=device)

        idx_list: List[int] = []
        q_list: List[torch.Tensor] = []
        logps_worker: List[torch.Tensor] = []
        values_worker: List[torch.Tensor] = []
        entropies_worker: List[float] = []
        logps_manager: List[torch.Tensor] = []
        values_manager: List[torch.Tensor] = []
        entropies_manager: List[float] = []
        rewards: List[float] = []
        rewards_int_worker: List[float] = []

        visited = set([start_idx])
        success = False

        t = 0
        current_proto_idx = None
        current_proto_emb = None
        while t < num_hops:
            # Manager decision if t % K == 0
            if t % self.manager_horizon == 0:
                if amr_node_emb is not None:
                    # ----- AMR-based manager: choose AMR node as subgoal -----
                    # first manager pick: choose node with highest cosine similarity
                    # to the graph node at start_idx; after that, pick random neighbor.
                    if (not hasattr(self, '_amr_manager_state')) or getattr(self, '_amr_manager_state', None) is None:
                        # initialize AMR manager state
                        self._amr_manager_state = {}
                        self._amr_manager_state['chosen_amr_idx'] = None
                        self._amr_manager_state['visits'] = 0

                    if self._amr_manager_state['chosen_amr_idx'] is None:
                        # compute cosine similarities between amr nodes and rgat_nodes[start_idx]
                        # rgat_nodes[start_idx] is (D,)
                        base = rgat_nodes[start_idx].to(device).unsqueeze(0)  # (1, D)
                        sims = F.cosine_similarity(amr_node_emb, base, dim=1)  # (N,)
                        chosen_amr_idx = int(torch.argmax(sims).item())
                    else:
                        prev = self._amr_manager_state['chosen_amr_idx']
                        neighs = amr_neighbors.get(prev, [])
                        neighs = [n for n in neighs if 0 <= n < amr_node_emb.size(0)]
                        if len(neighs) == 0:
                            neighs = [prev]
                            chosen_amr_idx = self._amr_manager_state['chosen_amr_idx']
                        else:
                            chosen_amr_idx = random.choice(neighs)

                    self._amr_manager_state['chosen_amr_idx'] = chosen_amr_idx
                    self._amr_manager_state['visits'] += 1

                    # set the current_proto_emb to the chosen AMR node embedding
                    current_proto_idx = None  # prototypes unused in AMR mode
                    # safety: ensure chosen_amr_idx is valid
                    if not (0 <= chosen_amr_idx < amr_node_emb.size(0)):
                        # fallback strategy: choose the closest valid index
                        chosen_amr_idx = max(0, min(chosen_amr_idx, amr_node_emb.size(0) - 1))

                    current_proto_emb = amr_node_emb[chosen_amr_idx].to(device)


                    # Record a "manager" logp/value/entropy proxy so the rest of training pipeline sees something.
                    # We set these to zero/placeholders because AMR manager is deterministic/randomized external logic.
                    logps_manager.append(torch.tensor(0.0, device=device))
                    entropies_manager.append(0.0)
                    values_manager.append(torch.tensor(0.0, device=device))
                else:
                    # ---------- original learned-prototype manager ----------
                    mgr_input = torch.cat([q_t, h_t, path_h, (rgat_nodes[target_idx] if target_idx is not None else torch.zeros_like(h_t))], dim=0)
                    mgr_logits = self.manager_policy(mgr_input)
                    mgr_probs = F.softmax(mgr_logits, dim=-1)
                    mgr_dist = torch.distributions.Categorical(mgr_probs)
                    if deterministic:
                        chosen_mgr = int(torch.argmax(mgr_probs).item())
                    else:
                        chosen_mgr = int(mgr_dist.sample().item())
                    mgr_logp = mgr_dist.log_prob(torch.tensor(chosen_mgr, device=device))
                    mgr_entropy = float(mgr_dist.entropy().mean().item())

                    # record manager action
                    logps_manager.append(mgr_logp)
                    entropies_manager.append(mgr_entropy)
                    values_manager.append(self.manager_value(mgr_input.unsqueeze(0)).squeeze(0))

                    current_proto_idx = chosen_mgr
                    current_proto_emb = self.prototypes.weight[chosen_mgr].to(device)

            # Worker step
            # get neighbors of current node
            neighbors = torch.nonzero(adj[start_idx] if (t == 0 and len(idx_list) == 0) else adj[idx_list[-1]], as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                # dead-end
                rewards.append(float(self.reward_config["R_fail"]))
                rewards_int_worker.append(0.0)
                break

            cand_indices = neighbors.tolist() if neighbors.dim() else [int(neighbors.item())]
            if mask_visited:
                filtered = [c for c in cand_indices if c not in visited]
                if len(filtered) > 0:
                    cand_indices = filtered

            cand_h = rgat_nodes[torch.tensor(cand_indices, device=device)].to(device)

            # build worker context w_t = [q_t; path_h; h_t; g_m]
            w_t = torch.cat([q_t, path_h, h_t, current_proto_emb], dim=0)
            logits = self._score_worker_candidate(w_t, cand_h)
            value_w = self.worker_value(w_t.unsqueeze(0)).squeeze(0)
            chosen_pos, logp_w, entropy_w, probs_w = self._select_action_from_logits(logits, deterministic=deterministic)
            chosen_idx = cand_indices[chosen_pos]

            # record worker transition (logp, value, entropy)
            logps_worker.append(logp_w)
            entropies_worker.append(entropy_w)
            values_worker.append(value_w)

            # compute rewards: extrinsic via similarity to target (like model_1), intrinsic via distance to subgoal
            prev_idx = idx_list[-1] if len(idx_list) >= 1 else start_idx
            extrinsic = self.compute_env_reward(prev_idx, chosen_idx, target_idx, rgat_nodes, visited=visited)
            # intrinsic: reduction in distance to current prototype (embedding distance)
            dist_prev = float(self.cosine_distance(rgat_nodes[prev_idx].to(device), current_proto_emb).item())
            dist_next = float(self.cosine_distance(rgat_nodes[chosen_idx].to(device), current_proto_emb).item())
            r_int = self.reward_config["alpha_int"] * (dist_prev - dist_next) - self.reward_config["beta_len"]
            # cycle penalty already included in extrinsic
            r = extrinsic + r_int

            # check if subgoal reached (within threshold)
            reached_subgoal = (dist_next <= 0.1)  # threshold can be tuned
            if reached_subgoal:
                r += self.reward_config["R_reach_subgoal"]

            # terminal check: reached final target
            if target_idx is not None and chosen_idx == target_idx:
                r += self.reward_config["R_success"]
                rewards.append(r)
                rewards_int_worker.append(r_int)
                idx_list.append(chosen_idx)
                q_list.append(q_t.detach().clone())
                visited.add(chosen_idx)
                # update path and query before breaking
                path_h = self.path_gru(rgat_nodes[chosen_idx].to(device), path_h)
                hx, cx = self._lstm_step(rgat_nodes[chosen_idx].to(device), q_t, cx)
                q_t = hx.squeeze(0)
                h_t = rgat_nodes[chosen_idx].to(device)
                success = True
                break

            # append transition normal
            rewards.append(r)
            rewards_int_worker.append(r_int)
            idx_list.append(chosen_idx)
            q_list.append(q_t.detach().clone())
            visited.add(chosen_idx)

            # update path hidden and query hidden
            path_h = self.path_gru(rgat_nodes[chosen_idx].to(device), path_h)
            hx, cx = self._lstm_step(rgat_nodes[chosen_idx].to(device), q_t, cx)
            q_t = hx.squeeze(0)
            h_t = rgat_nodes[chosen_idx].to(device)

            t += 1

        proto_choices = []
        if amr is not None and amr_graph is not None:
            # include the chosen amr var id / name in the returned info
            if hasattr(self, '_amr_manager_state') and self._amr_manager_state.get('chosen_amr_idx') is not None:
                idx = self._amr_manager_state['chosen_amr_idx']
                proto_choices = [amr_graph.var_list[idx]]
        else:
            proto_choices = [int(current_proto_idx)] if current_proto_idx is not None else []


        return dict(
            idx_list=idx_list,
            q_list=q_list,
            logps_worker=logps_worker,
            values_worker=values_worker,
            entropies_worker=entropies_worker,
            logps_manager=logps_manager,
            values_manager=values_manager,
            entropies_manager=entropies_manager,
            rewards=rewards,
            rewards_int_worker=rewards_int_worker,
            success=success,
            proto_choices=proto_choices,
            target_idx=target_idx
        )

    def run_inference(self, start_idx: int, question: Any, adj: torch.Tensor, rgat_nodes: torch.Tensor, num_hops: Optional[int] = None, mask_visited: bool = True) -> Dict[str, Any]:
        return self.run_episode(start_idx, question, adj, rgat_nodes, num_hops=num_hops, deterministic=True, mask_visited=mask_visited)

    # ----------------------------- Reward helper -----------------------------
    def compute_env_reward(self, prev_idx: int, next_idx: int, target_idx: Optional[int], rgat_nodes: torch.Tensor, node_feats: Optional[torch.Tensor] = None, visited: Optional[set] = None) -> float:
        cfg = self.reward_config
        r = -cfg["beta_len"]
        if target_idx is None:
            return r
        v_prev = rgat_nodes[prev_idx].to(self.device)
        v_next = rgat_nodes[next_idx].to(self.device)
        v_target = rgat_nodes[target_idx].to(self.device)
        sim_prev = F.cosine_similarity(v_prev.unsqueeze(0), v_target.unsqueeze(0)).item()
        sim_next = F.cosine_similarity(v_next.unsqueeze(0), v_target.unsqueeze(0)).item()
        r += cfg["alpha_sim"] * (sim_next - sim_prev)
        if visited is not None and next_idx in visited:
            r -= cfg["gamma_cycle"]
        return float(r)

    def _lstm_step(self, e_t, q_t, cx):
        e_t = e_t.view(1, -1)
        q_t = q_t.view(1, -1)
        cx = cx.view(1, -1)
        return self.lstm(e_t, (q_t, cx))

    # ----------------------------- Value helpers -----------------------------
    def get_worker_value(self, q_t: torch.Tensor, path_h: torch.Tensor, h_t: torch.Tensor, g_m: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([q_t, path_h, h_t, g_m], dim=0).unsqueeze(0)
        return self.worker_value(inp).squeeze(0)

    def get_manager_value(self, q_t: torch.Tensor, path_h: torch.Tensor, h_t: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([q_t, h_t, path_h, v_target], dim=0).unsqueeze(0)
        return self.manager_value(inp).squeeze(0)

    def get_regularization_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        for p in self.parameters():
            reg_loss = reg_loss + torch.sum(p ** 2)
        return self.reg_lambda * reg_loss
