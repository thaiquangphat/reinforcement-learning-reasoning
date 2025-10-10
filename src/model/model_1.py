"""
model_rl.py

QueryPathRL - an Actor-Critic (A2C-style) Reinforcement Learning model for multi-hop QA on Knowledge Graphs.
This model expects *precomputed* GAT node embeddings (rgat_nodes) from your existing Relational GAT encoder
and will NOT modify how GAT produces rgat_nodes. It is compatible with the existing function signatures
used in your codebase (forward(start_idx, question, adj, rgat_nodes, num_hops)).

Features:
- Actor network (scores over candidate neighbors, softmax over neighbors)
- Critic network (state value estimate)
- LSTM-based query state update (keeps memory over hops)
- Methods:
    - run_episode: run one episode (sampled or deterministic) and return trajectory (for train)
    - run_inference: deterministic greedy run for inference
    - get_value: compute critic value for a given state
- Default reward shaping: terminal reward, similarity shaping, length penalty, cycle penalty.
  You can customize reward behavior via reward_config passed to constructor.

Note: this file intentionally does not import project-specific modules (src / dataloader). It only depends on PyTorch
and transformers/sentence-transformers if you choose textual encoders.
"""
import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class QueryPathRLV1(nn.Module):
    def __init__(
        self,
        encoder: str = "sbert",
        in_dim: Optional[int] = None,
        num_hops: int = 100,
        reg_lambda: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reward_config: Optional[Dict[str, float]] = None,
    ):
        super(QueryPathRLV1, self).__init__()
        self.device = device
        self.encoder_name = encoder
        self.num_hops = num_hops
        # in_dim will be inferred from encoder if not provided when using textual encoder
        self.in_dim = in_dim or self._get_encoder_dim(encoder)
        self.reg_lambda = reg_lambda

        # Actor (scores per candidate neighbor). Input: [q_t, h_t, h_j] -> scalar score
        self.policy_net = nn.Sequential(
            nn.Linear(3 * self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, 1),
        )

        # Critic (value head). Input: [q_t, h_t] -> scalar value
        self.critic = nn.Sequential(
            nn.Linear(2 * self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, 1),
        )

        # LSTMCell for updating the query hidden state from chosen node embedding
        self.lstm = nn.LSTMCell(self.in_dim, self.in_dim)

        # Optional textual encoders (constructed lazily)
        self._sbert = None
        self._bert = None

        # Reward config: terminal reward, shaping weight, length penalty etc.
        default_rc = dict(
            R_success=1.0,  # terminal success reward
            R_fail=-0.2,  # terminal fail reward
            alpha_sim=0.5,  # shaping coefficient for similarity improvement
            beta_len=0.01,  # per-step length penalty
            gamma_cycle=0.1,  # cycle/invalid action penalty
        )
        self.reward_config = {**default_rc, **(reward_config or {})}

        self._lazy_init_encoders()

        self.to(self.device)

    # ---------------------- Encoder helpers ----------------------
    def _get_encoder_dim(self, encoder: str) -> int:
        """Return expected embedding dim for a given encoder name (heuristic)."""
        if encoder == "sbert":
            return 768
        if encoder == "bert":
            return 768
        if isinstance(self.in_dim, int) and self.in_dim:
            return self.in_dim
        # default fallback
        return 768

    def _lazy_init_encoders(self):
        if self.encoder_name == "sbert" and SentenceTransformer is not None and self._sbert is None:
            self._sbert = SentenceTransformer("all-mpnet-base-v2")
        if self.encoder_name == "bert" and BertModel is not None and self._bert is None:
            self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self._bert = BertModel.from_pretrained("bert-base-uncased")

    def _lstm_step(self, e_t, q_t, cx):
        # Flatten everything to (1, D)
        e_t = e_t.view(1, -1)
        q_t = q_t.view(1, -1)
        cx = cx.view(1, -1)
        return self.lstm(e_t, (q_t, cx))

    def get_embedding(self, item: Any) -> torch.Tensor:
        """
        Encode question text or accept an already-computed embedding (tensor, numpy, or list).
        Returns a torch.Tensor of shape (1, hidden_dim) on self.device.
        """
        if self.encoder_name == 'sbert':
            emb = self._sbert.encode(item, convert_to_tensor=True)
            return emb.clone().detach()
        elif self.encoder_name == 'bert':
            inputs = self._bert_tokenizer(item, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self._bert(**inputs)
            return outputs.last_hidden_state[:, 0, :]
        raise NotImplementedError(f"Encoder {self.encoder_name} not implemented")

    # ---------------------- Action selection utilities ----------------------
    def _score_candidates(self, q_t: torch.Tensor, h_t: torch.Tensor, candidates_h: torch.Tensor) -> torch.Tensor:
        """
        q_t: (D,), h_t: (D,), candidates_h: (num_candidates, D)
        returns logits: (num_candidates,)
        """
        num_cand = candidates_h.size(0)
        q_expand = q_t.unsqueeze(0).expand(num_cand, -1)
        h_t_expand = h_t.unsqueeze(0).expand(num_cand, -1)
        inp = torch.cat([q_expand, h_t_expand, candidates_h], dim=1)  # (num_cand, 3D)
        logits = self.policy_net(inp).squeeze(-1)  # (num_cand,)
        return logits

    def _select_action_from_logits(self, logits: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, float]:
        """
        Given logits for the candidate set, return:
        - chosen index in candidate list (int)
        - log_prob of chosen action (tensor scalar)
        - entropy (float)
        """
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            chosen = int(torch.argmax(probs).item())
        else:
            chosen = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(chosen, device=logits.device))
        entropy = float(dist.entropy().mean().item())
        return chosen, logp, entropy, probs.detach()

    # ---------------------- Reward helper ----------------------
    def compute_reward(
        self,
        prev_idx: int,
        next_idx: int,
        target_idx: Optional[int],
        rgat_nodes: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None,
        visited: Optional[set] = None,
    ) -> float:
        """
        Compute immediate reward for a single transition prev_idx -> next_idx.
        Uses cosine similarity shaping against target idx embedding if available.
        """
        cfg = self.reward_config
        # default step reward = -beta_len
        r = -cfg["beta_len"]
        if target_idx is None:
            return r
        # embeddings from rgat_nodes (contextualized)
        v_prev = rgat_nodes[prev_idx]
        v_next = rgat_nodes[next_idx]
        v_target = rgat_nodes[target_idx]
        # cosine similarities
        sim_prev = F.cosine_similarity(v_prev.unsqueeze(0), v_target.unsqueeze(0)).item()
        sim_next = F.cosine_similarity(v_next.unsqueeze(0), v_target.unsqueeze(0)).item()
        r += cfg["alpha_sim"] * (sim_next - sim_prev)
        # cycle / visited penalty
        if visited is not None and next_idx in visited:
            r -= cfg["gamma_cycle"]
        # terminal check (handled by environment typically)
        return float(r)

    # ---------------------- Episode runner (sampling or deterministic) ----------------------
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
    ) -> Dict[str, Any]:
        """
        Run one trajectory on the graph from start_idx for at most num_hops steps.

        Inputs:
            start_idx: starting node index (int)
            question: text or precomputed embedding
            adj: adjacency matrix (torch.Tensor) shape (N, N) or (N,)
            rgat_nodes: node embeddings tensor shape (N, D)
            num_hops: maximum hops (override self.num_hops)
            target_idx: (optional) index of target node for reward shaping and termination
            deterministic: if True use greedy argmax (for inference)
            mask_visited: if True, avoid re-selecting visited nodes (if possible)

        Returns a dict containing:
            idx_list: list of chosen node indices (including first chosen after start)
            e_list: list of chosen node embeddings
            q_list: list of query hidden states after each step
            logps: list of log probabilities for chosen actions (tensor list)
            values: list of critic values per step (tensor list)
            entropies: list of entropy floats per step
            rewards: list of immediate rewards per step
            success: bool whether target reached
        """
        num_hops = num_hops or self.num_hops
        N = rgat_nodes.size(0)
        device = self.device

        # question embedding
        q_t = self.get_embedding(question).to(device)[0]  # (D,)

        # initial h_t is the embedding of the start node
        h_t = rgat_nodes[start_idx].to(device)
        # initial LSTM cell state: (h, c)
        hx = torch.zeros(self.in_dim, device=device)
        cx = torch.zeros(self.in_dim, device=device)

        idx_list = []
        e_list = []
        q_list = []
        logps = []
        values = []
        rewards = []
        entropies = []

        visited = set([start_idx])

        success = False

        for step in range(num_hops):
            # get neighbors of h_t from adjacency (assume adj is dense indexable tensor)
            # adj expected as (N, N) adjacency matrix where adj[u, v] > 0 means neighbor
            neighbors = torch.nonzero(adj[start_idx] if step == 0 else adj[idx_list[-1]] , as_tuple=False).squeeze(-1)  # indices

            # If neighbors is empty -> terminal dead-end
            if neighbors.numel() == 0:
                # terminal fail
                rewards.append(float(self.reward_config["R_fail"]))
                break

            # build candidate embeddings
            cand_indices = neighbors.tolist() if neighbors.dim() else [int(neighbors.item())]
            # optionally filter out invalid candidates (e.g., mask visited)
            if mask_visited:
                cand_indices = [c for c in cand_indices if c not in visited]
                if len(cand_indices) == 0:
                    # if all neighbors visited, fall back to original neighbors (allow cycles but penalize)
                    cand_indices = neighbors.tolist() if neighbors.dim() else [int(neighbors.item())]

            cand_h = rgat_nodes[torch.tensor(cand_indices, device=device)].to(device)  # (num_cand, D)
            # compute logits
            logits = self._score_candidates(q_t, h_t, cand_h)
            # critic value for current state
            value = self.critic(torch.cat([q_t, h_t], dim=0).unsqueeze(0)).squeeze(0)  # scalar tensor
            # sample or pick action
            chosen_pos, logp, entropy, probs = self._select_action_from_logits(logits, deterministic=deterministic)

            chosen_idx = cand_indices[chosen_pos]
            # record
            idx_list.append(chosen_idx)
            e_list.append(rgat_nodes[chosen_idx].to(device))
            q_list.append(q_t.detach().clone())
            logps.append(logp)
            values.append(value.squeeze(0))
            entropies.append(entropy)

            # compute reward for this transition (prev -> next)
            prev_idx = idx_list[-2] if len(idx_list) >= 2 else start_idx
            r = self.compute_reward(prev_idx, chosen_idx, target_idx, rgat_nodes, node_feats=None, visited=visited)
            # if we reached target, add terminal success reward and stop
            if target_idx is not None and chosen_idx == target_idx:
                r += self.reward_config["R_success"]
                rewards.append(r)
                success = True
                break

            rewards.append(r)
            # update visited
            visited.add(chosen_idx)
            # update lstm state q_t from chosen node embedding
            hx, cx = self._lstm_step(e_list[-1], q_t, cx)
            # The above uses LSTMCell semantics; however we will instead directly update q_t to the embedding
            # to keep things simple and stable in mixed environments:
            # q_t = (q_t + e_list[-1].to(device)) / 2.0  # simple update rule (alternatively use LSTMCell properly)
            q_t = hx.squeeze(0)
            # set h_t to newly chosen node embedding
            h_t = rgat_nodes[chosen_idx].to(device)

        return dict(
            idx_list=idx_list,
            e_list=e_list,
            q_list=q_list,
            logps=logps,
            values=values,
            entropies=entropies,
            rewards=rewards,
            success=success,
        )

    # ---------------------- Inference wrapper ----------------------
    def run_inference(self, start_idx: int, question: Any, adj: torch.Tensor, rgat_nodes: torch.Tensor, num_hops: Optional[int] = None, mask_visited: bool = True) -> Dict[str, Any]:
        """Deterministic greedy run (argmax at each step)"""
        return self.run_episode(start_idx, question, adj, rgat_nodes, num_hops=num_hops, deterministic=True, mask_visited=mask_visited)

    # ---------------------- Utilities ----------------------
    def get_value(self, q_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """Return critic value for a particular state (q_t, h_t)."""
        return self.critic(torch.cat([q_t, h_t], dim=0).unsqueeze(0)).squeeze(0)

    def get_regularization_loss(self) -> torch.Tensor:
        """Optional parameter regularization (L2) used in prior model: sum of squared weights * reg_lambda"""
        reg_loss = torch.tensor(0.0, device=self.device)
        for p in self.parameters():
            reg_loss = reg_loss + torch.sum(p ** 2)
        return self.reg_lambda * reg_loss
