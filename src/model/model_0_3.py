# model_dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class QueryPathDQN(nn.Module):
    """
    DQN baseline for KG path reasoning.

    Q(s, a) is computed by scoring (query, current_node, candidate_node)
    """

    def __init__(
        self,
        encoder: str = "bert",
        in_dim: int = 768,
        hidden_dim: int = 256,
        num_hops: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.encoder_name = encoder
        self.in_dim = in_dim
        self.num_hops = num_hops

        # Q-network: outputs scalar Q for (s, a)
        self.q_net = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Encoders
        self._sbert = None
        self._bert = None
        self._lazy_init_encoders()

        self.to(device)

    # ---------------- Encoders ----------------
    def _lazy_init_encoders(self):
        if self.encoder_name == "sbert" and self._sbert is None:
            self._sbert = SentenceTransformer("all-mpnet-base-v2")
        if self.encoder_name == "bert" and self._bert is None:
            self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self._bert = BertModel.from_pretrained("bert-base-uncased")

    def get_embedding(self, item: Any) -> torch.Tensor:
        if isinstance(item, torch.Tensor):
            return item.to(self.device)
        if self.encoder_name == "sbert":
            return self._sbert.encode(item, convert_to_tensor=True).to(self.device)
        if self.encoder_name == "bert":
            inputs = self._bert_tokenizer(item, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._bert(**inputs)
            return out.last_hidden_state[:, 0, :].squeeze(0)
        raise NotImplementedError

    # ---------------- Q computation ----------------
    def q_values(
        self,
        q: torch.Tensor,
        h_t: torch.Tensor,
        cand_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        q: (D,)
        h_t: (D,)
        cand_h: (N, D)
        returns Q-values: (N,)
        """
        N = cand_h.size(0)
        q_expand = q.unsqueeze(0).expand(N, -1)
        h_expand = h_t.unsqueeze(0).expand(N, -1)
        inp = torch.cat([q_expand, h_expand, cand_h], dim=1)
        return self.q_net(inp).squeeze(-1)

    # ---------------- Episode rollout (ε-greedy) ----------------
    @torch.no_grad()
    def run_episode(
        self,
        start_idx: int,
        question: Any,
        adj: torch.Tensor,
        rgat_nodes: torch.Tensor,
        target_idx: int,
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:

        q = self.get_embedding(question)
        cur_idx = start_idx
        h_t = rgat_nodes[cur_idx].to(self.device)

        transitions = []
        idx_list: List[int] = []   # ← ADDED
        success = False

        for t in range(self.num_hops):
            neighbors = torch.nonzero(adj[cur_idx], as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                break

            cand_indices = neighbors.tolist()
            cand_h = rgat_nodes[cand_indices].to(self.device)

            q_vals = self.q_values(q, h_t, cand_h)

            # ε-greedy
            if torch.rand(1).item() < epsilon:
                action_idx = torch.randint(len(cand_indices), (1,)).item()
            else:
                action_idx = torch.argmax(q_vals).item()

            next_idx = cand_indices[action_idx]

            reward = 1.0 if next_idx == target_idx else 0.0
            done = reward > 0

            next_h = rgat_nodes[next_idx].detach()

            neighbors_next = torch.nonzero(adj[next_idx], as_tuple=False).squeeze(-1)
            cand_h_next = (
                rgat_nodes[neighbors_next.tolist()].detach()
                if neighbors_next.numel() > 0
                else None
            )

            transitions.append({
                "q": q.detach(),
                "h_t": h_t.detach(),
                "cand_h": cand_h.detach(),
                "action": action_idx,
                "reward": reward,
                "next_h": next_h,
                "cand_h_next": cand_h_next,
                "done": done,
            })

            idx_list.append(next_idx)   # ← ADDED

            if done:
                success = True
                break

            cur_idx = next_idx
            h_t = rgat_nodes[cur_idx].to(self.device)

        return {
            "transitions": transitions,
            "idx_list": idx_list,     # ← ADDED
            "start_idx": start_idx,   # ← ADDED
            "success": success,
        }
