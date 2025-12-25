# baseline_rl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class QueryPathRLPPO(nn.Module):
    """
    Simple (non-hierarchical) RL baseline for KG path reasoning.
    - Single policy over neighbors
    - Sparse terminal reward
    - REINFORCE + value baseline
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

        # Policy network: scores candidate neighbors
        self.policy = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value baseline
        self.value = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
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

    # ---------------- Core logic ----------------
    def score_actions(
        self,
        q: torch.Tensor,
        h_t: torch.Tensor,
        cand_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        q: (D,)
        h_t: (D,)
        cand_h: (N, D)
        returns logits: (N,)
        """
        N = cand_h.size(0)
        q_expand = q.unsqueeze(0).expand(N, -1)
        h_expand = h_t.unsqueeze(0).expand(N, -1)
        inp = torch.cat([q_expand, h_expand, cand_h], dim=1)
        return self.policy(inp).squeeze(-1)

    def run_episode(
        self,
        start_idx: int,
        question: Any,
        adj: torch.Tensor,
        rgat_nodes: torch.Tensor,
        target_idx: int,
        deterministic: bool = False,
    ) -> Dict[str, Any]:

        q = self.get_embedding(question)
        cur_idx = start_idx
        h_t = rgat_nodes[cur_idx].to(self.device)

        logps, values, rewards, idx_list = [], [], [], []
        success = False

        for t in range(self.num_hops):
            neighbors = torch.nonzero(adj[cur_idx], as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                break

            cand_indices = neighbors.tolist()
            cand_h = rgat_nodes[cand_indices].to(self.device)

            logits = self.score_actions(q, h_t, cand_h)
            probs = F.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)

            if deterministic:
                action = torch.argmax(probs)
            else:
                action = dist.sample()

            logp = dist.log_prob(action)
            next_idx = cand_indices[int(action.item())]

            # value baseline
            v = self.value(torch.cat([q, h_t], dim=0))

            logps.append(logp)
            values.append(v.squeeze(0))
            idx_list.append(next_idx)

            # terminal reward
            if next_idx == target_idx:
                rewards.append(1.0)
                success = True
                break
            else:
                rewards.append(0.0)

            cur_idx = next_idx
            h_t = rgat_nodes[cur_idx].to(self.device)

        return {
            "logps_old": logps,
            "values": values,
            "rewards": rewards,
            "idx_list": idx_list,
            "success": success,
        }
