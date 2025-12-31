import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class QueryPathLSTMAC(nn.Module):
    """
    LSTM Actor-Critic (GAE-compatible)
    """

    def __init__(
        self,
        encoder: str = "bert",
        in_dim: int = 768,
        hidden_dim: int = 256,
        lstm_dim: int = 256,
        num_hops: int = 20,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.encoder_name = encoder
        self.in_dim = in_dim
        self.num_hops = num_hops

        # ----------- LSTM memory -----------
        self.lstm = nn.LSTMCell(
            input_size=2 * in_dim,   # [q, h_t]
            hidden_size=lstm_dim,
        )

        # ----------- Policy head -----------
        self.policy = nn.Sequential(
            nn.Linear(lstm_dim + in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ----------- Value head -----------
        self.value = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ----------- Encoders -----------
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
    def score_actions(self, lstm_h, cand_h):
        N = cand_h.size(0)
        h_expand = lstm_h.unsqueeze(0).expand(N, -1)
        x = torch.cat([h_expand, cand_h], dim=1)
        return self.policy(x).squeeze(-1)

    def run_episode(
        self,
        start_idx,
        question,
        adj,
        rgat_nodes,
        target_idx,
        deterministic=False,
    ) -> Dict[str, Any]:

        q = self.get_embedding(question)
        cur_idx = start_idx
        h_t = rgat_nodes[cur_idx].to(self.device)

        hx = torch.zeros(1, self.lstm.hidden_size, device=self.device)
        cx = torch.zeros(1, self.lstm.hidden_size, device=self.device)

        logps, values, rewards = [], [], []
        success = False

        for _ in range(self.num_hops):
            neighbors = torch.nonzero(adj[cur_idx], as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                break

            cand_idx = neighbors.tolist()
            cand_h = rgat_nodes[cand_idx].to(self.device)

            lstm_in = torch.cat([q, h_t], dim=0).unsqueeze(0)
            hx, cx = self.lstm(lstm_in, (hx, cx))
            lstm_h = hx.squeeze(0)

            logits = self.score_actions(lstm_h, cand_h)
            probs = F.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)

            action = torch.argmax(probs) if deterministic else dist.sample()
            logp = dist.log_prob(action)

            next_idx = cand_idx[int(action.item())]
            v = self.value(lstm_h)

            logps.append(logp)
            values.append(v.squeeze(0))

            if next_idx == target_idx:
                rewards.append(1.0)
                success = True
                break
            else:
                rewards.append(0.0)

            cur_idx = next_idx
            h_t = rgat_nodes[cur_idx].to(self.device)

        return {
            "logps": logps,
            "values": values,
            "rewards": rewards,
            "success": success,
        }
