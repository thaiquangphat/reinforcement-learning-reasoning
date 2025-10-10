import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalGATHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RelationalGATHead, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.W_skip = nn.Linear(in_dim, out_dim, bias=False)

        self.a = nn.Parameter(torch.empty(out_dim * 3))
        nn.init.xavier_uniform_(self.a.view(1, -1))

        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_skip.weight)

        self.dropout = nn.Dropout(0.02)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, adj, rel_feat, features):
        """
        adj:      (B, N, N)
        rel_feat: (B, N, N, in_dim)
        features: (B, N, in_dim)
        Returns:  (B, N, out_dim)
        """
        B, N, _ = features.shape

        K = self.W_k(features)               # (B, N, out)
        Q = self.W_q(rel_feat)              # (B, N, N, out)
        V = self.W_v(features)              # (B, N, out)

        K_i = K.unsqueeze(2).expand(-1, -1, N, -1)
        V_j = V.unsqueeze(1).expand(-1, N, -1, -1)

        attn_input = torch.cat([Q, K_i, V_j], dim=-1)  # (B, N, N, 3*out)
        e_ij = self.leakyrelu(torch.sum(self.a * attn_input, dim=-1) / (self.out_dim ** 0.5))

        e_ij = e_ij.masked_fill(adj == 0, -1e9)
        alpha = F.softmax(e_ij, dim=-1)
        alpha = self.dropout(alpha)

        h_prime = torch.bmm(alpha, V)  # (B, N, out_dim)
        skip = self.W_skip(features)   # (B, N, out_dim)
        return h_prime + skip


class RelationalGATV3(nn.Module):
    def __init__(self, in_dim=384, num_layers=3, num_heads=4):
        super(RelationalGATV3, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        assert self.out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            heads = nn.ModuleList([RelationalGATHead(in_dim, self.head_dim) for _ in range(num_heads)])
            self.layers.append(nn.ModuleDict({
                'heads': heads,
                'proj': nn.Linear(self.out_dim, self.out_dim),
                'norm': nn.LayerNorm(self.out_dim)
            }))
            in_dim = self.out_dim  # for next layer

        self.activation = nn.ReLU()

    def forward(self, adj, rel_feat, features):
        h = features  # (B, N, in_dim)

        for layer in self.layers:
            head_outputs = [head(adj, rel_feat, h) for head in layer['heads']]  # List of (B, N, head_dim)
            h_cat = torch.cat(head_outputs, dim=-1)  # (B, N, out_dim)
            h_proj = layer['proj'](h_cat)            # Linear projection
            h = layer['norm'](h_proj)                # LayerNorm
            h = self.activation(h)

        return h  # (B, N, out_dim)