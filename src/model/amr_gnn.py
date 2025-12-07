import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------ Simple AMR GNN ------------------------
class AMRGNN(nn.Module):
    """
    Two-layer GCN-style network:
      X' = D^-0.5 (A+I) D^-0.5 X W
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super(AMRGNN, self).__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        adj: (N, N) adjacency (0/1 floats)
        returns: (N, out_dim)
        """
        device = x.device
        N = adj.size(0)
        I = torch.eye(N, device=device)
        A_hat = adj.to(device) + I
        deg = torch.sum(A_hat, dim=1)  # (N,)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # (N,N)

        h = norm @ x
        h = self.lin1(h)
        h = F.relu(h)
        h = norm @ h
        h = self.lin2(h)
        return h
