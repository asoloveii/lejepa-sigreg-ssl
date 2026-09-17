import torch, torch.nn as nn


class SIGReg(nn.Module):

    def __init__(self, n_proj=256, knots=17):
        super().__init__()
        self.n_proj = n_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt 
        window = torch.exp(-t.square()/2)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projs: torch.Tensor):
        A = torch.randn((projs.size(-1), self.n_proj), device=projs.device) 
        A.div_(A.norm(p=2, dim=0))
        x_t = (projs @ A).unsqueeze_(-1) @ self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * projs.size(-2)
        return statistic.mean()

class Encoder(nn.Module):

    def __init__(self, backbone, dim, proj_dim):
        super().__init__()

        self.backbone = backbone 
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(dim * 4),
            nn.Linear(dim*4, proj_dim, bias=False)
        )

    def forward(self, x: torch.Tensor):
        N, V = x.shape[:2]
        embs = self.backbone(x.flatten(0, 1))
        return embs, self.proj(embs).reshape(N, V, -1).transpose(0, 1)
