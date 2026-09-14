import torch, torch.nn as nn, torch.nn.functional as F

from .module import Derf


def create_rope_embeddings(max_seq_len, head_dim, theta):
    inv_freqs = 1 / (theta ** torch.arange(0, head_dim, 2).float() / head_dim)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # i, j -> ij outer product
    freqs = torch.outer(t, inv_freqs)
    # repeat for cos and sin
    embs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return embs


def apply_rope(x, rope_embs):
    seq_len = x.shape[1]
    embs = rope_embs[:seq_len, :]

    embs = embs.unsqueeze(0).unsqueeze(2)
    cos, sin = embs.cos(), embs.sin()

    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    x_partner = x_partner.flatten(-2)

    return (x * cos + x_partner * sin).type_as(x)


class PatchEmbedding(nn.Module):
    """Patch Embeddng. Divides each image into patches and projects each onto
    an embedding dimension to later use for Vision Transformer."""

    def __init__(self, in_chans, patch_size, embed_dim):
        super().__init__()

        self.patch_size = patch_size 
        self.embed_dim = embed_dim 

        self.patchify = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patchify(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  
        return x


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit. Two gates projects each embedding onto a 
    higher dimensional space with one gate also applying silu activation
    (x * sigmoid(x)). The element-wise product result between those two gates are
     then projected back onto embed_dim dimension.""" 

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        self.content_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.silu_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.out_proj(self.content_proj(x) * F.silu(self.silu_proj(x)))


class Attention(nn.Module):
    """Multi-Head Self-Attention.""" 

    def __init__(self, embed_dim, n_heads, out_dropout=0.2, qkv_dropout=0.2, qkv_bias=False):
        super().__init__()

        self.embed_dim = embed_dim 
        self.n_heads = n_heads 
        self.head_dim = embed_dim // n_heads 
        self.qkv_dropout = qkv_dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_dropout = nn.Dropout(p=out_dropout)

    def forward(self, x: torch.Tensor, rope_embs: torch.Tensor):
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv # (b, nh, n, d)

        q, k = apply_rope(q, rope_embs), apply_rope(k, rope_embs)

        attn_outs = F.scaled_dot_product_attention(q, k, v, dropout_p=self.qkv_dropout, is_causal=False)
        attn_outs = attn_outs.transpose(1, 2).contiguous().reshape(B, N, D)

        return self.out_dropout(self.out_proj(attn_outs))


class Block(nn.Module):
    """Transformer Block. Consists of Derf normalization -> MHA block with
    a residual connection, followed by another Derf normalization -> SwiGLU with 
    residual connection."""

    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 n_heads,
                 out_dropout,
                 qkv_dropout,
                 qkv_bias,
                 init_scale,
                 init_shift):
        super().__init__()

        self.mha = Attention(embed_dim, n_heads, out_dropout, qkv_dropout, qkv_bias)
        self.swiglu = SwiGLU(embed_dim, hidden_dim)

        self.derf1 = Derf(embed_dim, init_scale, init_shift)
        self.derf2 = Derf(embed_dim, init_scale, init_shift)

    def forward(self, x: torch.Tensor, rope_embs: torch.Tensor):
        x = x + self.mha(self.derf1(x), rope_embs)
        x = x + self.swiglu(self.derf2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer"""

    def __init__(self,
                 in_chans=3,
                 patch_size=16,
                 embed_dim=192,
                 depth=12,
                 n_heads=3,
                 hidden_dim=512,
                 out_dropout=0.2,
                 qkv_dropout=0.2,
                 qkv_bias=False,
                 init_scale=0.4,
                 init_shift=0.5,
                 max_seq_len=512,
                 theta=10000.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // n_heads

        self.patch_embedding = PatchEmbedding(in_chans, patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        self.blocks = nn.ModuleList([Block(embed_dim, hidden_dim, n_heads, out_dropout, qkv_dropout, 
                                           qkv_bias, init_scale, init_shift) for _ in range(depth)])

        self.register_buffer(
            "rope_embs",
            create_rope_embeddings(max_seq_len, self.head_dim, theta),
            persistent=False
        )

    def forward(self, x: torch.Tensor, return_only_cls=False):
        B = x.shape[0]

        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        for block in self.blocks:
            x = block(x, self.rope_embs)

        if return_only_cls:
            return x[:, 0, :]
        
        return x


def vit_tiny(patch_size=8, **kwargs):
    model = ViT(patch_size=patch_size, 
                embed_dim=192, 
                depth=12, 
                n_heads=3, 
                hidden_dim=512,
                qkv_bias=True, 
                **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = ViT(patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                n_heads=6, 
                hidden_dim=1024,
                qkv_bias=True, 
                **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = ViT(patch_size=patch_size, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                hidden_dim=2048,
                qkv_bias=True, 
                **kwargs)
    return model
