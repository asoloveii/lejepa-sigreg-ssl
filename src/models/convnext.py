import torch, torch.nn as nn, torch.nn.functional as F


class GRN(nn.Module):
    """Global Response Normalization Layer. Normalizes input tensor 
    (from a convolutional network) across channels, 
    which effectively mitigates feature collapse."""

    def __init__(self, dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(1, 1, 1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        gx = torch.norm(x, p=2, dim=[1, 2], keepdim=True) 
        nx = gx / (gx.mean(dim=-1, keepdim=True)+1e-6)
        return self.gamma * (x * nx) + self.beta + x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block. Input is first processed by conv layer with 7x7 kernel, 
    then upscaled to `4*in_channels` with 1x1 conv layer, downscaled back to 
    `in_channels` with a 1x1 conv and, finally, followed by residual connection"""

    def __init__(self, in_chans):
        super().__init__()
        # depthwise convolutional layer
        self.wide_dw = nn.Conv2d(in_chans, in_chans, kernel_size=7, stride=1, padding=3, groups=in_chans)
        self.norm1 = LayerNorm(in_chans, eps=1e-6, data_format="channels_last")
        # pointwise 1x1 convolutional layers implemented with linear
        self.up_conv = nn.Linear(in_chans, 4 * in_chans, bias=False)
        self.act = nn.GELU()
        self.norm2 = GRN(4 * in_chans)
        self.down_conv = nn.Linear(4 * in_chans, in_chans, bias=False)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.wide_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = self.up_conv(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.down_conv(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class ConvNeXtV2(nn.Module):

    def __init__(self, 
                 in_chans=3, 
                 dims=[48, 96, 192, 384], 
                 depths=[2, 2, 6, 2]):
        super().__init__()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.blocks.append(stage)
    
    def forward(self, x: torch.Tensor):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.blocks[i](x)
        return x.mean([-2, -1])


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model
