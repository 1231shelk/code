import torch
import torch.nn as nn
from linformer import Linformer
import torch.nn.functional as F
from vit_pytorch.efficient import ViT
from torchvision.models import resnet50

# ViT
def get_vit_model(image_size=224, patch_size=32, num_classes=5, device='cuda'):
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49 + 1,  # 7x7 patches + 1 cls tokenaa
        depth=12,
        heads=8,
        k=64
    )

    model = ViT(
        dim=128,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)

    return model




# Convolutional Stem + ViT
# 思路: 用卷积干净地下采样并提局部纹理 → 输出特征图作为“新图像”，再交给 ViT 做全局建模
# 使用: model = ConvStemViT(image_size=224, num_classes=2, dim=128, device='cuda')
class ConvStem(nn.Module):
    """将 224x224x3 -> 7x7xDIM（等价于 patch=32 的 token 网格）"""
    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # 112 -> 56
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            # 56 -> 28
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # 28 -> 14
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            # 14 -> 7（再下采样一次，得到与patch=32等价的 7x7 网格）
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)  # (B, DIM, 7, 7)

class ConvStemViT(nn.Module):
    def __init__(self, image_size=224, num_classes=2, dim=128, device='cuda'):
        super().__init__()
        self.stem = ConvStem(out_dim=dim)

        # 经过 stem 后得到“新图像”大小为 7x7、通道为 dim
        # trick：把 ViT 的 patch_size 设为 1，让每个网格位置就是一个 token
        efficient_transformer = Linformer(
            dim=dim, seq_len=49 + 1, depth=12, heads=8, k=64
        )
        self.vit = ViT(
            dim=dim,
            image_size=7,      # stem 输出的空间尺寸
            patch_size=1,      # 一个像素=一个token
            num_classes=num_classes,
            transformer=efficient_transformer,
            channels=dim,
        )
        self.to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        f = self.stem(x)        # (B, dim, 7, 7)
        logits = self.vit(f)    # 直接作为“图像”送入 ViT
        return logits




# Hybrid ViT（CNN Backbone + Transformer Encoder）
# 思路：用强CNN骨干（如 ResNet）提取金字塔特征 → 选一层特征图
# 线性投影成 Transformer 维度 → 加上位置编码与 [CLS] → 原生 TransformerEncoder 进行全局建模 → 分类头
# 使用: model = CVT(num_classes=2, dim=256, depth=8, heads=8, device='cuda')
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device
        # 可用更精致的正余弦编码，这里用可学习参数更简单
        pe = nn.Parameter(torch.zeros(1, C, H, W, device=device))
        return x + pe

class HybridResNetViT(nn.Module):
    def __init__(self, num_classes=2, dim=256, depth=8, heads=8, mlp_ratio=4.0, device='cuda'):
        super().__init__()
        backbone = resnet50(weights=None)  # 可替换 weights="IMAGENET1K_V1"
        # 取到 C3/C4/C5 等层，这里用 layer3 输出（28x28 或 14x14 视输入而定）
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )
        self.out_channels = 1024  # resnet50 layer3 输出通道

        self.proj = nn.Conv2d(self.out_channels, dim, kernel_size=1, bias=False)
        self.pos = PositionalEncoding2D(dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        f = self.backbone(x)                 # (B, Cb, H', W')
        f = self.proj(f)                     # (B, dim, H', W')
        f = self.pos(f)
        B, C, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2)  # (B, H'*W', dim)

        cls = self.cls_token.expand(B, -1, -1) # (B, 1, dim)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, dim)

        out = self.transformer(tokens)       # (B, 1+N, dim)
        cls_out = self.norm(out[:, 0])
        logits = self.head(cls_out)
        return logits



# CVT 卷积不仅在前端，还融入每一层 Transformer Block
# 增强了 ViT 的 局部感受性 和 平移不变性，更接近 CNN 的归纳偏置。
# 计算开销低于纯 ViT，但效果往往优于 ViT
# 使用: model = CVT(image_size=224, num_classes=5, dim=128, device='cuda')

# -----------------------
# Conv-based QKV projection (depthwise separable)
# -----------------------
class DWConvProj(nn.Module):
    """depthwise conv -> pointwise conv projection (preserves HxW spatial structure)"""
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.dw(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.pw(x)
        return x  # (B, C, H, W)

# -----------------------
# Convolutional Attention
# -----------------------
class ConvAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # conv-based projections for Q/K/V
        self.to_q = DWConvProj(dim)
        self.to_k = DWConvProj(dim)
        self.to_v = DWConvProj(dim)

        # output projection (pointwise conv)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W
        # conv projections -> (B, C, H, W)
        q = self.to_q(x).view(B, self.heads, C // self.heads, N)   # (B, heads, C_head, N)
        k = self.to_k(x).view(B, self.heads, C // self.heads, N)
        v = self.to_v(x).view(B, self.heads, C // self.heads, N)

        # attention (scaled dot-product)
        # attn: (B, heads, N, N)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)  # (B, heads, C_head, N)
        out = out.contiguous().view(B, C, H, W)         # (B, C, H, W)
        out = self.to_out(out)
        out = self.out_bn(out)
        return out

# -----------------------
# Transformer-like Block with ConvAttention + Conv-FFN
# -----------------------
class ConvTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.attn = ConvAttention(dim, heads=heads)
        # Conv-FFN: pointwise conv -> gelu -> pointwise conv
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        # small layer-scale/residuals are typical; simple residual here
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W)
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

# -----------------------
# CvT-like Stage: conv downsample + several ConvTransformerBlock
# -----------------------
class CvTStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, depth=1, heads=8):
        super().__init__()
        # conv tokenization / downsample (if stride>1)
        self.token_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # blocks
        self.blocks = nn.Sequential(*[ConvTransformerBlock(out_ch, heads=heads) for _ in range(depth)])

    def forward(self, x):
        x = self.token_conv(x)  # downsample + project channels
        x = self.blocks(x)
        return x

# -----------------------
# CvT-style model but keep same interface as CVT
# -----------------------
class CVT(nn.Module):
    """
    CvT-like model that keeps the original constructor signature.
    Usage:
      model = CVT(image_size=224, num_classes=5, dim=128, device='cuda')
    """
    def __init__(self, image_size=224, num_classes=2, dim=128, device='cuda',
                 stage_dims=None, stage_depths=None, heads=(4, 8, 8)):
        """
        - dim: base channel size used to construct stages if stage_dims not provided
        - stage_dims: list of channels for each stage, e.g. [dim//2, dim, dim*2]
        - stage_depths: number of transformer-blocks per stage, e.g. [1, 2, 3]
        - heads: tuple of attention heads per stage
        """
        super().__init__()
        self.device = device

        # default stage configs derived from dim
        if stage_dims is None:
            # ensure integers and sensible sizes
            d1 = max(32, dim // 2)
            d2 = dim
            d3 = max(dim * 2, 256)
            stage_dims = [d1, d2, d3]
        if stage_depths is None:
            stage_depths = [1, 2, 3]
        assert len(stage_dims) == len(stage_depths) == len(heads), "stage_dims, stage_depths, heads must align"

        # initial conv stem: small pre-processing before stage1 (keep light)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, stage_dims[0], kernel_size=7, stride=4, padding=3, bias=False),  # 224->56
            nn.BatchNorm2d(stage_dims[0]),
            nn.ReLU(inplace=True)
        )

        # build stages
        in_ch = stage_dims[0]
        stages = []
        for i, out_ch in enumerate(stage_dims):
            # for stage 0, stride=1 because initial conv already downsampled 4x.
            stride = 1 if i == 0 else 2
            stage = CvTStage(in_ch, out_ch, stride=stride, depth=stage_depths[i], heads=heads[i])
            stages.append(stage)
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )

        self.to(device)

    def forward(self, x):
        # x: (B,3,H,W) expected H=W=image_size
        x = x.to(self.device)
        x = self.init_conv(x)
        x = self.stages(x)
        logits = self.head(x)
        return logits





