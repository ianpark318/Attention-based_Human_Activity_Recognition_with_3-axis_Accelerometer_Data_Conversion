import torch
from torch import Tensor
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.backbone = models.vgg16(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.P = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Fuse the queries, keys, values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries, and vlaues in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion,
                    drop_p=forward_drop_p),
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 12,
                 depth: int = 12,
                 n_classes: int = 4,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes))


class Residual(nn.Module):
    def __init__(self, numIn, numOut, stride = 1):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.cab = CAB(numOut)
        self.stride = stride
        self.conv1 = nn.Conv2d(self.numIn, self.numOut, bias = False, kernel_size = 3,stride = self.stride,padding = 1)
        self.bn1 = nn.BatchNorm2d(self.numOut)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(self.numOut, self.numOut, bias = False, kernel_size = 3, stride = self.stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.numOut)
        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias = True, kernel_size = 1)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.cab(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.cab(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.numIn != self.numOut:
            residual = self.conv4(x)
        return out + residual
class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid())
    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, channel):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            ChannelAttention(channel)
            )
    def forward(self, x):
        return self.cab(x)

class AttResNet(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(AttResNet, self).__init__()
        self.in_planes = in_planes
        self.conv = nn.Conv2d(3, self.in_planes, kernel_size = 3, stride = 1, padding = 3, padding_mode='zeros', bias=False)
        self.conv_g = nn.Conv2d(2, self.in_planes, kernel_size = 3, stride = 1, padding = 1, padding_mode='zeros', bias=False)
        self.cab = CAB(in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.res1 = Residual(in_planes,in_planes)
        self.res2 = Residual(in_planes,in_planes)
        self.res3 = Residual(in_planes,in_planes)
        self.res4 = Residual(in_planes,in_planes)
        self.res5 = Residual(in_planes,in_planes)
        self.res6 = Residual(in_planes,in_planes)
        self.res7 = Residual(in_planes,in_planes)
        self.res8 = Residual(in_planes,in_planes)
        self.res7 = Residual(in_planes,in_planes)
        self.res8 = Residual(in_planes,in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(in_planes, num_classes)
    def forward(self, x):
        out = self.conv(x)
        out = self.cab(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.maxpool(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.maxpool(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.maxpool(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = Residual(in_planes, in_planes)
        self.res2 = Residual(in_planes, in_planes)
        self.res3 = Residual(in_planes, in_planes)
        self.res4 = Residual(in_planes, in_planes)
        self.res5 = Residual(in_planes, in_planes)
        self.res6 = Residual(in_planes, in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.maxpool(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.maxpool(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.maxpool(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out