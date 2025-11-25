import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.S3CA import S3CA_LM,S3CA_IMG
import numpy as np
from functools import partial
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention_img(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :self.img_chanel, :]
        x_lm = x[:, self.img_chanel:, :]

        B, N, C = x_img.shape
        kv = self.kv(x_img).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        q = x_lm.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_img = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_img = self.proj(x_img)
        x_img = self.proj_drop(x_img)

        return x_img

class Attention_lm(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :self.img_chanel, :]
        x_lm = x[:, self.img_chanel:, :]

        B, N, C = x_lm.shape
        kv = self.kv(x_lm).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        q = x_img.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_lm = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_lm = self.proj(x_lm)
        x_lm = self.proj_drop(x_lm)

        return x_lm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_ratio=4):
        super(ConvMLP, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_dim, in_dim * mlp_ratio, kernel_size=3, padding=1)
        # self.norm1 = LayerNorm(in_dim * mlp_ratio, eps=1e-6, data_format="channels_first")
        # self.norm1 = nn.LayerNorm(in_dim * mlp_ratio)
        self.act1 = nn.GELU()

        # MLP的卷积处理
        self.fc1 = nn.Conv1d(in_dim * mlp_ratio, in_dim * mlp_ratio, kernel_size=1)
        # self.norm2 = LayerNorm(in_dim * mlp_ratio, eps=1e-6, data_format="channels_first")
        # self.norm2 = nn.LayerNorm(in_dim * mlp_ratio)
        self.act2 = nn.GELU()

        # 第二层卷积
        self.conv2 = nn.Conv1d(in_dim * mlp_ratio, out_dim, kernel_size=3, padding=1)
        # self.norm3 = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")
        # self.norm3 = nn.LayerNorm(in_dim * mlp_ratio)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 第一步卷积+激活
        # x = self.norm1(self.act1(self.conv1(x)))
        x = self.act1(self.conv1(x))
        # 通过MLP卷积进行特征学习
        # x = self.norm2(self.act2(self.fc1(x)))
        x = self.act2(self.fc1(x))
        # 第二步卷积
        # x = self.norm3(self.conv2(x))
        x = self.conv2(x)
        return x



class SSCF(nn.Module):

    def __init__(self, in_chans=49, q_chanel = 49, num_classes=1000, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None,
                 act_layer=None, weight_init=''):

        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, in_chans + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        n_channels = (in_chans+1) + (q_chanel+1)
        self.downsample_m = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.downsample_s = nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=4)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            S3CT(
                dim=embed_dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)



    def forward(self, x, x_lm):
        B = x.shape[0]

        x_cls = torch.mean(x, 1).view(B,1,-1)
        x = torch.cat((x_cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        xlm_cls = torch.mean(x_lm, 1).view(B,1,-1)
        x_lm = torch.cat((xlm_cls, x_lm), dim=1)

        new_x = torch.cat((x, x_lm), dim=1)
        ###############################
        new_x_l = new_x
        new_x_m = self.downsample_m(new_x)
        new_x_s = self.downsample_s(new_x)
        new_x_in = [new_x_l,new_x_m,new_x_s]
        new_x_in = self.blocks(new_x_in)
        #############################
        # new_x_in = self.blocks(new_x)

        new_x_l = new_x_in[0]
        new_x_l = self.norm(new_x_l)
        x_class1 = new_x_l[:,0,:]
        x_class2 = new_x_l[:, self.in_chans+1, :]

        return x_class1



class S3CT(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block_l = Block_S3CT(
                dim=dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop=drop, attn_drop=attn_drop,
                drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_m = Block_S3CT(
            dim=dim//2, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        #
        self.block_s = Block_S3CT(
            dim=dim//4, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        n_channels = (in_chans+1) + (q_chanel+1)
        #
        self.upsample_m = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.upsample_s = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)

    def forward(self, x):

        x_l = x[0]
        x_m = x[1]
        x_s = x[2]

        x_l = self.block_l(x_l)
        x_m = self.block_m(x_m)
        x_s = self.block_s(x_s)


        ##
        ###
        x_m = self.upsample_s(x_s) + x_m
        x_l = x_l + self.upsample_m(x_m)
        x = [x_l, x_m ,x_s]


        return x


class Block_S3CT(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.img_chanel = in_chans + 1
        self.num_channels = in_chans + q_chanel + 2

        self.attn_img = S3CA_IMG(dim)
        self.attn_lm = S3CA_LM(dim)


        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        self.conv = nn.Conv1d(self.num_channels, self.num_channels, 1)
        # self.conv = ConvMLP(self.num_channels,self.num_channels,2)

    def forward(self, x):
        x_img = x[:,:self.img_chanel, :]
        x_lm = x[:,self.img_chanel:, :]

        x_img = self.attn_img(x)
        x_img = x_img +self.norm2(self.mlp1(x_img))

        x_lm = self.attn_lm(x)
        x_lm = x_lm + self.norm4(self.mlp2(x_lm))
        ##  default ##
        x = torch.cat((x_img, x_lm), dim=1)
        x = self.conv(x)
        ###
        return x




class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)