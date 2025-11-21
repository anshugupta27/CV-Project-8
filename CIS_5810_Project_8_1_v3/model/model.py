import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # TODO: Compute attention weights (A) as shown in the lecture notes and project doc
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # TODO: Compute weighted sum over values (SA)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # TODO: Implement the forward pass for the EncoderBlock as shown in Figure 2
        x = x + self.attn(self.norm1(x)) # MSA
        x = x + self.mlp(self.norm2(x))  # MLP
        return x


class PoseTransformer(nn.Module):
    def __init__(self, num_joints=21, in_chans=2, embed_dim=256, depth=4, num_heads=8, mlp_ratio=2.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # input projection
        self.input_proj = nn.Linear(in_chans, embed_dim)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # transformer encoder
        self.blocks = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        # output head
        self.norm = norm_layer(embed_dim)
        self.output_head = nn.Linear(embed_dim, 3)

    def forward(self, x, vis_flag=None):
        # TODO: Implement the forward pass for the PoseTransformer as shown in Figure 2
        B, _, _ = x.shape
        x = self.input_proj(x)  # (B, N, C)

        # add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_head(x)

        return x
