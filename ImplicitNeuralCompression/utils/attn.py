import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1(197), total_embed_dim(768)]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head],多头在token长度这里切
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim=3):
        super(SelfAttention, self).__init__()
        # linear layers to project input features into query, key and value
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # scaling factor for dot product
        self.scale = torch.sqrt(torch.tensor(dim))

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        # compute query, key and value
        q = self.query(x) # (batch_size, seq_len, dim)
        k = self.key(x) # (batch_size, seq_len, dim)
        v = self.value(x) # (batch_size, seq_len, dim)

        # compute attention scores by dot product of query and key
        scores = torch.matmul(q,k.transpose(-2,-1)) / self.scale # (batch_size ,seq_len ,seq_len)

        # apply softmax to get attention weights
        weights = torch.softmax(scores,dim=-1) # (batch_size ,seq_len ,seq_len)

        # compute output by weighted sum of value and weights
        output = torch.matmul(weights,v) # (batch_size ,seq_len ,dim)

        return output
    
#用卷积做attention
#coordinate-- [64,64,64,3]
class CA_block(nn.Module):
    def __init__(self,in_c) -> None:
        super(CA_block,self).__init__()
        self.in_c = in_c
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.ca = nn.Sequential(
            nn.Conv3d(in_c, in_c // 8, kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(),
            nn.Conv3d(in_c // 8, in_c, kernel_size=(1,1,1)),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
    
class PA_block(nn.Module):
    def __init__(self,in_c) -> None:
        super(PA_block,self).__init__()
        self.pa = nn.Sequential(
            nn.Conv3d(in_c, in_c // 8, kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(),
            nn.Conv3d(in_c // 8, 1, kernel_size=(1,1,1)),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        y = self.pa(x)
        return x * y
    
class block(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()
        self.ca = CA_block(in_c)
        self.pa = PA_block(in_c)
        
    def forward(self, x):
        y = self.ca(x)
        y = self.pa(y)
        
        return x + y
