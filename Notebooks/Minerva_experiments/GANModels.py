import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
#from einops import rearrange, reduce, repeat
#from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class Generator(nn.Module):
    def __init__(self, seq_len=150, patch_size=15, channels=3, num_classes=9, latent_dim=100, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(
                         depth=self.depth,
                         emb_size = self.embed_dim,
                         drop_p = self.attn_drop_rate,
                         forward_drop_p=self.forward_drop_rate
                        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)
        return output
    
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        '''
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        '''
        queries = self.rearrange_tensor(self.queries(x),  num_heads=self.num_heads)
        keys = self.rearrange_tensor(self.keys(x), num_heads=self.num_heads)
        values = self.rearrange_tensor(self.values(x),  num_heads=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        #print(f'out shape: {out.shape}')
        #out = rearrange(out, "b h n d -> b n (h d)")
        b, h, n, d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        #print(f'out final shape: {out.shape}')
        out = self.projection(out)
        return out
    
    def rearrange_tensor(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Rearrange tensor from shape (b, n, h * d) to (b, h, n, d).

        Args:
            x (Tensor): Input tensor with shape (b, n, h * d).
            num_heads (int): Number of heads (h).

        Returns:
            Tensor: Rearranged tensor with shape (b, h, n, d).
        """
        b, n, hd = x.shape
        d = hd // num_heads  # Calcula a dimensão `d` dividindo pela quantidade de cabeças

        # Redimensiona para (b, n, h, d) e reorganiza para (b, h, n, d)
        return x.view(b, n, num_heads, d).permute(0, 2, 1, 3)

    
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
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        '''
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        '''
        self.clshead = nn.Sequential(
            ReduceLayer(reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


    def forward(self, x):
        out = self.clshead(x)
        return out

class ReduceLayer(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(ReduceLayer, self).__init__()
        self.reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.mean(dim=1)  # Reduz ao longo da dimensão `n`, resultando em (b, e)
        elif self.reduction == 'sum':
            return x.sum(dim=1)
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")
    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, channels = 21, patch_size = 16, emb_size = 100, seq_len = 1024):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here
        '''
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*channels, emb_size))
        '''
        self.projection = nn.Sequential(
            RearrangeLayer(patch_size=patch_size, s1=1),  # Substitui o Rearrange do einops
            nn.Linear(patch_size * channels, emb_size))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_len // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=2) #for daghar (batch, 6, 60) -> (batch, 6, 1, 60)
        b, _, _, _ = x.shape
        x = self.projection(x)
        
        #cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        cls_tokens = self.cls_token.repeat(b, 1, 1) #Personal repeat from pytorch to transfer from einops

        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x     
    
    #For use in change of Rearrange einops layer
class RearrangeLayer(nn.Module):
    def __init__(self, patch_size: int, s1: int = 1):
        super(RearrangeLayer, self).__init__()
        self.patch_size = patch_size
        self.s1 = s1

    def forward(self, x: Tensor) -> Tensor:
        b, c, h_s1, w_s2 = x.shape
        h, s1 = h_s1, self.s1
        w, s2 = w_s2 // self.patch_size, self.patch_size
        
        # Rearrange tensor
        x = x.view(b, c, h, s1, w, s2)  # shape: (b, c, h, s1, w, s2)
        x = x.permute(0, 2, 4, 3, 5, 1)  # shape: (b, h, w, s1, s2, c)
        x = x.reshape(b, h * w, s1 * s2 * c)  # shape: (b, h * w, s1 * s2 * c)  

        return x
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 channels=3,
                 patch_size=15,
                 emb_size=50, 
                 seq_len = 150,
                 depth=3, 
                 n_classes=1, 
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(channels, patch_size, emb_size, seq_len),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
class Encoder(nn.Sequential):
    def __init__(self, 
                 channels=3,
                 patch_size=15,
                 emb_size=50, 
                 seq_len = 150,
                 depth=3,  
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(channels, patch_size, emb_size, seq_len),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
        )
