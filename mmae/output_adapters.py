from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, Tuple, Optional, Dict, List, Iterable

from .mmae_utils import *
from .output_adapter_utils import *


class PatchedOutputAdapterXA(nn.Module):
    '''
    Cross-attention adapter for spatial outputs, like images or feature maps.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image. 
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size. 
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens_enc: Dimension of tokens coming from encoder. Can be set using init method.
    :param dim_tokens: Dimension of decoder tokens
    :param depth: Number of additional (full self-attention) transformer layers after initial cross attention and MLP
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    :param mlp_ratio: MLP hidden dim ratio
    :param num_heads: Number of attention heads
    :param qkv_bias: Set to True to enable bias
    :param qk_scale: Set to override default of scale = head_dim ** -0.5
    :param drop_rate: Probability of dropping attention layer outputs
    :param attn_drop_rate: Probability of dropping attention matrix elements
    :param drop_path_rate: DropPath drop rate
    :param attn_head_dim: Set to override default of head_dim = dim_tokens // num_heads
    :param norm_layer: Type of normalization layer
    :param init_values: Optional initialization for Transformer block gamma values
    :param use_task_queries: Set to True to add task specific tokens from encoder (if available)
        to the corresponding query entries
    '''
    def __init__(self, 
                 num_channels: int, 
                 stride_level: int, 
                 patch_size_full: Union[int, Tuple[int,int]], 
                 dim_tokens_enc: Optional[int] = None,
                 dim_tokens: int = 256,
                 depth: int = 0,
                 learnable_pos_emb: int = False,
                 image_size: Union[int, Tuple[int]] = 224,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 use_task_queries: bool = True,
                 task: Optional[str] = None,
                 context_tasks: Optional[list] = None,
                 use_xattn: bool = True
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.use_task_queries = use_task_queries
        self.task = task
        self.use_xattn = use_xattn
        
        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if context_tasks is not None:
            self.task_embeddings = nn.ParameterDict({task: nn.Parameter(torch.zeros(1, 1, self.dim_tokens)) for task in context_tasks})
            for embedding in self.task_embeddings.values():
                trunc_normal_(embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if not self.learnable_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, h_posemb, w_posemb, self.dim_tokens))
            trunc_normal_(self.pos_emb, std=0.02)
        
        # One cross attention layer followed by MLP block, an optional transformer, and an output projection
        if self.use_xattn:
            self.decoder = CrossAttention(
                dim=self.dim_tokens, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)
            # TODO: consider using same norm for context and query
            self.context_norm = norm_layer(self.dim_tokens)
            self.query_norm = norm_layer(self.dim_tokens)
            self.out_norm = norm_layer(self.dim_tokens)

            mlp_hidden_dim = int(self.dim_tokens * mlp_ratio)
            self.mlp = Mlp(in_features=self.dim_tokens, hidden_features=mlp_hidden_dim)

        # Optional full self-attention transformer layers
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.decoder_transformer = nn.Sequential(*[
                Block(dim=self.dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        else:
            self.decoder_transformer = nn.Identity()
        
        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)
        
    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb', 'mask_token', 'task_embeddings'}

    def generate_context_embeddings(self, input_info,
                                    bs : int,
                                    size : Tuple[int, int],
                                    device: Optional[torch.device] = None):
        context_embeddings = []
        for task, info in input_info["tasks"].items():
            if self.task_embeddings is not None and task in self.task_embeddings:
                task_emb = repeat(self.task_embeddings[task], '() () d -> b n d', b=bs, n=info['num_tokens'])
            else:
                task_emb = torch.zeros((bs, info['num_tokens'], self.dim_tokens), device=device)

            if info['has_2d_posemb']:
                pos_emb = F.interpolate(self.pos_emb, size=size, mode='bilinear', align_corners=False)
                pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
                assert info['num_tokens'] == pos_emb.shape[1]
                task_emb = task_emb + pos_emb

            context_embeddings.append(task_emb)

        context_embeddings = torch.cat(context_embeddings, dim=1)

        return context_embeddings

    def get_queries_and_context(self, context_tokens, input_info, ids_keep, ids_restore):
        B = context_tokens.shape[0]
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        if 'num_global_tokens' in input_info:
            context_tokens_without_global = context_tokens[:, :-input_info['num_global_tokens']]
        else:
            context_tokens_without_global = context_tokens

        # Add mask tokens
        mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B, n=input_info['num_task_tokens'] - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)

        # Unshuffle context_with_mask
        context_with_mask = torch.gather(context_with_mask, dim=1,
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Generate context_emb and add them to context
        context_emb = self.generate_context_embeddings(input_info=input_info, bs=B, size=(N_H, N_W), device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb

        # Generate queries
        if self.use_task_queries and self.task in input_info['tasks']:
            start_idx = input_info['tasks'][self.task]['start_idx']
            end_idx = input_info['tasks'][self.task]['end_idx']
            queries = context_with_mask[:, start_idx:end_idx]
        else:
            queries = repeat(self.mask_token, '() () d -> b n d', b=B, n=N_H * N_W)
            queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bilinear', align_corners=False)
            queries_pos_emb = rearrange(queries_pos_emb, 'b d nh nw -> b (nh nw) d')
            queries = queries + queries_pos_emb
            if self.task_embeddings is not None and self.task in self.task_embeddings:
                queries_task_emb = repeat(self.task_embeddings[self.task], '() () d -> b n d', b=B, n=N_H * N_W)
                queries = queries + queries_task_emb

        # Unshuffle context and keep only initial context (yes, again)
        context_tokens_without_global = torch.gather(context_with_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Add back global tokens
        if 'num_global_tokens' in input_info:
            context_tokens = torch.cat(
                [context_tokens_without_global, context_tokens[:, -input_info['num_global_tokens']:]], dim=1)
        else:
            context_tokens = context_tokens_without_global

        return queries, context_tokens

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: Dict,
                ids_keep: torch.Tensor,
                ids_restore: torch.Tensor,
                ):
        '''
        Forward pass taking output tokens from encoder and optionally a subset of them corresponding
        to this output adapter's task (needs an additional mask describing position of these tokens in the queries).

        :param encoder_tokens: Output of encoder
        :param query_tokens: Optional subset of encoder output used as queries for this decoder
        :param query_tokens_mask: Optional mask describing position of query tokens
        :param image_size: Original image size
        '''
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Project encoder tokens to decoder tokens
        context_tokens = self.proj_context(encoder_tokens)

        # Get queries and context
        queries, context_tokens = self.get_queries_and_context(context_tokens, input_info, ids_keep, ids_restore)

        # Perform cross attention of queries to context tokens, followed by an MLP
        if self.use_xattn:
            x = self.decoder(self.query_norm(queries), self.context_norm(context_tokens))
            x = x + self.mlp(self.out_norm(x))
        else:
            x = queries

        # Optional transformer layers if depth > 0
        x = self.decoder_transformer(x)

        # Project each token to (C * P_H * P_W)
        x = self.out_proj(x)
        
        # Reshape sequence of patches into image
        x = rearrange(
            x, 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
            nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels
        )
        
        return x


class TransformerDecoderAdapter(nn.Module):
    '''
    Transformer decoder adapter

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens_enc: Dimension of tokens coming from encoder. Can be set using init method.
    :param dim_tokens: Dimension of decoder tokens
    :param depth: Number of additional (full self-attention) transformer layers after initial cross attention and MLP
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    :param mlp_ratio: MLP hidden dim ratio
    :param num_heads: Number of attention heads
    :param qkv_bias: Set to True to enable bias
    :param qk_scale: Set to override default of scale = head_dim ** -0.5
    :param drop_rate: Probability of dropping attention layer outputs
    :param attn_drop_rate: Probability of dropping attention matrix elements
    :param drop_path_rate: DropPath drop rate
    :param attn_head_dim: Set to override default of head_dim = dim_tokens // num_heads
    :param norm_layer: Type of normalization layer
    :param init_values: Optional initialization for Transformer block gamma values
    :param use_task_queries: Set to True to add task specific tokens from encoder (if available)
        to the corresponding query entries
    '''

    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int, int]],
                 dim_tokens_enc: Optional[int] = None,
                 dim_tokens: int = 256,
                 depth: int = 4,
                 learnable_pos_emb: int = False,
                 image_size: Union[int, Tuple[int]] = 224,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 use_task_queries: bool = True,
                 task: Optional[str] = None,
                 context_tasks: Optional[list] = None,
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.use_task_queries = use_task_queries
        self.task = task

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if context_tasks is not None:
            self.task_embeddings = nn.ParameterDict(
                {task: nn.Parameter(torch.zeros(1, 1, self.dim_tokens)) for task in context_tasks})
            for embedding in self.task_embeddings.values():
                trunc_normal_(embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if not self.learnable_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, h_posemb, w_posemb, self.dim_tokens))
            trunc_normal_(self.pos_emb, std=0.02)

        # Transformer Decoder layers (SA + XA + MLP)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=self.dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb', 'mask_token', 'task_embeddings'}

    def generate_context_embeddings(self, input_info,
                                    bs: int,
                                    size: Tuple[int, int],
                                    device: Optional[torch.device] = None):
        context_embeddings = []
        for task, info in input_info["tasks"].items():
            if self.task_embeddings is not None and task in self.task_embeddings:
                task_emb = repeat(self.task_embeddings[task], '() () d -> b n d', b=bs, n=info['num_tokens'])
            else:
                task_emb = torch.zeros((bs, info['num_tokens'], self.dim_tokens), device=device)

            if info['has_2d_posemb']:
                pos_emb = F.interpolate(self.pos_emb, size=size, mode='bilinear', align_corners=False)
                pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
                assert info['num_tokens'] == pos_emb.shape[1]
                task_emb = task_emb + pos_emb

            context_embeddings.append(task_emb)

        context_embeddings = torch.cat(context_embeddings, dim=1)

        return context_embeddings

    def get_queries_and_context(self, context_tokens, input_info, ids_keep, ids_restore):
        B = context_tokens.shape[0]
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        if 'num_global_tokens' in input_info:
            context_tokens_without_global = context_tokens[:, :-input_info['num_global_tokens']]
        else:
            context_tokens_without_global = context_tokens

        # Add mask tokens
        mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B,
                             n=input_info['num_task_tokens'] - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)

        # Unshuffle context_with_mask
        context_with_mask = torch.gather(context_with_mask, dim=1,
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Generate context_emb and add them to context
        context_emb = self.generate_context_embeddings(input_info=input_info, bs=B, size=(N_H, N_W),
                                                       device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb

        # Generate queries
        if self.use_task_queries and self.task in input_info['tasks']:
            start_idx = input_info['tasks'][self.task]['start_idx']
            end_idx = input_info['tasks'][self.task]['end_idx']
            queries = context_with_mask[:, start_idx:end_idx]
        else:
            queries = repeat(self.mask_token, '() () d -> b n d', b=B, n=N_H * N_W)
            queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bilinear', align_corners=False)
            queries_pos_emb = rearrange(queries_pos_emb, 'b d nh nw -> b (nh nw) d')
            queries = queries + queries_pos_emb
            if self.task_embeddings is not None and self.task in self.task_embeddings:
                queries_task_emb = repeat(self.task_embeddings[self.task], '() () d -> b n d', b=B, n=N_H * N_W)
                queries = queries + queries_task_emb

        # Unshuffle context and keep only initial context (yes, again)
        context_tokens_without_global = torch.gather(context_with_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Add back global tokens
        if 'num_global_tokens' in input_info:
            context_tokens = torch.cat(
                [context_tokens_without_global, context_tokens[:, -input_info['num_global_tokens']:]], dim=1)
        else:
            context_tokens = context_tokens_without_global

        return queries, context_tokens

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: Dict,
                ids_keep: torch.Tensor,
                ids_restore: torch.Tensor,
                ):
        '''
        Forward pass taking output tokens from encoder and optionally a subset of them corresponding
        to this output adapter's task (needs an additional mask describing position of these tokens in the queries).

        :param encoder_tokens: Output of encoder
        :param query_tokens: Optional subset of encoder output used as queries for this decoder
        :param query_tokens_mask: Optional mask describing position of query tokens
        :param image_size: Original image size
        '''
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Project encoder tokens to decoder tokens
        context = self.proj_context(encoder_tokens)

        # Get queries (x) and context
        x, context = self.get_queries_and_context(context, input_info, ids_keep, ids_restore)

        # Transformer Decoder
        for blk in self.decoder_blocks:
            x = blk(x, context)

        # Project each token to (C * P_H * P_W)
        x = self.out_proj(x)

        # Reshape sequence of patches into image
        x = rearrange(
            x, 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
            nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels
        )

        return x


class SemSegOutputAdapter(PatchedOutputAdapterXA):

    def __init__(self,
                 num_classes: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int,int]],
                 dim_tokens_enc: Optional[int] = None,
                 dim_tokens: int = 256,
                 depth: int = 0,
                 learnable_pos_emb: int = False,
                 image_size: Union[int, Tuple[int]] = 224,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 use_task_queries: bool = True,
                 task: Optional[str] = None,
                 context_tasks: Optional[list] = None,
                 ):
        super().__init__(
            num_channels=num_classes,
            stride_level=pair(patch_size_full)[0],  # Output num_classes values per token
            patch_size_full=patch_size_full,
            dim_tokens_enc=dim_tokens_enc,
            dim_tokens=dim_tokens,
            depth=depth,
            learnable_pos_emb=learnable_pos_emb,
            image_size=image_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_task_queries=use_task_queries,
            task=task,
            context_tasks=context_tasks,
        )
        patch_size = pair(patch_size_full)
        self.scale_factor = (patch_size[0] // stride_level, patch_size[1] // stride_level)

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: Dict,
                ids_keep: torch.Tensor,
                ids_restore: torch.Tensor,
                ):

        x = super().forward(encoder_tokens=encoder_tokens,
                            input_info=input_info,
                            ids_keep=ids_keep,
                            ids_restore=ids_restore)

        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return x


class LinearOutputAdapter(nn.Module):
    '''
    Linear output adapter.

    :param num_classes: Number of classes
    '''
    def __init__(self, 
                 num_classes: int,
                 dim_tokens_enc: Optional[int] = None,
                 use_mean_pooling: bool = True,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 init_scale: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dim_tokens_enc = dim_tokens_enc
        self.use_mean_pooling = use_mean_pooling # TODO: Doesn't do anything yet
        self.norm_layer = norm_layer
        self.init_scale = init_scale

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        self.norm = self.norm_layer(self.dim_tokens_enc)
        self.head = nn.Linear(dim_tokens_enc, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.head.weight.data.mul_(self.init_scale)
        self.head.bias.data.mul_(self.init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.init(dim_tokens_enc=self.dim_tokens_enc)

    def forward(self, 
                encoder_tokens: torch.Tensor, 
                **kwargs):
        x = encoder_tokens.mean(1)
        x = self.head(self.norm(x))
        return x


class SegmenterMaskTransformerAdapter(nn.Module):
    """Segmenter: Transformer for Semantic Segmentation.

    This head is the implementation of `Segmenter:　<https://arxiv.org/abs/2105.05633>`_.
    """

    def __init__(
            self,
            num_classes,
            depth: int = 2,
            num_heads: int = 12,
            embed_dim: int = 768,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            qkv_bias=True,
            main_task: str = 'rgb',
            patch_size: int = 16,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
    ):
        super().__init__()
        self.main_task = main_task
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.cls_emb = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        trunc_normal_(self.cls_emb, std=0.02)

        self.patch_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.classes_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.decoder_norm = norm_layer(embed_dim)
        self.mask_norm = norm_layer(num_classes)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.in_channels = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        start_idx = input_info['tasks'][self.main_task]['start_idx']
        end_idx = input_info['tasks'][self.main_task]['end_idx']
        x = encoder_tokens[:, start_idx:end_idx]

        # TODO: Consider option with stacking tokens and projecting when more than one modality
        return x

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_emb), 1)

        for blk in self.blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes])
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)

        # Interpolate to semseg res
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks


class CoolAdapter(nn.Module):
    """Cool adapter, inspired by segmenter
    Turns out this is very close to just predicting multiple values per patch, but
    with some normalization here and there
    TODO: experiment with a simple basic decoder
    """

    def __init__(
            self,
            num_classes,
            embed_dim: int = 768,
            preds_per_patch: int = 64,
            main_task: str = 'rgb',
            patch_size: int = 16,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            act_fn: nn.Module = nn.GELU,
            interpolate_mode: str = 'bilinear',
            **kwargs,
    ):
        super().__init__()
        self.main_task = main_task
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = embed_dim // preds_per_patch
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode

        self.cls_emb = nn.Parameter(torch.zeros(1, num_classes, self.class_dim))
        trunc_normal_(self.cls_emb, std=0.02)

        self.act = act_fn()
        self.preds_proj = nn.Linear(self.class_dim, self.class_dim)
        self.mask_norm = norm_layer(num_classes)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.in_channels = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        start_idx = input_info['tasks'][self.main_task]['start_idx']
        end_idx = input_info['tasks'][self.main_task]['end_idx']
        x = encoder_tokens[:, start_idx:end_idx]

        # TODO: Consider option with stacking tokens and projecting when more than one modality
        return x

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = self.proj_dec(x)
        x = self.act(x)
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H*N_W, p=self.preds_per_patch, c=self.class_dim)
        x = self.preds_proj(x)
        x = F.normalize(x, dim=-1, p=2)

        cls_emb = self.cls_emb.expand(x.shape[0], -1, -1)
        cls_emb = F.normalize(cls_emb, dim=-1, p=2)

        masks = x @ cls_emb.transpose(1, 2)
        masks = self.mask_norm(masks)

        masks = rearrange(masks, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                          nh=N_H, nw=N_W,
                          ph=int(self.preds_per_patch ** 0.5),
                          pw=int(self.preds_per_patch ** 0.5))

        # Interpolate to semseg res
        masks = F.interpolate(masks, size=(H, W), mode=self.interpolate_mode)

        return masks


class BasicAdapter(nn.Module):
    """Basic adapter, predicts multiple classes per patch
    """

    def __init__(
            self,
            num_classes,
            embed_dim: int = 768,
            preds_per_patch: int = 64,
            main_task: str = 'rgb',
            patch_size: int = 16,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            act_layer: nn.Module = nn.GELU,
            mlp_ratio: int = 4,
            interpolate_mode='bilinear',
            **kwargs,
    ):
        super().__init__()
        self.main_task = main_task
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = embed_dim // preds_per_patch
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode

        self.mlp = Mlp(self.class_dim, self.class_dim * int(mlp_ratio), self.class_dim, act_layer=act_layer)
        self.norm = norm_layer(self.class_dim)
        self.final_layer = nn.Linear(self.class_dim, self.num_classes)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.in_channels = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        start_idx = input_info['tasks'][self.main_task]['start_idx']
        end_idx = input_info['tasks'][self.main_task]['end_idx']
        x = encoder_tokens[:, start_idx:end_idx]

        # TODO: Consider option with stacking tokens and projecting when more than one modality
        return x

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = self.proj_dec(x)
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H*N_W, p=self.preds_per_patch, c=self.class_dim)
        x = x + self.mlp(self.norm(x))
        x = self.final_layer(x)

        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                          nh=N_H, nw=N_W,
                          ph=int(self.preds_per_patch ** 0.5),
                          pw=int(self.preds_per_patch ** 0.5))

        # Interpolate to semseg res
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)

        return x


class BasicWithConvAdapter(nn.Module):
    """Basic adapter, predicts multiple classes per patch
    Adds extra convolutions
    """

    def __init__(
            self,
            num_classes,
            embed_dim: int = 768,
            preds_per_patch: int = 64,
            main_task: str = 'rgb',
            patch_size: int = 16,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            act_layer: nn.Module = nn.GELU,
            mlp_ratio: int = 4,
            interpolate_mode='bilinear',
            **kwargs,
    ):
        super().__init__()
        self.main_task = main_task
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = embed_dim // preds_per_patch
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode

        self.mlp = Mlp(self.class_dim, self.class_dim * int(mlp_ratio), self.class_dim, act_layer=act_layer)
        self.norm1 = norm_layer(self.class_dim)
        self.norm2 = norm_layer(self.class_dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.class_dim, self.class_dim, 3, 1, padding=1),
            act_layer(),
            nn.Conv2d(self.class_dim, self.class_dim, 3, 1, padding=1),
        )
        self.final_layer = nn.Conv2d(self.class_dim, self.num_classes, 1)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.in_channels = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        start_idx = input_info['tasks'][self.main_task]['start_idx']
        end_idx = input_info['tasks'][self.main_task]['end_idx']
        x = encoder_tokens[:, start_idx:end_idx]

        # TODO: Consider option with stacking tokens and projecting when more than one modality
        return x

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = self.proj_dec(x)
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H*N_W, p=self.preds_per_patch, c=self.class_dim)
        x = x + self.mlp(self.norm1(x))
        x = self.norm2(x)
        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                          nh=N_H, nw=N_W,
                          ph=int(self.preds_per_patch ** 0.5),
                          pw=int(self.preds_per_patch ** 0.5))
        x = x + self.conv_block(x)
        x = self.final_layer(x)

        # Interpolate to semseg res
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)

        return x


class ConvNeXtAdapter(nn.Module):
    """Adapter with ConvNext blocks
    """

    def __init__(
            self,
            num_classes,
            embed_dim: int = 6144,
            preds_per_patch: int = 16,
            main_tasks: Iterable[str] = ('rgb', ),
            patch_size: int = 16,
            depth: int = 4,
            interpolate_mode: str = 'bilinear',
            **kwargs,
    ):
        super().__init__()
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = embed_dim // preds_per_patch
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=self.class_dim)
            for _ in range(depth)
        ])
        self.final_layer = nn.Conv2d(self.class_dim, self.num_classes, 1)
        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        x = []
        for task in self.main_tasks:
            start_idx = input_info['tasks'][task]['start_idx']
            end_idx = input_info['tasks'][task]['end_idx']
            x.append(encoder_tokens[:, start_idx:end_idx])

        x = torch.cat(x, dim=-1)
        return x

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = self.proj_dec(x)
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H*N_W, p=self.preds_per_patch, c=self.class_dim)
        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                          nh=N_H, nw=N_W,
                          ph=int(self.preds_per_patch ** 0.5),
                          pw=int(self.preds_per_patch ** 0.5))
        x = self.blocks(x)
        x = self.final_layer(x)

        # Interpolate to semseg res
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)

        return x


class ConvUpsampleAdapter(nn.Module):
    """Simple convolutional upsampling head
    Similar to SETR-PUP, but with Group Norm and Weight Standardized Convolutions
    """

    def __init__(
            self,
            num_classes,
            num_channels: int = 256,
            main_task: str = 'rgb',
            patch_size: int = 16,
            num_groups=32,
            **kwargs
    ):
        super().__init__()
        self.main_task = main_task
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.patch_size = patch_size

        self.block2 = nn.Sequential(
            Conv2dWS(num_channels, num_channels, 3, 1, padding=1),
            nn.GroupNorm(num_groups, num_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.block3 = nn.Sequential(
            Conv2dWS(num_channels, num_channels, 3, 1, padding=1),
            nn.GroupNorm(num_groups, num_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.block4 = nn.Sequential(
            Conv2dWS(num_channels, num_channels, 3, 1, padding=1),
            nn.GroupNorm(num_groups, num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_classes, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.in_channels = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.block1 = nn.Sequential(
            Conv2dWS(self.in_channels, self.num_channels, 3, 1, padding=1),
            nn.GroupNorm(self.num_groups, self.num_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        start_idx = input_info['tasks'][self.main_task]['start_idx']
        end_idx = input_info['tasks'][self.main_task]['end_idx']
        x = encoder_tokens[:, start_idx:end_idx]

        # TODO: Consider option with stacking tokens and projecting when more than one modality
        return x

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict):
        H, W = input_info['image_size']
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = rearrange(x, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x

class DPTOutputAdapter(nn.Module):
    '''
    DPT output adapter.

    :param num_classes: Number of output channels
    :param stride_level: 
    :param patch_size_full: 
    :param hooks: 
    :param layer_dims: 
    :param feature_dim: 
    :param use_bn: 
    :param dim_tokens_enc: 
    '''
    def __init__(self, 
                 num_classes: int = 3,
                 stride_level: int = 1, 
                 patch_size: Union[int, Tuple[int,int]] = 16, 
                 main_tasks: Iterable[str] = ('rgb', ),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 **kwargs):
        super().__init__()
        self.num_channels = num_classes
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type
        
        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        
        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)
        
        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn)
        
        if self.head_type == 'regression':
            # The "DPTDepthModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, self.num_channels, kernel_size=1, stride=1, padding=0)
            )
        elif self.head_type == 'semseg':
            # The "DPTSegmentationModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(feature_dim, self.num_channels, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            raise ValueError('DPT head_type must be "regression" or "semseg".')
        
        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)
        
    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks)
        
        # Set up activation postprocessing layers
        
        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )
        
        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )
        
        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )
        
        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc,
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])

    def adapt_tokens(self, encoder_tokens, input_info):
        # Adapt tokens
        x = []
        for task in self.main_tasks:
            start_idx = input_info['tasks'][task]['start_idx']
            end_idx = input_info['tasks'][task]['end_idx']
            x.append(encoder_tokens[:, start_idx:end_idx])

        x = torch.cat(x, dim=-1)
        return x
        
    def forward(self, encoder_tokens: List[torch.Tensor], input_info: Dict):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        
        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens. TODO: Add option to project global tokens like in DPT
        layers = [self.adapt_tokens(l, input_info) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]
        
        # Postprocess activations
        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        
        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        
        # Output head
        out = self.head(path_1)
        
        return out
