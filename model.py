from typing import Any
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
# from utils.utils import pair, PreNorm, FeedForward, Attention
# TSformer-VO的做法
from utils.utils import Block, Attention, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)
# class visual_encoder(nn.Module):
#     def __init__(self,opt,emb_dropout=0.1):
#         super(visual_encoder,self).__init__()
#         self.img_channel = 6
#         self.out_dim = 512
#         self.opt = opt
#         self.patch_height, self.patch_width = pair(opt.patch_size)
#         #token number and patch dimesion
#         self.img_num_patches = (self.opt.img_h // self.patch_height) * (self.opt.img_w // self.patch_width)
#         self.img_patch_dim = self.img_channel * self.patch_height * self.patch_width

#         self.img_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
#             nn.LayerNorm(self.img_patch_dim),
#             nn.Linear(self.img_patch_dim, self.out_dim),
#             nn.LayerNorm(self.out_dim),
#         )
#         self.pos_embedding = nn.Parameter(torch.randn(16, self.img_num_patches + 1, self.out_dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.out_dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(self.out_dim, 1, 8, self.out_dim / 8, self.out_dim, 0.1)
        
#         def forward(self,img):
#             # 使用两张图片concat，感觉也不能concat啊,单独对两张图做transformer感觉意义不是很大
#             # 这里需要参考TSformer的做法
#             v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
#             x = self.img_patch_embedding(v)
#             batch, seq, _ = x.shape
#             cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch)
#             x = torch.cat((cls_tokens, x), dim=1)
#             x += self.pos_embedding[:, :(seq + 1)]
#             x = self.dropout(x)

#             x = self.transformer(x)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # patch token 每一张图单独做，这里两张图并未联合在一起
        x = self.proj(x)#图变成[320,512,16,32],512是feat_dim
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class visual_encoder(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=(256, 512), patch_size=16, in_chans=3, num_classes=512, embed_dim=768, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super(visual_encoder,self).__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        # if self.attention_type == 'divided_space_time':
        #     i = 0
        #     for m in self.blocks.modules():
        #         m_str = str(m)
        #         if 'Block' in m_str:
        #             if i > 0:
        #               nn.init.constant_(m.temporal_fc.weight, 0)
        #               nn.init.constant_(m.temporal_fc.bias, 0)
        #             i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = x.transpose(1,2)
        x, T, W = self.patch_embed(x)# 两张图联合一起  x=[320,512,512],最后一维是dim，前面是token num
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x,_ = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)# shape = [2,1025,512]
        return x[:, 0] # here take away cls token,for predict

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
# # for TS-CAM
# class visual_encoder_CAM(visual_encoder):
#     def __call__(self, *args, **kwds):
#         super().__init__(*args, **kwds)
#     def forward(self, x):
#         # 输入是patch token [batch,N,dim]

#         return 

# for TS-CAM
class CAM(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=(256, 512), patch_size=16, in_chans=3, num_classes=512, embed_dim=768, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super(CAM,self).__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        # if self.attention_type == 'divided_space_time':
        #     i = 0
        #     for m in self.blocks.modules():
        #         m_str = str(m)
        #         if 'Block' in m_str:
        #             if i > 0:
        #               nn.init.constant_(m.temporal_fc.weight, 0)
        #               nn.init.constant_(m.temporal_fc.bias, 0)
        #             i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = x.transpose(1,2)
        x, T, W = self.patch_embed(x)# 两张图联合一起  x=[320,512,512],最后一维是dim，前面是token num
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 包含所有token cls and patch
        token = x.detach().clone()

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        # get space attn map
        attn_weight = []
        for blk in self.blocks:
            x, weight = blk(x, B, T, W)
            attn_weight.append(weight)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)# shape = [2,1025,512]
        cls_token = token[:,0]
        patch_token = token[:,1:]
        return x[:, 1:], attn_weight, T, cls_token, patch_token
        # here take away patch token,for predict

    def forward(self, x):
        x, attn_weight, T, cls_token, patch_token = self.forward_features(x) #得到的是patch token [2,1024,512]
        x = self.head(x)
    
        attn_weight = torch.stack(attn_weight)# depth,(batch*T),head,h,w
        # 得到cam时会用得着
        # if not self.training:
        #     attn_weight = torch.stack(attn_weight)# shape=[8,2,8,513,513] 0,2维度求平均，一个是block数，一个是head
        #     attn_weight = attn_weight[:,:,:,1:,1:]
        #     attn_weight.sum(0)
        #     # attn_weight = torch.mean(attn_weight,dim=0)# (batch*T),head,h,w
        #     attn_weight = torch.mean(attn_weight,dim=1)# (batch*T),h,w
        #     if attn_weight.size(0)/T > 1:
        #         batch = attn_weight.size(0)/2
        #         attn_weight = rearrange(attn_weight,'(b t) h w -> b t h w',b=batch,t=T)
        #         patch_token = rearrange(patch_token,'(b t) h w -> b t h w',b=batch,t=T)
        #         cams = 

        return x, attn_weight

class Encoder_CAM(nn.Module):
    def __init__(self, opt):
        super(Encoder_CAM, self).__init__()
        # CNN
        self.opt = opt
        self.visual_encoder = CAM((opt.img_h,opt.img_w), opt.patch_size, in_chans=3, num_classes=opt.v_f_len, embed_dim=opt.v_f_len, depth=8)
        self.head = nn.Linear(opt.v_f_len,6)
    def forward(self, img):
        # shape=[16,10,2,3,256,512]
        v = torch.cat((img[:, :-1].unsqueeze(2), img[:, 1:].unsqueeze(2)), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)
        device = img.device
        # image CNN
        # v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4), v.size(5))
        #自己加入TSformer内容
        v_f = []
        attn_weights = []
        for i in range(v.size(1)):
             tmp, attn_map = self.visual_encoder(v[:,i]) 
             v_f.append(tmp)
             attn_weights.append(attn_map)

        v_f = torch.stack(v_f,dim=1).to(device)
        # 对两张图求patch token 求mean
        vf_mean = torch.mean(v_f,dim=2)
        est_pose = self.head(vf_mean)
        # v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        # v = self.visual_head(v)  # (batch, seq_len, 256)
        decisions = torch.zeros(batch_size, seq_len, 2)
        probs = torch.zeros(batch_size, seq_len, 2)
        return est_pose, attn_map, decisions, probs

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # CNN
        self.opt = opt
        self.visual_encoder = visual_encoder((opt.img_h,opt.img_w), opt.patch_size, in_chans=3, num_classes=opt.v_f_len, embed_dim=opt.v_f_len, depth=8)
        self.inertial_encoder = Inertial_encoder(opt)

    def forward(self, img, imu):
        # shape=[16,10,2,3,256,512]
        v = torch.cat((img[:, :-1].unsqueeze(2), img[:, 1:].unsqueeze(2)), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)
        device = img.device
        # image CNN
        # v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4), v.size(5))
        #自己加入TSformer内容
        v_f = []
        for i in range(v.size(1)):
             tmp = self.visual_encoder(v[:,i]) 
             v_f.append(tmp)
        v_f = torch.stack(v_f,dim=1).to(device)
        # v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        # v = self.visual_head(v)  # (batch, seq_len, 256)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        return v_f, imu


# The fusion module
class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]

# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.i_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask

# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        self.fuse = Fusion_module(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select between fv and fv_alter
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc



class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        self.Policy_net = PolicyNet(opt)
        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses, decisions, logits= [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], None, hc)
            else:
                if selection == 'gumbel-softmax':
                    # Otherwise, sample the decision from the policy network
                    p_in = torch.cat((fi[:, i, :], hidden), -1)
                    logit, decision = self.Policy_net(p_in.detach(), temp)
                    decision = decision.unsqueeze(1)
                    logit = logit.unsqueeze(1)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
                elif selection == 'random':
                    decision = (torch.rand(fv.shape[0], 1, 2) < p).float()
                    decision[:,:,1] = 1-decision[:,:,0]
                    decision = decision.to(fv.device)
                    logit = 0.5*torch.ones((fv.shape[0], 1, 2)).to(fv.device)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)
        decisions = torch.cat(decisions, dim=1)
        logits = torch.cat(logits, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return poses, decisions, probs, hc


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
