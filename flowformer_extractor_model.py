# import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

# from utils.utils import coords_grid, bilinear_sampler, upflow8
# from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from flowformer.encoders import twins_svt_large_context, twins_svt_large
# from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
# from .twins import PosConv
# from flowformer.encoder import MemoryEncoder
# from .decoder import MemoryDecoder
from flowformer.cnn import BasicEncoder
import torch.nn.functional as F
# new MemoryEncoder for extractor_vo
class MemoryEncoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg

        if cfg.fnet == 'twins':
            self.feat_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            self.feat_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        else:
            exit()
        self.channel_convertor = nn.Conv2d(cfg.encoder_latent_dim, cfg.encoder_latent_dim, 1, padding=0, bias=False)

        # The output networks
        self.rnn_drop_out = nn.Dropout(0.2)
        # self.out_dim = cfg.cost_latent_token_num * cfg.cost_latent_dim * cfg.image_size[0]/8 * cfg.image_size[1]/8

    def corr(self, fmap1, fmap2): # fmap1 = source && fmap2 = target

        batch, dim, ht, wd = fmap1.shape
        fmap1 = rearrange(fmap1, 'b (heads d) h w -> b heads (h w) d', heads=self.cfg.cost_heads_num)# heads=1
        fmap2 = rearrange(fmap2, 'b (heads d) h w -> b heads (h w) d', heads=self.cfg.cost_heads_num)
        corr = einsum('bhid, bhjd -> bhij', fmap1, fmap2)
        corr = corr.permute(0, 2, 1, 3).view(batch*ht*wd, self.cfg.cost_heads_num, ht, wd)
        #corr = self.norm(self.relu(corr))
        corr = corr.view(batch, ht*wd, self.cfg.cost_heads_num, ht*wd).permute(0, 2, 1, 3)
        corr = corr.view(batch, self.cfg.cost_heads_num, ht, wd, ht, wd)

        return corr # [b,head,h,w,h,w]

    def forward(self, img1, img2, data, context=None):
        # The original implementation
        # feat_s = self.feat_encoder(img1)
        # feat_t = self.feat_encoder(img2)
        # feat_s = self.channel_convertor(feat_s)
        # feat_t = self.channel_convertor(feat_t)

        imgs = torch.cat([img1, img2], dim=0)
        feats = self.feat_encoder(imgs) # 与context feature 提取方式一样
        feats = self.channel_convertor(feats)
        B = feats.shape[0] // 2

        feat_s = feats[:B]
        feat_t = feats[B:]

        B, C, H, W = feat_s.shape
        size = (H, W)

        cost_volume = self.corr(feat_s, feat_t)
        
        return cost_volume, B, C, H, W

# flowformer transformer feature extractor
class FlowFormer_Extractor_VO(nn.Module):
    def __init__(self, cfg, regression_mode=2):
        super(FlowFormer_Extractor_VO, self).__init__()
        self.cfg = cfg
        self.memory_encoder = MemoryEncoder(cfg)

        # regress pose
        self.regression_mode = regression_mode
        # 1
        self.out_dim = cfg.cost_latent_token_num * cfg.cost_latent_dim
        if self.regression_mode == 1:
            self.regressor = nn.Sequential(
                nn.Linear(int(self.out_dim), 128),
                nn.LeakyReLU(0.1, inplace=True),
                # nn.Linear(1024, 128),
                # nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 6),
                )
        # 2 跟之前的regressor稍微有点区别，input[b,h,w,h,w]->reshape [b,h,w,-1]
        if self.regression_mode == 2:
            self.out_dim = cfg.image_size[0]*cfg.image_size[1]/64
            self.regressor_1 = nn.Sequential(
                nn.Linear(int(self.out_dim), 128),
                nn.LeakyReLU(0.1, inplace=True),
                # nn.Linear(1024, 128),
                # nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 8),
                )
            self.regressor_2 = nn.Sequential(
                nn.Linear(int(8*cfg.image_size[0]*cfg.image_size[1]/64), 2048),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(512, 128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 6),
                )
        # 3
        if self.regression_mode == 3:
            self.regressor_3 = nn.Sequential(
                nn.Linear(int(self.out_dim*(cfg.image_size[0]/8//7)*(cfg.image_size[1]/8//7)), 2048),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(512, 128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 6),
            )
    def forward(self, image1, image2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}
        cost_memory, B, C, H, W = self.memory_encoder(image1, image2, data, None)

        # regress pose
        # 1. 旧做法
        if self.regression_mode == 1:
            x = cost_memory.reshape(B, self.cfg.cost_latent_token_num, self.cfg.predictor_dim, -1)
            x = torch.mean(x, dim=-1)
            out = x.reshape(B,-1)
            out = self.rnn_drop_out(out)
            pose = self.regressor(out)
        # 2.先reshape 128*8->8，再reshapeH*W*8
        elif self.regression_mode == 2:
            x = cost_memory.reshape(B,H,W,-1)
            x = self.regressor_1(x)
            x = x.reshape(B,-1)
            pose = self.regressor_2(x)
        elif self.regression_mode == 3:
            avg_pool_feat = F.avg_pool2d(cost_memory.reshape(B,H,W,-1).permute(0,3,1,2), kernel_size=7, stride=7)
            pose = self.regressor_3(avg_pool_feat.reshape(B,-1))
        # 3. 先downsample H*W，再reshape到低维
        return pose.unsqueeze(1)
        # return cost_memory