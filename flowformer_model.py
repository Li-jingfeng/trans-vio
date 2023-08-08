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
from flowformer.encoder import MemoryEncoder
# from .decoder import MemoryDecoder
from flowformer.cnn import BasicEncoder
import torch.nn.functional as F

class FlowFormer_VO(nn.Module):
    def __init__(self, cfg, regression_mode=3):
        super(FlowFormer_VO, self).__init__()
        self.cfg = cfg
        self.regression_mode = regression_mode
        self.memory_encoder = MemoryEncoder(cfg)
        # self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        # regress pose
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
        # 2
        if self.regression_mode == 2:
            self.out_dim = cfg.cost_latent_token_num * cfg.cost_latent_dim
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

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)# transformer提context feature
            
        cost_memory, B, C, H, W = self.memory_encoder(image1, image2, data, context)

        # flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
        # return flow_predictions

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