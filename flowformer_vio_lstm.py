# import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from flowformer.encoders import twins_svt_large_context, twins_svt_large
from flowformer_lstm_encoder import MemoryEncoder # 针对lstm优化
from flowformer.cnn import BasicEncoder
import torch.nn.functional as F

class FlowFormer_VIO_LSTM(nn.Module):
    def __init__(self, cfg, regression_mode=2):
        super(FlowFormer_VIO_LSTM, self).__init__()
        self.cfg = cfg
        self.regression_mode = regression_mode
        self.memory_encoder = MemoryEncoder(cfg)
        # self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        # regress pose
        # imu encoder
        self.imu_dropout = 0.
        self.inertial_encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.imu_dropout))
        self.inertial_proj = nn.Linear(256 * 1 * 11, 256)
        # visual encoder
        # 1 
        self.out_dim = cfg.cost_latent_token_num * cfg.cost_latent_dim
        if self.regression_mode == 1:
            self.visual_regressor_vio = nn.Sequential(
                nn.Linear(int(self.out_dim), 128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 6),
                )
        # 2
        if self.regression_mode == 2:
            self.out_dim = cfg.cost_latent_token_num * cfg.cost_latent_dim
            self.visual_regressor_vio_1 = nn.Sequential(
                nn.Linear(int(self.out_dim), 128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 8),
                )
            self.visual_regressor_vio_2 = nn.Sequential(
                nn.Linear(int(8*cfg.image_size[0]*cfg.image_size[1]/64), 2048),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.1, inplace=True),
                # nn.Linear(512, 128),
                # nn.LeakyReLU(0.1, inplace=True),
                # nn.Linear(128, 6),
                )
        # 3
        if self.regression_mode == 3:
            self.visual_regressor_vio_3 = nn.Sequential(
                nn.Linear(int(self.out_dim*(cfg.image_size[0]/8//7)*(cfg.image_size[1]/8//7)), 2048),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.1, inplace=True),
                # nn.Linear(512, 128),
                # nn.LeakyReLU(0.1, inplace=True),
                # nn.Linear(128, 6),
            )
        self.pose_net = Pose_RNN()
        # self.pose_regressor = nn.Sequential(
        #     nn.Linear(768, 128),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(128, 6))
        

    def forward(self, imgs, imu, hc=None):
        # Following https://github.com/princeton-vl/RAFT/
        # imu encoder
        batch_size, seq_len, channel, oh,ow = imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len - 1)], dim=1)
        imu = imu.view(batch_size * (seq_len-1), imu.size(2), imu.size(3))    # x: (N x seq_len, 11, 6)
        imu = self.inertial_encoder_conv(imu.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        imu_feat = self.inertial_proj(imu.view(imu.shape[0], -1))      
        imu_feat = imu_feat.view(batch_size, seq_len-1, 256)# [b,s,d]
        all_visual_feat = []
        # for i in range(seq_len):
        # image1 = imgs[:,i,:3]
        # image2 = imgs[:,i,3:]
        # visual encoder
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image = 2 * (imgs / 255.0) - 1.0
        image = image.view(batch_size*seq_len,channel,oh,ow)
        data = {}
        if self.cfg.context_concat:# false
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image)# transformer提context feature
        # 与之前for循环相比，context输入是[b,seq,c,h,w], 
        context = context.view(batch_size,seq_len,context.shape[1],context.shape[2],context.shape[3])
        image = image.view(batch_size,seq_len,channel,oh,ow)
        cost_memory, B, C, H, W = self.memory_encoder(image, data, context[:,:-1])
        # flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
        # return flow_predictions
        # regress pose
        # 1. 旧做法
        if self.regression_mode == 1:
            x = cost_memory.reshape(B, self.cfg.cost_latent_token_num, self.cfg.predictor_dim, -1)
            x = torch.mean(x, dim=-1)
            out = x.reshape(B,-1)
            # out = self.rnn_drop_out(out)
            pose = self.visual_regressor_vio(out)
        # 2.先reshape 128*8->8，再reshapeH*W*8
        elif self.regression_mode == 2:
            x = cost_memory.reshape(B*(seq_len-1),H,W,-1)
            x = self.visual_regressor_vio_1(x)
            x = x.reshape(B*(seq_len-1),-1)
            visual_feat = self.visual_regressor_vio_2(x)
        # 3. 先downsample H*W，再reshape到低维
        elif self.regression_mode == 3:
            avg_pool_feat = F.avg_pool2d(cost_memory.reshape(B*(seq_len-1),H,W,-1).permute(0,3,1,2), kernel_size=7, stride=7)
            visual_feat = self.visual_regressor_vio_3(avg_pool_feat.reshape(B,-1))
        visual_feat = visual_feat.view(B,seq_len-1,-1)
        # concat visual_feat and imu_feat, predict pose
        poses = []
        for i in range(seq_len-1):
            pose, hc = self.pose_net(visual_feat[:,i:i+1,:], imu_feat[:,i:i+1,:], hc)
            poses.append(pose)
        poses = torch.cat(poses,dim=1)
        return poses,hc
        # return cost_memory

class Pose_RNN(nn.Module):
    def __init__(self):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = 768
        rnn_hidden_size = 1024
        rnn_dropout_out = 0.2
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            dropout=rnn_dropout_out,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fi, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        fused = torch.cat([fv, fi], dim=-1)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc