import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from svio_vo_corr_util import conv_flow, conv, predict_flow, deconv, crop_like, correlate, VOFlowRes
from einops import rearrange
from torch import einsum

__all__ = [
    'flownetc', 'flownetc_bn'
]


class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=False):
        super(FlowNetC,self).__init__()

        self.batchNorm = batchNorm
        self.conv1      = conv_flow(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2      = conv_flow(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3      = conv_flow(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir = conv_flow(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        self.conv3_1 = conv_flow(self.batchNorm, 473,  256)
        self.conv4   = conv_flow(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv_flow(self.batchNorm, 512,  512)
        self.conv5   = conv_flow(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv_flow(self.batchNorm, 512,  512)
        self.conv6   = conv_flow(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv_flow(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:,0]
        x2 = x[:,1]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a,out_conv3b) # 

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)# 16,473,24,43
        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        # if self.training:
        #     return flow2,flow3,flow4,flow5,flow6
        # else:
        # return in_conv3_1, out_correlation, out_conv3a, out_conv3b
        return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class svio_vo_corr(nn.Module):
    def __init__(self, args):
        super(svio_vo_corr, self).__init__()

        self.Feature_net = FlowNetC()
        self.flow_posenet = VOFlowRes()
        # self.out_dim = 473
        # self.out_dim1 = 441
        # self.visual_head_1 = nn.Sequential(
        #         nn.Linear(473, 128),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(128, 8),
        #         )
        # self.visual_head_2 = nn.Sequential(
        #         nn.Linear(int(8*args.img_w*args.img_h/64), 1024),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(1024, 256),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(256, 8),
        #         )

        # experiment_name = svio_vo_corr_conv
        # self.visual_head_1 = nn.Sequential(
        #         nn.Linear(int(args.img_w*args.img_h/64), 1024),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(1024, 128),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(128, 8),
        #         )
        # self.visual_head_2 = nn.Sequential(
        #         nn.Linear(int(8*self.out_dim), 1024),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(1024, 256),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(256, 128),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(128, 6),
        #         )

        # experiment_name = svio_vo_corr_conv_only_corr
        # self.visual_head_1 = nn.Sequential(
        #         nn.Linear(int(args.img_w*args.img_h/64), 1024),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(1024, 128),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(128, 8),
        #         )
        # self.visual_head_2 = nn.Sequential(
        #         nn.Linear(int(8*self.out_dim1), 1024),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(1024, 256),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(256, 128),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(128, 6),
        #         )

        # experiment_name = svio_vo_corr_conv_only_corr
        # self.visual_head_1 = nn.Sequential(
        #         nn.Linear(int((args.img_h/8)*((args.img_w/8))), 1024),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(1024, 128),
        #         nn.LeakyReLU(0.1, inplace=True),
        #         nn.Linear(128, 8),
        #         )
        # self.visual_head_2 = nn.Sequential(
        #     nn.Linear(int(8*(int(args.img_h/8)) * int(args.img_w/8)), 2048),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(2048, 512),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(512, 128),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(128, 6),
        #     )

    def forward(self, img, is_first=True):

        # fv1, corr_fv, out_conv3a, out_conv3b = self.Feature_net(img)# corr_fv: 16,441,32,60
        flow2 = self.Feature_net(img)
        # experiment_name == only corr_feature
        fv = flow2

        # batch_size = fv.shape[0]
        # source_patch = fv.shape[1]
        poses = []
        
        pose = self.flow_posenet(fv)
        poses.append(pose)

        poses = torch.cat(poses, dim=1)

        # experiment_name == dot_corr
        # cost_memory = corr(out_conv3a,out_conv3b)
        # B,D,H,W = out_conv3b.size()
        # x = cost_memory.reshape(B,H,W,-1)
        # # window_size注释掉regressor1
        # x = self.visual_head_1(x)
        # x = x.reshape(B,-1)
        # poses = self.visual_head_2(x)
        return poses.unsqueeze(1)

def flownetc(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownetc_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
# 本来是cuda和c++的库，现在改成之前的版本
def corr(fmap1, fmap2): # fmap1 = source && fmap2 = target

        batch, dim, ht, wd = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape
        fmap1 = rearrange(fmap1, 'b (heads d) h w -> b heads (h w) d', heads=1)# heads=1
        fmap2 = rearrange(fmap2, 'b (heads d) h w -> b heads (h w) d', heads=1)
        corr = einsum('bhid, bhjd -> bhij', fmap1, fmap2)
        corr = corr.permute(0, 2, 1, 3).view(batch*ht*wd, 1, ht2, wd2)
        #corr = self.norm(self.relu(corr))
        corr = corr.view(batch, ht*wd, 1, ht2*wd2).permute(0, 2, 1, 3)
        corr = corr.view(batch, 1, ht, wd, ht2, wd2)

        return corr # [b,head,h,w,h,w]