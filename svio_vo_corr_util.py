import torch.nn as nn
import torch.nn.functional as F
import torch
# from spatial_correlation_sampler import spatial_correlation_sample
# try:
#     from spatial_correlation_sampler import spatial_correlation_sample
# except ImportError as e:
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("default", category=ImportWarning)
    #     warnings.warn("failed to load custom correlation module"
    #                   "which is needed for FlowNetC", ImportWarning)

# 这是flownetC本身的conv
def conv_flow(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
# 这是为了回归flownetC预测到的flow2所声明
def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

# 增加了flownetC decoder(predict flow)部分，所以对应的linear回归pose需要修改
def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

class VOFlowRes(nn.Module):
    def __init__(self):
        super(VOFlowRes, self).__init__()
        inputnum = 2  # not 2+1+1+1
        blocknums = [2,2,3,4,6,7,3]
        outputnums = [32,64,64,128,128,256,256]

        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))

        self.inplanes = 32

        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1) # 40 x 28
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1) # 20 x 14
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1) # 10 x 7
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1) # 5 x 4
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1) # 3 x 2
        fcnum = 512 #outputnums[6] * 6

        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)


        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)


    def _make_layer(self, block, planes, blocks, stride, pad=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_rot, x_trans), dim=1)