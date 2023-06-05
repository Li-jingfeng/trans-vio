import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from einops import rearrange, repeat
import warnings

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
plt.switch_backend('agg')

_EPS = np.finfo(float).eps * 4.0

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def isRotationMatrix(R):
    '''
    check whether a matrix is a qualified rotation metrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_relative_pose(Rt1, Rt2):
    '''
    Calculate the relative 4x4 pose matrix between two pose matrices
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel

def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    Calculate the relative rotation and translation from two consecutive pose matrices 
    '''
    
    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

def rotationError(Rt1, Rt2):
    '''
    Calculate the rotation difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(Rt1, Rt2):
    '''
    Calculate the translational difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def eulerAnglesToRotationMatrix(theta):
    '''
    Calculate the rotation matrix from eular angles (roll, yaw, pitch)
    '''
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def normalize_angle_delta(angle):
    '''
    Normalization angles to constrain that it is between -pi and pi
    '''
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def pose_6DoF_to_matrix(pose):
    '''
    Calculate the 3x4 transformation matrix from Eular angles and translation vector
    '''
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)
    R = np.concatenate((R, t), 1)
    R = np.concatenate((R, np.array([[0, 0, 0, 1]])), 0)
    return R

def pose_accu(Rt_pre, R_rel):
    '''
    Calculate the accumulated pose from the latest pose and the relative rotation and translation
    '''
    Rt_rel = pose_6DoF_to_matrix(R_rel)
    return Rt_pre @ Rt_rel

def path_accu(pose):
    '''
    Generate the global pose matrices from a series of relative poses
    '''
    answer = [np.eye(4)]
    for index in range(pose.shape[0]):
        pose_ = pose_accu(answer[-1], pose[index, :])
        answer.append(pose_)
    return answer

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def rmse_err_cal(pose_est, pose_gt):
    '''
    Calculate the rmse of relative translation and rotation
    '''
    t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
    return t_rmse, r_rmse

def trajectoryDistances(poses):
    '''
    Calculate the distance and speed for each frame
    '''
    dist = [0]
    speed = [0]
    for i in range(len(poses) - 1):
        cur_frame_idx = i
        next_frame_idx = cur_frame_idx + 1
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        speed.append(np.sqrt(dx**2 + dy**2 + dz**2) * 10)
    return dist, speed

def lastFrameFromSegmentLength(dist, first_frame, len_):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + len_):
            return i
    return -1

def computeOverallErr(seq_err):
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)

    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err

def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values= np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt
    
def read_pose_from_text(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        values_p = read_pose(lines[0])
        poses_abs.append(values_p)            
        for i in range(1, len(lines)):
            values = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(values_p, values)) 
            values_p = values.copy()
            poses_abs.append(values) 
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
        
    return poses_abs, poses_rel

def saveSequence(poses, file_name):
    with open(file_name, 'w') as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            f.write(' '.join([str(r) for r in pose]))
            f.write('\n')

#vit transformer
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)
# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# From PyTorch internals
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(256, 512), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W
    
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)