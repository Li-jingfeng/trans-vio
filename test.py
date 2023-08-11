import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import DeepVIO,Encoder_CAM
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math
from vo_transformer import VisualOdometryTransformerActEmbed
from flowformer_model import FlowFormer_VO
from flowformer_vio import FlowFormer_VIO
from flowformer_vio_lstm import FlowFormer_VIO_LSTM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='/disk1/ljf/VO-Transformer/results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['04'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--experiment_name', type=str, default='test', help='experiment name')
parser.add_argument('--model', type=str, default='./model_zoo/vf_512_if_256_3e-05.model', help='path to the pretrained model')
parser.add_argument('--patch_size', type=int, default=16, help='patch token size')
parser.add_argument('--T', type=int, default=2, help='time transformer T=2')
parser.add_argument('--is_pretrained_mmae', type=bool, default=False, help='mmae backbone is_pretrained')
parser.add_argument('--model_type',type=str, default='cvpr', help='model type:[cvpr,deepvio,tscam]')
parser.add_argument('--use_cnn',default=False, action='store_true', help='use flownet get cls_token')
parser.add_argument('--use_imu',default=False, action='store_true', help='use imu_encoder as cls_token')
parser.add_argument('--stage', type=str, default="kitti", help="determines which dataset to use for training") 
parser.add_argument('--regression_mode', type=int, default=2, help="determines which regress_mode to use for flowformer_vo") 

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():

    # Create Dir
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    result_dir = file_dir.joinpath('files/')
    result_dir.mkdir_p()
    
    # GPU selections
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    # Initialize the tester
    tester = KITTI_tester(args)

    # Model initialization
    # model = DeepVIO(args)
    # model = Encoder_CAM(args)
    if args.model_type == 'cvpr':
        model = VisualOdometryTransformerActEmbed(cls_action=False,is_pretrained_mmae=True,use_cnn=args.use_cnn, use_imu=args.use_imu)
    elif args.model_type == 'flowformer_vo':
        from flowformer.config.kitti import get_cfg
        cfg = get_cfg()
        model = FlowFormer_VO(cfg['latentcostformer'])
    elif args.model_type == 'flowformer_vio':
        from flowformer.config.kitti import get_cfg
        cfg = get_cfg()
        model = FlowFormer_VIO(cfg['latentcostformer'], regression_mode=args.regression_mode)
    elif args.model_type == 'flowformer_vio_lstm':
        from flowformer.config.kitti import get_cfg
        cfg = get_cfg()
        model = FlowFormer_VIO_LSTM(cfg['latentcostformer'], regression_mode=args.regression_mode)
    
    weights = torch.load(args.model)
    new_state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(new_state_dict, strict=True)
    print('load model %s'%args.model)
        
    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)
    model.eval()

    errors = tester.eval(model, 'gumbel-softmax', num_gpu=len(gpu_ids))
    tester.generate_plots(result_dir, 30)
    tester.save_text(result_dir)
    
    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
        message += f"usage: {errors[i]['usage']:.4f}"
        print(message)
    
    

if __name__ == "__main__":
    main()




