import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
# from model import DeepVIO, Encoder_CAM, SVIO_VO, SVIO_VO_C
from original_deepvio import DeepVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math
import wandb 
from vo_transformer import VisualOdometryTransformerActEmbed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
# eccv2022 flowformer optical flow estimation
from flowformer_model import FlowFormer_VO
from flowformer_extractor_model import FlowFormer_Extractor_VO
from flowformer_extractor_nocorr_model import FlowFormer_Extractor_nocorr_VO
from FlowFormer_VO_part_corr_model import FlowFormer_VO_part_corr
from flowformer_vio import FlowFormer_VIO
from flowformer_vio_lstm import FlowFormer_VIO_LSTM
from gmflow.gmflow import GMFlow_VO
from svio_vo_corr import svio_vo_corr
from PWCnet_vo import pwcnet_vo
EPSILON = 1e-8

# from accelerate import Acceleraton
# accelerator = Accelerator(split_batches=True)
# device = accelerator.device
rank=0
# os.environ['CUDA_VISIBLE_DEVICES']='0,2'
# device = torch.device('cuda')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='/disk1/ljf/VO-Transformer/results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10', '01', '02', '04'], help='sequences for validation')
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

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=40, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_joint', type=float, default=5e-5, help='learning rate for joint training stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='debug', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

# parser.add_argument('--pretrain_flownet',type=str, default='./model_zoo/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain_flownet',type=str, default=None, help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')
parser.add_argument('--patch_size', type=int, default=16, help='patch token size')
parser.add_argument('--T', type=int, default=2, help='time transformer T=2')
parser.add_argument('--is_pretrained_mmae', type=bool, default=True, help='mmae backbone is_pretrained')
parser.add_argument('--teacher_model_type',type=str, default='deepvio', help='model type:[cvpr,deepvio,tscam]')
parser.add_argument('--student_model_type',type=str, default='flowformer_vio', help='model type:[cvpr,deepvio,tscam]')

parser.add_argument('--use_cnn',default=False, action='store_true', help='use flownet get cls_token')
parser.add_argument('--use_imu',default=False, action='store_true', help='use imu_encoder as cls_token')
# eccv2022 flowformer
parser.add_argument('--stage', type=str, default="kitti", help="determines which dataset to use for training") 
parser.add_argument('--regression_mode', type=int, default=2, help="determines which regress_mode to use for flowformer_vo") 
parser.add_argument('--add_part_weight', default=False, action='store_true', help="有此参数为true,是否加载flowformer部分权重文件") 
parser.add_argument('--kd_weight', default=0.1, type=float)

# 2022 GMFlow
parser.add_argument('--gmflow_feature_channels', default=128, type=int)
parser.add_argument('--gmflow_num_scales', default=1, type=int,
                    help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
parser.add_argument('--gmflow_upsample_factor', default=8, type=int)
parser.add_argument('--gmflow_num_head', default=1, type=int)
parser.add_argument('--gmflow_attention_type', default='swin', type=str)
parser.add_argument('--gmflow_ffn_dim_expansion', default=4, type=int)
parser.add_argument('--gmflow_num_transformer_layers', default=6, type=int)
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',help='self-attention radius for flow propagation, -1 indicates global attention')
parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',help='correlation radius for matching, -1 indicates global matching')

args = parser.parse_args()

if args.experiment_name != 'debug':
    wandb.init(
        # set the wandb project where this run will be logged
        entity="vio-research",
        project="VIO Research",
        name=args.experiment_name,
        
        # track hyperparameters and run metadata
        config={
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "lr_warmup": args.lr_warmup,
        "lr_joint": args.lr_joint,
        "lr_fine": args.lr_fine,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "epochs_warmup": args.epochs_warmup,
        "epochs_joint": args.epochs_joint,
        "epochs_fine": args.epochs_fine,
        "patch_size": args.patch_size,
        "workers": args.workers,
        "Lambda": args.Lambda,
        "v_f_len": args.v_f_len,
        "i_f_len":args.i_f_len,
        "T": args.T,
        "is_pretrained_mmae": args.is_pretrained_mmae,
        "teacher_model_type": args.teacher_model_type,
        "student_model_type": args.student_model_type,
        "use_cnn": args.use_cnn,
        "stage": args.stage,
        "regression_mode": args.regression_mode,
        "kd_weight": args.kd_weight,
        "add_part_weight": args.add_part_weight,
        "gmflow_feature_channels": args.gmflow_feature_channels,
        "gmflow_num_scales": args.gmflow_num_scales,
        "gmflow_upsample_factor": args.gmflow_upsample_factor,
        "gmflow_num_head": args.gmflow_num_head,
        "gmflow_attention_type": args.gmflow_attention_type,
        "attention_type": args.gmflow_attention_type,
        "ffn_dim_expansion": args.gmflow_ffn_dim_expansion,
        "num_transformer_layers": args.gmflow_num_transformer_layers,
        "prop_radius_list": args.prop_radius_list,
        "attn_splits_list": args.attn_splits_list,
        "corr_radius_list": args.corr_radius_list,
        }
    )

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# 修改超参数，按照cvpr2023的设置
def update_status(ep, args, model):
    if args.student_model_type == 'cvpr':
        if ep < 10:
            lr = (2e-4/10) * (ep+1)
        else:
            lr = 2e-4
        selection = 'gumbel-softmax'
        temp = args.temp_init
    else:
        if ep < args.epochs_warmup:  # Warmup stage
            lr = args.lr_warmup
            selection = 'random'
            temp = args.temp_init
            # for param in model.module.Policy_net.parameters(): # Disable the policy network
            #     param.requires_grad = False
        elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint: # Joint training stage
            lr = args.lr_joint
            selection = 'gumbel-softmax'
            temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
            # for param in model.module.Policy_net.parameters(): # Enable the policy network
            #     param.requires_grad = True
        elif ep >= args.epochs_warmup + args.epochs_joint: # Finetuning stage
            lr = args.lr_fine
            selection = 'gumbel-softmax'
            temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
    return lr, selection, temp

def train(model, teacher_model, optimizer, train_loader, selection, temp, logger, ep, iter, p=0.5, weighted=False, rank=0, model_type='cvpr', kd_lambda=0.1):
    # train中model_type指的都是student model
    mse_losses = []
    fkd_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight, intric) in enumerate(train_loader):

        imgs = imgs.cuda().float()# [b,s,c,h,w]
        imus = imus.cuda().float()# [b,101,6]
        gts = gts.cuda().float() 
        weight = weight.cuda().float()
        intric = intric.cuda().float()

        optimizer.zero_grad()
        # teacher model result
        teacher_poses, _, kd_teacher_feature = teacher_model(imgs, imus, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        kd_teacher_feature = kd_teacher_feature.squeeze(1)
        # 单单为添加为了mmae cvpr23所做的准备，别的模型删掉这些。
        batch_size,seq_len = imgs.shape[0], imgs.shape[1]
        device = imgs.device
        decisions = torch.zeros(batch_size, seq_len, 2).to(device)
        probs = torch.zeros(batch_size, seq_len, 2).to(device)

        # student model output
        if model_type == "flowformer_vio":
            img0 = imgs[:, 0]
            img1 = imgs[:, 1]
            imus = imus.unsqueeze(1)# 对于只估计两帧直接的变换来说，给imus添加seq维度
            poses, kd_student_feature = model(img0,img1,imus)

        # feature kd loss
        loss_fkd = kd_lambda * torch.dist(kd_teacher_feature, kd_student_feature, 2)

        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        # teacher model pose loss 可视化
        if not weighted:
            teacher_angle_loss = torch.nn.functional.mse_loss(teacher_poses[:,:,:3], gts[:, :, :3])
            teacher_translation_loss = torch.nn.functional.mse_loss(teacher_poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            teacher_angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (teacher_poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            teacher_translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (teacher_poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        teacher_poes_loss = 100 * teacher_angle_loss + teacher_translation_loss  
        teacher_poes_loss = teacher_poes_loss.float()
        # 100 -> 50
        pose_loss = 50 * angle_loss + translation_loss    
        pose_loss = pose_loss.float()    
        penalty = (decisions[:,:,0].float()).sum(-1).mean()

        loss = pose_loss + loss_fkd

        loss.backward()
        optimizer.step()

        iter = iter + 1
        if i % args.print_frequency == 0: 
            # message = f'Epoch: {ep}, iters: {i}/{data_len}, iter: {iter}/{data_len}, pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}, geo_inverse_loss: {geo_inverse_loss.item():.6f}, abs_diff_geo_inverse_rot: {abs_diff_geo_inverse_rot.item():.6f}, abs_diff_geo_inverse_pos_x: {abs_diff_geo_inverse_pos[0].item():.6f}, abs_diff_geo_inverse_pos_y: {abs_diff_geo_inverse_pos[1].item():.6f}, abs_diff_geo_inverse_pos_z: {abs_diff_geo_inverse_pos[2].item():.6f}'
            message = f'Epoch: {ep}, iters: {i}/{data_len}, iter: {iter}/{data_len}, teacher pose loss: {teacher_poes_loss.item():.6f}, student pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}, loss_fkd: {loss_fkd.item():.6f}'
            print(message)
            if args.experiment_name != 'debug':
                # wandb.log({"Epoch": ep, "iters": i, "iter": iter, "pose loss": pose_loss.item(),"penalty": penalty, "angle loss": angle_loss.item(), "translation loss": translation_loss.item(), "loss": loss.item(), "geo_inverse_loss": geo_inverse_loss.item(), "abs_diff_geo_inverse_rot": abs_diff_geo_inverse_rot.item(), "abs_diff_geo_inverse_pos_x": abs_diff_geo_inverse_pos[0].item(), "abs_diff_geo_inverse_pos_y": abs_diff_geo_inverse_pos[1].item(), "abs_diff_geo_inverse_pos_z": abs_diff_geo_inverse_pos[2].item()})
                wandb.log({"Epoch": ep, "iters": i, "iter": iter, "teacher pose loss": teacher_poes_loss.item(),"teacher angle loss": teacher_angle_loss.item(),"teacher translation loss": teacher_translation_loss.item(), "student pose loss": pose_loss.item(),"loss_fkd": loss_fkd.item(),"penalty": penalty, "student angle loss": angle_loss.item(), "student translation loss": translation_loss.item(), "loss": loss.item()})
            logger.info(message)

        mse_losses.append(pose_loss.item())
        fkd_losses.append(loss_fkd.item())
        penalties.append(penalty.item())

    return np.mean(mse_losses), np.mean(fkd_losses), np.mean(penalties), iter

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
# def main(rank, world_size):
def main():

    iter = 0
    ep = 0
    experiment_dir = Path('./new_results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()
    
    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # Load the dataset
    transform_train = [custom_transform.ToTensor(),
                       custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(args.data_dir,
                        sequence_length=args.seq_len,
                        train_seqs=args.train_seq,
                        transform=transform_train
                        )
    logger.info('train_dataset: ' + str(train_dataset))

    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device('cuda', local_rank)
    # Model initialization

    if args.teacher_model_type == 'deepvio':
        teacher_model = DeepVIO(args)
    if args.student_model_type == 'flowformer_vio':
        if args.stage == 'kitti':
            from flowformer.config.kitti import get_cfg
            cfg = get_cfg()
        student_model = FlowFormer_VIO(cfg['latentcostformer'],regression_mode=args.regression_mode)
    # teacher model加载权重，不进行训练
    teacher_state_dict = torch.load("model_zoo/vf_512_if_256_3e-05.model")
    teacher_model.load_state_dict(teacher_state_dict)
    # 冻结所有参数，包括BN以及dropout
    teacher_model.eval()
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
    # student load pretrained model
    if args.pretrain is not None:
        student_state_dict = torch.load(args.pretrain)
        student_state_dict = {k.replace("module.", "") if "module." in k else k: v for k, v in student_state_dict.items()}
        if args.student_model_type == "flowformer_vio": # 与flowformer_vo保持一致
            if args.add_part_weight:# flowformer_vio or 加载部分kitti权重
                for k in list(student_state_dict):
                    if "cost_perceiver_encoder" in k:
                        del student_state_dict[k]
            student_model.load_state_dict(student_state_dict,strict=False)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
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

    if args.student_model_type == 'flowformer_vio':
        for param in student_model.parameters():
            param.requires_grad = False
        for param in student_model.inertial_encoder_conv.parameters():
            param.requires_grad = True
        for param in student_model.inertial_proj.parameters():
            param.requires_grad = True
        for param in student_model.pose_regressor.parameters():
            param.requires_grad = True
        if args.regression_mode == 2:
            for param in student_model.visual_regressor_vio_1.parameters():
                param.requires_grad = True
            for param in student_model.visual_regressor_vio_2.parameters():
                param.requires_grad = True
        if args.add_part_weight:
            # 可以更新corss attention部分，CostPerceiverEncoder其他部分不参与forward也不参与更新
            for param in student_model.memory_encoder.cost_perceiver_encoder.input_layer.parameters():
                param.requires_grad = True
            for param in student_model.memory_encoder.cost_perceiver_encoder.patch_embed.parameters():
                param.requires_grad = True
            student_model.memory_encoder.cost_perceiver_encoder.latent_tokens.requires_grad = True

    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)

    # Feed teacher and student model to GPU
    student_model.cuda(gpu_ids[0])
    student_model = torch.nn.DataParallel(student_model, device_ids = gpu_ids)
    teacher_model.cuda(gpu_ids[0])
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids = gpu_ids)

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None and args.student_model_type=="deepvio" else 0    
    iter = init_epoch*(len(train_loader) // args.batch_size + 1) if args.pretrain is not None else 0
    
    best = 10000
    for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
    # while(iter < args.iter_warmup+args.iter_joint+args.iter_fine):
        lr, selection, temp = update_status(ep, args, student_model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, iter: {iter}, lr: {lr}, selection: {selection}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        student_model.train()
        avg_pose_loss, avg_kd_loss, avg_penalty_loss, iter = train(student_model, teacher_model, optimizer, train_loader, selection, temp, logger, ep, iter, p=0.5, rank=rank, model_type=args.student_model_type, kd_lambda=args.kd_weight)

        # Save the model after training
        # if rank==0:
        if(os.path.isfile(f'{checkpoints_dir}/{(ep-1):003}.pth')):# and ep%10!=1
            os.remove(f'{checkpoints_dir}/{(ep-1):003}.pth')
        torch.save(student_model.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        # dist.barrier()
        # Save the model after training
        message = f'Epoch {ep} iter {iter}training finished, pose loss: {avg_pose_loss:.6f}, penalty_loss: {avg_penalty_loss:.6f}, kd_loss: {avg_kd_loss:.6f}, model saved'
        print(message)
        logger.info(message)

        # if ep > args.epochs_warmup+args.epochs_joint or ep%10==0:
        # 这个也需要更改，这里是update all batch_size=16,所以所有的epoch除3
        if ep%10==9:
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad():
                student_model.eval()
                errors = tester.eval(student_model, selection='gumbel-softmax', num_gpu=len(gpu_ids))
        
            t_rel = np.mean([errors[i]['t_rel'] for i in range(3)])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(3)])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(3)])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(3)])
            usage = np.mean([errors[i]['usage'] for i in range(3)])

            if t_rel < best:
                best = t_rel 
                if best < 10:
                    torch.save(student_model.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')

            message = f'Epoch {ep} evaluation finished , Test_Average_t_rel: {t_rel:.4f}, Test_Average_r_rel: {r_rel:.4f}, Test_Average_t_rmse: {t_rmse:.4f}, Test_Average_r_rmse: {r_rmse:.4f}, Test_Average_usage: {usage:.4f}, Test_Average_best t_rel: {best:.4f}'
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "Test_Average_t_rel": round(t_rel, 4), "Test_Average_r_rel": round(r_rel, 4), "Test_Average_t_rmse": round(t_rmse, 4), "Test_Average_r_rmse": round(r_rmse, 4), "Test_Average_usage": round(usage, 4), "Test_Average_best t_rel": round(best, 4)})
            
            message = "Epoch {} iter {} evaluation Seq. 05 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[0]['t_rel'], 4), round(errors[0]['r_rel'], 4), round(errors[0]['t_rmse'], 4), round(errors[0]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "5. t_rel": round(errors[0]['t_rel'], 4), "5. r_rel": round(errors[0]['r_rel'], 4), "5. t_rmse": round(errors[0]['t_rmse'], 4), "5. r_rmse": round(errors[0]['r_rmse'], 4)})

            message = "Epoch {} iter {} evaluation Seq. 07 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[1]['t_rel'], 4), round(errors[1]['r_rel'], 4), round(errors[1]['t_rmse'], 4), round(errors[1]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "7. t_rel": round(errors[1]['t_rel'], 4), "7. r_rel": round(errors[1]['r_rel'], 4), "7. t_rmse": round(errors[1]['t_rmse'], 4), "7. r_rmse": round(errors[1]['r_rmse'], 4)})

            message = "Epoch {} iter {} evaluation Seq. 10 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[2]['t_rel'], 4), round(errors[2]['r_rel'], 4), round(errors[2]['t_rmse'], 4), round(errors[2]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "10. t_rel": round(errors[2]['t_rel'], 4), "10. r_rel": round(errors[2]['r_rel'], 4), "10. t_rmse": round(errors[2]['t_rmse'], 4), "10. r_rmse": round(errors[2]['r_rmse'], 4)})
            
            message = "Epoch {} iter {} evaluation Seq. 01 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[3]['t_rel'], 4), round(errors[3]['r_rel'], 4), round(errors[3]['t_rmse'], 4), round(errors[3]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "01. t_rel": round(errors[3]['t_rel'], 4), "01. r_rel": round(errors[3]['r_rel'], 4), "01. t_rmse": round(errors[3]['t_rmse'], 4), "01. r_rmse": round(errors[3]['r_rmse'], 4)})
            
            message = "Epoch {} iter {} evaluation Seq. 02 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[4]['t_rel'], 4), round(errors[4]['r_rel'], 4), round(errors[4]['t_rmse'], 4), round(errors[4]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "02. t_rel": round(errors[4]['t_rel'], 4), "02. r_rel": round(errors[4]['r_rel'], 4), "02. t_rmse": round(errors[4]['t_rmse'], 4), "02. r_rmse": round(errors[4]['r_rmse'], 4)})
            
            message = "Epoch {} iter {} evaluation Seq. 04 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[5]['t_rel'], 4), round(errors[5]['r_rel'], 4), round(errors[5]['t_rmse'], 4), round(errors[5]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "04. t_rel": round(errors[5]['t_rel'], 4), "04. r_rel": round(errors[5]['r_rel'], 4), "04. t_rmse": round(errors[5]['t_rmse'], 4), "04. r_rmse": round(errors[5]['r_rmse'], 4)})
            
            logger.info(message)
            print(message)
    
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)



if __name__ == "__main__":
    # DP training
    main()

# if model_type == "flowformer_vio_lstm":
        #     # imgs_seq = torch.cat((imgs[:, :-1], imgs[:, 1:]), dim=2)
        #     poses, _ = model(imgs,imus,hc=None)
        # elif model_type == "deepvio" or model_type == "svio_vo":
        #     poses = model(imgs)
        # elif model_type == "svio_vo_corr":
        #     poses = model(imgs)
        # elif model_type == "pwcnet_vo":
        #     poses = model(imgs)
        # elif model_type == "flowformer_vo":
        #     img0 = imgs[:, 0]
        #     img1 = imgs[:, 1]
        #     poses = model(img0,img1)
        # elif model_type == "flowformer_extractor_vo":
        #     img0 = imgs[:, 0]
        #     img1 = imgs[:, 1]
        #     poses = model(img0,img1)
        # elif model_type == "flowformer_vo_part_corr":
        #     img0 = imgs[:, 0]
        #     img1 = imgs[:, 1]
        #     poses = model(img0,img1)
        # elif model_type == "flowformer_extractor_nocorr_vo":
        #     img0 = imgs[:, 0]
        #     img1 = imgs[:, 1]
        #     poses = model(img0,img1)
        # elif model_type == "flowformer_vio":
        #     img0 = imgs[:, 0]
        #     img1 = imgs[:, 1]
        #     imus = imus.unsqueeze(1)# 对于只估计两帧直接的变换来说，给imus添加seq维度
        #     poses = model(img0,img1,imus)
        # elif model_type == "gmflow_vo":
        #     poses = model(imgs[:,0],imgs[:,1],attn_splits_list=args.attn_splits_list,corr_radius_list=args.corr_radius_list,prop_radius_list=args.prop_radius_list)