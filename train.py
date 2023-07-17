import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import DeepVIO, Encoder_CAM, FlowVO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math
import wandb 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

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
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
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

parser.add_argument('--pretrain_flownet',type=str, default='./model_zoo/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
# parser.add_argument('--pretrain_flownet',type=str, default=None, help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')
parser.add_argument('--patch_size', type=int, default=16, help='patch token size')
parser.add_argument('--T', type=int, default=2, help='time transformer T=2')
parser.add_argument('--iter_warmup', type=int, default=10000, help='for iter not epoch')
parser.add_argument('--iter_joint', type=int, default=20000, help='for iter not epoch')
parser.add_argument('--iter_fine', type=int, default=20000, help='for iter not epoch')
parser.add_argument('--epoch_or_iter', action='store_true', default=False, help='true is epoch, false is iter')
parser.add_argument('--model_type', type=str, default='flowVO', help='path to the pretrained model')


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
        "epoch_or_iter": args.epoch_or_iter,
        "iter_warmup": args.iter_warmup,
        "iter_joint": args.iter_joint,
        "iter_fine": args.iter_fine,
        "model_type": args.model_type,
        }
    )

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def update_status(ep, args, model):
    if args.epoch_or_iter:
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
    # var ep==iter
    else:
        if ep < args.iter_warmup:
            lr = args.lr_warmup
        elif ep >= args.iter_warmup and ep < args.iter_warmup + args.iter_joint:
            lr = args.lr_joint
        elif ep >= args.iter_warmup + args.iter_joint:
            lr = args.lr_fine
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))

    return lr, selection, temp

def train(model, optimizer, train_loader, selection, temp, logger, ep, iter, p=0.5, weighted=False):
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float() 
        weight = weight.cuda().float()

        optimizer.zero_grad()
        # 模型结果
        poses, decisions, probs = model(imgs)

        
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = 100 * angle_loss + translation_loss        
        penalty = (decisions[:,:,0].float()).sum(-1).mean()
        # loss = pose_loss + args.Lambda * penalty 
        loss = pose_loss
        
        loss.backward()
        optimizer.step()

        iter = iter + 1
        
        if i % args.print_frequency == 0: 
            message = f'Epoch: {ep}, iter: {iter}/{data_len}, pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}'
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter": iter, "pose loss": pose_loss.item(),"penalty": penalty, "angle loss": angle_loss.item(), "translation loss": translation_loss.item(), "loss": loss.item()})
            logger.info(message)

        mse_losses.append(pose_loss.item())
        penalties.append(penalty.item())
        if iter > (args.iter_warmup + args.iter_warmup + args.iter_joint):
            break

    return np.mean(mse_losses), np.mean(penalties), iter



def main():
    iter = 0
    ep = 0 
    # Create Dir
    experiment_dir = Path('./results')
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
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

    # Model initialization
    if args.model_type=='flowVO':
        model = FlowVO(args)
    elif args.model_type=='ts_cam':
        model = Encoder_CAM(args)
    elif args.model_type=='deepvio':
        model = DeepVIO(args)

    # Continual training or not
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    
    # Use the pre-trained flownet or not
    if args.model_type=='flowVO' and args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
        model_dict = model.Feature_net.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)

    pretrain = args.pretrain
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None else 0
    iter = init_epoch * (len(train_loader) // args.batch_size + 1) if args.pretrain is not None else 0
    
    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    
    best = 10000

    # 如果还是想使用ep来控制，那么就将main函数上面的ep=0给注释掉 最底下的ep=ep+1也给注释掉
    # for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
    while(iter < args.iter_warmup+args.iter_joint+args.iter_fine):
        
        lr, selection, temp = update_status(iter, args, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, iter: {iter} lr: {lr}, selection: {selection}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss, avg_penalty_loss, iter = train(model, optimizer, train_loader, selection, temp, logger, ep, iter=iter, p=0.5)
        
        # Save the model after training
        if(os.path.isfile(f'{checkpoints_dir}/{(ep-1):003}.pth') and ep%10!=0):
            os.remove(f'{checkpoints_dir}/{(ep-1):003}.pth')
        torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        # Save the model after training
        message = f'Epoch {ep} iter: {iter} training finished, pose loss: {avg_pose_loss:.6f}, penalty_loss: {avg_penalty_loss:.6f}, model saved'
        print(message)
        logger.info(message)
        
        if ep % 5 == 0 or iter == args.iter_warmup+args.iter_joint+args.iter_fine-1:
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad():  
                model.eval()
                errors = tester.eval(model, selection='gumbel-softmax', num_gpu=len(gpu_ids))
        
            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            usage = np.mean([errors[i]['usage'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel 
                if best < 10:
                    torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')

            message = "Epoch {} iter: {} evaluation Seq. 05 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[0]['t_rel'], 4), round(errors[0]['r_rel'], 4), round(errors[0]['t_rmse'], 4), round(errors[0]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "5. t_rel": round(errors[0]['t_rel'], 4), "5. r_rel": round(errors[0]['r_rel'], 4), "5. t_rmse": round(errors[0]['t_rmse'], 4), "5. r_rmse": round(errors[0]['r_rmse'], 4)})

            message = "Epoch {} iter: {} evaluation Seq. 07 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[1]['t_rel'], 4), round(errors[1]['r_rel'], 4), round(errors[1]['t_rmse'], 4), round(errors[1]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "7. t_rel": round(errors[1]['t_rel'], 4), "7. r_rel": round(errors[1]['r_rel'], 4), "7. t_rmse": round(errors[1]['t_rmse'], 4), "7. r_rmse": round(errors[1]['r_rmse'], 4)})

            message = "Epoch {} iter: {} evaluation Seq. 10 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[2]['t_rel'], 4), round(errors[2]['r_rel'], 4), round(errors[2]['t_rmse'], 4), round(errors[2]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "10. t_rel": round(errors[2]['t_rel'], 4), "10. r_rel": round(errors[2]['r_rel'], 4), "10. t_rmse": round(errors[2]['t_rmse'], 4), "10. r_rmse": round(errors[2]['r_rmse'], 4)})
            
            message = "Epoch {} iter: {} evaluation Seq. 01 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[3]['t_rel'], 4), round(errors[3]['r_rel'], 4), round(errors[3]['t_rmse'], 4), round(errors[3]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "1. t_rel": round(errors[3]['t_rel'], 4), "01. r_rel": round(errors[3]['r_rel'], 4), "01. t_rmse": round(errors[3]['t_rmse'], 4), "01. r_rmse": round(errors[3]['r_rmse'], 4)})
            
            message = "Epoch {} iter: {} evaluation Seq. 02 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[4]['t_rel'], 4), round(errors[4]['r_rel'], 4), round(errors[4]['t_rmse'], 4), round(errors[4]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "2. t_rel": round(errors[4]['t_rel'], 4), "02. r_rel": round(errors[4]['r_rel'], 4), "02. t_rmse": round(errors[4]['t_rmse'], 4), "02. r_rmse": round(errors[4]['r_rmse'], 4)})
            
            message = "Epoch {} iter: {} evaluation Seq. 04 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, iter, round(errors[5]['t_rel'], 4), round(errors[5]['r_rel'], 4), round(errors[5]['t_rmse'], 4), round(errors[5]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iter":iter, "4. t_rel": round(errors[5]['t_rel'], 4), "04. r_rel": round(errors[5]['r_rel'], 4), "04. t_rmse": round(errors[5]['t_rmse'], 4), "04. r_rmse": round(errors[5]['r_rmse'], 4)})
            
            logger.info(message)
            print(message)
            
        ep += 1
    
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()
