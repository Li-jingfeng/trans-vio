#batch2/6
python train.py --gpu_ids 0,3 --batch_size 6 --data_dir ./data --experiment_name batch6 --seq_len 11 --workers 4 --patch_size 16
python test.py --gpu_ids 2 --data_dir ./data --experiment_name test --seq_len 11 --workers 4 --patch_size 16 --model ./results/batch6/checkpoints/022_copy.pth --experiment_name test_batch6
#VO 
python train.py --gpu_ids 3 --batch_size 2 --data_dir ./data --experiment_name ts_cam --seq_len 11 --workers 4 --patch_size 16 --pretrain ./results/ts_cam/checkpoints/012.pth --epochs_warmup 10 --epochs_joint 10 --epochs_fine 5
python test.py --gpu_ids 2 --data_dir ./data --experiment_name test_batch6 --seq_len 11 --workers 4 --patch_size 16 --model ./results/batch6/checkpoints/051_copy.pth
# TS-CAM
python train.py --gpu_ids 2 --batch_size 2 --data_dir ./data --experiment_name ts_cam --seq_len 11 --workers 4 --patch_size 16 --pretrain ./results/ts_cam/checkpoints/012.pth
python test.py --gpu_ids 2 --data_dir ./data --experiment_name test_ts_cam --seq_len 11 --workers 4 --patch_size 16 --model ./results/ts_cam/checkpoints/012.pth
# TS-CAM-cls token
python train.py --gpu_ids 2,3 --batch_size 4 --data_dir ./data --experiment_name ts_cam_cls --seq_len 11 --workers 4 --patch_size 16 --epochs_warmup 20 --epochs_joint 20 --epochs_fine 10
python test.py --gpu_ids 3 --data_dir ./data --experiment_name test_ts_cam_cls --seq_len 11 --workers 4 --patch_size 16 --model ./results/ts_cam_cls/checkpoints/049.pth
# 还是接着TS_CAM_cls token继续训练50个epoch，看结果有没有提升
python train.py --gpu_ids 2,3 --batch_size 4 --data_dir ./data --experiment_name ts_cam_cls_100epoch --seq_len 11 --workers 4 --patch_size 16 --epochs_warmup 40 --epochs_joint 40 --epochs_fine 20 --pretrain ./results/ts_cam_cls/checkpoints/049.pth
python test.py --gpu_ids 3 --data_dir ./data --experiment_name test_ts_cam_cls_100epoch --seq_len 11 --workers 4 --patch_size 16 --model ./results/ts_cam_cls_100epoch/checkpoints/099.pth
# 2023/6/29 使用cvpr2023 mmae的预训练模型
# 可能出现分布式训练无法释放显存的问题，执行fuser -v /dev/nvidia，然后再使用kill -9 PID
CUDA_VISIBLE_DEVICES="0,2" accelerate launch --main_process_port 4488 train.py --batch_size 2 --data_dir ./data --experiment_name mmae50epoch --seq_len 11 --workers 4 --patch_size 16 --epochs_warmup 20 --epochs_joint 20 --epochs_fine 10 --pretrain ./results/mmae50epoch/checkpoints/013.pth
python test.py --gpu_ids 3 --data_dir ./data --experiment_name test_mmae50epoch --seq_len 11 --workers 4 --model ./results/mmae50epoch/checkpoints/049.pth
# 2023/7/6 add is_pretrained_mmae==true
CUDA_VISIBLE_DEVICES="0" accelerate launch train.py --batch_size 12 --data_dir ./data --experiment_name mmaebatch16-pretrained --seq_len 2 --workers 4 --patch_size 16 --is_pretrained_mmae True
# is_pretrained_mmae==False
CUDA_VISIBLE_DEVICES="0" accelerate launch train.py --batch_size 12 --data_dir ./data --experiment_name mmaebatch16-no-pretrained --seq_len 2 --workers 4 --patch_size 16 --is_pretrained_mmae False

# 2023/7/7 - 7/8 使用cvpr中的训练参数 and is_pretrained_mmae==True  这个实验没有跑，现在最重要的是解决过拟合的问题
CUDA_VISIBLE_DEVICES="1" accelerate launch train.py --batch_size 12 --data_dir ./data --experiment_name mmaebatch12-pretrained-hyperparameter --seq_len 2 --workers 8 --patch_size 16 --is_pretrained_mmae True --epochs_warmup 10 --epochs_joint 45 --epochs_fine 45 --lr_warmup 2e-4 --lr_joint 2e-4 --lr_fine 2e-4 --optimizer adamW
# 2023/7/8 解决ts_cam_cls过拟合的问题

# 加载预训练模型，只更新head部分，结果update_head很差
python train.py --model_type cvpr --gpu_ids 2 --batch_size 16 --data_dir ./data --experiment_name update_head --seq_len 2 --workers 8 --patch_size 16 --pretrain ./model_zoo/remove_module_fix_model.pth --epochs_warmup 40 --epochs_joint 40 --epochs_fine 20 --is_pretrained_mmae False --img_w 341 --img_h 192
#update only head结果不能说明继续训练会不会有提升，训练集和测试集出现越训越差，序列4结果很好
# 冻结前面部分，后面3个block(9,10,11)+head梯度反传  并且增加几何一致loss  update_3block_head_add_geo_loss
python train.py --model_type cvpr --gpu_ids 1 --batch_size 16 --data_dir ./data --experiment_name update_3block_head_add_geo_loss --seq_len 2 --workers 8 --patch_size 16 --pretrain ./model_zoo/remove_module_fix_model.pth --epochs_warmup 80 --epochs_joint 80 --epochs_fine 40 --is_pretrained_mmae False --img_w 341 --img_h 192
# 对照实验，冻结前面部分，后面3个block(9,10,11)+head梯度反传  不增加几何一致loss
python train.py --model_type cvpr --gpu_ids 1 --batch_size 16 --data_dir ./data --experiment_name update_3block_head --seq_len 2 --workers 8 --patch_size 16 --pretrain ./model_zoo/remove_module_fix_model.pth --epochs_warmup 80 --epochs_joint 80 --epochs_fine 40 --is_pretrained_mmae False --img_w 341 --img_h 192

# DDP 运行方式 不能运行，不知为什么
CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node=2 train.py --model_type cvpr --batch_size 128 --data_dir ./data --experiment_name b128_mmae_pretrained --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 200 --epochs_joint 200 --epochs_fine 100 --is_pretrained_mmae True --img_w 341 --img_h 192
CUDA_VISIBLE_DEVICES="1,2" python train.py --model_type cvpr --batch_size 128 --data_dir ./data --experiment_name b128_mmae_pretrained --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 200 --epochs_joint 200 --epochs_fine 100 --is_pretrained_mmae True --img_w 341 --img_h 192
# 上述方法出现启动多个wandb实验的问题，使用accelerate
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --main_process_port 22238 train.py --model_type cvpr --batch_size 128 --data_dir ./data --experiment_name new_b128_mmae_pretrained --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 200 --epochs_joint 200 --epochs_fine 100 --is_pretrained_mmae True --img_w 341 --img_h 192
CUDA_VISIBLE_DEVICES="0" python train.py --model_type cvpr --batch_size 128 --data_dir ./data --experiment_name new_b128_mmae_pretrained --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 200 --epochs_joint 200 --epochs_fine 100 --is_pretrained_mmae True --img_w 341 --img_h 192
CUDA_VISIBLE_DEVICES="0" python train.py --model_type svio_vo --batch_size 16 --data_dir ./data --experiment_name b16_svio_vo --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 40 --epochs_joint 40 --epochs_fine 20 --img_w 341 --img_h 192
CUDA_VISIBLE_DEVICES="0" python train.py --model_type svio_vo --batch_size 128 --data_dir ./data --experiment_name b128_svio_vo --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 200 --epochs_joint 200 --epochs_fine 100 --img_w 341 --img_h 192

CUDA_VISIBLE_DEVICES="0" python train.py --model_type cvpr --batch_size 128 --data_dir ./data --experiment_name cvpr_fix_loss --seq_len 2 --workers 8 --patch_size 16 --epochs_warmup 200 --epochs_joint 200 --epochs_fine 100 --is_pretrained_mmae True --img_w 341 --img_h 192
