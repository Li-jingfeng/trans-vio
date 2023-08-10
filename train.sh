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

CUDA_VISIBLE_DEVICES="0" python train.py --model_type cvpr --use_cnn --batch_size 128 --pretrain_flownet ./model_zoo/flownets_bn_EPE2.459.pth.tar --data_dir ./data --experiment_name add_flownet_cls_token --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 100 --epochs_joint 100 --epochs_fine 50 --is_pretrained_mmae True --img_w 341 --img_h 192
CUDA_VISIBLE_DEVICES="0" python train.py --model_type cvpr --use_imu --batch_size 128 --data_dir ./data --experiment_name add_imu_cls_token --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 100 --epochs_joint 100 --epochs_fine 50 --is_pretrained_mmae True --img_w 341 --img_h 192

# 2023/7/31 add flowformer encoder, transformer的使用方式
CUDA_VISIBLE_DEVICES="0,1" python train.py --model_type flowformer_vo --batch_size 2 --data_dir ./data --experiment_name flowformer_vo --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 100 --epochs_joint 100 --epochs_fine 50 --is_pretrained_mmae True --img_w 960 --img_h 432 --stage kitti
# 2023/8/3 change image resolution /2 add pretrained checkpoints
CUDA_VISIBLE_DEVICES="2,0" python train.py --model_type flowformer_vo --pretrain ./model_zoo/flowformer_kitti.pth --batch_size 8 --data_dir ./data --experiment_name flowformer_vo_pretrained --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 100 --epochs_joint 100 --epochs_fine 50 --is_pretrained_mmae True --img_w 480 --img_h 216 --stage kitti
CUDA_VISIBLE_DEVICES="1,3" python train.py --model_type flowformer_vo --pretrain ./model_zoo/flowformer_kitti.pth --batch_size 48 --data_dir ./data --experiment_name flowformer_vo_pretrained_before_AGT --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 100 --epochs_joint 100 --epochs_fine 50 --is_pretrained_mmae True --img_w 480 --img_h 216 --stage kitti
# 修改cvpr图片尺寸
CUDA_VISIBLE_DEVICES="1" python train.py --model_type cvpr --gpu_ids 0 --batch_size 16 --data_dir ./data --experiment_name cvpr_size --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 80 --epochs_joint 80 --epochs_fine 40 --is_pretrained_mmae True --img_w 512 --img_h 256
# flowformer_vo regression_mode=2
CUDA_VISIBLE_DEVICES="3" python train.py --model_type flowformer_vo --regression_mode 2 --gpu_ids 0 --batch_size 18 --data_dir ./data --experiment_name flowformer_vo_regress_mode_2 --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 80 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# flowformer_vo regression_mode=3
CUDA_VISIBLE_DEVICES="2" python train.py --model_type flowformer_vo --regression_mode 3 --gpu_ids 0 --batch_size 16 --data_dir ./data --experiment_name flowformer_vo_regress_mode_3 --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 80 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# flowformer_vo regression_mode=3 only update regressor3
CUDA_VISIBLE_DEVICES="3" python train.py --model_type flowformer_vo --regression_mode 3 --gpu_ids 0 --batch_size 50 --data_dir ./data --experiment_name flowformer_vo_regress_mode_3_update_regressor --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 20 --epochs_joint 100 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# flowformer_vo regression_mode=2 only update regressor1 and 2
# 这个版本达到过拟合的效果
CUDA_VISIBLE_DEVICES="2" python train.py --model_type flowformer_vo --regression_mode 2 --gpu_ids 0 --batch_size 50 --data_dir ./data --experiment_name flowformer_vo_regress_mode_2_update_regressor --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 20 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# 针对上一过拟合的版本，不修改参数，在最后加入imu feature concat做预测，看能不能缓减过拟合效果，同时达到更好的结果---->结果是可以缓减过拟合，结果更好
python train.py --model_type flowformer_vio --regression_mode 2 --gpu_ids 3 --batch_size 50 --data_dir ./data --experiment_name flowformer_vio_regress_mode_2_freeze_backbone --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 20 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# 根据上述实验结果，修改cvpr参数，重新实验
python train.py --model_type cvpr --gpu_ids 0 --batch_size 128 --data_dir ./data --experiment_name cvpr_fix_lr --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 100 --epochs_joint 100 --epochs_fine 50 --is_pretrained_mmae True --img_w 341 --img_h 192
# 这里已经将freeze给注释掉，所有的参与训练，同时训练参数与过拟合保持一致 因为batch=16，所以epoch都除以3
python train.py --model_type flowformer_vo --regression_mode 2 --gpu_ids 3 --batch_size 16 --data_dir ./data --experiment_name flowformer_vo_regress_mode_2_update_all --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 7 --epochs_joint 26 --epochs_fine 14 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# 感觉效果不是很好，重新做一次实验，将batch_size改为32
CUDA_VISIBLE_DEVICES="1,3" python train.py --model_type flowformer_vo --regression_mode 2 --batch_size 32 --data_dir ./data --experiment_name flowformer_vo_regress_mode_2_update_all_b32 --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 20 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# 2023/8/9  除了regressor之外，同时将cross attention纳入更新，而且加载权重时不加载这部分权重  多了一个add_part_weight参数
python train.py --model_type flowformer_vio --regression_mode 2 --add_part_weight True --gpu_ids 0 --batch_size 48 --data_dir ./data --experiment_name flowformer_vio_regress_mode_2_update_cross_attn --seq_len 2 --workers 16 --patch_size 16 --epochs_warmup 20 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth
# 加入lstm 仍然是flowformer_vio_lstm 更新lstm，不更新cross attention，看一下对估计有无影响
CUDA_VISIBLE_DEVICES="0,1" python train.py --model_type flowformer_vio_lstm --regression_mode 2 --batch_size 48 --data_dir ./data --experiment_name flowformer_vio_lstm_regress_mode_2_update_lstm --seq_len 11 --workers 16 --patch_size 16 --epochs_warmup 20 --epochs_joint 80 --epochs_fine 40 --img_w 480 --img_h 216 --stage kitti --pretrain ./model_zoo/flowformer_kitti.pth

