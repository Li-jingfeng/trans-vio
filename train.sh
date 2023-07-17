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

# 2023/7/8解决ts_cam_cls_100epoch过拟合的问题
python train.py --gpu_ids 1 --batch_size 16 --data_dir ./data --experiment_name ts_cam_cls_iter50k --seq_len 2 --workers 8 --patch_size 16 --iter_warmup 10000 --iter_joint 20000 --iter_fine 20000 --epoch_or_iter False
# 7/9 iter增加，到300K
python train.py --gpu_ids 0 --batch_size 16 --data_dir ./data --experiment_name ts_cam_cls_iter300k --seq_len 2 --workers 8 --patch_size 16 --iter_warmup 60000 --iter_joint 120000 --iter_fine 120000 --epoch_or_iter False
# 7/10 fix update_status lr
python train.py --gpu_ids 1 --batch_size 16 --data_dir ./data --experiment_name ts_cam_cls_fix_lr_iter300k --seq_len 2 --workers 8 --patch_size 16 --iter_warmup 60000 --iter_joint 120000 --iter_fine 120000
# 7/11 fix wandb.log seq 01,02,04结果重复的bug
python train.py --gpu_ids 0 --batch_size 16 --data_dir ./data --experiment_name ts_cam_cls_fix_log_iter300k --seq_len 2 --workers 8 --patch_size 16 --iter_warmup 60000 --iter_joint 120000 --iter_fine 120000
python test.py --gpu_ids 1 --data_dir ./data --experiment_name test_ts_cam_cls_fix_lr_iter300k --seq_len 2 --workers 8 --patch_size 16 --model ./results/ts_cam_cls_fix_lr_iter300k/checkpoints/099.pth
# 调整batch_size和seq_len 2023/7/6
python train.py --gpu_ids 0,1 --batch_size 16 --data_dir ./data --experiment_name ts_cam_cls_b16 --seq_len 2 --workers 48 --patch_size 16
