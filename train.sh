#batch2/6
python train.py --gpu_ids 0,3 --batch_size 6 --data_dir ./data --experiment_name batch6 --seq_len 11 --workers 4 --patch_size 16
python test.py --gpu_ids 2 --data_dir ./data --experiment_name test --seq_len 11 --workers 4 --patch_size 16 --model ./results/batch6/checkpoints/022_copy.pth --experiment_name test_batch6
#VO TS-CAM
python train.py --gpu_ids 0,3 --batch_size 16 --data_dir ./data --experiment_name ts_cam --seq_len 11 --workers 48 --patch_size 16
