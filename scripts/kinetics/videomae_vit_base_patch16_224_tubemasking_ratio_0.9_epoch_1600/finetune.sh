# Set the path to save checkpoints
OUTPUT_DIR='/home/maggie/VideoMAE_checkpoints_curriculum/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/curriculum'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/home/maggie/VideoMAE_curriculum/labels/k400'
# path to pretrain model
MODEL_PATH='/home/maggie/VideoMAE_checkpoints/pretrain_checkpoint/pretrain_checkpoint.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port 12320 --nnodes=1 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 75 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --enable_deepspeed 
