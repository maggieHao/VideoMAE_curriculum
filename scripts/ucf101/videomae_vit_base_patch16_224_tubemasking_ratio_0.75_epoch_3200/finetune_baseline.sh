# Set the path to save checkpoints
OUTPUT_DIR='/home/maggie/VideoMAE_checkpoints_curriculum/ucf101_baseline'
# path to UCF101 annotation file (train.csv/val.csv/test.csv)
DATA_PATH='/home/maggie/VideoMAE_curriculum/labels/ucf101'
# path to pretrain model
MODEL_PATH='/home/maggie/VideoMAE_checkpoints/pretrain_checkpoint_ucf101/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 8 GPUs (1 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 \
    --master_port 12320  run_class_finetuning_baseline.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --data_set UCF101 \
    --nb_classes 101 \
    --batch_size 16 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 2 \
    --opt adamw \
    --lr 5e-4 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --fc_drop_rate 0.5 \
    --drop_path 0.2 \
    --use_checkpoint \
    --dist_eval \
    --enable_deepspeed 
