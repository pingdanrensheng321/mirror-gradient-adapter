# --num_epochs 500 \
# --checkpoint_path /home/gpuadmin/projects/siren/siren/logs/noise_pretrain/checkpoints/model_epoch_0500.pth \

# CUDA_VISIBLE_DEVICES=0 python experiment_scripts/train_img.py \
# --model_type=sine \
# --experiment_name basic_origninal_Epoc_500_counter \
# --rgb \
# --num_epochs 500 \
# --loss_function mse \
# --steps_til_summary 1 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 train.py --config configs/glass_vit_l.yaml
# CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mirror_mobile_test.yaml --model /home/gpuadmin/hds/SAM-Adapter-PyTorch/pretrained/mobile_sam.pt
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/mirror_mobile_test.yaml --model /home/gpuadmin/hds/SAM-Adapter-PyTorch/save/_mirror_mobile85.60_aaai/model_epoch_best.pth