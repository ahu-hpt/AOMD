#!/usr/bin/env bash
# base experiment
# Teacher Network
CUDA_VISIBLE_DEVICES=0 python run.py --dataset cars196 --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 128\
              --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --save_dir cars196_resnet50_512

# Student Network(triplet)
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --triplet_ratio 1 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_triplet > cars196_triplet.log 2>&1 &

# AOMD
# mode 1
# mode 2
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode1_l2 > cars196_aomd_mode1_l2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode2_l2 > cars196_aomd_mode2_l2.log 2>&1 &

# contrast experiment
# sp
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --sp_ratio 2e1  \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_sp > cars196_sp.log 2>&1 &
# at
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --at_ratio 1e3\
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_at > cars196_at.log 2>&1 &

# fitnet
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_fitnet.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --fitnet_ratio 1 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_fitnet > cars196_fitnet.log 2>&1 &

# Ablation experiment
# mode1 a=1 , b=0
# mode1 a=0 , b=1
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode1_l2_a_1_b_0 > cars196_aomd_mode1_l2_a_1_b_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode1_l2_a_0_b_1 > cars196_aomd_mode1_l2_a_0_b_1.log 2>&1 &

# mode2 a=1 , b=0
# mode2 a=0 , b=1
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode2_l2_a_1_b_0 > cars196_aomd_mode2_l2_a_1_b_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode2_l2_a_0_b_1 > cars196_aomd_mode2_l2_a_0_b_1.log 2>&1 &
