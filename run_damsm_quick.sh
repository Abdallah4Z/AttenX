#!/usr/bin/env bash
cd /home/skyvision/AttenX
BASE="results/fair_parallel_10e_cub_damsm"
mkdir -p $BASE/{baseline,attenx_sa,attenx_mhsa}/{checkpoints,logs}

CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --variant baseline --epochs 10 --batch-size 4 --num-workers 4 --gamma-damsm 5.0 --lambda-kl 2.0 --seed 42 --lr-g 1e-4 --lr-d 4e-4 --D-steps 1 --damsm-text-path ./checkpoints/damsm_quick/damsm_latest.pth --damsm-image-path ./checkpoints/damsm_quick/damsm_latest.pth --checkpoint-dir $BASE/baseline/checkpoints --log-dir $BASE/baseline/logs --checkpoint-interval 5 --validate-interval 1 > $BASE/baseline/train.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --variant attenx_sa --epochs 10 --batch-size 4 --num-workers 4 --gamma-damsm 5.0 --lambda-kl 2.0 --seed 42 --lr-g 1e-4 --lr-d 4e-4 --D-steps 1 --damsm-text-path ./checkpoints/damsm_quick/damsm_latest.pth --damsm-image-path ./checkpoints/damsm_quick/damsm_latest.pth --checkpoint-dir $BASE/attenx_sa/checkpoints --log-dir $BASE/attenx_sa/logs --checkpoint-interval 5 --validate-interval 1 > $BASE/attenx_sa/train.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --variant attenx_mhsa --epochs 10 --batch-size 4 --num-workers 4 --gamma-damsm 5.0 --lambda-kl 2.0 --seed 42 --lr-g 1e-4 --lr-d 4e-4 --D-steps 1 --damsm-text-path ./checkpoints/damsm_quick/damsm_latest.pth --damsm-image-path ./checkpoints/damsm_quick/damsm_latest.pth --checkpoint-dir $BASE/attenx_mhsa/checkpoints --log-dir $BASE/attenx_mhsa/logs --checkpoint-interval 5 --validate-interval 1 --mhsa-heads 4 > $BASE/attenx_mhsa/train.out 2>&1 &

echo "All 3 variants launched. PIDs:"
ps aux | grep "train.py --variant" | grep -v grep | awk '{print $2, $11, $12}'
