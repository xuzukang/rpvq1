#!/bin/bash

# 设置CUDA_VISIBLE_DEVICES环境变量
export CUDA_VISIBLE_DEVICES=5

# 启动Python脚本并将所有参数传递给它
nohup python /data01/home/xuzk/workspace/VPTQ/run_rpvq.py \
  --num_res_centroids -1 -1 -1 -1 -1 \
  --group_num 1 \
  --low_rank_hessian True \
  --gptq True \
  --npercent 0 \
  --vq_type rpvq_v3 \
  --finetune True \
  --finetune_type e2e \
  --train_dataset wikitext2 \
  --batch_size 4 \
  --devset_size 141 \
  --ctx_size 2048 \
  --ft_valid_size 5 \
  --ft_valid_freq 1 \
  --ft_bs 2 \
  --ft_lr 5e-5 \
  --ft_epochs 1 \
  --ft_early_stop 2 \
  --ft_loss_type kl \
  --finetune_blocks 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 \
  --loss_direct False \
  --tasks arc_challenge arc_easy hellaswag piqa winogrande \
  --new_eval \
  --seq_len 2048 \
  --blocksize 128 \
  --kmeans_mode hessian \
  --enable_perm \
  --enable_norm \
  --ktol 1e-5 \
  --kiter 100 \
  --vector_lens -1 6 6 6 6 \
  --num_res_layers -1 2 2 2 2 \
  --model_name weights/llama3-8b-hf \
  --hessian_path weights/hessian/llama3-8b \
  --inv_hessian_path weights/invhessian/llama3-8b \
  --output_dir 2.25bit \
  --num_centroids -1 512 128 64 32 \
  --gpu_ids 5 &

#   --output_dir 4bit \
#   --num_centroids -1 32768 8192 2048 512 \
#   --gpu_ids 5 &

#   --output_dir 3bit \
#   --num_centroids -1 2048 512 256 256 \
#   --gpu_ids 5 &

#   --output_dir 2.25bit \
#   --num_centroids -1 512 128 64 32 \
#   --gpu_ids 5 &

#   --output_dir 2bit \
#   --num_centroids -1 256 64 32 32 \
#   --gpu_ids 5 &

# 提示脚本已启动
echo "Script has been started in the background."
