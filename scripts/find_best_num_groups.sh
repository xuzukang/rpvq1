#!/bin/bash

# Define output log file
LOG_DIR="outputs/llama3-8b-hf/logs/find_best_num_groups"
TIME=$(date +'%Y-%m-%d_%H-%M-%S')
mkdir -p "$LOG_DIR/$TIME"


NUM_GROUP="1"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V1.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" \
  --num_centroids "-1" "384" \
  --num_res_layers "-1" "2" \
  --num_res_centroids "-1" "-1" \
  --group_num "1" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "64" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "2" "2" "2" "2" "2" "2" "2" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1


NUM_GROUP="2"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V2.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" \
  --num_centroids "-1" "1024" "8192" \
  --num_res_layers "-1" "2" "1" \
  --num_res_centroids "-1" "-1" \
  --group_num "2" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "64" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "2" "2" "2" "2" "2" "2" "2" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1


NUM_GROUP="3"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V3.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" \
  --num_centroids "-1" "512" "512" "4096" \
  --num_res_layers "-1" "2" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1"\
  --group_num "3" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "80" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "2" "2" "2" "2" "2" "2" "2" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

NUM_GROUP="4"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V4.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "4096" \
  --num_res_layers "-1" "2" "2" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "64" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "2" "2" "2" "2" "2" "2" "2" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

NUM_GROUP="5"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V5.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" "6"\
  --num_centroids "-1" "4096" "512" "256" "128" "1024" \
  --num_res_layers "-1" "2" "2" "2" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" "-1"\
  --group_num "5" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "64" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "2" "2" "2" "2" "2" "2" "2" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

NUM_GROUP="6"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V6.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" "6" "6"\
  --num_centroids "-1" "512" "256" "256" "128" "4096" "256"\
  --num_res_layers "-1" "3" "3" "2" "2" "1" "1"\
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" "-1" "-1"\
  --group_num "6" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "64" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "2" "2" "2" "2" "2" "2" "2" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1




