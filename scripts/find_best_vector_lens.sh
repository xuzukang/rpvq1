#!/bin/bash

# Define output log file
LOG_DIR="outputs/llama3-8b-hf/logs/find_best_vector_lens"
TIME=$(date +'%Y-%m-%d_%H-%M-%S')
mkdir -p "$LOG_DIR/$TIME"

# 2.5-3.5 bit之间
## 分成4组，每组16个
NUM_GROUP="4"
VECTOR_LENS="16"
echo "RPVQ Algorithm start with 2.5-3.5 bit group 4 vector 16"  
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.5_V0.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "0" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.76_V1.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.76_V2.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "False" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.76_V3.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.01_V4.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "128" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.01_V5.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "4" "4" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.01_V6.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "6" "6" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "False" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.01_V7.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "8" "8" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1


LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.01_V8.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "16" "16" "16" "16" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "10" "6" "4" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

echo "RPVQ Algorithm end with 2.5-3.5 bit group 4 vector 16"  


echo "RPVQ Algorithm star with 2.5-3.5 bit group 4 vector 12"  
NUM_GROUP="4"
VECTOR_LENS="12"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.5_V0.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "0" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V1.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V2.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "False" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V3.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V4.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "2" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V5.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "3" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V6.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "4" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V7.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "5" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V8.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "12" "12" "12" "12" \
  --num_centroids "-1" "4096" "4096" "4096" "4096" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "128" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

echo "RPVQ Algorithm end with 2.5-3.5 bit group 4 vector 16"  

echo "RPVQ Algorithm star with 2.5-3.5 bit group 4 vector 8"  
NUM_GROUP="4"
VECTOR_LENS="8"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.5_V0.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "0" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V1.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V2.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "False" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V3.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V4.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "2" "2" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V5.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "3" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V6.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "4" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V7.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "5" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V8.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "8" "8" "8" "8" \
  --num_centroids "-1" "256" "256" "256" "256" \
  --num_res_layers "-1" "4" "3" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "128" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

echo "RPVQ Algorithm end with 2.5-3.5 bit group 4 vector 8"  

echo "RPVQ Algorithm star with 2.5-3.5 bit group 4 vector 6"  
NUM_GROUP="4"
VECTOR_LENS="6"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.5_V0.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "256" \
  --num_res_layers "-1" "2" "2" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "0" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V1.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "256" \
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
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V2.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "256" \
  --num_res_layers "-1" "2" "2" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "False" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V3.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "256" \
  --num_res_layers "-1" "2" "2" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

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
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V5.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "256" \
  --num_res_layers "-1" "2" "2" "2" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V6.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "4096" "4096" "256" \
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
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V7.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "1024" "512" "4096" "256" \
  --num_res_layers "-1" "3" "2" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V8.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "6" "6" "6" "6" \
  --num_centroids "-1" "4096" "512" "4096" "256" \
  --num_res_layers "-1" "2" "2" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "128" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

echo "RPVQ Algorithm end with 2.5-3.5 bit group 4 vector 6"  

echo "RPVQ Algorithm star with 2.5-3.5 bit group 4 vector 4"  
NUM_GROUP="4"
VECTOR_LENS="4"
LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.5_V0.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "256" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "0" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V1.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "256" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V2.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "256" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "False" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b2.75_V3.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "256" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "False" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V4.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "256" "256" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V5.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "4096" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V6.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "256" "256" "16" \
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
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V7.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "1024" "4096" "256" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "64" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

LOG_FILE="$LOG_DIR/$TIME/g${NUM_GROUP}v${VECTOR_LENS}"+"b3.0_V8.log"
python3 run_vptq.py \
  --model_name "weights/llama3-8b-hf" \
  --output_dir "outputs/llama3-8b-hf" \
  --hessian_path "weights/hessian/llama-31-8b" \
  --inv_hessian_path "weights/invhessian/llama-31-8b" \
  --vector_lens "-1" "4" "4" "4" "4" \
  --num_centroids "-1" "256" "4096" "256" "16" \
  --num_res_layers "-1" "2" "1" "1" "1" \
  --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
  --group_num "4" \
  --low_rank "128" \
  --low_rank_hessian "True" \
  --gptq "True" \
  --vq_type "rpvq_v3" \
  --npercent "0" \
  --new_eval \
  --seq_len "8192" \
  --blocksize "128" \
  --kmeans_mode "hessian" \
  --enable_perm \
  --enable_norm \
  --ktol "1e-5" \
  --kiter "100" \
  --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
  > "$LOG_FILE" 2>&1

echo "RPVQ Algorithm end with 2.5-3.5 bit group 4 vector 4"  


# 当前最好的实验配置
# LOG_FILE="$LOG_DIR/$TIME/V1.log"
#  python3 run_vptq.py \
#   --model_name "weights/llama3-8b-hf" \
#   --output_dir "outputs/llama3-8b-hf" \
#   --hessian_path "weights/hessian/llama-31-8b" \
#   --inv_hessian_path "weights/invhessian/llama-31-8b" \
#   --vector_lens "-1" "6" "6" "6" "6" \
#   --num_centroids "-1" "4096" "512" "256" "256" \
#   --num_res_layers "-1" "2" "2" "2" "2" \
#   --num_res_centroids "-1" "-1" "-1" "-1" "-1" \
#   --group_num "4" \
#   --low_rank "64" \
#   --low_rank_hessian "True" \
#   --gptq "True" \
#   --vq_type "rpvq_v3" \
#   --npercent "0" \
#   --new_eval \
#   --seq_len "8192" \
#   --blocksize "128" \
#   --kmeans_mode "hessian" \
#   --enable_perm \
#   --enable_norm \
#   --ktol "1e-5" \
#   --kiter "100" \
#   --gpu_ids "6" "6" "6" "6" "6" "6" "6" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "7" "7" "7" "7" \
#   > "$LOG_FILE" 2>&1








