# This script is adapted from 
# https://github.com/FranxYao/Long-Context-Data-Engineering.git
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


METHOD='semblock'     
MAX_CAPACITY_PROMPT=96  
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
TAG=test

(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001\
    --model_provider LLaMA3 \
    --model_name /data/hfhub/Llama-3.1-8B-Instruct \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

