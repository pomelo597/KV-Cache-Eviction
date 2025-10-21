export CUDA_VISIBLE_DEVICES=0

method="SemBlock" # Support SemBlock, PyramidKV, SnapKV, H2O, StreamingLLM, SentenceKV, ChunkKV
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager".
source_path=""
model_path="/data/hfhub/Llama-3.1-8B-Instruct"
save_dir=${source_path}"/data/SemBlock/result/all_sample_baseline_llama3.1-instruct" # path to result save_dir


for max_capacity_prompts in 64 128 256 512 1024 2048

do
python3 -u run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True
done


