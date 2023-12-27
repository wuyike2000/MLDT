eval "$(conda shell.bash hook)"
conda activate MLDT

base_port=6225

base_port=$((base_port))

python llm_eval.py \
--base-port ${base_port} \
--eval \
--interactive_eval \
--max_retry 1 \
--llm ../../pretrain/chatglm3-6b \
--demo \
--mode multi-layer \
--api YOUR_API
