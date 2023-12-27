eval "$(conda shell.bash hook)"
conda activate MLDT

base_port=4940

python llm_eval.py \
--base-port ${base_port} \
--eval \
--interactive_eval \
--max_retry 1 \
--llm ../../pretrain/LongAlpaca-13B \
--lora output-react/LongAlpaca-13B \
--mode react \
--api YOUR_API
