eval "$(conda shell.bash hook)"
conda activate MLDT

base_port=6574

python llm_eval.py \
--base-port ${base_port} \
--eval \
--interactive_eval \
--max_retry 1 \
--llm gpt-4 \
--lora None \
--mode embodied \
--demo \
--api YOUR_API
