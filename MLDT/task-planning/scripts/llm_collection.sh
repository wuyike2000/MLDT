eval "$(conda shell.bash hook)"
conda activate MLDT

base_port=4040

for seed in 0; do
for subset in InDistributation; do

base_port=$((base_port))

python llm_collection.py \
--seed ${seed} \
--base-port ${base_port} \
--eval \
--subset ${subset} \
--interactive_eval \
--interactive_eval_path interactive_eval/${subset}/seed${seed} \
--max_retry 1 \
--llm gpt-3.5-turbo \
--lora None \
--mode multi-layer \
--demo \
--api YOUR_API \
--output data_collection/${subset}.json \
--collection \
--interval 500

done
done
