source .venv/bin/activate
# pip3 install -r requirements.txt

# source ./.env

# test1: HumanEval
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"max_new_tokens": 512, "do_sample": false, "num_beams": 1}' \
    --dataset_path openai/openai_humaneval \
    --template_path ./templates/humaneval_template.json \
    --metric_path code_eval \
    --result_path results/chico_humaneval.jsonl \
    --debug_mode

# test2: JHumanEval
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"max_new_tokens": 512, "do_sample": false, "num_beams": 1}' \
    --dataset_path kogi-jwu/jhumaneval \
    --template_path ./templates/humaneval_template.json \
    --metric_path code_eval \
    --result_path results/chico_jhumaneval.jsonl \
    --debug_mode

# test3: MIHE-EN
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"max_new_tokens": 512, "do_sample": false, "num_beams": 1}' \
    --dataset_path kogi-jwu/multilingual_instruction_humaneval \
    --dataset_args '{"split": "test", "subset":"en"}' \
    --template_path ./templates/MIHE-EN.json \
    --metric_path code_eval \
    --result_path results/chico_mihe-en.jsonl \
    --debug_mode

# test4: MIHE-JA
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"max_new_tokens": 512, "do_sample": false, "num_beams": 1}' \
    --dataset_path kogi-jwu/multilingual_instruction_humaneval \
    --dataset_args '{"split": "test", "subset":"ja"}' \
    --template_path ./templates/MIHE-JA.json \
    --metric_path code_eval \
    --result_path results/chico_mihe-ja.jsonl \
    --debug_mode

# test3: MIHE-EN-CoT
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"max_new_tokens": 512, "do_sample": false, "num_beams": 1}' \
    --dataset_path kogi-jwu/multilingual_instruction_humaneval \
    --dataset_args '{"split": "test", "subset":"en"}' \
    --template_path ./templates/MIHE-EN-CoT.json \
    --metric_path code_eval \
    --result_path results/chico_mihe-en-cot.jsonl \
    --debug_mode

# test4: MIHE-JA-CoT
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"max_new_tokens": 512, "do_sample": false, "num_beams": 1}' \
    --dataset_path kogi-jwu/multilingual_instruction_humaneval \
    --dataset_args '{"split": "test", "subset":"ja"}' \
    --template_path ./templates/MIHE-JA-CoT.json \
    --metric_path code_eval \
    --result_path results/chico_mihe-ja-cot.jsonl \
    --debug_mode


