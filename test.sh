source .venv/bin/activate
pip3 install -r requirements.txt

source ./.env

# test1: HumanEval
python3 ./scripts/main.py \
    --model_path gpt-3.5-turbo \
    --model_args '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 512, "n": 1}' \
    --openai_api_key $OPENAI_API_KEY \
    --dataset_path openai_humaneval \
    --template_path ./templates/humaneval_template.json \
    --metric_path code_eval \
    --result_path results/gpt-3.5_humaneval.jsonl \
    --debug_mode

# test2: JHumanEval
python3 ./scripts/main.py \
    --model_path kkuramitsu/tinycodellama-jp-0.13b-50k \
    --model_args '{"temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "num_return_sequences": 1}' \
    --dataset_path kogi-jwu/jhumaneval \
    --template_path ./templates/humaneval_template.json \
    --metric_path code_eval \
    --result_path results/tinycodellama_jhumaneval.jsonl \
    --debug_mode

# test3: DataSet (English)
python3 ./scripts/main.py \
    --model_path gpt-3.5-turbo \
    --model_args '{"temperature": 0.8, "top_p": 0.95, "max_tokens": 512, "n": 3}' \
    --openai_api_key $OPENAI_API_KEY \
    --dataset_path ./datasets/HumanEvalSet_top37.jsonl \
    --dataset_args '{"n": 1}' \
    --template_path ./templates/humanevalset_en.json \
    --metric_path code_eval \
    --result_path results/gpt-3.5-turbo_humaneval.jsonl \
    --debug_mode

# test4: DataSet (Japanese)
python3 ./scripts/main.py \
    --model_path gpt-3.5-turbo \
    --model_args '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 512, "n": 3}' \
    --openai_api_key $OPENAI_API_KEY \
    --dataset_path ./datasets/HumanEvalSet_top37.jsonl \
    --dataset_args '{"n": 1}' \
    --template_path ./templates/humanevalset_ja.json \
    --metric_path code_eval \
    --result_path results/gpt-3.5-turbo_jhumaneval.jsonl \
    --debug_mode

# test5: JPythonCodeQA2023
python3 ./scripts/main.py \
    --model_path gpt-3.5-turbo \
    --openai_api_key $OPENAI_API_KEY \
    --model_args '{"max_tokens": 1}' \
    --dataset_path ./datasets/JPythonCodeQA2023.jsonl \
    --dataset_args '{"num":10}' \
    --template_path ./templates/jpythoncodeqa2023.json \
    --metric_path  exact_match \
    --result_path results/gpt-3.5_JPythonCodeQA2023.jsonl \
    --debug_mode
