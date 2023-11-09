source .venv/bin/activate
pip3 install -r requirements.txt

source ./.env

# test1
python3 ./scripts/main.py \
    --model_path test \
    --model_args '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 512, "n": 1}' \
    --dataset_path ./datasets/ex_humaneval.jsonl \
    --template_path ./templates/humaneval_template.json \
    --metric_path test \
    --result_path result1.jsonl \
    --debug_mode

# test2
python3 ./scripts/main.py \
    --model_path gpt-3.5-turbo \
    --model_args '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 512, "n": 1}' \
    --openai_api_key $OPENAI_API_KEY \
    --dataset_path openai_humaneval \
    --dataset_args '{"num":1}' \
    --template_path ./templates/humaneval_template.json \
    --metric_path test \
    --result_path result2.jsonl \
    --debug_mode

# test3
python3 ./scripts/main.py \
    --model_path kkuramitsu/momogpt-neox-testing \
    --model_args '{"temperature": 0.2, "top_p": 0.95}' \
    --hf_token $HF_TOKEN \
    --dataset_path shunk031/JGLUE \
    --dataset_args '{"num":1}' \
    --dataset_args '{"subset": "JCommonsenseQA", "split": "validation"}' \
    --template_path ./templates/jglue_template.json \
    --metric_path test \
    --result_path result3.jsonl \
    --debug_mode