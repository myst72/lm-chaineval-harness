import argparse
import json
import os
from tqdm import tqdm
from collections import defaultdict
from models import load_model
from dataloaders import load_testdata
from templates import load_template
from evaluators import load_evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--model_args', type=json.loads, default=None, help='Model arguments in JSON format')
    parser.add_argument('--quantize_model', action='store_true', help='Enable model quantization with bitsandbytes')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API token')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--aws_access_key_id', type=str, default=None, help='AWS access key ID')
    parser.add_argument('--aws_secret_access_key', type=str, default=None, help='AWS secret access key')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--dataset_args', type=json.loads, default=None, help='Dataset arguments in JSON format')
    parser.add_argument('--template_path', type=str, required=True, help='Path to the template file')
    parser.add_argument('--metric_path', type=str, default=False, help='Path to the metric file')
    parser.add_argument('--metric_args', type=json.loads, default=None, help='Metric arguments in JSON format')
    parser.add_argument('--result_path', type=str, default='./log/result.jsonl', help='Path to the result file')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for verbose output')
    return parser.parse_args()


def debug_print(debug_mode, *messages):
    """Print debug messages if debug mode is enabled."""
    if debug_mode:
        print("🐥", *messages)


def load_existing_results(result_path):
    """Load existing results from the file."""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []


def group_and_aggregate_results(results):
    grouped_results = defaultdict(lambda: defaultdict(list))
    for result in results:
        id_value = result.get('id')
        if id_value:
            for key, value in result.items():
                if key in ['model_output', 'formatted_output']:
                    grouped_results[id_value][key].append(value)
                else:
                    grouped_results[id_value][key] = value
    return [dict(result) for result in grouped_results.values()]


def find_id_value(data):
    for key in data.keys():
        if 'id' in key:
            return data[key]
    return None


def find_unprocessed_data(dataset, existing_results):
    """Find data in the dataset that has not been processed yet."""
    processed_ids = {find_id_value(result) for result in existing_results}
    return [data for data in dataset if find_id_value(data) not in processed_ids]


def save_results(result_path, dataset, record, total_score=-1):
    """Save the evaluation results to a file."""
    mode = 'a' if total_score == -1 else 'w'
    
    # 結果を保存するディレクトリを確認し、存在しない場合は作成
    directory = os.path.dirname(result_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(result_path, mode, encoding='utf-8') as f:
        for data in dataset:
            filtered_data = record.copy()
            for k, v in data.items():
                if 'id' in k:
                    filtered_data['id'] = v
            filtered_data['model_input'] = data.get('model_input', '')
            # filtered_data['model_output'] = data.get('model_output', '')
            # filtered_data['output_format'] = data.get('output_format', '')
            model_outputs = data.get('model_output', [])
            formatted_outputs = data.get('formatted_output', [])
            
            for i, (model_output, formatted_output) in enumerate(zip(model_outputs, formatted_outputs)):
                # 各要素を個別に処理
                filtered_data['model_output'] = model_output
                filtered_data['formatted_output'] = formatted_output
                
                # その他のデータを処理
                filtered_data['output_format'] = data.get('output_format', '')
                filtered_data['formatted_correctly'] = data.get('formatted_correctly', '')
                filtered_data['reference'] = data.get('reference', '')
                filtered_data['item_score'] = data.get('item_score', '')
                filtered_data['total_score'] = total_score
                # 結果をファイルに書き込み
                f.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    
    quantize = args.quantize_model
    debug_print(args.debug_mode, "Quantization:\n", quantize)

    model = load_model(args.model_path, args.openai_api_key, args.aws_access_key_id, args.aws_secret_access_key, args.hf_token, args.model_args, quantize)
    debug_print(args.debug_mode, "Model loaded:\n", model)

    dataset = load_testdata(args.dataset_path, args.dataset_args)
    debug_print(args.debug_mode, "Dataset loaded:\n", len(dataset), "entries")

    template = load_template(args.template_path)
    debug_print(args.debug_mode, "Template loaded:\n", template)

    evaluator = load_evaluator(args.metric_path, args.metric_args)
    debug_print(args.debug_mode, "Evaluator loaded:\n", evaluator)

    record = {
        'model': args.model_path,
        'dataset': args.dataset_path,
        'template': args.template_path,
        'metrics': args.metric_path,
    } 

    loaded_results = load_existing_results(args.result_path)
    existing_results = group_and_aggregate_results(loaded_results)
    unprocessed_data = find_unprocessed_data(dataset, existing_results)

    for data in tqdm(unprocessed_data):
            
        prompt = template.process(data)
        data['reference'] = template.process_reference(data)
        data['model_input'] = prompt
        debug_print(args.debug_mode, "Input:\n", data['model_input'])
        data['model_output'] = model.generate(prompt)
        debug_print(args.debug_mode, "Output_Sample:\n", data['model_output'][0])
        output_lang, output_format, formatted_output_list = template.collate(prompt, data['model_output'])
        data['output_format'] = output_format
        if isinstance(formatted_output_list, dict):
            data['formatted_output'] = formatted_output_list["output"]
            data['formatted_correctly'] = formatted_output_list["formatted_correctly"]
        else:
            data['formatted_output'] = formatted_output_list
        debug_print(args.debug_mode, "Formatted_Sample:\n", data['formatted_output'][0])

        if evaluator:
            if data['formatted_output'] is None:
                data['item_score'] = 0.0
            else:
                data['item_score'] = evaluator.item_calculate(data, record, output_lang)
                debug_print(args.debug_mode, "Score:\n", data['item_score'])

        save_results(args.result_path, [data], record)
    
    if evaluator:
        if 'output_lang' not in locals():
            raise ValueError("output_lang is not defined. Cannot execute as all data has already been processed.")

        all_data = existing_results + unprocessed_data
        total_score = evaluator.total_calculate(all_data, record, output_lang)
        save_results(args.result_path, all_data, record, total_score)


if __name__ == '__main__':
    main()
