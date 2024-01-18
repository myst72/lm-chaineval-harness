import json
import argparse
from tqdm import tqdm
from results_handling import load_existing_results, group_and_aggregate_results, find_id_value, find_unprocessed_data, save_results
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
        print(args.debug_mode, "Total_Score:\n", total_score)


if __name__ == '__main__':
    main()
