import argparse
import json
from tqdm import tqdm
from models import load_model
from dataloaders import load_testdata
from templates import load_template
from evaluators import load_evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--model_args', type=json.loads, default=None, help='Model arguments in JSON format')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API token')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--dataset_args', type=json.loads, default=None, help='Dataset arguments in JSON format')
    parser.add_argument('--template_path', type=str, required=True, help='Path to the template file')
    parser.add_argument('--metric_path', type=str, default=False, help='Path to the metric file')
    parser.add_argument('--metric_args', type=json.loads, default=None, help='Metric arguments in JSON format')
    parser.add_argument('--result_path', type=str, default='./result.jsonl', help='Path to the result file')
    parser.add_argument('--no_quantize_model', action='store_false', help='Disable model quantization with bitsandbytes')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for verbose output')
    return parser.parse_args()


def debug_print(debug_mode, *messages):
    """Print debug messages if debug mode is enabled."""
    if debug_mode:
        print("🐥", *messages)


def save_results(result_path, results, record, final=False):
    """Save the evaluation results to a file."""

    # 評価算出までした場合には全て上書きで保存をする
    mode = 'w' if final else 'a'

    with open(result_path, mode, encoding='utf-8') as f:
        for result in results:
            filtered_result = record
            for k, v in result.items():
                if 'id' in k:
                    filtered_result['id'] = v
            filtered_result['model_input'] = result.get('model_input', '')
            filtered_result['model_output'] = result.get('model_output', '')
            filtered_result['formatted_output'] = result.get('formatted_output', '')
            for k, v in result.items():
                if 'score' in k:
                    filtered_result[k] = v
            f.write(json.dumps(filtered_result, ensure_ascii=False) + '\n')
                    
            filtered_result = {
                key: value for key, value in result.items()
                if key in ['id', 'score'] or key in ['model_input', 'model_output', 'formatted_output']
            }



def main():
    args = parse_args()
    quantize = not args.no_quantize_model

    model = load_model(args.model_path, args.openai_api_key, args.hf_token, args.model_args, quantize)
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

    for data in tqdm(dataset):
        prompt = template.process(data)
        data['reference'] = template.process_reference(data)
        data['model_input'] = prompt
        data['model_output'] = model.generate(prompt)
        data['formatted_output'] = template.collate(prompt, data['model_output'])

        # debug_print(args.debug_mode, "Reference:\n", data['reference'])
        debug_print(args.debug_mode, "Input:\n", data['model_input'])
        debug_print(args.debug_mode, "Output:\n", data['model_output'])
        debug_print(args.debug_mode, "Formatted:\n", data['formatted_output'])

        save_results(args.result_path, [data], record)
    
    if evaluator: 
        score, dataset = evaluator.calculate(dataset, record)
        print("score:\n", score)

    save_results(args.result_path, dataset, record, final=True)


if __name__ == '__main__':
    main()