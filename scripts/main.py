import argparse
import json
from models import load_model
from dataloaders import load_testdata
from templates import load_template
from evaluators import load_evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--model_args', type=json.loads, help='Model arguments in JSON format')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API token')
    parser.add_argument('--hf_token', type=str, help='HuggingFace API token')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--dataset_args', type=json.loads, help='Dataset arguments in JSON format')
    parser.add_argument('--template_path', type=str, required=True, help='Path to the template file')
    parser.add_argument('--metric_path', type=str, required=True, help='Path to the metric file')
    parser.add_argument('--metric_args', type=json.loads, help='Metric arguments in JSON format')
    parser.add_argument('--result_path', type=str, required=True, help='Path to the result file')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for verbose output')
    return parser.parse_args()


def debug_print(debug_mode, *messages):
    """Print debug messages if debug mode is enabled."""
    if debug_mode:
        print("🐥", *messages)


def save_results(result_path, results):
    """Save the evaluation results to a file."""
    with open(result_path, 'a', encoding='utf-8') as f:
        for result in results:
            filtered_result = {}
            for k, v in result.items():
                if k in ['model_input', 'model_output', 'formatted_output']:
                    filtered_result[k] = v
                elif 'id' in k:
                    filtered_result['id'] = v
            f.write(json.dumps(filtered_result, ensure_ascii=False) + '\n')



def main():
    args = parse_args()

    model = load_model(args.model_path, args.openai_api_key, args.hf_token, args.model_args)
    dataset = load_testdata(args.dataset_path, args.dataset_args)
    template = load_template(args.template_path)
    evaluator = load_evaluator(args.metric_path, args.metric_args)

    debug_print(args.debug_mode, "Model loaded:", model)
    debug_print(args.debug_mode, "Dataset loaded:", len(dataset), "entries")
    debug_print(args.debug_mode, "Template loaded:", template)
    debug_print(args.debug_mode, "Evaluator loaded:", evaluator)

    for data in dataset:
        prompt = template.process(data)
        data['model_input'] = prompt
        data['model_output'] = model.generate(prompt)
        data['formatted_output'] = template.collate(data['model_output'])

        debug_print(args.debug_mode, "Input:", data['model_input'])
        debug_print(args.debug_mode, "Output:", data['model_output'])
        debug_print(args.debug_mode, "Formatted:", data['formatted_output'])

    if evaluator:
        record = {
            'model': args.model_path,
            'dataset': args.dataset_path,
            'template': args.template_path,
            'metrics': args.metric_path,
        }
        
        score = evaluator.calculate(dataset, record)
        print("score:", score)

    save_results(args.result_path, dataset)


if __name__ == '__main__':
    main()