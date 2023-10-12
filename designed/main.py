from .dataloaders import load_testdata
from .models import load_model
from .templates import TemplateProcessor
from .evaluators import Evaluator

def main():
    keymap = {}
    model = load_model(model_path, model_args)
    dataset = load_testdata(source)
    template = load_template(template_path, keymap)
    evaluator = load_evaluator(metrics_path)

    result = keymap.get('result', 'result')
    for data in dataset:
        prompt = template(data)
        data['model_input'] = prompt
        data['model_output'] = model.generate(prompt, n)
        data[result] = template.collate(data['model_output'])

    if evaluator:
        record = dict(
            model=model_path,
            dataset=source,
            template=template_path,
            metrics=metrics,
        )
        evaluator.calculate(data, keymap, record)



    # print(data[:2])  # 最初の2つのデータを表示