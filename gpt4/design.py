import argparse
import json

"""
基本的なソフトウェア設計を以下のモジュールに分けて行います。

CLI Interface: コマンドラインからの引数をパースし、適切な処理を呼び出すインターフェース。
Model Loader: HuggingFaceや商用APIモデル(ChatGPTなど)をロードする。
Dataset Loader: 指定されたデータセットやJSONLファイルをロードする。
Template Processor: 複数のプロンプトテンプレートをロードして、実際のプロンプトを生成する。
Evaluator: モデルにプロンプトを投げ、出力を受け取り、指定されたメトリックスで評価を行う。
Result Saver: 実験結果やモデルの出力結果をJSONL形式で保存する。

"""


# 1. CLI Interface
class CLIInterface:
    """
    argparse や click を使用して、コマンドライン引数を解析。
    必要な引数やオプションを定義し、他のモジュールの関数やクラスを呼び出す。
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="lm_chain_harness tool")
        self._init_args()

    def _init_args(self):
        self.parser.add_argument("--model", type=str, required=True)
        self.parser.add_argument("--model_args", type=str)
        self.parser.add_argument("--dataset", type=str, required=True)
        self.parser.add_argument("--prompt_template", type=str, required=True)
        self.parser.add_argument("--prompt_args", type=str)
        self.parser.add_argument("--metrics", type=str, required=True)
        self.parser.add_argument("--result_path", type=str, required=True)

    def parse_args(self):
        return self.parser.parse_args()

# 2. Model Loader
class ModelLoader:
    """
    指定されたモデルをロードする基底クラス。
    HuggingFaceモデルの場合は、Transformersライブラリを使用。
    商用APIモデルの場合は、そのAPIに対応するクライアントライブラリを使用。
    """
    def __init__(self, model_name, model_args=None):
        self.model_name = model_name
        self.model_args = model_args

    def load(self):
        print(f"Loading model: {self.model_name} with args: {self.model_args}")
        return "Sample Model"

# 3. Dataset Loader
class DatasetLoader:
    """
    指定されたデータセット名やJSONLファイルパスからデータをロード。
    データセットは内部的にリストや辞書などのデータ構造で保持。
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):
        print(f"Loading dataset: {self.dataset}")
        return ["Sample Data"]

# 4. Template Processor
class TemplateProcessor:
    """
    指定されたテンプレートファイルをロード。
    テンプレートから実際のプロンプトを生成。(prompt_argsに応じて処理を変更)
    """

    def __init__(self, template_file, prompt_args=None):
        self.template_file = template_file
        self.prompt_args = prompt_args

    def process(self):
        print(f"Processing template: {self.template_file} with args: {self.prompt_args}")
        return "Processed Prompt"

# 5. Evaluator
class Evaluator:
    """
    モデルを使用して、生成されたプロンプトに対する回答を取得。
    metricsオプションに基づいて評価を行い、スコアを算出。
    """
    
    def __init__(self, model, prompt, metrics):
        self.model = model
        self.prompt = prompt
        self.metrics = metrics

    def evaluate(self):
        print(f"Evaluating using model: {self.model} and prompt: {self.prompt}")
        return "Sample Score"

# 6. Result Saver
class ResultSaver:
    """
    評価結果やモデルの出力を、指定されたパスにJSONL形式で保存。
    """

    def __init__(self, result_path):
        self.result_path = result_path

    def save(self, result):
        print(f"Saving result to: {self.result_path}")
        with open(self.result_path, 'w') as f:
            json.dump(result, f)

if __name__ == "__main__":
    cli = CLIInterface()
    args = cli.parse_args()

    model_loader = ModelLoader(args.model, args.model_args)
    model = model_loader.load()

    dataset_loader = DatasetLoader(args.dataset)
    dataset = dataset_loader.load()

    template_processor = TemplateProcessor(args.prompt_template, args.prompt_args)
    prompt = template_processor.process()

    evaluator = Evaluator(model, prompt, args.metrics)
    result = evaluator.evaluate()

    result_saver = ResultSaver(args.result_path)
    result_saver.save(result)
