from evaluate import load
import os

# =====================
# Base Class
# =====================

class Evaluator:
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified metrics, and calculate scores.
    """

    def __init__(self, metric_args):
        self.metric_args = metric_args

    def calculate(self, dataset, record):
        raise NotImplementedError("Must override calculate in subclass")


# =====================
# Testing Code
# =====================

class TestEvaluator(Evaluator):
    """
    テスト用Evaluatorクラス。単に文字列の長さでスコアを算出する。
    """

    def calculate(self, dataset, record):
        # データセット内のすべてのoutputの長さの平均を計算してスコアとする
        total_length = sum(len(data['formatted_output']) for data in dataset)
        return total_length / len(dataset) if dataset else 0


# =====================
# Evaluation Evaluator
# =====================

class CodeEvalEvaluator(Evaluator):
    """
    コード評価用Evaluatorクラス。HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """
    def calculate(self, dataset, record):
        
        # code_evalメトリックをロード
        code_eval_metric = load("code_eval")

        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        pass_at_k_scores = []

        # データセット内の各データに対してcode_evalを実行
        for data in dataset:
            # テストケースと候補のコードを取得
            test_cases = [data['test']]
            candidates = [[data['formatted_output']]]

            # code_evalメトリックを計算
            try:
                pass_at_k, results = code_eval_metric.compute(references=test_cases, predictions=candidates, k=[1])
                pass_at_k_scores.append(pass_at_k['pass@1']) # k=1 のスコアをリストに追加
            except Exception as e:
                print(f"Error evaluating code: {e}")
                pass_at_k_scores.append(0)  # エラーが発生した場合は0を追加

        # pass_at_kの平均値を最終スコアとする
        average_pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0
        return average_pass_at_k



class AccuracyEvaluator(Evaluator):
    # 正確性評価用のEvaluatorクラスの実装は省略
    pass


class BLEUEvaluator(Evaluator):
    # BLEUスコア評価用のEvaluatorクラスの実装は省略
    pass


# =====================
# Evaluator Loader Factory
# =====================

class EvaluatorLoaderFactory:
    """
    Evaluatorのインスタンスを生成するためのファクトリクラス。
    metric_pathに応じて適切なEvaluatorクラスをロードする。
    """
    @staticmethod
    def create(metric_path, metric_args):
        if metric_path == "test":
            return TestEvaluator(metric_args)
        elif metric_path == "code_eval":
            return CodeEvalEvaluator(metric_args)
        elif metric_path == "accuracy":
            return AccuracyEvaluator(metric_args)
        elif metric_path == "BLEU":
            return BLEUEvaluator(metric_args)
        else:
            raise ValueError(f"Unknown metric path: {metric_path}")



# =====================
# Utility Function
# =====================

def load_evaluator(metric_path, metric_args):
    return EvaluatorLoaderFactory.create(metric_path, metric_args)