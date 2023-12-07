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
        result_score = total_length / len(dataset) if dataset else 0
        for data in dataset:
            data['test_result_score'] = result_score
        return result_score, dataset


# =====================
# Evaluation Evaluator
# =====================

class CodeEvalEvaluator(Evaluator):
    """
    コード評価用Evaluatorクラス。HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """
    def is_blank(self, candidates):
        if isinstance(candidates, list):
            for sublist in candidates:
                if not isinstance(sublist, list) or not all(isinstance(item, str) and item.strip() == '' for item in sublist):
                    return False
            return True
        return False

    def calculate(self, dataset, record):
        # code_evalメトリックをロード
        code_eval_metric = load("code_eval")

        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        # データセット内の各データに対してcode_evalを実行
        for data in dataset:
            # テストケースと候補のコードを取得
            test_cases = [data['reference']]
            candidates = [[data['formatted_output']]]

            if not self.is_blank(candidates):
                pass_at_k, results = code_eval_metric.compute(references=test_cases, predictions=candidates, k=[1])
                data['item_pass@1_score'] = pass_at_k['pass@1'] # k=1 のスコアをリストに追加
            else:
                data['item_pass@1_score'] = 0.0

        # 各要素からitem_scoreを取り出して平均を算出
        item_scores = [data['item_pass@1_score'] for data in dataset]
        average_pass_at_k = sum(item_scores) / len(item_scores) if item_scores else 0

        for data in dataset:
            data['all_pass@1_score'] = average_pass_at_k

        return average_pass_at_k, dataset


class AccuracyEvaluator(Evaluator):
    # 正確性評価用のEvaluatorクラスの実装は省略
    def calculate(self, dataset, record):
        
        # 正解データと予測データのリストを準備
        references = [d['reference'] for d in dataset]
        candidates = [d['model_output'] for d in dataset]
        # accuracy スコアを計算
        score = self.metric.compute(predictions=candidates, references=references)['accuracy']

        for data in dataset:
            data['accuracy_score'] = score
 
        return score, dataset


class BLEUEvaluator(Evaluator):
    def calculate(self, dataset, record):

        # BLEUメトリック用のデータ準備
        references = [[d['reference'].split()] for d in dataset]  # リストのリストとして分割された参照文
        candidates = [d['model_output'].split() for d in dataset]  # 分割された予測文のリスト
        # BLEU スコアを計算
        score = self.metric.compute(predictions=candidates, references=references)['bleu']

        for data in dataset:
            data['bleu_score'] = score
 
        return score, dataset


class F1Evaluator(Evaluator):
    def calculate(self, dataset, record):
        
        # F1スコアの計算に必要な正解ラベルと予測ラベルのリストを準備
        references = [d['reference'] for d in dataset]
        candidates = [d['model_output'] for d in dataset]
        # F1スコアを計算
        score = self.metric.compute(predictions=candidates, references=references)["f1"]
        # `score` には通常、precision, recall, f1 のキーが含まれている
        #f1_score = score['f1']
        #score = f1_score

        for data in dataset:
            data['f1_score'] = score
 
        return score, dataset


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
        elif metric_path == "F1":
            return F1Evaluator(metric_args)
        else:
            raise ValueError(f"Unknown metric path: {metric_path}")



# =====================
# Utility Function
# =====================

def load_evaluator(metric_path, metric_args):
    if metric_path:
        return EvaluatorLoaderFactory.create(metric_path, metric_args)
    else:
        return False