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


    @staticmethod
    def create(evaluator_type, model, prompts, metrics):
        if evaluator_type == "SampleEvaluator":
            return SampleEvaluator(model, prompts, metrics)
        # Add other evaluator conditions here
        raise ValueError(f"Unknown evaluator: {evaluator_type}")


class SampleEvaluator:
    def __init__(self, model, prompts, metrics):
        self.model = model
        self.prompts = prompts
        self.metrics = metrics
    
    def evaluate(self):
        results = []
        for prompt in self.prompts:
            output = self.model.generate(prompt)
            results.append({"prompt": prompt, "output": output, "score": output})
        return results