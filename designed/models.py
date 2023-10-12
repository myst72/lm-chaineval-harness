# 2. Model Loader and the LM class
class Model:
    """
    学習済みモデルを抽象化したクラス. 
    """
    def generate(self, prompt: str)->str:
        """
        プロンプトを受け取り、モデルが出力した結果を返す。
        """
        return f"Generated response for: {prompt}"

class ModelLoader:
    """
    モデル名とオプションからModelインスタンスをロードするクラス
    """
    def __init__(self, model_name, model_args:dict):
        self.model_name = model_name
        self.model_args = model_args

    def load()->Model:
        return Model()

class TestModel:
    def generate(self, prompt: str)->str:
        return f"Generated response for: {prompt}"

class TestModelLoader(ModelLoader):
    def load()->Model:
        return TestModel()

class ModelLoaderFactory:
    @staticmethod
    def create(model_name, model_args=None):
        if model_name == "test":
            return ModelLoader(model_name, model_args)
        else:
            raise ValueError(f"Unknown model: {model_name}")


# # Example usage:
# model = ModelLoader.create("test", {"option1": "value1", "option2": "value2"})
# print(model.generate("How's the weather?"))
