from abc import ABC, abstractmethod
import json
from datasets import load_dataset

"""
DataLoaderのサンプルデータは、HumanEval のデータ形式に変更できますか？
"""

# 1. Abstract DataLoader
class DataLoader(ABC):
    @abstractmethod
    def load(self) -> list[dict]:
        pass

# 2. HuggingFace DataLoader for HumanEval
class HFDataLoader(DataLoader):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load(self) -> list[dict]:
        dataset = load_dataset(self.dataset_path)
        # テストセットを辞書のリストとして返す
        return [{k: v for k, v in zip(dataset["test"].features, item)} for item in dataset["test"]]

# 3. JSONL DataLoader
class JSONDataLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[dict]:
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

# 4. Test DataLoader
class TestDataLoader(DataLoader):
    def load(self) -> list[dict]:
        # テスト用のデータを返します
        return [{"task_id": "test_1", "prompt": "test_prompt_1", "canonical_solution": "test_solution_1", "test": "test_test_1", "entry_point": "test_entry_1"},
                {"task_id": "test_2", "prompt": "test_prompt_2", "canonical_solution": "test_solution_2", "test": "test_test_2", "entry_point": "test_entry_2"}]

# 5. DataLoader Factory
class DataLoaderFactory:
    @staticmethod
    def create(source: str) -> DataLoader:
        if source == "huggingface":
            return HFDataLoader(source)
        elif source.endswith(".jsonl"):
            return JSONDataLoader(source)
        elif source == "test":
            return TestDataLoader()
        else:
            raise ValueError(f"Unknown data source: {source}")

def load_testdata(source: str):
    loader = DataLoaderFactory.create(source)
    return loader.load()
