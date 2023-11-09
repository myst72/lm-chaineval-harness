from abc import ABC, abstractmethod
import json
from datasets import load_dataset


# =====================
# Base Class
# =====================

class DataLoader(ABC):
    """An abstract base class for all data loaders to enforce the implementation of the load method."""
    @abstractmethod
    def load(self) -> list[dict]:
        pass


# =====================
# Testing Code
# =====================

class TestDataLoader(DataLoader):
    """A test data loader that returns a predefined list of dictionaries."""
    def load(self) -> list[dict]:
        return [
            {"task_id": "test_1", "prompt": "test_prompt_1", "canonical_solution": "test_solution_1", "test": "test_test_1", "entry_point": "test_entry_1"},
            {"task_id": "test_2", "prompt": "test_prompt_2", "canonical_solution": "test_solution_2", "test": "test_test_2", "entry_point": "test_entry_2"}
        ]


# =====================
# JSON Data Loader
# =====================

class JSONDataLoader(DataLoader):
    """A data loader for JSONL files."""
    def __init__(self, dataset_path: str, dataset_args: dict = None):
        self.dataset_path = dataset_path
        self.dataset_args = dataset_args if dataset_args is not None else {}
        self.dataset_num = self.dataset_args.get('num')

    def load(self) -> list[dict]: 
        dataset = []
        try:
            with open(self.dataset_path, 'r') as f:
                dataset = [json.loads(line.strip()) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.dataset_path} does not exist.")
        dataset = dataset if self.dataset_num is None else dataset[:self.dataset_num]
        return dataset


# =====================
# HuggingFace Data Loader
# =====================

class HFDataLoader(DataLoader):
    """A data loader for datasets available through the Hugging Face datasets library."""
    def __init__(self, dataset_path: str, dataset_args: dict = None):
        self.dataset_path = dataset_path
        self.dataset_args = dataset_args if dataset_args is not None else {}
        self.dataset_num = self.dataset_args.get('num')

    def load(self) -> list[dict]:
        split = self.dataset_args.get("split", "test")
        subset = self.dataset_args.get("subset")

        if subset:
            dataset = load_dataset(self.dataset_path, subset, split=split)
        else:
            dataset = load_dataset(self.dataset_path, split=split)
        
        dataset = [{k: v for k, v in item.items()} for item in dataset]
        dataset = dataset if self.dataset_num is None else dataset[:self.dataset_num]
        return dataset


# =====================
# Data Loader Factory
# =====================

class DataLoaderFactory:
    """Factory class to create appropriate data loader instances based on the provided dataset path."""
    @staticmethod
    def create(dataset_path: str, dataset_args: dict = None) -> DataLoader:
        if dataset_path == "test":
            return TestDataLoader()
        elif dataset_path.endswith(".jsonl"):
            return JSONDataLoader(dataset_path, dataset_args)
        else:
            return HFDataLoader(dataset_path, dataset_args)



# =====================
# Utility Function
# =====================

def load_testdata(dataset_path: str, dataset_args: dict = None):
    loader = DataLoaderFactory.create(dataset_path, dataset_args)
    return loader.load()