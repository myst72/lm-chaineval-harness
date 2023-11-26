from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai
from openai import OpenAI
import torch

# =====================
# Base Classes
# =====================

class Model:
    """Base class for abstracting a pretrained model."""
    def generate(self, prompt: str)->str:
        return f"Generated response for: {prompt}"


class ModelLoader:
    """Loads a Model instance based on a model name and additional arguments."""
    def __init__(self, model_name, model_args:dict):
        self.model_name = model_name
        self.model_args = model_args

    def load(self)->Model:
        return Model()


# =====================
# Testing Code
# =====================

class TestModel:
    def generate(self, prompt: str, model_args=None)->str:
        return f"Generated response for: {prompt} \n with args: {model_args}"

class TestModelLoader(ModelLoader):
    def load(self)->Model:
        return TestModel()


# =====================
# HuggingFace Model Integration
# =====================

class HFModel(Model):
    def __init__(self, model_name, hf_token=None, model_args=None):
        default_args = {
            "max_length": 512,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.2,
            "return_full_text": False,
        }
        combined_args = {**default_args, **(model_args or {})}

        # super().__init__()
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=hf_token if hf_token else None,
            trust_remote_code=True, 
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # pipelineなしで実装----------------------------------
        # # Initialize the model
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name, 
        #     use_auth_token=hf_token if hf_token else None,
        #     trust_remote_code=True
        # )

        # # Set the device to GPU if available
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        # ----------------------------------

        self.model_args = combined_args

        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            # use_auth_token=hf_token if hf_token else None,
            **self.model_args
        )
    
    def generate(self, prompt: str) -> str:
        # pipelineなしで実装----------------------------------
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # generated_ids = self.model.generate(input_ids, **self.model_args)
        # return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # ----------------------------------
        generated_texts = self.generator(prompt, **self.model_args)
        return generated_texts[0]['generated_text']

class HFModelLoader(ModelLoader):
    def __init__(self, model_name, hf_token=None, model_args=None):
        super().__init__(model_name, model_args)
        self.hf_token = hf_token

    def load(self) -> HFModel:
        return HFModel(self.model_name, self.hf_token, self.model_args)


# =====================
# OpenAI Model Integration
# =====================

class OpenAIModel(Model):
    def __init__(self, openai_api_key, model_name, model_args=None):
        # Default arguments for OpenAI API
        default_args = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 512, 
            "n": 1}
        # Override defaults with any user-provided arguments
        combined_args = {**default_args, **(model_args or {})}

        super().__init__()
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.model_args = combined_args

    def generate(self, prompt: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.model_args
        )
        # prompt_and_response = prompt + "\n" + response.choices[0].message.content
        return response.choices[0].message.content

class OpenAIModelLoader(ModelLoader):
    def __init__(self, openai_api_key, model_name, model_args=None):
        super().__init__(model_name, model_args)
        self.openai_api_key = openai_api_key

    def load(self) -> OpenAIModel:

        return OpenAIModel(self.openai_api_key, self.model_name, self.model_args)


# =====================
# Model Loader Factory
# =====================

class ModelLoaderFactory:
    @staticmethod
    def create(model_name, openai_api_key=None, hf_token=None, model_args=None):
        try:
            if model_name == "test":
                return TestModelLoader(model_name, model_args)
            elif model_name.startswith("gpt"):
                return OpenAIModelLoader(openai_api_key, model_name, model_args)
            else:
                return HFModelLoader(model_name, hf_token, model_args)
        except Exception as e:
            print(f"Failed to load the model. Error message: {e}")
            raise e



# =====================
# Utility Function
# =====================

def load_model(model_path, openai_api_key, hf_token, model_args):
    model_loader = ModelLoaderFactory.create(model_path, openai_api_key, hf_token, model_args)
    model = model_loader.load()
    return model