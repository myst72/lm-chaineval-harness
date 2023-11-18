import json
import re
import ast

class TemplateProcessor:
    def __init__(self, template_path):
        self.template_path = template_path
        self.template_data = self.load()
        self.template_string = self.template_data.get("template", "")
        self.reference_string = self.template_data.get("reference", "")

    def load(self):
        """Loads the template file."""
        if self.template_path.endswith('.json'):
            with open(self.template_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            raise ValueError("Unsupported template format. Please provide a .json file.")

    def process(self, data):
        """Creates a prompt using the loaded template and provided data."""
        try:
            prompt = self.template_string.format(**data)
            return prompt
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for template: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in template formatting: {e}")
    
    def process_reference(self, data):
        """Creates a reference using the loaded template and provided data."""
        try:
            reference = self.reference_string.format(**data)
            return reference
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for reference: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in reference formatting: {e}")

    # def format_natural_language(self, model_output):
    #     formatted_output = re.sub(r"<outpuT>|```.*?```", "", model_output, flags=re.IGNORECASE | re.DOTALL) # Remove special token <outpuT> and code blocks
    #     return formatted_output.strip()
    
    def extract_textblock(self, text):
        """テキストブロック抽出"""
        pattern = r'"""'
        matches = [match.start() for match in re.finditer(pattern, text)]

        # マッチが2つ未満の場合は元のテキストを返す
        if len(matches) < 2:
            return text  

        # マッチの数に基づいて最も内側のブロックの開始と終了インデックスを計算
        middle_index = len(matches) // 2
        innermost_start = matches[middle_index - 1] + 3  # 開始インデックスの直後
        innermost_end = matches[middle_index]  # 終了インデックス

        # 最も内側のブロックを抽出
        innermost_block = text[innermost_start:innermost_end]
        return innermost_block.strip()  # 余分な空白を削除


    def format_natural_language(self, model_output, prompt=''):
        """自然言語後処理"""
        #テキストブロック抽出
        format_output = self.extract_textblock(model_output) 

        if '<outpuT>' in format_output:
            start_index = format_output.find('<outpuT>') + len('<outpuT>')
            format_output = format_output[start_index:].strip() # '<outpuT>'以降を返す

        #プロンプトと重複行抽出
        promptlines = [line.strip() for line in prompt.split('\n')]
        format_outputlines = format_output.split('\n')
        format_output = [line for line in format_outputlines if line.strip() not in promptlines]
        format_output = '\n'.join(format_output)
        return format_output
        
    def format_programming_language(self, model_output): #FIXME
        code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", model_output, flags=re.DOTALL) # Extract content within code blocks
        extracted_code = '\n'.join(code_blocks).strip() if code_blocks else model_output
        functions = []
        for match in re.finditer(r'\bdef\b.*?\breturn\b.*?(?=\n|$)', extracted_code, flags=re.DOTALL): # Attempt to parse and validate each function-like string
            function_code = match.group(0)
            try:
                ast.parse(function_code)# Check syntax
                functions.append(function_code)
            except SyntaxError:
                pass
        formatted_code = '\n\n'.join(functions).strip()
        return formatted_code

    def collate(self, prompt, model_output):
        """Collates the model output according to the specified output language in the template data."""
        output_lang = self.template_data.get('output_lang', 'NL')

        if output_lang in ['NL', 'en', 'ja', 'ko', 'zh']: # Format for natural language
            return self.format_natural_language(model_output, prompt)

        elif output_lang in ['PL', 'py', 'cpp', 'js', 'ru']: # Format for programming language
            return self.format_programming_language(model_output)

        else:
            raise ValueError(f"Unsupported output language: {output_lang}")



# =====================
# Utility Function
# =====================

def load_template(template_path):
    template = TemplateProcessor(template_path)
    return template