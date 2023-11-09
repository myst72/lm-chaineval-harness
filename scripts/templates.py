import json
import re
import ast

class TemplateProcessor:
    def __init__(self, template_path):
        self.template_path = template_path
        self.template_data = self.load()
        self.template_string = self.template_data.get("template", "")

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
    
    def format_natural_language(self, model_output):
        formatted_output = re.sub(r"<outpuT>|```.*?```", "", model_output, flags=re.IGNORECASE | re.DOTALL) # Remove special token <outpuT> and code blocks
        return formatted_output.strip()
        
    def format_programming_language(self, model_output):
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

    def collate(self, model_output):
        """Collates the model output according to the specified output language in the template data."""
        output_lang = self.template_data.get('output_lang', 'NL')

        if output_lang in ['NL', 'en', 'jp']: # Format for natural language
            return self.format_natural_language(model_output)

        elif output_lang in ['PL', 'py']: # Format for programming language
            return self.format_programming_language(model_output)

        else:
            raise ValueError(f"Unsupported output language: {output_lang}")



# =====================
# Utility Function
# =====================

def load_template(template_path):
    template = TemplateProcessor(template_path)
    return template