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
