# lm-chaineval-harness
An chain version of lm evaluation framework.

このツールは、言語モデルの自動評価用フレームワークです。  
以下の機能を提供します。

- 既存の評価用データを用いた言語モデルの評価
- 既存の評価用データを活用した、逆翻訳による実行ベース評価での言語モデルの評価

## 環境設定

### Install

```shell
git clone https://github.com/KuramitsuLab/lm-chaineval-harness.git
cd lm-chaineval-harness
pip3 install -r requirements.txt
```

### 環境変数

評価に使用するモデルで、HuggingFace のアクセストークンや、OpenAI のAPI キーが必要な場合には、`.env` ファイルに記載して保存してください。

```plaintext:envファイル
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
HF_TOKEN=HF_TOKEN
```

## 評価方法

1. [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) からテンプレートファイルを選ぶ
2. 任意のパス名に変更後、`chain.sh` として保存する
    ```sh
    MODEL_PATH="MODEL_PATH"
    DATASET_PATH="DATASET_PATH"
    TEMPLATE_PATH="TEMPLATE_PATH"
    METRIC_PATH="METRIC_PATH"

    python3 ./scripts/main.py \
        --model_path $MODEL_PATH \
        --dataset_path $DATASET_PATH \
        --template_path $TEMPLATE_PATH \
        --metric_path $METRIC_PATH \
        --result_path result.jsonl \
    ```
    
    - `MODEL_PATH` : 評価したいモデルのパス名を指定
    - `DATASET_PATH` : HuggingFace Hub 上で提供されているデータセットのパス名を指定
        - 個人がローカルに所有するjsonl 形式のデータを指定することも可能
    - `TEMPLATE_PATH` : [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) から選んだテンプレートのパス名を指定
        - 個人で新たに作成したテンプレートのパス名の指定も可能
    - `METRIC_PATH` : 評価指標のパス名を指定
        - [HuggingFaceのevaluate-metric](https://huggingface.co/evaluate-metric)で提供されているパス名で指定する (e.g., pass@k: `code_eval`)


3. 評価を実行する
    ```sh
    sh chain.sh
    ```

### アクセストークンやAPI が必要なモデルの評価

- HuggingFace アクセストークンの場合：

    ```sh
    MODEL_PATH="MODEL_PATH"
    DATASET_PATH="DATASET_PATH"
    TEMPLATE_PATH="TEMPLATE_PATH"
    METRIC_PATH="METRIC_PATH"

    source ./.env

    python3 ./scripts/main.py \
        --model_path $MODEL_PATH \
        --hf_token $HF_TOKEN \
        --dataset_path $DATASET_PATH \
        --template_path $TEMPLATE_PATH \
        --metric_path $METRIC_PATH \
        --result_path result.jsonl \
    ```

- OpenAI APIキーの場合：

    ```sh
    MODEL_PATH="MODEL_PATH"
    DATASET_PATH="DATASET_PATH"
    TEMPLATE_PATH="TEMPLATE_PATH"
    METRIC_PATH="METRIC_PATH"

    source ./.env

    python3 ./scripts/main.py \
        --model_path $MODEL_PATH \
        --openai_api_key $OPENAI_API_KEY \
        --dataset_path $DATASET_PATH \
        --template_path $TEMPLATE_PATH \
        --metric_path $METRIC_PATH \
        --result_path result.jsonl \
    ```