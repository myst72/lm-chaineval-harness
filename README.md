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

1. [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) からテンプレートファイルを選ぶ、もしくは作成する
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

## 評価方法 - BackCodeEval

逆翻訳を活用した実行ベースでの評価方法をサポートしています。


1. [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) からテンプレートファイルを選ぶ、もしくは作成する
2. 任意のパス名に変更後、`chain.sh` として保存する
    ```sh
    # 評価したいタスク
    python3 ./scripts/main.py \
        --model_path <MODEL_PATH> \
        --dataset_path <DATASET_PATH> \
        --template_path <TEMPLATE_1_PATH> \
        --result_path <RESULT_1_PATH> \
    
    # BackCodeEval
    python3 ./scripts/main.py \
        --model_path <MODEL_PATH> \
        --dataset_path <RESULT_1_PATH> \
        --template_path <TEMPLATE_2_PATH> \
        --metric_path <METRIC_PATH> \
        --result_path result_all.jsonl \
    ```
    
    - `MODEL_PATH` : 評価したいモデルのパス名を指定
    - `DATASET_PATH` : HuggingFace Hub 上で提供されているデータセットのパス名を指定
        - 個人がローカルに所有するjsonl 形式のデータを指定することも可能
    - `TEMPLATE_1_PATH`, `TEMPLATE_2_PATH` : [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) から選んだテンプレートのパス名を指定
        - 個人で新たに作成したテンプレートのパス名の指定も可能
    - `METRIC_PATH` : 評価指標のパス名を指定
        - [HuggingFaceのevaluate-metric](https://huggingface.co/evaluate-metric)で提供されているパス名で指定する (e.g., pass@k: `code_eval`)


3. 評価を実行する
    ```sh
    sh chain.sh
    ```

## その他のオプション

### モデルのパラメータ設定

HuggingFace のPipeline で使用できるパラメータを個別で設定可能です。  
`model_args` として与えてください。

```sh
MODEL_PATH="MODEL_PATH"
DATASET_PATH="DATASET_PATH"
TEMPLATE_PATH="TEMPLATE_PATH"
METRIC_PATH="METRIC_PATH"

python3 ./scripts/main.py \
    --model_path $MODEL_PATH \
    --model_args '{"temperature": 0.1, "top_p": 0.90, "max_new_tokens": 512}' \
    --dataset_path $DATASET_PATH \
    --template_path $TEMPLATE_PATH \
    --metric_path $METRIC_PATH \
    --result_path result.jsonl \
```

### 量子化の有効化

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) を使用した4bitでの量子化を指定することができます。  
量子化を行う際には、コマンドライン引数として `quantize_model` を追加してください。

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
    --quantize_model
```

### デバッグモード

`debug_mode` を追加するとデバッグモードになります。

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
    --debug_mode
```
