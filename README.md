# LLM Evaluation CLI

CLIツールで複数クラウドプロバイダーのLLMを同一プロンプトで実行し、回答・トークン数・レイテンシーを比較します。

## 対象モデル

| キー | プロバイダー | モデル |
|---|---|---|
| `gemini-3-flash-preview` | Google Cloud | Gemini 3 Flash Preview |
| `gemini-2-5-flash` | Google Cloud | Gemini 2.5 Flash |
| `gemini-2-5-flash-lite` | Google Cloud | Gemini 2.5 Flash Lite |
| `claude-haiku-4-5` | AWS Bedrock | Claude Haiku 4.5 |
| `claude-3-haiku` | AWS Bedrock | Claude 3 Haiku |
| `amazon-nova2-lite` | AWS Bedrock | Amazon Nova2 Lite |

## セットアップ

### DevContainer を使う場合（推奨）

VS Code と [Dev Containers 拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) または GitHub Codespaces を使うと、依存パッケージのインストールなしにすぐ動作する環境が起動します。

#### 認証情報の渡し方

**方法 1: 環境変数（ホスト側で設定済みの場合）**

ホスト OS に以下の環境変数を設定しておくと、コンテナに自動で引き継がれます。

```bash
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
export GOOGLE_CLOUD_LOCATION=us-central1          # 省略時は us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json  # ADC使用時は不要
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_DEFAULT_REGION=us-east-1
```

**方法 2: `.env` ファイル**

```bash
cp .env.example .env
# .env を編集してキーを設定
```

コンテナ起動後、`python-dotenv` が `.env` を自動読み込みします。

**方法 3: Google Cloud ADC（Application Default Credentials）**

`gcloud auth application-default login` で設定した ADC を使う場合、`GOOGLE_APPLICATION_CREDENTIALS` は不要です。
ホストの `~/.config/gcloud` ディレクトリがコンテナに読み取り専用でマウントされるため、ADC がそのまま利用できます。

**方法 4: AWS プロファイル（`~/.aws/credentials`）**

ホストの `~/.aws` ディレクトリがコンテナに読み取り専用でマウントされます。IAM ロールや AWS SSO も利用可能です。

#### 起動方法

1. このリポジトリを VS Code で開く
2. コマンドパレットで `Dev Containers: Reopen in Container` を実行
3. コンテナのビルドが完了したら、ターミナルで実行可能になります

### ローカル環境でセットアップする場合

#### 1. 依存パッケージのインストール

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. 認証情報の設定

```bash
cp .env.example .env
# .env を編集してキーを設定
```

`.env` の内容:

```
GOOGLE_CLOUD_PROJECT=...           # GCP プロジェクト ID
GOOGLE_CLOUD_LOCATION=us-central1  # GCP リージョン（省略時は us-central1）
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json  # ADC 使用時は不要
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

Google Cloud は `gcloud auth application-default login` による ADC でも認証可能です。
AWS は `~/.aws/credentials` や IAM ロールでも認証可能です。

## 使い方

```bash
# 全モデルで実行
python main.py "量子もつれをわかりやすく説明してください。"

# 特定モデルのみ指定
python main.py --models gemini-2-5-flash,claude-3-haiku "AIの未来は？"

# ファイルからプロンプトを読み込む
python main.py --prompt-file prompt.txt

# stdin から読み込む
echo "ハイクを作ってください。" | python main.py

# 順番に実行（デフォルトは並列）
python main.py --sequential "テスト"

# コンソール出力なし（ログのみ保存）
python main.py --quiet "テスト"
```

## 出力

### コンソール出力（比較表 + 回答）

```
------------------------------------------------------------------------------------------
Model                                         In    Out  Think        ms  Status
------------------------------------------------------------------------------------------
gemini-2.5-flash-preview-04-17               120    340    512    1234.5  OK
anthropic.claude-3-haiku-20240307-v1:0        98    280      -     890.2  OK
...
------------------------------------------------------------------------------------------
```

### JSON ログ（`logs/<run-id>.json`）

```json
{
  "run_id": "uuid",
  "timestamp": "2025-01-01T00:00:00+00:00",
  "prompt": "...",
  "results": [
    {
      "run_id": "uuid",
      "timestamp": "...",
      "provider": "google_cloud",
      "model": "gemini-2.5-flash-preview-04-17",
      "prompt": "...",
      "response": "...",
      "tokens": {
        "input": 120,
        "output": 340,
        "thinking": 512,
        "total": 972
      },
      "latency_ms": 1234.5,
      "error": null,
      "raw_usage": { ... }
    }
  ]
}
```

## 設定（`config.yaml`）

各モデルの有効/無効、モデルID、オプション（temperature、最大トークン数、thinking設定など）を変更できます。

```yaml
models:
  gemini-2-5-flash:
    provider: google_cloud
    model_id: gemini-2.5-flash-preview-04-17
    enabled: true
    options:
      enable_thinking: true
      thinking_budget: 8192
      max_output_tokens: 8192
```

## トークン計測について

| プロバイダー | 入力 | 出力 | 思考 |
|---|---|---|---|
| Google Cloud (Gemini) | `prompt_token_count` | `candidates_token_count` | `thoughts_token_count` |
| AWS Bedrock (Claude) | `input_tokens` | `output_tokens` | thinkingブロックの単語数（推計） |
| AWS Bedrock (Nova2) | `inputTokens` | `outputTokens` | API提供時に対応 |
