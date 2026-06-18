# nanochat フォルダ構成・役割まとめ

## 全体像

```
nanochat/          ← コアライブラリ（再利用可能なモジュール群）
scripts/           ← 実行スクリプト（各ステージのエントリポイント）
tasks/             ← 評価・SFT用タスク定義
runs/              ← シェルスクリプト（実験レシピ）
```

---

## `nanochat/` — コアライブラリ

LLM の学習・推論に必要な基盤モジュールをまとめたパッケージ。`scripts/` から import して使われる。

### モデル定義

| ファイル | 役割 |
|---|---|
| `gpt.py` | GPT トランスフォーマー本体（`nn.Module`）。`GPTConfig`・`CausalSelfAttention`・`MLP`・`Block`・`GPT` クラスを定義。Rotary Embeddings、QK Norm、GQA、relu²、スライディングウィンドウアテンション、Value Embeddings（ResFormer スタイル）を実装 |
| `flash_attention.py` | Flash Attention 3（Hopper GPU）と PyTorch SDPA の自動切り替えラッパー。`flash_attn` モジュールとして FA3 と同一 API を提供 |
| `fp8.py` | FP8 トレーニング実装。`Float8Linear` で `nn.Linear` を置き換え、cuBLAS FP8 カーネルを使用（torchao 不要の軽量実装、約150行） |

### 最適化

| ファイル | 役割 |
|---|---|
| `optim.py` | `MuonAdamW`（単一GPU）と `DistMuonAdamW`（分散）を実装。2D 行列パラメータには Muon（Newton-Schulz 直交化）、それ以外には AdamW を使用。`DistMuonAdamW` は ZeRO-2 スタイルのオプティマイザ状態シャーディングと3フェーズ非同期通信を実装 |

### データ

| ファイル | 役割 |
|---|---|
| `dataloader.py` | 分散対応の事前学習用データローダー。BOS アライン + Best-Fit パッキングで 100% トークン利用率を実現。Parquet ファイルを読み込み、マルチスレッドでトークン化 |
| `dataset.py` | 事前学習データ（Parquet 形式）のダウンロード・一覧取得ユーティリティ |
| `tokenizer.py` | GPT-4 スタイル BPE トークナイザー。`HuggingFaceTokenizer` と `RustBPETokenizer`（tiktoken ベース）の2実装。会話のトークン化（`render_conversation`）、RL 用の `render_for_completion` も担当。特殊トークン（`<|bos|>`、`<|user_start|>` など）を管理 |

### 推論

| ファイル | 役割 |
|---|---|
| `engine.py` | KV キャッシュを使った効率的な推論エンジン。`KVCache` クラスと `Engine.generate()` を提供。prefill を batch=1 で実行後、KV キャッシュを複製して複数サンプルを並列生成するストリーミングジェネレータ。Python 電卓ツール使用のステートマシンも内蔵 |

### 評価・ユーティリティ

| ファイル | 役割 |
|---|---|
| `core_eval.py` | DCLM 論文の CORE スコア評価（ベースモデル用）。多肢選択・スキーマ・言語モデリングタスクを分散評価 |
| `loss_eval.py` | バリデーション損失をバイト毎ビット（bpb）で評価 |
| `checkpoint_manager.py` | チェックポイントの保存・読み込み。`save_checkpoint` / `load_model` / `load_optimizer_state` を提供。`base` / `sft` / `rl` の3ステージに対応 |
| `common.py` | `COMPUTE_DTYPE`、`compute_init`、`print0`、`get_base_dir` などの共通ユーティリティ |
| `execution.py` | LLM が生成した Python コードをサンドボックス実行する機能（タイムアウト・メモリ制限・危険関数無効化）。HumanEval 評価と RL 報酬計算に使用 |
| `report.py` | 学習結果のレポート生成ユーティリティ |
| `ui.html` | チャット Web UI のフロントエンド（HTML/CSS/JS） |

---

## `scripts/` — 実行スクリプト

LLM パイプラインの各ステージに対応するエントリポイント。`python -m scripts.XXX` または `torchrun` で実行する。

### トークナイザー

| ファイル | 役割 |
|---|---|
| `tok_train.py` | BPE トークナイザーを FineWeb データから学習・保存 |
| `tok_eval.py` | トークナイザーの圧縮率（バイト/トークン）を評価 |

### ベースモデル（事前学習）

| ファイル | 役割 |
|---|---|
| `base_train.py` | ベースモデルの事前学習メインスクリプト。`--depth` 1つで全ハイパーパラメータを自動決定（モデル幅・ヘッド数・学習率・バッチサイズ・学習ステップ数）。wandb ロギング・チェックポイント保存・分散学習対応。FP8 トレーニングオプションあり |
| `base_eval.py` | ベースモデルの評価（CORE スコア・bpb・サンプル生成） |

### チャットモデル（SFT・RL・推論）

| ファイル | 役割 |
|---|---|
| `chat_sft.py` | Supervised Fine-Tuning（SFT）。SmolTalk・MMLU・GSM8K・SpellingBee などのタスク混合でファインチューニング。事前学習チェックポイントからハイパーパラメータを継承 |
| `chat_rl.py` | 強化学習（GRPO ベース、実質 REINFORCE）。GSM8K の正解報酬でオンポリシー学習。KL 正則化なし |
| `chat_eval.py` | チャットモデルの評価（ARC・MMLU・GSM8K・HumanEval・SpellingBee の ChatCORE スコア計算） |
| `chat_web.py` | FastAPI ベースの Web チャットサーバー。複数 GPU への `WorkerPool` 分散対応、SSE ストリーミング応答 |
| `chat_cli.py` | CLI でチャットモデルと対話 |

---

## `tasks/` — タスク定義

SFT 学習データおよびチャットモデル評価に使うタスクを定義するパッケージ。各タスクは「会話形式のデータセット」として統一インターフェースを持つ。

### 基底クラス

| ファイル | 役割 |
|---|---|
| `common.py` | `Task`（基底クラス）・`TaskMixture`（複数タスクをシャッフル混合）・`TaskSequence`（順番に連結）・`render_mc`（多肢選択フォーマット）を定義。`Task` は `__getitem__` で会話オブジェクトを返し、`evaluate(conversation, completion)` で正誤判定する統一インターフェースを持つ |

### 各タスク

| ファイル | eval_type | 内容 |
|---|---|---|
| `arc.py` | categorical | ARC-Easy / ARC-Challenge（Allen AI の多肢選択科学問題） |
| `mmlu.py` | categorical | MMLU（幅広いトピックの多肢選択問題、100K 問） |
| `gsm8k.py` | generative | GSM8K（小学校レベルの数学 8K 問）。Python ツール呼び出し形式の会話に変換。`reward()` メソッドで RL にも使用 |
| `humaneval.py` | generative | HumanEval（Python コーディングタスク）。`execution.py` でコードを実行して正誤判定 |
| `spellingbee.py` | generative | `SpellingBee`（単語中の文字数カウント）と `SimpleSpelling`（単語のスペル）。Python ツール呼び出しを含む合成データを生成 |
| `smoltalk.py` | — | HuggingFace SmolTalk（460K 件の汎用会話データ）。SFT の主要データソース |
| `customjson.py` | — | 任意の JSONL ファイルから会話データを読み込む汎用タスク（アイデンティティ注入などに使用） |

---

## フォルダ間の依存関係

```
scripts/base_train.py  →  nanochat/gpt.py
                       →  nanochat/dataloader.py
                       →  nanochat/optim.py
                       →  nanochat/checkpoint_manager.py

scripts/chat_sft.py    →  tasks/common.py
                       →  tasks/gsm8k.py, tasks/smoltalk.py, ...

scripts/chat_rl.py     →  nanochat/engine.py
                       →  tasks/gsm8k.py

scripts/chat_web.py    →  nanochat/engine.py
                       →  nanochat/checkpoint_manager.py

nanochat/engine.py     →  nanochat/gpt.py
nanochat/gpt.py        →  nanochat/flash_attention.py
                       →  nanochat/optim.py

tasks/humaneval.py     →  nanochat/execution.py
```

---

## LLM パイプラインの流れ

```
tok_train.py          # 1. トークナイザー学習
    ↓
base_train.py         # 2. 事前学習（ベースモデル）
    ↓
chat_sft.py           # 3. SFT（チャットモデル化）
    ↓
chat_rl.py            # 4. RL（GSM8K で強化）
    ↓
chat_web.py           # 5. Web UI で会話
```

---

ファイル名は `FOLDER_GUIDE.md` として、リポジトリのルートに配置してください。
