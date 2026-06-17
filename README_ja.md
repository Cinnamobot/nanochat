# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat は LLM（大規模言語モデル）を訓練するための、最もシンプルな実験用ハーネスです。単一の GPU ノード上で動作するよう設計されており、コードは最小限でハック可能です。トークナイズ、事前学習、ファインチューニング、評価、推論、チャット UI など、LLM の主要なすべてのステージをカバーしています。たとえば、2019年に約 $43,000 かかった GPT-2 相当の LLM を、わずか $48（8×H100 GPU ノードで約2時間）で自分でトレーニングし、おなじみの ChatGPT 風 Web UI で会話することができます。スポットインスタンスを使えば、総コストは約 $15 に近づきます。より一般的には、nanochat は `--depth`（GPT トランスフォーマーモデルのレイヤー数）という単一の複雑さダイヤルを設定するだけで、計算効率の良いモデルのミニシリーズ全体をトレーニングできるよう、すぐに使える状態で設定されています（GPT-2 相当の能力は現在のコードでは depth 26 前後です）。その他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率調整、トレーニング期間、重み減衰など）は自動的に最適な形で決定されます。

質問は [DeepWiki](https://deepwiki.com/karpathy/nanochat)、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)、または Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルをご利用ください。

## GPT-2 到達時間リーダーボード

現在、開発の主な焦点は最も多くの計算量を必要とする事前学習ステージのチューニングです。modded-nanogpt リポジトリに触発され、進捗とコミュニティの協力を促進するために、nanochat は「GPT-2 スピードラン」のリーダーボードを維持しています。これは DCLM CORE スコアで測定した GPT-2 相当の能力に到達するまでの実時間です。[runs/speedrun.sh](runs/speedrun.sh) スクリプトは常に GPT-2 相当モデルのトレーニングと会話の参照方法を反映しています。現在のリーダーボードは以下の通りです：

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

主な指標は「GPT-2 到達時間」— 8×H100 GPU ノードで GPT-2（1.6B）の CORE メトリクスを上回るために必要な実時間です。GPT-2 の CORE スコアは 0.256525 です。2019年には GPT-2 のトレーニングに約 $43,000 かかりましたが、7年間にわたるスタック全体の多くの進歩により、現在ははるかに速く、$100 以下（例：現在の約 $3/GPU/時間で、8×H100 ノードは約 $24/時間、2時間で約 $48）で実現できます。

リーダーボードの解釈と貢献方法については [dev/LEADERBOARD.md](dev/LEADERBOARD.md) を参照してください。

## はじめに

### セットアップ

nanochat は依存関係管理に [uv](https://docs.astral.sh/uv/) を使用しています。インストール方法：

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

開発用（pytest、matplotlib、ipykernel、transformers などを追加）：

```bash
uv sync --extra gpu --group dev
```

### GPT-2 を再現して会話する

最も楽しい体験は、自分の GPT-2 をトレーニングして会話することです。そのためのパイプライン全体が [runs/speedrun.sh](runs/speedrun.sh) という単一ファイルに含まれており、8×H100 GPU ノードで実行するよう設計されています。お好みのプロバイダー（例：[Lambda](https://lambda.ai/service/gpu-cloud)）から新しい 8×H100 GPU ボックスを起動し、トレーニングスクリプトを開始してください：

```bash
bash runs/speedrun.sh
```

これには約3時間かかるため、screen セッションで実行することをお勧めします。完了したら、ChatGPT 風 Web UI で会話できます。ローカルの uv 仮想環境がアクティブであることを確認し（`source .venv/bin/activate` を実行）、サーバーを起動してください：

```bash
python -m scripts.chat_web
```

表示された URL にアクセスしてください。例えば Lambda では、ノードのパブリック IP にポートを続けて、[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/) のようにアクセスします。通常 ChatGPT に話しかけるように LLM と会話してください！物語や詩を書かせたり、自分が誰かを聞いてハルシネーションを見たり、空が青い理由を聞いたり、なぜ緑なのかを聞いたりしてみてください。スピードランは 4e19 FLOPs の能力モデルなので、幼稚園児と話しているような感じです :)

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

追加メモ：

- Ampere 8×A100 GPU ノードでも問題なく動作しますが、少し遅くなります。
- `torchrun` を省略することで単一 GPU でも問題なく動作し、ほぼ同一の結果が得られます（コードは自動的に勾配累積に切り替わります）が、8倍長く待つ必要があります。
- GPU の VRAM が 80GB 未満の場合、一部のハイパーパラメータを調整しないと OOM（メモリ不足）になります。スクリプト内の `--device-batch-size` を探して、32（デフォルト）から 16、8、4、2、または 1 に減らしてください。それ以下にする場合は、もう少し仕組みを理解して工夫する必要があります。
- ほとんどのコードはかなり標準的な PyTorch なので、xpu、mps などでも動作するはずですが、すべてのコードパスを個人的に確認したわけではないため、問題が生じる可能性があります。

## 研究

研究者で nanochat の改善に協力したい場合、[runs/scaling_laws.sh](runs/scaling_laws.sh) と [runs/miniseries.sh](runs/miniseries.sh) の2つのスクリプトが参考になります。関連ドキュメントは [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) を参照してください。クイック実験（約5分の事前学習実行）には、12層モデル（GPT-1 サイズ）のトレーニングが好みです。例えば次のようにします：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

これは wandb（実行名 "d12"）を使用し、最終ステップのみ CORE メトリクスを実行し、中間チェックポイントのサンプリングと保存は行いません。コードを変更して d12（または d16 など）を再実行し、改善されたかどうかを確認するというイテレーションループが好みです。実行が改善されたかどうかを確認するために、wandb プロットで以下を監視します：

1. `val_bpb`（語彙サイズに依存しない単位のバリデーション損失、バイト毎ビット）を `step`、`total_training_time`、`total_training_flops` の関数として
2. `core_metric`（DCLM CORE スコア）
3. VRAM 使用率、`train/mfu`（モデル FLOPS 使用率）、`train/tok_per_sec`（トレーニングスループット）

例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照してください。

重要な点として、nanochat はトランスフォーマーの深さという単一の複雑さダイヤルを中心に書かれ設定されています。この単一の整数が他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率調整、トレーニング期間、重み減衰など）を自動的に決定し、トレーニングされたモデルが計算効率的になるようにします。ユーザーはこれらについて考えたり設定したりする必要はなく、`--depth` を使って小さいモデルか大きいモデルかを指定するだけで、すべてが「うまく機能」します。深さを変えることで、さまざまなサイズの計算効率的なモデルの nanochat ミニシリーズが得られます。GPT-2 相当の能力モデル（現時点で最も興味深いもの）は、現在のコードでは d24〜d26 の範囲のどこかにあります。ただし、リポジトリへのあらゆる変更候補は、すべての深さの設定で機能するよう十分に原理的でなければなりません。

## CPU / MPS での実行

[runs/runcpu.sh](runs/runcpu.sh) スクリプトは、CPU または Apple Silicon での実行の非常にシンプルな例を示しています。数十分のトレーニングという合理的な時間に収めるために、トレーニングする LLM を大幅に縮小しています。この方法では強力な結果は得られません。

## 精度 / dtype

nanochat は `torch.amp.autocast` を使用しません。代わりに、精度は `nanochat/common.py` で定義された単一のグローバル `COMPUTE_DTYPE` を通じて明示的に管理されます。デフォルトではハードウェアに基づいて自動検出されます：

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

デフォルトは `NANOCHAT_DTYPE` 環境変数で上書きできます：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

仕組み：モデルの重みは fp32 で保存されます（オプティマイザの精度のため）が、カスタム `Linear` レイヤーがフォワードパス中にそれらを `COMPUTE_DTYPE` にキャストします。埋め込みはメモリ節約のために直接 `COMPUTE_DTYPE` で保存されます。これにより、autocast と同じ混合精度の利点が得られますが、どの精度で何が実行されるかを完全に明示的に制御できます。

注意：`float16` トレーニングは勾配アンダーフローを防ぐために `base_train.py` で自動的に `GradScaler` を有効にします。SFT もこれをサポートしていますが、RL は現在サポートしていません。fp16 での推論はどこでも問題なく動作します。

## ガイド

役立つ情報を含む可能性のあるガイドを最新順に公開しています：

- [2026年2月1日: $100 以下で GPT-2 を超える：nanochat の旅](https://github.com/karpathy/nanochat/discussions/481)
- [1月7日 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) は最初の nanochat ミニシリーズモデルを文書化しています。
- nanochat に新しい能力を追加するには、[ガイド：strawberry の r を数える（および一般的な能力の追加方法）](https://github.com/karpathy/nanochat/discussions/164)を参照してください。
- nanochat をカスタマイズするには、Discussions の[ガイド：nanochat にアイデンティティを注入する](https://github.com/karpathy/nanochat/discussions/139)を参照してください。合成データ生成と SFT ステージへのデータ混合を通じて nanochat のパーソナリティを調整する方法が説明されています。
- [2025年10月13日：nanochat オリジナル投稿](https://github.com/karpathy/nanochat/discussions/1) は nanochat を紹介していますが、現在は一部の情報が古くなっており、モデルも現在の master より古い（結果も悪い）です。

## ファイル構成

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # アイデンティティ用の合成データ例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 事前学習データシャード生成
├── nanochat
│   ├── __init__.py                 # 空
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存/読み込み
│   ├── common.py                   # 各種小ユーティリティ
│   ├── core_eval.py                # ベースモデルの CORE スコア評価（DCLM 論文）
│   ├── dataloader.py               # トークナイズ分散データローダー
│   ├── dataset.py                  # 事前学習データのダウンロード/読み込みユーティリティ
│   ├── engine.py                   # KV キャッシュを使った効率的なモデル推論
│   ├── execution.py                # LLM がツールとして Python コードを実行できるようにする
│   ├── gpt.py                      # GPT nn.Module トランスフォーマー
│   ├── logo.svg
│   ├── loss_eval.py                # バイト毎ビット（損失の代わり）の評価
│   ├── optim.py                    # AdamW + Muon オプティマイザ（単一 GPU および分散）
│   ├── report.py                   # nanochat レポート作成ユーティリティ
│   ├── tokenizer.py                # GPT-4 スタイルの BPE トークナイザーラッパー
│   └── ui.html                     # nanochat フロントエンドの HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # ミニシリーズトレーニングスクリプト
│   ├── runcpu.sh                   # CPU/MPS での実行例
│   ├── scaling_laws.sh             # スケーリング則実験
│   └── speedrun.sh                 # 約 $100 の nanochat d20 をトレーニング
├── scripts
│   ├── base_eval.py                # ベースモデル：CORE スコア、バイト毎ビット、サンプル
│   ├── base_train.py               # ベースモデル：トレーニング
│   ├── chat_cli.py                 # チャットモデル：CLI で会話
│   ├── chat_eval.py                # チャットモデル：タスク評価
│   ├── chat_rl.py                  # チャットモデル：強化学習
│   ├── chat_sft.py                 # チャットモデル：SFT トレーニング
│   ├── chat_web.py                 # チャットモデル：WebUI で会話
│   ├── tok_eval.py                 # トークナイザー：圧縮率評価
│   └── tok_train.py                # トークナイザー：トレーニング
├── tasks
│   ├── arc.py                      # 多肢選択式科学問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 任意の jsonl 会話から Task を作成
│   ├── gsm8k.py                    # 8K 小学校数学問題
│   ├── humaneval.py                # 誤称；シンプルな Python コーディングタスク
│   ├── mmlu.py                     # 多肢選択式問題、幅広いトピック
│   ├── smoltalk.py                 # HF の SmolTalk 複合データセット
│   └── spellingbee.py              # モデルにスペル/文字数を教えるタスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## コントリビューション

nanochat の目標は、$1,000 未満の予算でエンドツーエンドで扱えるマイクロモデルの最先端を改善することです。アクセシビリティは総コストだけでなく、認知的複雑さについても言えます。nanochat は網羅的に設定可能な LLM「フレームワーク」ではありません。巨大な設定オブジェクト、モデルファクトリー、if-then-else の怪物はコードベースにありません。最初から最後まで実行して会話できる ChatGPT モデルを生成するよう設計された、単一の、一貫した、最小限の、読みやすい、ハック可能な、最大限フォーク可能な「強力なベースライン」コードベースです。現在、個人的に最も興味深いのは GPT-2 への到達レイテンシを短縮すること（つまり CORE スコアを 0.256525 以上にすること）です。現在これには約3時間かかりますが、事前学習ステージを改善することでさらに向上できます。

現在の AI ポリシー：開示。PR を提出する際は、LLM が実質的に貢献した部分、自分で書いていない部分、または完全に理解していない部分を申告してください。

## 謝辞

- 名前（nanochat）は、事前学習のみをカバーしていた以前のプロジェクト [nanoGPT](https://github.com/karpathy/nanoGPT) に由来しています。
- nanochat は [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) にも触発されており、明確なメトリクスとリーダーボードで nanoGPT リポジトリをゲーム化し、事前学習のアイデアと実装の多くを借用しています。
- [HuggingFace](https://huggingface.co/) の fineweb と smoltalk に感謝します。
- このプロジェクトの開発に使用した計算リソースを提供してくれた [Lambda](https://lambda.ai/service/gpu-cloud) に感謝します。
- アドバイスと指導をくれた LLM ウィスパラー 🧙‍♂️ Alec Radford に感謝します。
- issue、プルリクエスト、ディスカッションの管理を手伝ってくれたリポジトリ管理者 Sofie [@svlandeg](https://github.com/svlandeg) に感謝します。

## 引用

研究で nanochat が役立った場合は、以下のように引用してください：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MIT
