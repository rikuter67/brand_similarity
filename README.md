# ファッションブランドライン類似度分析のためのスタイル情報拡張プロジェクト

## 1. プロジェクト概要

本プロジェクトは、ファッションブランドの各ラインに関する説明文（description）に対し、大規模言語モデル（LLM）を用いてファッションスタイルに関する情報を補完・生成することを目的としています。

**目的:**
- 既存の説明文（歴史や沿革中心）をクリーニング
- LLMを活用してスタイル情報を付加
- ファッション的観点からブランドライン間の類似度分析
- 質の高いテキストデータの生成

**最終ゴール:**
- 生成された説明文をベクトル化
- ブランドライン間の類似度計算、クラスタリング、可視化
- 「共通施設数」に基づく物理的な近さの分析を補完
- より多角的で深い洞察の獲得

## 2. 使用データ

- **blines.csv**: 主要入力データ（ブランドラインID、名称、既存の説明文）
- **brands.csv**: ブランド情報（ID、名称、カテゴリ）
- データは `/datasets/bline_similarity/` ディレクトリに配置

## 3. 主要な処理フロー

### データ読み込みと前処理
- blines.csv を読み込み
- HTMLタグ除去、不要な改行や空白の整理
- `description_cleaned` カラムの作成

### LLMによるスタイル情報特化型ディスクリプション生成
- ローカルLLM（例: ELYZA-japanese-Llama-2-7b）を使用
- Few-shotの例とプロンプトテンプレートを活用

### 既存説明文の長さに応じた条件分岐処理
1. **長文の場合** (`desc_len_chars > presummary_threshold`):
   - LLMによる事前要約（400字程度）
   - 要約結果を基にスタイル情報の要約・再構成
   
2. **中程度の文の場合** (`desc_len_chars > generate_complement_threshold`):
   - スタイル情報の補完・生成

3. **短文またはほぼ情報がない場合**:
   - ブランドライン名を手がかりに完全新規生成

### 結果の保存
- 処理後のデータフレーム（`description_styled` カラム含む）をCSV保存
- 処理ログも別途CSV保存

## 4. 必要なライブラリ・環境

- Python 3.x
- pandas
- transformers
- torch
- tqdm
- BeautifulSoup4
- accelerate
- LLMモデルファイル: ELYZA-japanese-Llama-2-7b など

## 5. ファイル構成（推奨）
```bash
.
├── generate_styled_descriptions.py  # メイン処理スクリプト
├── few_shot.py                      # Few-shotの例とプロンプトテンプレート定義
├── datasets/
│   └── bline_similarity/
│       ├── blines.csv               # 入力するブランドラインデータ
│       ├── brands.csv               # (任意) ブランド情報データ
│       └── (ここに生成されたCSVファイルが出力)
├── model/                           # LLMモデルファイル
│   └── (ELYZA-japanese-Llama-2-7b など)
└── README.md
```

## 6. 実行方法

```bash
python generate_styled_descriptions.py [オプション]
```

主要なオプション

- --mode TEXT: 実行モード (validation または production)
- --num_samples INTEGER: 検証モードで処理するサンプル数（デフォルト: 3）
- --max_tokens INTEGER: 生成時の最大トークン数（デフォルト: 500）
- --max_summary_tokens INTEGER: 事前要約時の最大トークン数（デフォルト: 150）
- --temperature FLOAT: 生成時のtemperature（デフォルト: 0.75）
- --presummary_threshold INTEGER: 事前要約の対象となる文字数（デフォルト: 800）
- --summarize_restructure_threshold INTEGER: 要約・再構成プロンプト使用の閾値（デフォルト: 200）
- --generate_complement_threshold INTEGER: 補完・生成プロンプト使用の閾値（デフォルト: 30）

実行例
検証モードで5サンプル処理:
```bash
python generate_styled_descriptions.py --mode validation --num_samples 5 --max_tokens 600
```

本番モードで全件処理:
```
bash
python generate_styled_descriptions.py --mode production
```