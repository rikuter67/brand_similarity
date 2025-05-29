# ファッションブランドライン類似度分析のためのスタイル情報拡張プロジェクト

## 1. プロジェクト概要

本プロジェクトは、ファッションブランドの各ラインに関する説明文（description）に対し、大規模言語モデル（LLM）を用いてファッションスタイルに関する情報を補完・生成することを目的としています。

**目的:**
- 既存の説明文（歴史や沿革中心）をクリーニングし、スタイル情報を付加
- LLMを活用してファッション的特徴を豊富に含む新しい説明文を生成
- ファッション的観点からブランドライン間の類似度分析の基盤構築
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

### 3.1 データ読み込みと前処理
- blines.csv を読み込み
- HTMLタグ除去、不要な改行や空白の整理
- `description_cleaned` カラムの作成

### 3.2 LLMによるスタイル情報特化型ディスクリプション生成
- ローカルLLM（例: ELYZA-japanese-Llama-2-7b）を使用
- シンプルで効率的なプロンプト設計
- Few-shot例を最小限に抑制（1-2個）

### 3.3 既存説明文の長さに応じた条件分岐処理
1. **長文の場合** (`desc_len_chars > 250`):
   - LLMによる事前要約（80字程度）
   - 要約結果を基にスタイル情報の補完・生成
   
2. **中程度の文の場合** (`30 < desc_len_chars <= 150`):
   - 既存情報を活用したスタイル情報の補完・生成

3. **短文またはほぼ情報がない場合** (`desc_len_chars <= 30`):
   - ブランドライン名を手がかりに完全新規生成

### 3.4 高品質なテキスト抽出
- Few-shot例の混入を自動検出・除去
- 不完全な文章の自動切り詰め
- ブランド名の混同を防ぐ仕組み

### 3.5 結果の保存
- 成功したもののみ `description` カラムを更新
- 指定されたカラム構成でCSV保存
- 処理成功率と詳細ログの出力

## 4. 必要なライブラリ・環境

```bash
pip install pandas transformers torch tqdm beautifulsoup4 accelerate
```

**必要なファイル:**
- Python 3.8+
- CUDA対応GPU（推奨）
- LLMモデルファイル: ELYZA-japanese-Llama-2-7b など

## 5. ファイル構成

```
project/
├── run.py                           # メイン処理スクリプト
├── few_shot.py                      # プロンプトテンプレートとFew-shot例
├── datasets/
│   └── bline_similarity/
│       ├── blines.csv               # 入力データ
│       └── blines_updated_desc_*.csv # 出力データ
├── model/                           # LLMモデルファイル
│   └── (ELYZA-japanese-Llama-2-7b/)
└── README.md
```

## 6. 実行方法

### 6.1 基本的な実行コマンド

```bash
# 検証モード（推奨：初回実行時）
python run.py --mode validation --num_samples 5 --max_tokens 150 --temperature 0.4 --top_p 0.7 --quantize none

# 本番モード（全件処理）
python run.py --mode production --max_tokens 180 --temperature 0.4 --top_p 0.7 --quantize none
```

### 6.2 主要なオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--mode` | validation | 実行モード (validation/production) |
| `--num_samples` | 5 | 検証モードで処理するサンプル数 |
| `--max_tokens` | 180 | 生成時の最大トークン数 |
| `--max_summary_tokens` | 60 | 事前要約時の最大トークン数 |
| `--temperature` | 0.4 | 生成時のtemperature（0.1-1.0） |
| `--top_p` | 0.7 | 生成時のtop_p（0.1-1.0） |
| `--quantize` | none | モデル量子化（none/8bit/4bit） |

### 6.3 実行例

```bash
# 少量テスト（3件）
python run.py --mode validation --num_samples 3 --max_tokens 120

# 中規模テスト（20件）
python run.py --mode validation --num_samples 20 --max_tokens 150

# 本番実行（全件）
python run.py --mode production --max_tokens 180 --temperature 0.4
```

## 7. 出力データ形式

成功した処理のみが以下の形式でCSVファイルとして保存されます：

```csv
id,name,name_second,name_en,description,map_genre,gourmet,brand_id,enabled,created_at,updated_at
2,マックイーン,アレキサンダー・マックイーン,McQueen,革新的でドラマティックなデザインが特徴。伝統的なテーラリングと前衛的な要素を融合させ、スカルモチーフや極端なシルエットで知られる。高級素材を使用し、強い自己表現を求める層に支持される。,,False,2,True,,2025-03-14 14:38:45 +0900
```

## 8. 品質管理と最適化

### 8.1 自動品質チェック
- Few-shot例の混入検出・除去
- 不完全な文章の自動修正
- ブランド名の一貫性チェック

### 8.2 パフォーマンス最適化  
- プロンプト長の厳格な制限（600トークン以下）
- メモリ効率的な処理
- エラー発生時の継続処理

### 8.3 エラーハンドリング
- 各ブランドラインの個別エラー処理
- 詳細なデバッグ情報の出力
- 処理成功率の追跡

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

**メモリ不足エラー:**
```bash
# より軽量な設定で実行
python run.py --max_tokens 100 --quantize 8bit
```

**生成品質が低い場合:**
```bash
# より保守的なパラメータで実行
python run.py --temperature 0.3 --top_p 0.6 --max_tokens 120
```

**処理速度が遅い場合:**
```bash
# 量子化を有効にして実行
python run.py --quantize 8bit --max_tokens 150
```

### 9.2 デバッグ情報の確認
実行中に表示される詳細なデバッグ情報で問題箇所を特定できます：
- プロンプト文字数とトークン数
- 生成パラメータの実際の値  
- 各処理ステップの成功/失敗状況

## 10. 次のステップ

生成された高品質な説明文を用いて：
1. **テキストエンベディング** (Sentence-BERT等)
2. **類似度計算とクラスタリング**
3. **可視化分析** (t-SNE, UMAP等)
4. **既存の共通施設数分析との比較・統合**

これにより、ファッションブランドライン間の真のスタイル的類似性を定量的に分析できるようになります。