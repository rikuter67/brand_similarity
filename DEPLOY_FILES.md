# Streamlit Cloud デプロイ用 - 最低限必要ファイル

## 📁 必須ファイル（7ファイル）

### 1. メインアプリケーション
- `streamlit_brand_app_improved.py` - メインStreamlitアプリ

### 2. 依存関係ファイル
- `brand_similarity.py` - ブランド類似度分析
- `brand_store_matching.py` - ブランド・店舗マッチング 
- `location_bias_reranking.py` - 位置情報リランキング
- `integrated_dimensionality_reduction.py` - 次元削減処理

### 3. 設定ファイル
- `requirements.txt` - Python依存関係
- `.streamlit/config.toml` - Streamlit設定

### 4. ドキュメント（オプション）
- `STREAMLIT_DEPLOY.md` - デプロイ手順書

---

## 🚫 除外されるファイル (.gitignoreで設定済み)

### 大容量データ
- `*.npy`, `*.npz` - NumPy配列
- `ruri_v3_downloaded_from_hub/` - モデルファイル
- `*_results/` - 結果フォルダ
- `*.csv` - データファイル（requirements.txt除く）

### テスト・デバッグファイル
- `test_*.py`, `debug_*.py`
- `gpt_oss_*.py` - GPT-OSS関連
- `app.py`, `gemini.py` など

### Docker関連
- `Dockerfile`, `docker-compose.yml`
- `*.sh` - シェルスクリプト

---

## 📋 デプロイ前チェック

```bash
# 実際にpushされるファイルを確認
git status
git add .
git status

# 不要ファイルが含まれていないか確認
git ls-files | grep -E "\.(npy|csv|png)$" || echo "✅ 大容量ファイルなし"
```

## 🎯 推定リポジトリサイズ

- **合計**: 約200KB未満
- **Python ファイル**: ~150KB
- **設定ファイル**: ~10KB
- **ドキュメント**: ~30KB

これにより高速なクローン・デプロイが可能です！