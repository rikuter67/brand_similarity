# Streamlit Community Cloud デプロイガイド 🚀

## 📋 デプロイ手順

### 1. GitHubリポジトリの準備
```bash
# リポジトリをGitHubにプッシュ
git add .
git commit -m "Streamlit Cloud deployment ready"
git push origin main
```

### 2. Streamlit Community Cloudでデプロイ
1. **https://share.streamlit.io** にアクセス
2. **GitHubアカウントでログイン**
3. **"New app"** をクリック
4. リポジトリ設定:
   - **Repository**: あなたのリポジトリを選択
   - **Branch**: `main`
   - **Main file path**: `streamlit_brand_app_improved.py`
   - **App URL**: 任意のURL名を設定

### 3. Secrets設定（APIキー）
1. デプロイ後、**アプリダッシュボード** → **"⚙️ Settings"** 
2. **"Secrets"** タブをクリック
3. 以下の形式でAPIキーを設定:

```toml
# Gemini API Key（必須）
GEMINI_API_KEY = "AIzaSyXXXXXXXXXXXXXX"

# Google Custom Search（オプション）
GOOGLE_API_KEY = "AIzaSyYYYYYYYYYYYYY"
GOOGLE_CSE_ID = "your_search_engine_id"
```

4. **"Save"** をクリック

### 4. アプリの再起動
- Settings → "Reboot app" でアプリを再起動

## 🔑 APIキー取得方法

### Gemini API Key
1. **https://aistudio.google.com** にアクセス
2. **"Get API key"** をクリック
3. **"Create API key"** で新しいキーを作成
4. キーをコピーしてStreamlit Secretsに設定

## 📱 アプリ機能

- ✅ ブランド名入力
- ✅ Gemini APIで説明文自動生成
- ✅ 類似ブランド分析
- ✅ インタラクティブ可視化
- ✅ 位置情報リランキング

## ⚠️ 注意事項

### Streamlit Cloud制限
- **GPT-OSS**: サポートされていません（ローカル実行のみ）
- **メモリ**: 1GB制限
- **CPU**: 共有リソース
- **ストレージ**: 一時的（再起動で消去）

### 推奨設定
- **Gemini API使用**: クラウドデプロイでは必須
- **小さなデータセット**: パフォーマンスのため
- **キャッシュ活用**: `@st.cache_data`で高速化

## 🔧 トラブルシューティング

### デプロイエラー
```bash
# requirements.txtの依存関係確認
pip install -r requirements.txt
```

### APIキーエラー
- Streamlit Cloud Secretsでキー設定を確認
- Gemini API有効化を確認

### メモリエラー
- データサイズを削減
- 不要な処理をコメントアウト

## 📊 パフォーマンス最適化

```python
# キャッシュ使用例
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

# セッション状態活用
if 'data' not in st.session_state:
    st.session_state.data = load_data()
```

## 🔄 更新・管理

### アプリ更新
```bash
git add .
git commit -m "Update app"
git push origin main
# → Streamlit Cloudが自動でデプロイ
```

### ログ確認
- アプリダッシュボード → "View logs"

### アプリ削除
- Settings → "Delete app"

---

## 🎯 デプロイ完了！

デプロイが完了すると、以下のようなURLでアクセスできます：
**https://your-app-name.streamlit.app**