import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel
import torch
import google.generativeai as genai
import os
import time
import re
from sklearn.cluster import KMeans
import umap
from anchor_based_embedding import AnchorBasedEmbedding, compare_embedding_methods

genai.configure(api_key="AIzaSyCV5BV5i514ouP0fqp_1lCoMyJKaDoboqA")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Ruri v3モデルのパスとエンベディングローダーのセットアップ
# 前回のダウンロード場所に合わせて './ruri_v3_downloaded_from_hub' を指定
RURI_MODEL_PATH = "./ruri_v3_downloaded_from_hub" 
try:
    @st.cache_resource # Streamlitでモデルロードをキャッシュする
    def load_ruri_model():
        tokenizer = AutoTokenizer.from_pretrained(RURI_MODEL_PATH)
        # float16でロードしてメモリを節約
        model = AutoModel.from_pretrained(RURI_MODEL_PATH, torch_dtype=torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    
    ruri_tokenizer, ruri_model, ruri_device = load_ruri_model()
except Exception as e:
    st.error(f"Ruri v3モデルのロードに失敗しました。Ruriモデルが {RURI_MODEL_PATH} に存在するか、必要なライブラリがインストールされているか確認してください。エラー: {e}")
    st.stop()

# 既存のエンベディングとデータのロード
@st.cache_data # Streamlitでデータロードをキャッシュする
def load_existing_data():
    try:
        df = pd.read_csv("description.csv")
        # 列名のリネーム処理
        if 'ブランド名' in df.columns and 'name' not in df.columns:
            df.rename(columns={'ブランド名': 'name'}, inplace=True)
        if 'bline_id' in df.columns and 'id' not in df.columns:
            df.rename(columns={'bline_id': 'id'}, inplace=True)
        elif 'id' not in df.columns:
            df['id'] = range(len(df)) # ダミーID

        # Ruri v3 raw エンベディングファイルを使用
        embeddings_path = "./ruri_embeddings_results/ruri_description_embeddings_v3_raw_hub.npy"
        
        # ファイルが存在しない場合は警告を出して空のエンベディングを返すか、エラーで停止
        if not os.path.exists(embeddings_path):
            st.warning(f"既存のRuri v3エンベディングファイルが見つかりません: {embeddings_path}。新しいブランドの追加は可能ですが、既存ブランドのマップは表示されません。")
            return df, np.array([]) # 既存のエンベディングがない場合は空配列を返す

        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != len(df):
            st.warning(f"既存のエンベディング数 ({embeddings.shape[0]}) がブランドデータ数 ({len(df)}) と一致しません。可視化に影響が出る可能性があります。")
            # 不一致の場合の処理（例：少ない方に合わせるなど）
            min_rows = min(embeddings.shape[0], len(df))
            df = df.iloc[:min_rows].copy()
            embeddings = embeddings[:min_rows]
            
        return df, embeddings
    except Exception as e:
        st.error(f"既存データのロードに失敗しました。description.csvとエンベディングファイルが存在するか確認してください。エラー: {e}")
        st.stop()

df_existing, embeddings_existing = load_existing_data()

# UMAPリデューサーもキャッシュして高速化
@st.cache_resource
def get_umap_reducer(embeddings_data, method="Standard UMAP", n_anchors=75, lambda_anchor=0.1):
    if embeddings_data.size == 0: # 既存エンベディングがない場合はUMAPをfitできない
        return None
    # 選択された可視化手法に応じてリデューサーを作成
    if method == "アンカーベース埋め込み":
        reducer = AnchorBasedEmbedding(
            n_anchors=min(n_anchors, embeddings_data.shape[0]//5),
            lambda_anchor=lambda_anchor,
            n_components=2,
            n_neighbors=min(30, embeddings_data.shape[0] -1),
            min_dist=0.05,
            spread=1.5,
            random_state=42,
            n_epochs=300
        )
    else:  # Standard UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, embeddings_data.shape[0] -1),
            min_dist=0.05,
            spread=1.5,
            metric='cosine',
            random_state=42,
            n_epochs=500
        )
    # For AnchorBasedEmbedding, we don't pre-fit since it only has fit_transform
    if isinstance(reducer, AnchorBasedEmbedding):
        pass  # Will use fit_transform when needed
    else:
        reducer.fit(embeddings_data)
    return reducer


def create_visualization_plot(df, brand_name, method_name, x_col='umap_x', y_col='umap_y'):
    """可視化プロットを作成する関数"""
    # より効果的な可視化のために、強調表示用の色とサイズを個別に設定
    df['marker_size'] = df['is_highlighted'].astype(int) * 6 + 4  # False=4, True=10 (サイズを小さく)
    df['cluster_str'] = df['cluster'].astype(str)
    
    # クラスター数に応じた最適な色パレットを選択
    n_clusters = df['cluster'].nunique()
    
    # より明確に区別できる色パレット（彩度と明度を調整）
    color_palette = [
        '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
        '#1ABC9C', '#E67E22', '#34495E', '#F1C40F', '#E91E63',
        '#8E44AD', '#27AE60', '#D35400', '#2980B9', '#C0392B',
        '#16A085', '#7F8C8D', '#BDC3C7', '#95A5A6', '#F4F6F7'
    ]
    
    # クラスター数が色パレットより多い場合は繰り返し
    if n_clusters > len(color_palette):
        color_palette = color_palette * ((n_clusters // len(color_palette)) + 1)
    
    # クラスターIDのリストを取得してcategory_ordersを明示的に設定
    cluster_ids = sorted(df['cluster_str'].unique())
    
    # 基本のPlotlyで可視化 (離散的な色マッピングを使用)
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color='cluster_str',
        size='marker_size',
        size_max=2,  # 最大サイズを縮小
        hover_data=['name', 'description_preview'],
        hover_name='name',
        title=f'ブランド類似度マップ ({method_name}) - {brand_name} を追加',
        width=1200,  # 幅を拡大して見やすく
        height=800,   # 高さを拡大
        color_discrete_sequence=color_palette[:n_clusters],  # 離散的な色を使用
        category_orders={'cluster_str': cluster_ids}  # クラスターを離散カテゴリとして明示
    )
    
    # 新しく追加されたブランドを特別に強調表示
    highlighted_data = df[df['is_highlighted'] == True]
    if not highlighted_data.empty:
        fig.add_trace(
            go.Scatter(
                x=highlighted_data[x_col],
                y=highlighted_data[y_col],
                mode='markers',
                marker=dict(
                    size=15,  # 新規追加ブランドのサイズも調整
                    color='#FF0000',
                    symbol='diamond',  # より目立つシンボル
                    line=dict(width=2, color='#8B0000')
                ),
                name=f'新規追加: {brand_name}',
                text=highlighted_data['name'],
                hovertemplate='<b>%{text}</b><br>' +
                            'X: %{x:.2f}<br>' +
                            'Y: %{y:.2f}<br>' +
                            '<i>新しく追加されたブランド</i><extra></extra>'
            )
        )
    
    # マーカーのスタイルを改善
    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color='white'),  # マーカーの縁を細く
            opacity=0.9,  # 不透明度を上げてクラスター色をより明確に
            sizemin=3,    # 最小サイズを設定
            sizemode='diameter'  # サイズモードを指定
        )
    )
    
    fig.update_layout(
        title_x=0.5,
        font=dict(size=14),
        hovermode='closest',
        legend=dict(
            title='クラスター',
            itemsizing='constant', 
            tracegroupgap=0,
            orientation='v',
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


# --- Streamlit UI ---
st.title("👗 ファッションブランド類似度探索アプリ")
st.markdown("ブランド名を入力して、Geminiが生成した説明文から類似ブランドを可視化します。")

# 可視化手法の選択
st.sidebar.header("🎨 可視化設定")
visualization_method = st.sidebar.selectbox(
    "可視化手法を選択",
    ["Standard UMAP", "アンカーベース埋め込み", "手法比較モード"],
    index=1
)

# Default values
n_anchors = 75
lambda_anchor = 0.1

if visualization_method == "アンカーベース埋め込み":
    st.sidebar.subheader("🎯 アンカー設定")
    n_anchors = st.sidebar.slider("アンカー数", 20, 200, 75, 5)
    lambda_anchor = st.sidebar.slider("アンカー引力重み (λ)", 0.01, 0.5, 0.1, 0.01)
    
elif visualization_method == "手法比較モード":
    st.sidebar.info("複数の手法を同時に比較表示します")

# Initialize UMAP reducer after UI components are defined
umap_reducer = get_umap_reducer(embeddings_existing, visualization_method, n_anchors, lambda_anchor)

# --- ブランド名入力 ---
brand_name_input = st.text_input("分析したいブランド名を入力してください", "ZARA") # デフォルトをZARAに

# --- Gemini説明文生成とエンベディング、可視化 ---
if st.button("Geminiで説明文を生成 & マップに表示"):
    if not brand_name_input:
        st.warning("ブランド名を入力してください。")
    else:
        generated_description = ""
        gemini_prompt = f"""あなたはファッション業界の専門ライターです。ファッションブランド「{brand_name_input}」の特徴を300文字程度で簡潔に説明してください。特に、ブランドのスタイル、デザイン哲学、ターゲット層、代表的なアイテムに焦点を当ててください。
        
        ### 良質な例1:
        ブランド名: コム デ ギャルソン 参考情報: コム デ ギャルソン(COMME des GARÇONS)は日本のファッションブランド。創業者川久保玲。黒などモノトーンを多様、孤高の女性を描いた。 説明文: コム デ ギャルソンは、川久保玲が手掛ける革新的なファッションブランド。黒を基調としたモノトーンの色使い、左右非対称なカッティング、身体のラインを曖昧にするルーズなシルエットが特徴。代表的なアイテムは独創的なフォルムのジャケットやドレス、穴あきニット。ウールギャバジンなど独自加工の素材を多用し、流行に左右されず個性を表現したいアート志向の層に支持される。
        ### 良質な例2:
        ブランド名: ジル サンダー 参考情報: ジル サンダーはMs.ジル・サンダーが設立。洗練されて繊細、かつ品質にこだわったミニマルなデザインが特徴。 説明文: ジル サンダーは、ドイツ発のミニマリズムを代表するブランド。純粋さと品質を追求し、不要な装飾を削ぎ落としたクリーンで洗練されたデザインが特徴。代表的なアイテムは完璧なカッティングのシャツ、構築的シルエットのコート。カシミア、ウール、シルク等の高品質な天然素材を好み、ニュートラルカラーを基調とした落ち着いたパレット。流行に左右されないタイムレスなエレガンスを求める知的な大人に支持される。
        ### 処理対象:
        ブランド名: {brand_name_input}
        説明文:"""
        
        max_retries = 3 
        for attempt in range(max_retries):
            try:
                with st.spinner(f"Geminiで説明文を生成中... (試行 {attempt + 1}/{max_retries})"):
                    response = gemini_model.generate_content(gemini_prompt)
                    generated_description = response.text.strip()
                st.success("説明文の生成が完了しました！")
                st.subheader(f"{brand_name_input} の生成された説明文:")
                st.write(generated_description)
                break 
            except Exception as e:
                error_message = str(e)
                st.warning(f"'{brand_name_input}' の説明生成中にエラーが発生しました: {error_message}")
                if "quota_id: \"GenerateRequestsPerDayPerProjectPerModel-FreeTier\"" in error_message:
                    st.error("1日の無料リクエストクォータを超過しました。翌日以降に再試行してください。")
                    break 
                elif "retry_delay" in error_message:
                    match = re.search(r'seconds:\s*(\d+)', error_message)
                    delay = int(match.group(1)) if match else 10
                    st.info(f"一時的なエラーです。{delay}秒待機して再試行します (残り{max_retries - 1 - attempt}回)。")
                    time.sleep(delay + 1)
                else:
                    st.error(f"予期せぬエラーが発生しました: {error_message}")
                    break
        else: 
            st.error(f"'{brand_name_input}' の説明生成に失敗しました。")
            generated_description = "" 

        if generated_description:
            with st.spinner(f"{brand_name_input}をRuri v3でエンベディングし、マップを更新中..."):
                new_inputs = ruri_tokenizer(
                    [generated_description],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                new_inputs = {k: v.to(ruri_device) for k, v in new_inputs.items()}
                with torch.no_grad():
                    new_outputs = ruri_model(**new_inputs)
                new_embedding = new_outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # 既存のデータフレームとエンベディングに新しいブランドを追加
                new_brand_id = df_existing['id'].max() + 1 if 'id' in df_existing.columns and not df_existing.empty else 0
                new_row = pd.DataFrame([{
                    'name': brand_name_input,
                    'description': generated_description, 
                    'id': new_brand_id,
                    'cluster': -1, 
                    'is_highlighted': True # ★新しく追加したブランドにフラグ★
                }])
                
                if 'is_highlighted' not in df_existing.columns:
                    df_existing['is_highlighted'] = False
                else:
                    # 以前のハイライトをリセットしたい場合は False に設定
                    # df_existing がコピーされているので、元の df_existing には影響しない
                    # df_combined 作成時に新しい is_highlighted を使う
                    pass 


                df_combined = pd.concat([df_existing, new_row], ignore_index=True)
                
                # ここで df_combined の 'is_highlighted' 列が正しく初期化されていることを確認
                # concat後のdf_combinedに、is_highlightedが存在しない既存の行がある場合、NaNになるのでFalseで埋める
                df_combined['is_highlighted'] = df_combined['is_highlighted'].fillna(False)


                embeddings_combined = np.vstack([embeddings_existing, new_embedding])

                # UMAPで次元削減（エラーハンドリングを強化）
                try:
                    if umap_reducer is None: 
                         st.info("既存データがないため、新しいブランドのみで埋め込みを再学習します。")
                         current_reducer = get_umap_reducer(embeddings_combined, visualization_method, n_anchors, lambda_anchor)
                         if current_reducer is not None:
                             if isinstance(current_reducer, AnchorBasedEmbedding):
                                 coords_2d_combined = current_reducer.fit_transform(embeddings_combined, df_combined['name'].tolist())
                             else:
                                 coords_2d_combined = current_reducer.fit_transform(embeddings_combined)
                         else:
                             st.error("埋め込みデータが不足しているため、可視化できません。")
                             st.stop()
                    else:
                        current_reducer = umap_reducer
                        # 既存のリデューサーがある場合の処理
                        if hasattr(umap_reducer, 'transform'):
                            coords_2d_combined = umap_reducer.transform(embeddings_combined)
                        else:
                            # アンカーベース手法の場合は再学習が必要
                            if isinstance(umap_reducer, AnchorBasedEmbedding):
                                coords_2d_combined = umap_reducer.fit_transform(embeddings_combined, df_combined['name'].tolist())
                            else:
                                coords_2d_combined = umap_reducer.fit_transform(embeddings_combined)
                except Exception as e:
                    st.error(f"次元削減処理でエラーが発生しました: {str(e)}")
                    st.error("設定を調整するか、別の可視化手法を試してください。")
                    st.stop()

                df_combined['umap_x'] = coords_2d_combined[:, 0]
                df_combined['umap_y'] = coords_2d_combined[:, 1]
                
                # UMAPのクラスタリングを再実行して、新しいブランドを含める（大量データ対応）
                # ブランド数に応じて適切なクラスター数を自動調整
                n_brands = len(df_combined)
                if n_brands < 10:
                    n_clusters = min(3, n_brands - 1)
                elif n_brands < 50:
                    n_clusters = min(8, n_brands // 3)
                elif n_brands < 200:
                    n_clusters = min(15, n_brands // 8)
                else:
                    n_clusters = min(25, n_brands // 12)  # 大量データの場合はより多くのクラスター
                
                if n_clusters > 0:
                    try:
                        kmeans = KMeans(
                            n_clusters=n_clusters, 
                            random_state=42, 
                            n_init=10,  # 大量データ時は計算量を抑制
                            max_iter=300,  # 収束時間を短縮
                            algorithm='elkan' if n_brands > 100 else 'lloyd'  # 大量データ時は高速アルゴリズム
                        )
                        df_combined['cluster'] = kmeans.fit_predict(embeddings_combined)
                        st.success(f"クラスタリング完了: {n_clusters}個のクラスターに分類しました")
                    except Exception as e:
                        st.warning(f"クラスタリングでエラーが発生しました: {str(e)}. 単一クラスターとして処理します。")
                        df_combined['cluster'] = 0
                else:
                    df_combined['cluster'] = 0

                # 説明文のプレビュー列を再生成（より効率的に）
                try:
                    df_combined['description_preview'] = df_combined['description'].apply(
                        lambda x: (x[:50] + '...') if (isinstance(x, str) and len(x) > 50) else str(x)
                    )
                except Exception as e:
                    st.warning(f"説明文プレビュー生成でエラー: {str(e)}")
                    df_combined['description_preview'] = df_combined['name']  # フォールバック

                # 手法比較モードの場合は複数の可視化を表示
                if visualization_method == "手法比較モード":
                    st.subheader("🔬 複数手法の比較")
                    
                    # 複数手法で埋め込みを実行
                    comparison_results = compare_embedding_methods(embeddings_combined, df_combined['name'].tolist())
                    
                    # タブで各手法を表示
                    tabs = st.tabs(list(comparison_results.keys()))
                    
                    for tab, (method_name, result) in zip(tabs, comparison_results.items()):
                        with tab:
                            embedding = result['embedding']
                            df_temp = df_combined.copy()
                            df_temp['x'] = embedding[:, 0]
                            df_temp['y'] = embedding[:, 1]
                            
                            # 個別の可視化
                            fig_comp = create_visualization_plot(
                                df_temp, brand_name_input, method_name, 
                                x_col='x', y_col='y'
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                            # メトリクス表示
                            if 'metrics' in result:
                                metrics = result['metrics']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("PCA分散寄与率", f"{metrics.get('pca_variance_ratio', 0):.3f}")
                                with col2:
                                    st.metric("平均アンカー間距離", f"{metrics.get('anchor_distance_mean', 0):.3f}")
                                with col3:
                                    st.metric("クラスター内分散", f"{metrics.get('intra_cluster_variance_mean', 0):.3f}")
                else:
                    # 単一手法での可視化
                    fig = create_visualization_plot(
                        df_combined, brand_name_input, visualization_method
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # アンカーベース手法の場合は追加情報を表示
                    if visualization_method == "アンカーベース埋め込み" and hasattr(current_reducer, 'get_anchor_info'):
                        anchor_info = current_reducer.get_anchor_info()
                        if anchor_info:
                            with st.expander("📍 アンカー情報", expanded=False):
                                anchor_df = pd.DataFrame({
                                    'ブランド名': anchor_info['names'],
                                    'X座標': anchor_info['coords'][:, 0],
                                    'Y座標': anchor_info['coords'][:, 1]
                                })
                                st.dataframe(anchor_df, use_container_width=True)
                                
                                st.markdown("**アンカーとは**: 埋め込み空間での基準点として選ばれたブランドです。")
                    
                    # 分析メトリクスの表示
                    if hasattr(current_reducer, 'get_analysis_metrics'):
                        metrics = current_reducer.get_analysis_metrics(embeddings_combined, coords_2d_combined)
                        if metrics:
                            st.subheader("📊 分析メトリクス")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("PCA分散寄与率", f"{metrics.get('pca_variance_ratio', 0):.3f}")
                            with col2:
                                st.metric("平均アンカー間距離", f"{metrics.get('anchor_distance_mean', 0):.3f}")
                            with col3:
                                st.metric("アンカー間距離標準偏差", f"{metrics.get('anchor_distance_std', 0):.3f}")
                            with col4:
                                st.metric("クラスター内分散", f"{metrics.get('intra_cluster_variance_mean', 0):.3f}")
                
                st.success(f"{brand_name_input}が{visualization_method}マップに追加されました。")
        else:
            st.error("説明文が生成されなかったため、マップに表示できません。")



# # --- アプリの実行方法 ---
# st.markdown("---")
# st.subheader("アプリの実行方法")
# st.code("""
# 1. 必要なライブラリをインストールします:
#    pip install streamlit google-generativeai transformers torch umap-learn scikit-learn pandas tqdm

# 2. `GOOGLE_API_KEY` 環境変数を設定します:
#    export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# 3. Ruri v3モデルをダウンロードします (もし未実施なら):
#    git clone https://huggingface.co/cl-nagoya/ruri-v3-310m ./ruri_v3_downloaded_from_hub

# 4. Ruri v3エンベディングを生成します (初回のみ):
#    python embedding.py

# 5. この `app.py` ファイルがあるディレクトリでStreamlitを実行します:
#    streamlit run app.py
# """)