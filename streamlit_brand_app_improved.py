import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from brand_similarity import LLMStyleEmbeddingAnalyzer
from brand_store_matching import BrandStoreAnalyzer
from location_bias_reranking import create_reranker_from_streamlit_data
import os
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

# Streamlit page configuration
st.set_page_config(
    page_title="Brand Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_brand_data():
    """ブランドデータの読み込み"""
    try:
        if os.path.exists('integrated_brands.csv'):
            csv_file = 'integrated_brands.csv'
        else:
            csv_file = 'description.csv'
        
        analyzer = LLMStyleEmbeddingAnalyzer()
        analyzer.load_data(csv_file)
        
        # 埋め込みファイルの読み込み
        embedding_files = [
            "./ruri_embeddings_results/ruri_description_embeddings_v3_raw_hub.npy",
            "./ruri_embeddings_results/ruri_description_embeddings.npy",
            "./ruri_embeddings_results/updated_embeddings.npy"
        ]
        
        for emb_path in embedding_files:
            if os.path.exists(emb_path):
                try:
                    raw_embeddings = np.load(emb_path)
                    from sklearn.preprocessing import normalize
                    analyzer.embeddings = normalize(raw_embeddings, norm='l2', axis=1)
                    st.success(f"✅ 埋め込み読み込み: {os.path.basename(emb_path)}")
                    break
                except Exception as e:
                    st.warning(f"⚠️ 埋め込み読み込みエラー: {e}")
                    continue
        
        if analyzer.embeddings is None:
            analyzer.generate_advanced_embeddings()
        
        analyzer.calculate_similarity_matrix()
        return analyzer
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None

@st.cache_data
def load_store_analyzer():
    """店舗分析器の読み込み"""
    try:
        return BrandStoreAnalyzer("datasets/bline_similarity/maps.csv")
    except Exception as e:
        st.error(f"店舗データ読み込みエラー: {e}")
        return None

def generate_brand_description_gemini(api_key, brand_name):
    """Gemini APIを使ったブランド説明文生成"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        prompt = f"""
        Please write a comprehensive brand description for "{brand_name}" in Japanese.
        Include information about:
        - Brand history and origin
        - Design philosophy and aesthetic
        - Target demographic
        - Key products or specialties
        - Price range and positioning
        
        Keep it between 200-400 characters.
        """
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024
            }
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text'].strip()
        else:
            st.error(f"API エラー: {response.status_code}")
        
        return None
        
    except Exception as e:
        st.error(f"説明文生成エラー: {e}")
        return None

def generate_brand_description_gpt_oss(brand_name):
    """GPT-OSSを使ったブランド説明文生成"""
    try:
        from gpt_oss_direct import GPTOSSDirect
        
        gpt = GPTOSSDirect()
        if not gpt.is_available():
            st.error("GPT-OSSモデルが利用できません")
            return None
        
        description = gpt.generate_brand_description(brand_name)
        if description and len(description.strip()) > 10:
            return description.strip()
        else:
            st.warning("GPT-OSSから有効な説明文を取得できませんでした")
            return None
        
    except Exception as e:
        st.error(f"GPT-OSS説明文生成エラー: {e}")
        return None

def generate_brand_description(api_key, brand_name, use_gpt_oss=False):
    """ブランド説明文生成 - GPT-OSSまたはGemini"""
    if use_gpt_oss:
        return generate_brand_description_gpt_oss(brand_name)
    else:
        return generate_brand_description_gemini(api_key, brand_name)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Brand Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("データ読み込み中..."):
        analyzer = load_brand_data()
        store_analyzer = load_store_analyzer()
    
    if analyzer is None:
        st.error("ブランドデータの読み込みに失敗しました。")
        return
    
    # Sidebar menu
    st.sidebar.header("📊 Analysis Menu")
    
    page = st.sidebar.selectbox(
        "機能を選択:",
        [
            "🏠 ブランド類似度検索", 
            "🏪 店舗一致率分析", 
            "➕ 新ブランド追加",
            "📊 データ概観"
        ]
    )
    
    # Main content
    if page == "🏠 ブランド類似度検索":
        show_brand_similarity_search(analyzer)
    elif page == "🏪 店舗一致率分析":
        show_store_overlap_analysis(store_analyzer)
    elif page == "➕ 新ブランド追加":
        show_new_brand_addition(analyzer)
    elif page == "📊 データ概観":
        show_data_overview(analyzer, store_analyzer)

def show_brand_similarity_search(analyzer):
    """ブランド類似度検索"""
    st.header("🔍 ブランド類似度検索")
    
    if analyzer.df is None:
        st.error("データがありません")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_brand = st.selectbox(
            "検索するブランドを選択:",
            analyzer.df['name'].tolist()
        )
    
    with col2:
        top_k = st.slider("表示件数:", 5, 20, 10)
        min_similarity = st.slider("最小類似度:", 0.0, 1.0, 0.1, 0.05)
    
    # 位置情報リランキング設定
    enable_location_boost = st.checkbox("🏪 店舗情報によるリランキング", value=True)
    
    if enable_location_boost:
        bias_strength = st.slider("店舗バイアス強度:", 0.0, 1.0, 0.3, 0.1)
    else:
        bias_strength = 0.0
    
    if st.button("🚀 検索実行", type="primary"):
        with st.spinner("類似ブランドを検索中..."):
            # 基本的な類似度計算
            try:
                target_idx = analyzer.df[analyzer.df['name'] == target_brand].index[0]
                target_embedding = analyzer.embeddings[target_idx]
                
                # 類似度計算
                similarities = cosine_similarity([target_embedding], analyzer.embeddings)[0]
                
                # 結果整理
                results = []
                for i, sim in enumerate(similarities):
                    if i != target_idx and sim >= min_similarity:
                        brand_name = analyzer.df.iloc[i]['name']
                        results.append({
                            'brand_name': brand_name,
                            'similarity': sim,
                            'description': analyzer.df.iloc[i]['description']
                        })
                
                # 位置情報リランキング適用
                if enable_location_boost and results:
                    try:
                        reranker = create_reranker_from_streamlit_data(analyzer)
                        if reranker and target_brand in reranker.brand_locations:
                            similarity_dict = {r['brand_name']: r['similarity'] for r in results}
                            reranked = reranker.rerank_similarity_with_location_bias(
                                similarity_dict, target_brand, bias_strength
                            )
                            for r in results:
                                if r['brand_name'] in reranked:
                                    r['final_similarity'] = reranked[r['brand_name']]
                                    r['location_boost'] = reranked[r['brand_name']] - r['similarity']
                                else:
                                    r['final_similarity'] = r['similarity']
                                    r['location_boost'] = 0
                            
                            # 最終類似度でソート
                            results.sort(key=lambda x: x['final_similarity'], reverse=True)
                        else:
                            for r in results:
                                r['final_similarity'] = r['similarity']
                                r['location_boost'] = 0
                    except Exception as e:
                        st.warning(f"位置情報リランキングエラー: {e}")
                        for r in results:
                            r['final_similarity'] = r['similarity']
                            r['location_boost'] = 0
                else:
                    for r in results:
                        r['final_similarity'] = r['similarity']
                        r['location_boost'] = 0
                    results.sort(key=lambda x: x['similarity'], reverse=True)
                
                # 上位K件に絞る
                top_results = results[:top_k]
                
                if top_results:
                    st.success(f"✅ {len(top_results)} 件の類似ブランドが見つかりました")
                    
                    # 結果表示
                    display_data = []
                    for i, result in enumerate(top_results, 1):
                        display_data.append({
                            '順位': f"#{i}",
                            'ブランド名': result['brand_name'],
                            'ベース類似度': f"{result['similarity']:.4f}",
                            '最終類似度': f"{result['final_similarity']:.4f}",
                            '位置ブースト': f"+{result['location_boost']:.3f}" if result['location_boost'] > 0.001 else "---"
                        })
                    
                    df_display = pd.DataFrame(display_data)
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # 可視化
                    if len(top_results) > 1:
                        brands = [r['brand_name'] for r in top_results[:8]]
                        base_sims = [r['similarity'] for r in top_results[:8]]
                        final_sims = [r['final_similarity'] for r in top_results[:8]]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='ベース類似度',
                            x=brands,
                            y=base_sims,
                            marker_color='lightblue'
                        ))
                        fig.add_trace(go.Bar(
                            name='最終類似度',
                            x=brands,
                            y=final_sims,
                            marker_color='darkblue'
                        ))
                        
                        fig.update_layout(
                            title=f"類似度比較: {target_brand}",
                            xaxis_title="ブランド名",
                            yaxis_title="類似度",
                            barmode='group',
                            height=400,
                            xaxis={'tickangle': 45}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("条件に一致するブランドが見つかりませんでした。")
                
            except Exception as e:
                st.error(f"検索エラー: {e}")

def show_store_overlap_analysis(store_analyzer):
    """店舗一致率分析"""
    st.header("🏪 店舗一致率分析")
    
    if store_analyzer is None:
        st.error("店舗データが利用できません。")
        return
    
    # 品質レポート
    report = store_analyzer.get_separation_quality_report()
    
    st.subheader("📊 データ品質レポート")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総店舗数", report['総エントリー数'])
    with col2:
        st.metric("分離成功率", f"{report['店舗名分離率']:.1%}")
    with col3:
        st.metric("検出ブランド数", report['検出ブランド数'])
    with col4:
        st.metric("平均店舗数/ブランド", f"{report['平均店舗数']:.1f}")
    
    # ブランド・店舗サマリー
    st.subheader("🏆 店舗数ランキング")
    summary = store_analyzer.get_brand_store_summary()
    st.dataframe(summary.head(20), use_container_width=True, hide_index=True)
    
    # 高い店舗一致率ペア
    st.subheader("🔗 高い店舗一致率ペア")
    threshold = st.slider("一致率閾値:", 0.1, 1.0, 0.3, 0.05)
    
    high_overlap = store_analyzer.find_high_overlap_brands(threshold=threshold)
    
    if high_overlap:
        st.info(f"📍 {len(high_overlap)} 組のペアが見つかりました")
        
        display_data = []
        for pair in high_overlap[:20]:
            display_data.append({
                'ブランド1': pair['ブランド1'],
                'ブランド2': pair['ブランド2'],
                '店舗一致率': f"{pair['店舗一致率']:.3f}",
                '共通店舗数': pair['共通店舗数'],
                '共通店舗': pair['共通店舗'][:100] + "..." if len(pair['共通店舗']) > 100 else pair['共通店舗']
            })
        
        df_pairs = pd.DataFrame(display_data)
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)
    else:
        st.warning("指定された閾値以上の一致率を持つペアが見つかりませんでした。")

def show_new_brand_addition(analyzer):
    """新ブランド追加"""
    st.header("➕ 新ブランド追加")
    
    st.markdown("""
    <div class="success-box">
    <h4>🚀 新ブランド分析</h4>
    <p>ブランド名を入力し、GPT-OSSまたはGemini APIを使って説明文を生成後、既存ブランドとの類似度を分析します。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    model_option = st.radio(
        "🤖 生成モデルを選択:",
        ["GPT-OSS (ローカル)", "Gemini API (オンライン)"],
        help="GPT-OSSはローカルで動作し、Gemini APIはオンライン接続が必要です"
    )
    
    use_gpt_oss = model_option.startswith("GPT-OSS")
    
    # API Key input (only for Gemini)
    api_key = None
    if not use_gpt_oss:
        api_key = st.text_input(
            "🔑 Gemini API Key:",
            type="password",
            help="Google AI StudioでAPIキーを取得してください"
        )
    
    # Brand input
    brand_name = st.text_input(
        "🏷️ ブランド名:", 
        placeholder="例: Stone Island, Fear of God, Acne Studios"
    )
    
    # CSV批量処理セクション
    st.markdown("---")
    st.subheader("📊 CSV一括処理")
    
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロード:",
        type=['csv'],
        help="name列を含むCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            # CSV読み込み
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSVファイル読み込み完了: {len(df)} 行")
            
            if 'name' not in df.columns:
                st.error("'name' 列が見つかりません。")
                return
            
            st.write("データプレビュー:")
            st.dataframe(df.head(), use_container_width=True)
            
            # 一括処理設定
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("バッチサイズ:", 1, 20, 5)
            with col2:
                start_from = st.number_input("開始行 (0から):", 0, len(df)-1, 0)
            
            # 一括処理実行
            if st.button("🚀 一括説明文生成開始", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                # 処理対象データ
                process_df = df[start_from:start_from+batch_size].copy()
                
                for i, row in process_df.iterrows():
                    brand_name_batch = row['name']
                    
                    # 進捗更新
                    progress = (i - start_from + 1) / len(process_df)
                    progress_bar.progress(progress)
                    status_text.text(f"処理中: {brand_name_batch} ({i-start_from+1}/{len(process_df)})")
                    
                    # 説明文生成
                    try:
                        description = generate_brand_description(api_key, brand_name_batch, use_gpt_oss)
                        if description:
                            results.append({
                                'ブランド名': brand_name_batch,
                                '生成された説明文': description,
                                '状態': '成功'
                            })
                        else:
                            results.append({
                                'ブランド名': brand_name_batch,
                                '生成された説明文': '生成失敗',
                                '状態': 'エラー'
                            })
                    except Exception as e:
                        results.append({
                            'ブランド名': brand_name_batch,
                            '生成された説明文': f'エラー: {str(e)}',
                            '状態': 'エラー'
                        })
                
                # 結果表示
                if results:
                    result_df = pd.DataFrame(results)
                    success_count = len(result_df[result_df['状態'] == '成功'])
                    
                    st.success(f"✅ 処理完了: {success_count}/{len(results)} 件成功")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # CSV出力
                    csv_output = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 結果をCSVダウンロード",
                        data=csv_output,
                        file_name=f"brand_descriptions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                progress_bar.empty()
                status_text.empty()
        except Exception as e:
            st.error(f"CSV処理エラー: {e}")
    
    # 単一ブランド処理
    st.markdown("---")
    st.subheader("🎯 単一ブランド分析")
    
    # 条件チェック
    can_proceed = brand_name and (use_gpt_oss or api_key)
    
    if not can_proceed:
        if use_gpt_oss:
            st.info("ブランド名を入力してください。")
        else:
            st.info("APIキーとブランド名を入力してください。")
        return
    
    # Analysis button
    if st.button("🚀 単一ブランド分析開始", type="primary"):
        try:
            # Step 1: Generate description
            with st.spinner("説明文を生成中..."):
                description = generate_brand_description(api_key, brand_name, use_gpt_oss)
                if not description:
                    st.error("説明文生成に失敗しました。")
                    return
            
            st.success("✅ 説明文生成完了")
            with st.expander("生成された説明文", expanded=True):
                st.write(description)
            
            # Step 2: Calculate similarities (simplified)
            st.info("🔍 類似度計算は実装中です。現在は説明文生成のみ対応しています。")
            
        except Exception as e:
            st.error(f"エラー: {e}")

def show_data_overview(analyzer, store_analyzer):
    """データ概観"""
    st.header("📊 データ概観")
    
    if analyzer is None:
        st.error("ブランドデータがありません")
        return
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総ブランド数", len(analyzer.df))
    with col2:
        if analyzer.embeddings is not None:
            st.metric("埋め込み次元数", analyzer.embeddings.shape[1])
        else:
            st.metric("埋め込み次元数", "未生成")
    with col3:
        avg_desc_len = analyzer.df['description'].str.len().mean()
        st.metric("平均説明文長", f"{avg_desc_len:.0f}文字")
    with col4:
        if store_analyzer:
            st.metric("店舗データブランド数", store_analyzer.get_separation_quality_report()['検出ブランド数'])
        else:
            st.metric("店舗データ", "未読み込み")
    
    # データサンプル
    st.subheader("📋 ブランドデータサンプル")
    st.dataframe(analyzer.df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()