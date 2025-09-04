#!/usr/bin/env python3
"""
統合ブランド検索システム
新ブランド追加 → Gemini説明文生成 → Ruri v3ベクトル化 → 類似度検索 → リランキング → 可視化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from brand_similarity import LLMStyleEmbeddingAnalyzer
# from location_bias_reranking import create_reranker_from_streamlit_data
from gpt_oss_direct import GPTOSSDirect
import warnings
warnings.filterwarnings('ignore')
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Streamlit page configuration
st.set_page_config(
    page_title="統合ブランド検索システム",
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
    .search-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
    }
    .new-brand-highlight {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffecb3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedBrandSearchSystem:
    """統合ブランド検索システム"""
    
    def __init__(self):
        self.base_analyzer = None
        self.ruri_model = None
        self.embeddings = None
        self.brand_names = []
        self.brand_descriptions = []
        # self.reranker = None
        self.added_brands = []  # 新規追加されたブランド
        self.dimensionality_reducer = None
        self.reduction_results = {}
        self.cluster_results = {}
        
        # ブランドマッピング用
        self.brand_mapping = {}  # integrated_brand -> maps_brand
        
        # ジャンル情報管理
        self.maps_data = None  # maps.csvデータ
        self.brand_genre_mapping = {}  # ブランド名 -> genres
        self.genre_brands = {}  # genre -> [brand_names]
        
        # GPT-OSS直接実行インスタンス
        self.gpt_oss = None
        
    @st.cache_resource
    def load_ruri_model(_self):
        """Ruri v3モデルの読み込み"""
        try:
            model = SentenceTransformer('cl-nagoya/ruri-v3-310m')
            return model
        except Exception as e:
            st.error(f"Ruri v3モデル読み込みエラー: {e}")
            return None
    
    @st.cache_data
    def load_base_data(_self):
        """既存ブランドデータの読み込み"""
        try:
            # 既存アナライザーを使用
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
                        analyzer.embeddings = normalize(raw_embeddings, norm='l2', axis=1)
                        st.success(f"✅ 埋め込み読み込み: {os.path.basename(emb_path)}")
                        break
                    except Exception as e:
                        continue
            
            if analyzer.embeddings is None:
                st.warning("既存埋め込みなし - 新規生成が必要")
                
            return analyzer
            
        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            return None
    
    def initialize(self):
        """システム初期化"""
        self.base_analyzer = self.load_base_data()
        self.ruri_model = self.load_ruri_model()
        
        if self.base_analyzer and self.base_analyzer.embeddings is not None:
            self.embeddings = self.base_analyzer.embeddings.copy()
            # ベースデータからブランド名と説明を復元
            self.brand_names = self.base_analyzer.df['name'].tolist()
            self.brand_descriptions = self.base_analyzer.df['description'].tolist()
            
            # リランカー初期化
            # try:
            #     self.reranker = create_reranker_from_streamlit_data(self.base_analyzer)
            #     # ブランドマッピングを初期化
            #     self._initialize_brand_mapping()
            # except:
            #     self.reranker = None
                
            # ジャンル情報読み込み
            self.load_genre_data()
            
            # 次元削減システム初期化
            self.initialize_dimensionality_reducer()
            
        
        return self.base_analyzer is not None and self.ruri_model is not None
    
    def initialize_dimensionality_reducer(self):
        """次元削減システム初期化"""
        try:
            if self.embeddings is not None and len(self.embeddings) > 0:
                # ブランドレベルでの埋め込み集約
                brand_embeddings, unique_brand_names, unique_descriptions = self.aggregate_embeddings_by_brand()
                
                if brand_embeddings is not None and len(brand_embeddings) > 0:
                    from integrated_dimensionality_reduction import IntegratedDimensionalityReduction
                    self.dimensionality_reducer = IntegratedDimensionalityReduction(
                        embeddings=brand_embeddings,
                        brand_names=unique_brand_names,
                        descriptions=unique_descriptions
                    )
                    print(f"✅ 次元削減システム初期化完了: {len(brand_embeddings)}ブランド")
                else:
                    self.dimensionality_reducer = None
            else:
                self.dimensionality_reducer = None
        except Exception as e:
            print(f"次元削減システム初期化エラー: {e}")
            self.dimensionality_reducer = None
    
    def aggregate_embeddings_by_brand(self):
        """ブランドレベルでの埋め込み集約"""
        try:
            if not hasattr(self, 'base_analyzer') or self.base_analyzer is None:
                return None, None, None
                
            df = self.base_analyzer.df
            embeddings = self.embeddings
            
            if len(df) != len(embeddings):
                print(f"⚠️ データ長不一致: CSV {len(df)} vs 埋め込み {len(embeddings)}")
                # データ長が違う場合でも、CSVの範囲内で処理を続行
                print("📝 CSVの範囲内で埋め込み集約を実行します...")
                max_index = min(len(df), len(embeddings))
                df = df.iloc[:max_index]
                embeddings = embeddings[:max_index]
            
            # ブランド名でグループ化して平均埋め込みを計算
            brand_groups = df.groupby('name')
            brand_embeddings = []
            unique_brand_names = []
            unique_descriptions = []
            
            for brand_name, group in brand_groups:
                indices = group.index.tolist()
                
                # インデックスがembeddingsの範囲内にあることを確認
                valid_indices = [i for i in indices if i < len(embeddings)]
                
                if valid_indices:
                    # 該当ブランドの埋め込みを平均化
                    brand_embedding = np.mean(embeddings[valid_indices], axis=0)
                    brand_embeddings.append(brand_embedding)
                    unique_brand_names.append(brand_name)
                    
                    # 代表的な説明文を取得
                    desc = group.iloc[0].get('description', brand_name)
                    unique_descriptions.append(desc)
            
            if brand_embeddings:
                brand_embeddings = np.array(brand_embeddings)
                print(f"📊 ブランド集約完了: {len(unique_brand_names)}ブランド, 埋め込み形状: {brand_embeddings.shape}")
                return brand_embeddings, unique_brand_names, unique_descriptions
            else:
                return None, None, None
                
        except Exception as e:
            print(f"埋め込み集約エラー: {e}")
            return None, None, None
    
    def _initialize_brand_mapping(self):
        """ブランドマッピングの初期化、部分一致で自動マッピング"""
        # if not self.reranker or not self.brand_names:
        #     return
        
        # available_brands = set(self.reranker.get_available_brands_for_location_analysis())
        return
        
        # 部分一致でマッピングを作成
        for integrated_brand in self.brand_names:
            best_match = None
            best_score = 0.0
            
            integrated_lower = integrated_brand.lower().replace(' ', '').replace('-', '')
            
            for available_brand in available_brands:
                available_lower = available_brand.lower().replace(' ', '').replace('-', '')
                
                # 部分一致チェック（双方向）
                if integrated_lower in available_lower or available_lower in integrated_lower:
                    # 一致率を計算（簡易版）
                    score = len(set(integrated_lower) & set(available_lower)) / len(set(integrated_lower) | set(available_lower))
                    if score > best_score and score > 0.3:  # 30%以上の一致
                        best_match = available_brand
                        best_score = score
            
            if best_match:
                self.brand_mapping[integrated_brand] = best_match
        
        print(f"🔗 ブランドマッピング初期化: {len(self.brand_mapping)} 件のマッピングを作成")
        
        # マッピング例を表示（最初の5件）
        for i, (integrated, mapped) in enumerate(list(self.brand_mapping.items())[:5]):
            print(f"  {integrated} -> {mapped}")
    
    def load_genre_data(self):
        """maps.csvからジャンル情報を読み込み"""
        try:
            import pandas as pd
            import os
            
            maps_path = "datasets/bline_similarity/maps.csv"
            if not os.path.exists(maps_path):
                st.warning("⚠️ maps.csvが見つかりません - ジャンル機能は無効化されます")
                return
            
            # maps.csvを読み込み
            self.maps_data = pd.read_csv(maps_path)
            st.info(f"✅ maps.csvを読み込みました（{len(self.maps_data)}件）")
            
            # ブランド名を抽出（店舗名からブランド名を推定）
            self._extract_brand_genre_mapping()
            
        except Exception as e:
            st.error(f"ジャンル情報読み込みエラー: {e}")
            self.maps_data = None
    
    def _extract_brand_genre_mapping(self):
        """maps.csvからブランド名とジャンルのマッピングを作成"""
        if self.maps_data is None:
            return
        
        # ジャンルがnullでない行のみ処理
        valid_genre_data = self.maps_data.dropna(subset=['genre'])
        
        for _, row in valid_genre_data.iterrows():
            shop_name = str(row['name'])
            genre_str = str(row['genre'])
            
            # ジャンル文字列を分割
            genres = [g.strip() for g in genre_str.split(',') if g.strip()]
            
            # 店舗名からブランド名を推定（簡易版）
            brand_name = self._extract_brand_name_from_shop(shop_name)
            
            if brand_name and brand_name in self.brand_names:
                # ブランドとジャンルのマッピングを作成
                if brand_name not in self.brand_genre_mapping:
                    self.brand_genre_mapping[brand_name] = set()
                
                self.brand_genre_mapping[brand_name].update(genres)
                
                # ジャンルからブランドの逆マッピングも作成
                for genre in genres:
                    if genre not in self.genre_brands:
                        self.genre_brands[genre] = set()
                    self.genre_brands[genre].add(brand_name)
        
        # setをlistに変換
        for brand in self.brand_genre_mapping:
            self.brand_genre_mapping[brand] = list(self.brand_genre_mapping[brand])
        
        for genre in self.genre_brands:
            self.genre_brands[genre] = list(self.genre_brands[genre])
        
        st.info(f"🎯 ジャンルマッピング完了: {len(self.brand_genre_mapping)}ブランド、{len(self.genre_brands)}ジャンル")
    
    def _extract_brand_name_from_shop(self, shop_name):
        """店舗名からブランド名を推定"""
        # 店舗名から場所情報を除去してブランド名を抽出
        shop_parts = shop_name.split()
        if len(shop_parts) == 0:
            return None
        
        # 最初の部分をブランド名として使用（簡易版）
        potential_brand = shop_parts[0]
        
        # integrated_brands.csvのブランド名と部分一致チェック
        for brand_name in self.brand_names:
            brand_lower = brand_name.lower().replace(' ', '').replace('-', '')
            potential_lower = potential_brand.lower().replace(' ', '').replace('-', '')
            
            # 部分一致または包含関係チェック
            if (potential_lower in brand_lower or brand_lower in potential_lower) and len(potential_lower) > 2:
                return brand_name
        
        return None
    
    def generate_brand_description_template(self, brand_name):
        """オフライン用のデフォルト説明文テンプレート生成"""
        # ブランド名をベースにした基本的な説明文テンプレート
        template = f"""ファッションブランド「{brand_name}」は、独自のデザイン哲学とスタイルを持つブランドです。
コンテンポラリーなデザインと機能性を重視した製品展開を行っており、
幅広い年齢層に支持されています。革新的なアプローチとクラフトマンシップを
大切にしながら、現代的なライフスタイルに合わせた商品を提供しています。"""
        return template.strip()
    
    def initialize_gpt_oss(self):
        """GPT-OSS直接実行モデルの初期化"""
        if self.gpt_oss is None:
            with st.spinner("GPT-OSSモデルを読み込み中..."):
                self.gpt_oss = GPTOSSDirect()
        return self.gpt_oss.is_available()
    
    def check_gpt_oss_connection(self):
        """GPT-OSS接続状態確認"""
        try:
            if self.gpt_oss is None:
                self.gpt_oss = GPTOSSDirect()
            return self.gpt_oss.is_available()
        except Exception:
            return False
    
    def generate_brand_description(self, api_key, brand_name, fallback_mode="template"):
        """GPT-OSS ローカルAPIでブランド説明文生成（フォールバック機能付き）"""
        
        # モデル初期化
        model_available = self.initialize_gpt_oss()
        
        if not model_available:
            st.warning("⚠️ GPT-OSSモデルを読み込めません")
            
            if fallback_mode == "template":
                st.info("📝 デフォルトテンプレートを使用します")
                return self.generate_brand_description_template(brand_name)
            elif fallback_mode == "manual":
                st.info("✏️ 手動入力モードに切り替えてください")
                return None
            else:
                st.error("❌ フォールバック機能が設定されていません")
                return None
        
        try:
            # 直接推論実行
            st.info("🚀 GPT-OSSで説明文生成中...")
            description = self.gpt_oss.generate_brand_description(brand_name)
            
            if description and not description.startswith("エラー:"):
                return description.strip()
            else:
                st.error(f"生成エラー: {description}")
                # エラー時のフォールバック
                if fallback_mode == "template":
                    st.info("📝 生成エラーのためデフォルトテンプレートを使用します")
                    return self.generate_brand_description_template(brand_name)
                return None
        except Exception as e:
            st.error(f"説明文生成エラー: {e}")
            
            # 例外発生時のフォールバック
            if fallback_mode == "template":
                st.info("📝 エラーのためデフォルトテンプレートを使用します")
                return self.generate_brand_description_template(brand_name)
            
            return None
    
    def check_brand_exists(self, brand_name, strict=True):
        """ブランドの重複チェック"""
        if not self.brand_names:
            return False, None
        
        # 完全一致チェック
        if brand_name in self.brand_names:
            return True, brand_name
        
        # 非厳密モードの場合、類似ブランド名チェック
        if not strict:
            brand_lower = brand_name.lower().replace(' ', '').replace('-', '')
            for existing_brand in self.brand_names:
                existing_lower = existing_brand.lower().replace(' ', '').replace('-', '')
                if brand_lower == existing_lower or brand_lower in existing_lower or existing_lower in brand_lower:
                    return True, existing_brand
        
        return False, None
    
    def add_new_brand(self, brand_name, description):
        """新ブランドを系統に追加"""
        if not self.ruri_model:
            st.error("Ruri v3モデルが利用できません")
            return False
        
        # 既存ブランドチェック
        exists, existing_name = self.check_brand_exists(brand_name, strict=True)
        if exists:
            st.warning(f"⚠️ '{brand_name}' は既に登録されています（既存: '{existing_name}'）。スキップします。")
            return False
        
        try:
            # Ruri v3で新ブランドをベクトル化
            new_embedding = self.ruri_model.encode(
                [description], 
                normalize_embeddings=True,
                convert_to_tensor=False
            )
            
            # 既存システムに統合
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embedding])
            else:
                self.embeddings = new_embedding
            
            self.brand_names.append(brand_name)
            self.brand_descriptions.append(description)
            self.added_brands.append({
                'name': brand_name,
                'description': description,
                'index': len(self.brand_names) - 1
            })
            
            # データ永続化と次元削減システム更新
            self._save_updated_data()
            self._update_dimensionality_reducer()
            
            st.success(f"✅ {brand_name} を追加しました")
            return True
            
        except Exception as e:
            st.error(f"ブランド追加エラー: {e}")
            return False
    
    def process_csv_batch(self, csv_data, api_key="", progress_callback=None, use_existing_descriptions=False, force_regenerate=False, strict_check=True):
        """CSV一括処理 - ブランド名検索→重複チェック→説明文生成→Ruriv3ベクトル化→辞書追加のサイクル"""
        if not self.ruri_model:
            st.error("Ruri v3モデルが利用できません")
            return []
        
        results = []
        total_brands = len(csv_data)
        
        for i, row in enumerate(csv_data):
            if progress_callback:
                progress_callback(i, total_brands, f"ブランド名検索中: {row['brand_name']}")
            
            try:
                brand_name = row['brand_name']
                
                # ステップ1: ブランド名による既存ベクトル辞書での重複チェック
                exists, existing_name = self.check_brand_exists(brand_name, strict=strict_check)
                if exists:
                    error_msg = f'既にベクトル化済み（既存: {existing_name}）' if existing_name != brand_name else '既にベクトル化済み'
                    results.append({
                        'brand_name': brand_name,
                        'status': 'skipped',
                        'error': error_msg,
                        'description': 'ベクトル辞書に存在',
                        'generated': False,
                        'step': 'duplicate_check'
                    })
                    continue
                
                if progress_callback:
                    progress_callback(i, total_brands, f"説明文生成中: {brand_name}")
                
                # ステップ2: ブランド名から説明文生成
                description = None
                generated = False
                
                if force_regenerate:
                    # 強制的に新規生成（フォールバック機能付き）
                    description = self.generate_brand_description(api_key, brand_name, fallback_mode="template")
                    generated = True
                elif use_existing_descriptions and row.get('description') and str(row['description']).strip():
                    # 既存説明文を使用
                    description = row['description']
                    generated = False
                else:
                    # GPT-OSSで説明文生成（フォールバック機能付き）
                    description = self.generate_brand_description(api_key, brand_name, fallback_mode="template")
                    generated = True
                    
                if not description:
                    results.append({
                        'brand_name': brand_name,
                        'status': 'failed',
                        'error': '説明文生成失敗',
                        'description': '',
                        'generated': generated,
                        'step': 'description_generation'
                    })
                    continue
                
                if progress_callback:
                    progress_callback(i, total_brands, f"Ruriv3ベクトル化中: {brand_name}")
                
                # ステップ3: Ruriv3によるベクトル化
                try:
                    new_embedding = self.ruri_model.encode(
                        [description], 
                        normalize_embeddings=True,
                        convert_to_tensor=False
                    )
                except Exception as e:
                    results.append({
                        'brand_name': brand_name,
                        'status': 'failed',
                        'error': f'ベクトル化失敗: {str(e)}',
                        'description': description[:100] + '...',
                        'generated': generated,
                        'step': 'vectorization'
                    })
                    continue
                
                if progress_callback:
                    progress_callback(i, total_brands, f"ベクトル辞書追加中: {brand_name}")
                
                # ステップ4: ベクトル辞書への追加
                try:
                    if self.embeddings is not None:
                        self.embeddings = np.vstack([self.embeddings, new_embedding])
                    else:
                        self.embeddings = new_embedding
                    
                    self.brand_names.append(brand_name)
                    self.brand_descriptions.append(description)
                    self.added_brands.append({
                        'name': brand_name,
                        'description': description,
                        'index': len(self.brand_names) - 1
                    })
                    
                    results.append({
                        'brand_name': brand_name,
                        'status': 'success',
                        'description': description[:100] + '...',
                        'generated': generated,
                        'step': 'completed',
                        'embedding_shape': new_embedding.shape,
                        'vector_index': len(self.brand_names) - 1
                    })
                    
                except Exception as e:
                    results.append({
                        'brand_name': brand_name,
                        'status': 'failed',
                        'error': f'辞書追加失敗: {str(e)}',
                        'description': description[:100] + '...',
                        'generated': generated,
                        'step': 'dictionary_addition'
                    })
                
            except Exception as e:
                results.append({
                    'brand_name': row.get('brand_name', 'Unknown'),
                    'status': 'error',
                    'error': f'予期しないエラー: {str(e)}',
                    'description': '',
                    'generated': False,
                    'step': 'unknown_error'
                })
        
        # 処理完了後の統計情報をログ
        success_count = sum(1 for r in results if r['status'] == 'success')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')
        failed_count = len(results) - success_count - skipped_count
        
        # 新規追加があった場合はデータ永続化と次元削減システム更新
        if success_count > 0:
            self._save_updated_data()
            self._update_dimensionality_reducer()
        
        print(f"📊 CSV一括処理完了: 成功 {success_count}, スキップ {skipped_count}, 失敗 {failed_count}")
        
        return results
    
    def _save_updated_data(self):
        """更新されたデータをディスクに保存"""
        try:
            # CSVファイル更新
            if self.base_analyzer and hasattr(self.base_analyzer, 'df'):
                # 新規追加ブランドを既存データフレームに追加
                new_rows = []
                for brand in self.added_brands:
                    new_rows.append({
                        'name': brand['name'],
                        'description': brand['description']
                    })
                
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    updated_df = pd.concat([self.base_analyzer.df, new_df], ignore_index=True)
                    updated_df.to_csv('integrated_brands.csv', index=False)
            
            # 埋め込みベクトル保存
            if self.embeddings is not None:
                os.makedirs('./ruri_embeddings_results/', exist_ok=True)
                np.save('./ruri_embeddings_results/ruri_description_embeddings_v3_raw_hub.npy', self.embeddings)
                
        except Exception as e:
            st.warning(f"データ保存エラー: {e}")
    
    def _update_dimensionality_reducer(self):
        """次元削減システムを更新されたデータで再初期化"""
        try:
            if self.embeddings is not None and len(self.embeddings) > 0:
                # ブランドレベルでの埋め込み集約
                brand_embeddings, unique_brand_names, unique_descriptions = self.aggregate_embeddings_by_brand()
                
                if brand_embeddings is not None and len(brand_embeddings) > 0:
                    from integrated_dimensionality_reduction import IntegratedDimensionalityReduction
                    self.dimensionality_reducer = IntegratedDimensionalityReduction(
                        embeddings=brand_embeddings,
                        brand_names=unique_brand_names,
                        descriptions=unique_descriptions
                    )
                    print(f"✅ 次元削減システム更新完了: {len(brand_embeddings)}ブランド")
                else:
                    self.dimensionality_reducer = None
            else:
                self.dimensionality_reducer = None
        except Exception as e:
            st.warning(f"次元削減システム更新エラー: {e}")
            self.dimensionality_reducer = None
    
    def search_similar_brands(self, target_brand, top_k=10, min_similarity=0.1, genre_filter=None, normalize_similarity=False):
        """類似ブランド検索（ジャンルフィルタリング対応）"""
        if self.embeddings is None:
            return []
        
        try:
            # ブランド名検索
            if target_brand not in self.brand_names:
                st.error(f"ブランド '{target_brand}' が見つかりません")
                return []
            
            target_idx = self.brand_names.index(target_brand)
            target_embedding = self.embeddings[target_idx].reshape(1, -1)
            
            # ジャンルフィルタリング用の候補ブランドを決定
            candidate_brands = set(self.brand_names)
            
            if genre_filter and self.brand_genre_mapping:
                # 指定されたジャンルを持つブランドのみに限定
                if isinstance(genre_filter, str):
                    genre_filter = [genre_filter]
                
                filtered_brands = set()
                for genre in genre_filter:
                    if genre in self.genre_brands:
                        filtered_brands.update(self.genre_brands[genre])
                
                if filtered_brands:
                    candidate_brands = filtered_brands
                    st.info(f"🎯 ジャンル '{', '.join(genre_filter)}' でフィルタリング: {len(candidate_brands)}ブランドが対象")
                else:
                    st.warning(f"⚠️ 指定されたジャンル '{', '.join(genre_filter)}' にブランドが見つかりません")
                    return []
            
            # 類似度計算
            similarities = cosine_similarity(target_embedding, self.embeddings)[0]
            
            # 結果整理
            results = []
            for i, sim in enumerate(similarities):
                if i < len(self.brand_names) and i < len(self.brand_descriptions):  # bounds check
                    brand_name = self.brand_names[i]
                    if i != target_idx and sim >= min_similarity and brand_name in candidate_brands:
                        is_new_brand = any(added['index'] == i for added in self.added_brands)
                        brand_genres = self.brand_genre_mapping.get(brand_name, [])
                        
                        results.append({
                            'brand_name': brand_name,
                            'similarity': sim,
                            'description': self.brand_descriptions[i],
                            'is_new': is_new_brand,
                            'genres': brand_genres
                        })
            
            # 類似度の正規化（オプション）
            if normalize_similarity and results:
                similarities = [r['similarity'] for r in results]
                min_sim = min(similarities)
                max_sim = max(similarities)
                
                # Min-Max正規化で0-1の範囲に再スケーリング
                if max_sim > min_sim:
                    for result in results:
                        original_sim = result['similarity']
                        normalized_sim = (original_sim - min_sim) / (max_sim - min_sim)
                        result['original_similarity'] = original_sim
                        result['similarity'] = normalized_sim
                        result['normalized'] = True
                else:
                    # 全て同じ値の場合は正規化しない
                    for result in results:
                        result['original_similarity'] = result['similarity']
                        result['normalized'] = False
            else:
                for result in results:
                    result['original_similarity'] = result['similarity']
                    result['normalized'] = False
            
            # 類似度順でソート
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"検索エラー: {e}")
            return []
    
    # def apply_fixed_boost_reranking(self, results, target_brand, location_method="comprehensive"):
        """固定ブースト戦略によるリランキング（分析結果ベース）"""
        if not results:
            return results
        
        # 固定ブースト値設定（分析結果に基づく）
        boost_tiers = {
            0.8: 0.025,   # 80%以上一致 -> +0.025
            0.5: 0.015,   # 50%以上一致 -> +0.015
            0.2: 0.005,   # 20%以上一致 -> +0.005
            0.0: 0.000    # 一致なし -> +0.000
        }
        
        # 初期値設定
        for result in results:
            result['similarity_score'] = result['similarity']  # ベーススコアを保存
            result['original_similarity'] = result['similarity']  # 元の類似度
            result['final_similarity'] = result['similarity']
            result['location_boost'] = 0.0
            result['location_similarity'] = 0.0
            result['boost_tier'] = '0%'
            result['rerank_method'] = location_method
            result['rerank_mode'] = 'fixed_boost'
        
        if not self.reranker:
            st.warning("リランカーが利用できません - 元の類似度スコアを使用")
            return results
        
        try:
            # 位置類似度を計算して固定ブーストを適用
            for result in results:
                brand_name = result['brand_name']
                
                # ブランドマッピングを使用して位置類似度を計算
                mapped_target = self.brand_mapping.get(target_brand, target_brand)
                mapped_brand = self.brand_mapping.get(brand_name, brand_name)
                
                if mapped_target in self.reranker.brand_locations and mapped_brand in self.reranker.brand_locations:
                    location_sim = self.reranker.calculate_location_similarity(
                        mapped_target, mapped_brand, method=location_method
                    )
                    result['location_similarity'] = location_sim
                    
                    # 位置一致率に応じた固定ブーストを適用
                    boost_value = 0.0
                    tier_label = '0%'
                    
                    for threshold in sorted(boost_tiers.keys(), reverse=True):
                        if location_sim >= threshold:
                            boost_value = boost_tiers[threshold]
                            tier_label = f'{int(threshold*100)}%+'
                            break
                    
                    result['location_boost'] = boost_value
                    result['boost_tier'] = tier_label
                    result['final_similarity'] = result['similarity'] + boost_value
                else:
                    result['location_similarity'] = 0.0
                    result['location_boost'] = 0.0
                    result['boost_tier'] = 'データなし'
                    result['final_similarity'] = result['similarity']
            
            # 最終類似度でソート
            results.sort(key=lambda x: x['final_similarity'], reverse=True)
            return results
            
        except Exception as e:
            st.warning(f"固定ブーストリランキングエラー: {e}")
            return results
    
    # def apply_location_reranking(self, results, target_brand, bias_strength=0.015, 
    #                             location_method="comprehensive", rerank_mode="weighted_average"):
        """拡張された位置情報リランキング適用（従来方式）"""
        if not results:
            return results
        
        # まず初期値を設定
        for result in results:
            result['similarity_score'] = result['similarity']  # API結果形式に合わせる
            result['original_similarity'] = result['similarity']
            result['final_similarity'] = result['similarity']
            result['location_boost'] = 0.0
            result['location_similarity'] = 0.0
            result['rerank_method'] = location_method
            result['rerank_mode'] = rerank_mode
            result['bias_strength'] = bias_strength
        
        # リランカーがない場合は初期値のまま返す
        if not self.reranker:
            st.warning("リランカーが利用できません - 元の類似度スコアを使用")
            return results
        
        try:
            # 類似度辞書作成
            similarity_dict = {r['brand_name']: r['similarity'] for r in results}
            
            # リランキング実行（ブランドマッピング付き）
            reranked_similarities = self.reranker.rerank_similarity_with_location_bias(
                similarity_dict, target_brand, bias_strength, location_method, rerank_mode,
                brand_mapping=self.brand_mapping
            )
            
            # 各ブランドの位置類似度も計算
            for result in results:
                brand_name = result['brand_name']
                
                # ブランドマッピングを使用して位置類似度を計算
                mapped_target = self.brand_mapping.get(target_brand, target_brand)
                mapped_brand = self.brand_mapping.get(brand_name, brand_name)
                
                if mapped_target in self.reranker.brand_locations and mapped_brand in self.reranker.brand_locations:
                    location_sim = self.reranker.calculate_location_similarity(
                        mapped_target, mapped_brand, method=location_method
                    )
                    result['location_similarity'] = location_sim
                else:
                    result['location_similarity'] = 0.0
                
                # リランキング後の類似度を設定
                if brand_name in reranked_similarities:
                    result['similarity_score'] = reranked_similarities[brand_name]  # API結果形式
                    result['final_similarity'] = reranked_similarities[brand_name]
                    result['location_boost'] = reranked_similarities[brand_name] - result['similarity']
                else:
                    result['similarity_score'] = result['similarity']
                    result['final_similarity'] = result['similarity']
                    result['location_boost'] = 0.0
            
            # 最終類似度でソート
            results.sort(key=lambda x: x['final_similarity'], reverse=True)
            return results
            
        except Exception as e:
            st.warning(f"リランキングエラー: {e}")
            # エラー時も初期値は既に設定済み
            return results
    
    def create_embedding_visualization(self, highlight_brands=None):
        """特徴量空間の可視化（ブランドレベル）"""
        if self.embeddings is None or len(self.embeddings) < 2:
            st.warning("可視化に十分なデータがありません")
            return None
        
        try:
            # ブランドレベルの埋め込みを取得
            brand_embeddings, unique_brand_names, unique_descriptions = self.aggregate_embeddings_by_brand()
            
            if brand_embeddings is None or len(brand_embeddings) < 2:
                st.warning("ブランドレベルの可視化に十分なデータがありません")
                return None
            
            # UMAP次元削減
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(brand_embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            coords_2d = reducer.fit_transform(brand_embeddings)
            
            # 色分け設定
            colors = []
            hover_texts = []
            sizes = []
            
            for i, (name, desc) in enumerate(zip(unique_brand_names, unique_descriptions)):
                # インデックス範囲チェック
                if i >= len(coords_2d):
                    break
                    
                # 新規追加ブランドかチェック（ブランド名ベース）
                is_new = any(added['brand_name'] == name for added in self.added_brands)
                # ハイライト対象かチェック
                is_highlight = highlight_brands and name in highlight_brands
                
                if is_new:
                    colors.append('red')
                    sizes.append(12)
                elif is_highlight:
                    colors.append('orange')
                    sizes.append(10)
                else:
                    colors.append('steelblue')
                    sizes.append(8)
                
                # ホバーテキスト
                hover_text = f"<b>{name}</b><br>"
                hover_text += f"座標: ({coords_2d[i, 0]:.3f}, {coords_2d[i, 1]:.3f})<br>"
                if is_new:
                    hover_text += "<b>🆕 新規追加</b><br>"
                hover_text += f"説明: {desc[:100]}..."
                hover_texts.append(hover_text)
            
            # Plotly可視化
            fig = go.Figure()
            
            # データの長さを一致させる
            max_length = min(len(coords_2d), len(colors), len(sizes), len(hover_texts), len(unique_brand_names))
            
            # メインの散布図
            fig.add_trace(go.Scatter(
                x=coords_2d[:max_length, 0],
                y=coords_2d[:max_length, 1],
                mode='markers',
                marker=dict(
                    color=colors[:max_length],
                    size=sizes[:max_length],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=unique_brand_names[:max_length],
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts[:max_length],
                showlegend=False
            ))
            
            # 新規追加ブランドの強調
            if self.added_brands:
                # インデックス範囲チェック
                valid_indices = [added['index'] for added in self.added_brands 
                               if added['index'] < len(coords_2d) and added['index'] < len(self.brand_names)]
                if valid_indices:
                    new_coords = coords_2d[valid_indices]
                    new_names = [self.brand_names[i] for i in valid_indices]
                
                    fig.add_trace(go.Scatter(
                        x=new_coords[:, 0],
                        y=new_coords[:, 1],
                        mode='markers+text',
                        marker=dict(
                            color='red',
                            size=15,
                            symbol='star',
                            line=dict(width=2, color='darkred')
                        ),
                        text=new_names,
                        textposition='top center',
                        textfont=dict(size=10, color='darkred'),
                        name='新規追加ブランド',
                        showlegend=True
                    ))
            
            fig.update_layout(
                title={
                    'text': '🧠 ブランド特徴量空間 (Ruri v3 + UMAP)',
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis_title='UMAP次元1',
                yaxis_title='UMAP次元2',
                width=900,
                height=600,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"可視化エラー: {e}")
            return None
    
    def create_genre_center_visualization(self, selected_genres=None):
        """ジャンル中心ベクトルの特徴量空間可視化"""
        if self.embeddings is None or len(self.embeddings) < 2:
            st.warning("可視化に十分なデータがありません")
            return None
        
        if not self.brand_genre_mapping:
            st.warning("ジャンル情報が利用できません")
            return None
        
        try:
            # UMAP次元削減
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(self.embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            coords_2d = reducer.fit_transform(self.embeddings)
            
            # ジャンル中心ベクトルを計算
            genre_centers = {}
            genre_coords = {}
            
            for genre, brand_list in self.genre_brands.items():
                if selected_genres and genre not in selected_genres:
                    continue
                
                # ジャンルに属するブランドのインデックスを取得
                brand_indices = []
                for brand_name in brand_list:
                    if brand_name in self.brand_names:
                        brand_indices.append(self.brand_names.index(brand_name))
                
                if len(brand_indices) > 0:
                    # インデックス範囲チェック
                    valid_indices = [idx for idx in brand_indices if idx < len(self.embeddings) and idx < len(coords_2d)]
                    if valid_indices:
                        # 中心ベクトルを計算（埋め込み空間）
                        genre_embeddings = self.embeddings[valid_indices]
                        center_embedding = np.mean(genre_embeddings, axis=0)
                        genre_centers[genre] = center_embedding
                        
                        # 2D座標での中心も計算
                        genre_2d_coords = coords_2d[valid_indices]
                        center_2d = np.mean(genre_2d_coords, axis=0)
                        genre_coords[genre] = center_2d
            
            # 可視化作成
            fig = go.Figure()
            
            # 全ブランドをプロット（薄く表示）
            colors = []
            hover_texts = []
            sizes = []
            
            for i, (name, desc) in enumerate(zip(self.brand_names, self.brand_descriptions)):
                # インデックス範囲チェック
                if i >= len(coords_2d):
                    break
                    
                brand_genres = self.brand_genre_mapping.get(name, [])
                
                # 選択されたジャンルに属するかチェック
                if selected_genres:
                    has_selected_genre = any(g in selected_genres for g in brand_genres)
                    if has_selected_genre:
                        colors.append('lightblue')
                        sizes.append(6)
                    else:
                        colors.append('lightgray')
                        sizes.append(3)
                else:
                    colors.append('lightblue')
                    sizes.append(6)
                
                # ホバーテキスト
                hover_text = f"<b>{name}</b><br>"
                hover_text += f"ジャンル: {', '.join(brand_genres) if brand_genres else 'なし'}<br>"
                hover_text += f"座標: ({coords_2d[i, 0]:.3f}, {coords_2d[i, 1]:.3f})<br>"
                hover_text += f"説明: {desc[:100]}..."
                hover_texts.append(hover_text)
            
            # データの長さを一致させる
            max_length = min(len(coords_2d), len(colors), len(sizes), len(hover_texts), len(self.brand_names))
            
            # ブランド散布図
            fig.add_trace(go.Scatter(
                x=coords_2d[:max_length, 0],
                y=coords_2d[:max_length, 1],
                mode='markers',
                marker=dict(
                    color=colors[:max_length],
                    size=sizes[:max_length],
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                text=self.brand_names[:max_length],
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts[:max_length],
                name='ブランド',
                showlegend=True
            ))
            
            # ジャンル中心をプロット
            center_colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
            
            for i, (genre, coord) in enumerate(genre_coords.items()):
                color = center_colors[i % len(center_colors)]
                brand_count = len(self.genre_brands[genre])
                
                fig.add_trace(go.Scatter(
                    x=[coord[0]],
                    y=[coord[1]],
                    mode='markers+text',
                    marker=dict(
                        color=color,
                        size=20,
                        symbol='diamond',
                        line=dict(width=3, color='black')
                    ),
                    text=[f"{genre}"],
                    textposition='top center',
                    textfont=dict(size=12, color='black'),
                    name=f'{genre} ({brand_count}ブランド)',
                    hovertemplate=f"<b>{genre}</b><br>ブランド数: {brand_count}<br>座標: ({coord[0]:.3f}, {coord[1]:.3f})<extra></extra>",
                    showlegend=True
                ))
            
            fig.update_layout(
                title={
                    'text': '🎯 ジャンル中心ベクトル可視化 (Ruri v3 + UMAP)',
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis_title='UMAP次元1',
                yaxis_title='UMAP次元2',
                width=1000,
                height=700,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"ジャンル中心可視化エラー: {e}")
            return None

def main():
    st.markdown('<h1 class="main-header">🎯 統合ブランド検索システム</h1>', unsafe_allow_html=True)
    st.markdown("**新ブランド追加 → Gemini説明文生成 → Ruri v3ベクトル化 → 類似度検索 → リランキング → 可視化**")
    st.markdown("---")
    
    # システム初期化
    if 'search_system' not in st.session_state:
        with st.spinner("システム初期化中..."):
            search_system = IntegratedBrandSearchSystem()
            if search_system.initialize():
                st.session_state.search_system = search_system
                st.success("✅ システム初期化完了")
            else:
                st.error("❌ システム初期化失敗")
                return
    
    search_system = st.session_state.search_system
    
    # メインページ - タブ形式
    tab1, tab2, tab3, tab4 = st.tabs(["🔍🆕 ブランド検索 & 追加", "📂 CSV一括追加", "🧠 特徴量空間解析", "📊 システム情報"])
    
    with tab1:
        show_integrated_brand_interface(search_system)
    
    with tab2:
        show_csv_batch_interface(search_system)
    
    with tab3:
        show_advanced_visualization(search_system)
    
    with tab4:
        show_system_status(search_system)

def show_integrated_brand_interface(search_system):
    """ブランド検索と新ブランド追加の統合インターフェース"""
    st.header("🔍🆕 ブランド検索 & 新ブランド追加")
    
    # 2列レイアウト
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("🔍 ブランド検索")
        show_search_panel(search_system)
    
    with col2:
        st.subheader("➕ 新ブランド追加")
        show_addition_panel(search_system)
    
    # 共通の結果表示エリア
    st.markdown("---")
    show_unified_results(search_system)

def show_search_panel(search_system):
    """検索パネル"""
    if not search_system.brand_names:
        st.warning("検索可能なブランドがありません")
        return
    
    with st.container():
        st.markdown('<div class="search-card">', unsafe_allow_html=True)
        
        target_brand = st.selectbox(
            "🎯 検索するブランド:",
            search_system.brand_names,
            key="search_brand"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("表示件数:", 5, 100, 10, key="search_top_k")
        with col2:
            min_similarity = st.slider("最小類似度:", 0.0, 1.0, 0.1, 0.05, key="search_min_sim")
        
        # 正規化オプション
        normalize_similarity = st.checkbox("📊 類似度をギャラリ内で正規化（差を強調）", 
                                          value=False, 
                                          help="検索結果の類似度をmin-max正規化し、差を分かりやすくします")
        
        # ジャンルフィルタリング設定
        st.markdown("##### 🎯 ジャンルフィルタリング")
        if search_system.genre_brands:
            available_genres = sorted(search_system.genre_brands.keys())
            enable_genre_filter = st.checkbox("ジャンル限定検索を有効にする", value=False, key="enable_genre_filter")
            
            if enable_genre_filter:
                selected_genres = st.multiselect(
                    "対象ジャンル（複数選択可）:",
                    available_genres,
                    key="selected_genres",
                    help="選択されたジャンルの店舗を持つブランドのみが検索対象になります"
                )
            else:
                selected_genres = None
        else:
            enable_genre_filter = False
            selected_genres = None
            st.info("ℹ️ ジャンル情報が利用できません")
        
        # リランキング設定
        # st.markdown("##### 🏪 店舗リランキング設定")
        # enable_reranking = st.checkbox("位置情報リランキングを有効にする", value=True, key="search_rerank")
        
        # if enable_reranking:
        #     col_bias, col_method = st.columns(2)
        #     with col_bias:
        #         # 分析結果に基づく推奨値: 0.005-0.025
        #         bias_strength = st.slider("バイアス強度:", 0.001, 0.050, 0.015, 0.005, key="search_bias", 
        #                                 help="推奨値: 0.005-0.025 (分析結果より)")
        #     with col_method:
        #         rerank_mode = st.selectbox(
        #             "リランキング方式:",
        #             ["weighted_average", "linear_addition", "location_rerank"],
        #             index=0,
        #             key="search_rerank_mode",
        #             help="weighted_average: 重み付き平均, linear_addition: 線形加算, location_rerank: 位置重視"
        #         )
        #     
        #     location_method = st.selectbox(
        #         "位置類似度計算方法:",
        #         ["comprehensive", "tenant", "building", "floor", "geographic", "area"],
        #         index=0,
        #         key="search_location_method",
        #         help="comprehensive: 総合, tenant: テナント, building: ビル, floor: フロア, geographic: 地理的距離, area: エリア"
        #     )
        #     
        #     # 固定ブースト戦略の追加
        #     use_fixed_boost = st.checkbox("🎯 固定ブースト戦略を使用", value=False, key="search_fixed_boost",
        #                                 help="位置一致率に応じた段階的固定ブースト値を適用")
        # else:
        #     bias_strength = 0.0
        #     rerank_mode = "weighted_average"
        #     location_method = "comprehensive"
        #     use_fixed_boost = False
        enable_reranking = False
        bias_strength = 0.0
        rerank_mode = "weighted_average"
        location_method = "comprehensive"
        use_fixed_boost = False
        
        if st.button("🚀 検索実行", type="primary", key="search_execute"):
            with st.spinner("類似ブランド検索中..."):
                # ジャンルフィルタを適用して検索
                genre_filter = selected_genres if enable_genre_filter and selected_genres else None
                results = search_system.search_similar_brands(target_brand, top_k, min_similarity, genre_filter=genre_filter, normalize_similarity=normalize_similarity)
                
                # if enable_reranking and results:
                #     # 固定ブースト戦略または従来方式を選択
                #     if use_fixed_boost:
                #         results = search_system.apply_fixed_boost_reranking(
                #             results, target_brand, location_method
                #         )
                #     else:
                #         results = search_system.apply_location_reranking(
                #             results, target_brand, bias_strength, location_method, rerank_mode
                #         )
                
                st.session_state['search_results'] = results
                st.session_state['search_target'] = target_brand
                
                if results:
                    st.success(f"✅ {len(results)} 件の類似ブランドを発見")
                else:
                    st.warning("条件に一致するブランドが見つかりませんでした")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_addition_panel(search_system):
    """新ブランド追加パネル"""
    with st.container():
        st.markdown('<div class="search-card">', unsafe_allow_html=True)
        
        api_key = st.text_input(
            "🔑 GPT-OSS設定 (任意入力):",
            type="password",
            help="ローカルGPT-OSSサーバーを使用（APIキー不要）",
            key="add_api_key",
            placeholder="ローカル実行のため入力不要"
        )
        
        brand_name = st.text_input(
            "🏷️ ブランド名:", 
            placeholder="例: Stone Island, Fear of God",
            key="add_brand_name"
        )
        
        # フォールバック設定
        st.markdown("##### 🔧 説明文生成設定")
        fallback_mode = st.radio(
            "GPT-OSS接続失敗時の動作:",
            options=["template", "manual"],
            format_func=lambda x: "📝 テンプレート使用" if x == "template" else "✏️ 手動入力",
            index=0,
            key="add_fallback_mode",
            help="template: デフォルトテンプレートを使用, manual: 手動で説明文を入力"
        )
        
        # 重複チェック設定
        strict_duplicate_check = st.checkbox(
            "🔍 厳密な重複チェック",
            value=True,
            key="add_strict_check",
            help="有効: 完全一致のみチェック / 無効: 類似ブランド名もチェック"
        )
        
        # 手動入力用のテキストエリア（条件付き表示）
        manual_description = None
        if fallback_mode == "manual":
            manual_description = st.text_area(
                "✏️ ブランド説明文 (手動入力):",
                placeholder="ブランドの特徴、起源、スタイルなどを200-400文字程度で入力してください",
                height=120,
                key="add_manual_description"
            )
        
        # 新ブランド追加時のリランキング設定
        # st.markdown("##### 🏪 追加後リランキング設定")
        # with st.expander("リランキング詳細設定"):
        #     add_enable_rerank = st.checkbox("追加後に位置情報リランキングを適用", value=True, key="add_rerank_enable")
        #     if add_enable_rerank:
        #         col_add_bias, col_add_method = st.columns(2)
        #         with col_add_bias:
        #             # 推奨値に基づく調整
        #             add_bias_strength = st.slider("バイアス強度:", 0.001, 0.050, 0.015, 0.005, key="add_bias",
        #                                          help="推奨値: 0.005-0.025")
        #         with col_add_method:
        #             add_rerank_mode = st.selectbox(
        #                 "リランキング方式:",
        #                 ["weighted_average", "linear_addition", "location_rerank"],
        #                 index=0,
        #                 key="add_rerank_mode"
        #             )
        #         
        #         add_location_method = st.selectbox(
        #             "位置類似度計算方法:",
        #             ["comprehensive", "tenant", "building", "floor", "geographic", "area"],
        #             index=0,
        #             key="add_location_method"
        #         )
        #         
        #         add_use_fixed_boost = st.checkbox("🎯 固定ブースト戦略を使用", value=False, key="add_fixed_boost")
        #     else:
        #         add_bias_strength = 0.015  # 推奨値に変更
        #         add_rerank_mode = "weighted_average"  
        #         add_location_method = "comprehensive"
        #         add_use_fixed_boost = False
        add_enable_rerank = False
        add_bias_strength = 0.015
        add_rerank_mode = "weighted_average"
        add_location_method = "comprehensive"
        add_use_fixed_boost = False
        
        # ボタンの有効化判定
        button_disabled = not brand_name or (fallback_mode == "manual" and not manual_description)
        
        if st.button("🚀 ブランド追加", type="primary", disabled=button_disabled, key="add_execute"):
            
            # 既存ブランドチェック
            exists, existing_name = search_system.check_brand_exists(brand_name, strict=strict_duplicate_check)
            if exists:
                if existing_name == brand_name:
                    st.warning(f"⚠️ '{brand_name}' は既に登録されています。")
                else:
                    st.warning(f"⚠️ '{brand_name}' に類似するブランド '{existing_name}' が既に登録されています。")
                return
            
            description = None
            
            # 手動入力モードの場合
            if fallback_mode == "manual" and manual_description:
                description = manual_description.strip()
                st.markdown('<div class="new-brand-highlight">', unsafe_allow_html=True)
                st.success("✅ 手動入力の説明文を使用")
                with st.expander("📝 入力された説明文", expanded=True):
                    st.write(description)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Step 1: GPT-OSSで説明文生成（フォールバック機能付き）
                with st.spinner("GPT-OSSで説明文生成中..."):
                    description = search_system.generate_brand_description(api_key, brand_name, fallback_mode=fallback_mode)
                    
                if description:
                    st.markdown('<div class="new-brand-highlight">', unsafe_allow_html=True)
                    st.success("✅ 説明文生成完了")
                    with st.expander("📝 生成された説明文", expanded=True):
                        st.write(description)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Step 2: Ruri v3でベクトル化＆追加
                    with st.spinner("Ruri v3でベクトル化中..."):
                        success = search_system.add_new_brand(brand_name, description)
                    
                    if success:
                        st.balloons()
                        st.success(f"🎉 {brand_name} をシステムに追加しました！")
                        
                        # 即座に類似検索実行（リランキング適用）
                        similar_brands = search_system.search_similar_brands(brand_name, top_k=5)
                        
                        # 位置情報リランキング適用（新ブランド追加時）
                        # if add_enable_rerank and search_system.reranker and similar_brands:
                        #     if add_use_fixed_boost:
                        #         similar_brands = search_system.apply_fixed_boost_reranking(
                        #             similar_brands, brand_name, add_location_method
                        #         )
                        #     else:
                        #         similar_brands = search_system.apply_location_reranking(
                        #             similar_brands, brand_name, bias_strength=add_bias_strength, 
                        #             location_method=add_location_method, rerank_mode=add_rerank_mode
                        #         )
                        
                        st.session_state['add_results'] = similar_brands
                        st.session_state['added_brand'] = brand_name
                        
                        if similar_brands:
                            st.info(f"🔍 {brand_name} の類似ブランドを確認できます（下の結果エリアをご覧ください）")
                else:
                    st.error("説明文生成に失敗しました")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_unified_results(search_system):
    """統合結果表示エリア"""
    st.subheader("📊 検索・追加結果")
    
    # 検索結果表示
    if 'search_results' in st.session_state and st.session_state['search_results']:
        st.markdown("### 🔍 検索結果")
        
        results = st.session_state['search_results']
        target_brand = st.session_state.get('search_target', '')
        
        # 表形式表示
        display_data = []
        for i, result in enumerate(results, 1):
            brand_indicator = "🆕" if result.get('is_new', False) else ""
            
            # 計算式の明確化
            base_sim = result['similarity']
            final_sim = result.get('final_similarity', base_sim)
            boost = result.get('location_boost', 0.0)
            location_sim = result.get('location_similarity', 0.0)
            
            # 固定ブーストの場合の特別表示
            rerank_mode = result.get('rerank_mode', 'none')
            boost_info = ''
            if rerank_mode == 'fixed_boost':
                tier = result.get('boost_tier', '0%')
                boost_info = f"{tier} ({boost:+.3f})"
            else:
                boost_info = f"{boost:+.4f}"
            
            display_data.append({
                '順位': f"#{i}",
                'ブランド': f"{brand_indicator} {result['brand_name']}",
                '類似度': f"{final_sim:.4f}"
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # 詳細情報展開
        with st.expander("📋 詳細情報"):
            for i, result in enumerate(results[:5], 1):
                st.markdown(f"**{i}. {result['brand_name']}**")
                if result.get('normalized', False):
                    st.write(f"類似度: {result['similarity']:.4f} (正規化後)")
                    st.write(f"元の類似度: {result['original_similarity']:.4f}")
                else:
                    st.write(f"類似度: {result['similarity']:.4f}")
                st.write(f"説明: {result['description'][:200]}...")
                st.markdown("---")
    
    # 追加結果表示
    if 'add_results' in st.session_state and st.session_state['add_results']:
        st.markdown("### 🆕 新規追加ブランドの類似度")
        
        results = st.session_state['add_results']
        added_brand = st.session_state.get('added_brand', '')
        
        # 表形式表示
        display_data = []
        for i, result in enumerate(results, 1):
            brand_indicator = "🆕" if result.get('is_new', False) else ""
            
            # 計算式の明確化
            base_sim = result['similarity']
            final_sim = result.get('final_similarity', base_sim)
            boost = result.get('location_boost', 0.0)
            location_sim = result.get('location_similarity', 0.0)
            
            # 固定ブーストの場合の特別表示
            rerank_mode = result.get('rerank_mode', 'none')
            boost_info = ''
            if rerank_mode == 'fixed_boost':
                tier = result.get('boost_tier', '0%')
                boost_info = f"{tier} ({boost:+.3f})"
            else:
                boost_info = f"{boost:+.4f}"
            
            display_data.append({
                '順位': f"#{i}",
                'ブランド': f"{brand_indicator} {result['brand_name']}",
                '類似度': f"{final_sim:.4f}"
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # 詳細情報展開
        with st.expander("📋 詳細情報"):
            for i, result in enumerate(results, 1):
                st.markdown(f"**{i}. {result['brand_name']}**")
                if result.get('normalized', False):
                    st.write(f"類似度: {result['similarity']:.4f} (正規化後)")
                    st.write(f"元の類似度: {result['original_similarity']:.4f}")
                else:
                    st.write(f"類似度: {result['similarity']:.4f}")
                st.write(f"説明: {result['description'][:200]}...")
                st.markdown("---")
    

def show_system_status(search_system):
    """システム状態表示"""
    st.header("📊 システム情報")
    
    # 統計情報
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総ブランド数（デフォルト表示）", len(search_system.brand_names))
    with col2:
        st.metric("新規追加数", len(search_system.added_brands))
    with col3:
        if search_system.embeddings is not None:
            st.metric("埋め込み次元", search_system.embeddings.shape[1])
    with col4:
        # reranker_status = "有効" if search_system.reranker else "無効"
        # st.metric("リランキング", reranker_status)
        pass
    
    # 分析結果サマリーを追加
    st.subheader("📊 分析結果サマリー")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎯 **推奨ブースト値**: 0.005-0.025")
    with col2:
        st.info("🔗 **ブランドマッピング**: 自動部分一致")
    with col3:
        st.info("⚙️ **固定ブースト**: 位置一致率ベース")
    
    # ベクトル辞書の詳細情報
    st.subheader("📚 ベクトル辞書情報")
    col1, col2 = st.columns(2)
    
    with col1:
        if search_system.embeddings is not None:
            st.write(f"**辞書サイズ**: {search_system.embeddings.shape}")
            st.write(f"**総ベクトル数**: {len(search_system.brand_names)}")
            st.write(f"**次元数**: {search_system.embeddings.shape[1] if len(search_system.embeddings.shape) > 1 else 'N/A'}")
        else:
            st.write("ベクトル辞書が読み込まれていません")
    
    with col2:
        if search_system.brand_names:
            st.write(f"**最新ブランド**: {search_system.brand_names[-1] if search_system.brand_names else 'N/A'}")
            st.write(f"**最古ブランド**: {search_system.brand_names[0] if search_system.brand_names else 'N/A'}")
            new_brands_count = len(search_system.added_brands)
            st.write(f"**今回セッション追加数**: {new_brands_count}")

    # 新規追加ブランド一覧
    if search_system.added_brands:
        st.subheader("🆕 新規追加ブランド（今回セッション）")
        for i, added in enumerate(search_system.added_brands, 1):
            with st.expander(f"**{i}. {added['name']}** (インデックス: {added['index']})"):
                st.write(f"**説明:** {added['description']}")
                st.write(f"**ベクトル辞書インデックス:** {added['index']}")
                st.write(f"**処理サイクル:** ✅ 完了")
    
    # システム設定情報
    st.subheader("⚙️ システム設定")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**モデル情報:**")
        st.write("- 埋め込みモデル: Ruri v3 (cl-nagoya/ruri-v3-310m)")
        st.write("- 説明文生成: GPT-OSS 20B (ローカル)")
        st.write("- 次元削減: UMAP")
        
    with col2:
        st.write("**機能状態:**")
        st.write(f"- ベースアナライザー: {'✅' if search_system.base_analyzer else '❌'}")
        st.write(f"- Ruriモデル: {'✅' if search_system.ruri_model else '❌'}")
        # GPT-OSS接続状態をチェック
        server_status = search_system.check_gpt_oss_connection()
        st.write(f"- GPT-OSSサーバー: {'✅' if server_status else '❌'}")
        
        if not server_status:
            st.info("📝 フォールバック機能により、テンプレートまたは手動入力で説明文生成可能")

def show_brand_addition(search_system):
    """新ブランド追加機能"""
    st.header("➕ 新ブランド追加")
    
    with st.container():
        st.markdown('<div class="search-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            api_key = st.text_input(
                "🔑 GPT-OSS設定 (任意入力):",
                type="password",
                help="ローカルGPT-OSSサーバーを使用（APIキー不要）",
                placeholder="ローカル実行のため入力不要"
            )
        
        with col2:
            brand_name = st.text_input(
                "🏷️ ブランド名:", 
                placeholder="例: Stone Island, Fear of God"
            )
        
        if st.button("🚀 ブランド追加", type="primary", disabled=not brand_name):
            
            # Step 1: GPT-OSSで説明文生成（フォールバック機能付き）
            with st.spinner("GPT-OSSで説明文生成中..."):
                description = search_system.generate_brand_description(api_key, brand_name, fallback_mode="template")
                
            if description:
                st.markdown('<div class="new-brand-highlight">', unsafe_allow_html=True)
                st.success("✅ 説明文生成完了")
                st.write("**生成された説明文:**")
                st.write(description)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Step 2: Ruri v3でベクトル化＆追加
                with st.spinner("Ruri v3でベクトル化中..."):
                    success = search_system.add_new_brand(brand_name, description)
                
                if success:
                    st.balloons()
                    st.success(f"🎉 {brand_name} をシステムに追加しました！")
                    
                    # 即座に類似検索実行（リランキング適用）
                    st.info("🔍 追加したブランドの類似度を確認...")
                    similar_brands = search_system.search_similar_brands(brand_name, top_k=5)
                    
                    # 位置情報リランキング適用
                    # if search_system.reranker and similar_brands:
                    #     similar_brands = search_system.apply_location_reranking(
                    #         similar_brands, brand_name, bias_strength=0.3
                    #     )
                    
                    if similar_brands:
                        st.write("**類似ブランド:**")
                        for i, result in enumerate(similar_brands, 1):
                            brand_indicator = "🆕" if result.get('is_new', False) else ""
                            final_sim = result.get('final_similarity', result['similarity'])
                            boost = result.get('location_boost', 0)
                            if boost > 0.001:
                                st.write(f"{i}. {brand_indicator} {result['brand_name']} (類似度: {result['similarity']:.4f})")
            else:
                st.error("説明文生成に失敗しました")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_brand_search(search_system):
    """ブランド検索機能"""
    st.header("🔍 ブランド類似度検索")
    
    if not search_system.brand_names:
        st.warning("検索可能なブランドがありません")
        return
    
    with st.container():
        st.markdown('<div class="search-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            target_brand = st.selectbox(
                "🎯 検索するブランド:",
                search_system.brand_names
            )
        
        with col2:
            top_k = st.slider("表示件数:", 5, 20, 10)
        
        with col3:
            min_similarity = st.slider("最小類似度:", 0.0, 1.0, 0.1, 0.05)
        
        # 正規化オプション
        normalize_similarity_simple = st.checkbox("📊 類似度をギャラリ内で正規化", value=False)
        
        # リランキング設定
        # enable_reranking = st.checkbox("🏪 位置情報リランキング", value=True)
        # if enable_reranking:
        #     bias_strength = st.slider("バイアス強度:", 0.0, 1.0, 0.3, 0.1)
        enable_reranking = False
        
        if st.button("🚀 検索実行", type="primary"):
            with st.spinner("類似ブランド検索中..."):
                # 基本検索
                results = search_system.search_similar_brands(target_brand, top_k, min_similarity, normalize_similarity=normalize_similarity_simple)
                
                # リランキング適用
                # if enable_reranking and results:
                #     results = search_system.apply_location_reranking(
                #         results, target_brand, bias_strength
                #     )
                
                # 結果表示
                if results:
                    st.success(f"✅ {len(results)} 件の類似ブランドを発見")
                    
                    # 表形式表示
                    display_data = []
                    for i, result in enumerate(results, 1):
                        brand_indicator = "🆕" if result.get('is_new', False) else ""
                        display_data.append({
                            '順位': f"#{i}",
                            'ブランド': f"{brand_indicator} {result['brand_name']}",
                            '類似度': f"{result['similarity']:.4f}"
                        })
                    
                    df_display = pd.DataFrame(display_data)
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # 詳細情報展開
                    with st.expander("📋 詳細情報"):
                        for i, result in enumerate(results[:5], 1):
                            st.markdown(f"**{i}. {result['brand_name']}**")
                            if result.get('normalized', False):
                                st.write(f"類似度: {result['similarity']:.4f} (正規化後)")
                                st.write(f"元の類似度: {result['original_similarity']:.4f}")
                            else:
                                st.write(f"類似度: {result['similarity']:.4f}")
                            st.write(f"説明: {result['description'][:200]}...")
                            st.markdown("---")
                else:
                    st.warning("条件に一致するブランドが見つかりませんでした")
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_advanced_visualization(search_system):
    """特徴量空間解析"""
    st.header("🧠 特徴量空間解析")
    
    if not search_system.dimensionality_reducer:
        st.error("❌ 次元削減システムが利用できません")
        st.info("📝 ヒント: まず「ブランド検索 & 追加」タブでブランドデータを読み込んでください")
        return
    
    # サイドバーで設定
    st.sidebar.header("⚙️ 解析設定")
    
    # 次元削減手法選択
    available_methods = search_system.dimensionality_reducer.get_available_methods()
    selected_methods = st.sidebar.multiselect(
        "🔬 次元削減手法:",
        available_methods,
        default=['PCA', 'UMAP', 't-SNE']  # 軽量な手法をデフォルトに
    )
    
    if not selected_methods:
        st.sidebar.warning("⚠️ 少なくとも1つの手法を選択してください")
    
    # クラスタリング手法選択
    available_clustering = search_system.dimensionality_reducer.get_available_clustering()
    clustering_method = st.sidebar.selectbox(
        "🎯 クラスタリング手法:",
        available_clustering,
        index=0
    )
    
    # パラメータ設定
    st.sidebar.subheader("📊 パラメータ")
    
    # クラスター数設定
    if clustering_method in ['KMeans', 'Hierarchical', 'GMM']:
        n_clusters = st.sidebar.slider("クラスター数:", 2, 15, 5)
    else:
        n_clusters = 5
    
    # DBSCAN固有パラメータ
    if clustering_method == 'DBSCAN':
        eps = st.sidebar.slider("ε (eps):", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("最小サンプル数:", 2, 20, 5)
        clustering_kwargs = {'eps': eps, 'min_samples': min_samples}
    else:
        clustering_kwargs = {}
    
    # 可視化オプション
    st.sidebar.subheader("🎨 可視化オプション")
    show_brand_names = st.sidebar.checkbox("ブランド名表示", value=False)
    
    # 解析タイプ選択
    analysis_type = st.selectbox(
        "解析タイプ:",
        ["🎨 基本可視化", "🎯 ジャンル中心ベクトル", "📊 比較可視化", "🎯 単一手法詳細解析", "📈 全手法一括解析"]
    )
    
    if analysis_type == "🎨 基本可視化":
        show_basic_visualization(search_system)
    elif analysis_type == "🎯 ジャンル中心ベクトル":
        show_genre_center_visualization(search_system)
    elif analysis_type == "📊 比較可視化":
        show_comparison_analysis(search_system, selected_methods, clustering_method, 
                               n_clusters, clustering_kwargs, show_brand_names)
    elif analysis_type == "🎯 単一手法詳細解析":
        show_single_method_analysis(search_system, selected_methods, clustering_method,
                                  n_clusters, clustering_kwargs)
    elif analysis_type == "📈 全手法一括解析":
        show_comprehensive_analysis(search_system, clustering_method, n_clusters, clustering_kwargs)

def show_basic_visualization(search_system):
    """基本的な特徴量空間可視化"""
    st.subheader("🎨 基本特徴量空間可視化")
    
    # データ情報表示
    total_brands = len(search_system.brand_names)
    st.info(f"📊 システム内全 **{total_brands}** ブランドの特徴量空間を可視化（デフォルト表示）")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 次元削減手法選択
        basic_method = st.selectbox(
            "🔬 次元削減手法:",
            ['UMAP', 'PCA', 't-SNE'],
            index=0
        )
    
    with col2:
        # ハイライト設定
        if search_system.brand_names:
            highlight_brands = st.multiselect(
                "🎯 ハイライトするブランド:",
                search_system.brand_names[:20],  # 最初の20ブランドのみ表示
                default=[]
            )
        else:
            highlight_brands = []
    
    if st.button("🎨 可視化生成", type="primary"):
        with st.spinner(f"{basic_method}次元削減実行中..."):
            if basic_method == 'UMAP':
                fig = search_system.create_embedding_visualization(highlight_brands)
            else:
                # 他の手法も対応
                search_system.dimensionality_reducer.apply_selected_methods([basic_method])
                fig = search_system.dimensionality_reducer.create_comparison_visualization_for_streamlit(show_brand_names=False)
            
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # 統計情報
            st.subheader("📊 システム統計")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("総ブランド数（デフォルト表示）", len(search_system.brand_names))
            with col2:
                st.metric("新規追加数", len(search_system.added_brands))
            with col3:
                if search_system.embeddings is not None:
                    st.metric("埋め込み次元", search_system.embeddings.shape[1])
            with col4:
                # reranker_status = "有効" if search_system.reranker else "無効"
                # st.metric("リランキング", reranker_status)
                pass
            
            # 新規追加ブランド一覧
            if search_system.added_brands:
                st.subheader("🆕 新規追加ブランド")
                for added in search_system.added_brands:
                    st.markdown(f"- **{added['name']}**")
                    st.write(f"  {added['description'][:100]}...")

def show_genre_center_visualization(search_system):
    """ジャンル中心ベクトル可視化"""
    st.subheader("🎯 ジャンル中心ベクトル可視化")
    
    # ジャンル情報が利用可能かチェック
    if not search_system.genre_brands:
        st.warning("⚠️ ジャンル情報が利用できません。maps.csvが読み込まれているか確認してください。")
        return
    
    # ジャンル統計情報表示
    st.info(f"📊 利用可能ジャンル: {len(search_system.genre_brands)} 種類")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ジャンル選択
        available_genres = sorted(search_system.genre_brands.keys())
        
        # デフォルトでよく使われそうなジャンルを選択
        default_genres = []
        common_genres = ['womens', 'mens', 'accessary', 'goods', 'shoes', 'beauty', 'sports_and_outdoor']
        for genre in common_genres:
            if genre in available_genres:
                default_genres.append(genre)
        
        selected_genres = st.multiselect(
            "表示するジャンル（空=全て）:",
            available_genres,
            default=default_genres[:5] if default_genres else [],  # 最初の5つだけデフォルト選択
            help="選択したジャンルの中心ベクトルを特徴量空間上にプロット"
        )
        
        # 全ジャンル表示オプション
        show_all_genres = st.checkbox("全ジャンルを表示", value=False)
        if show_all_genres:
            selected_genres = available_genres
    
    with col2:
        # ジャンル統計
        st.write("**ジャンル別ブランド数（上位10）:**")
        genre_counts = {genre: len(brands) for genre, brands in search_system.genre_brands.items()}
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for genre, count in top_genres:
            st.write(f"- {genre}: {count}ブランド")
    
    if st.button("🎯 ジャンル中心ベクトル可視化実行", type="primary"):
        if not selected_genres and not show_all_genres:
            st.warning("表示するジャンルを選択してください")
            return
        
        with st.spinner("ジャンル中心ベクトル計算・可視化中..."):
            # 選択されたジャンルまたは全ジャンル
            target_genres = selected_genres if selected_genres else available_genres
            
            # 可視化実行
            fig = search_system.create_genre_center_visualization(target_genres)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # ジャンル詳細情報
            st.subheader("📊 表示ジャンル詳細")
            for genre in target_genres:
                if genre in search_system.genre_brands:
                    brand_count = len(search_system.genre_brands[genre])
                    with st.expander(f"🏷️ {genre} ({brand_count}ブランド)"):
                        brands = search_system.genre_brands[genre][:10]  # 最初の10ブランドのみ表示
                        st.write(", ".join(brands))
                        if len(search_system.genre_brands[genre]) > 10:
                            st.write(f"...他 {len(search_system.genre_brands[genre]) - 10} ブランド")
        else:
            st.error("可視化の生成に失敗しました")

def show_comparison_analysis(search_system, selected_methods, clustering_method, 
                           n_clusters, clustering_kwargs, show_brand_names):
    """比較可視化解析"""
    if not selected_methods:
        st.warning("最低1つの次元削減手法を選択してください")
        return
    
    # データ確認 - デフォルトで全244ブランドシステム（現在は約1769ブランド）を表示
    total_brands = len(search_system.brand_names)
    st.info(f"📊 システム内ブランド総数: **{total_brands}** ブランド（デフォルト表示対象）")
    st.write(f"🔬 選択手法: {', '.join(selected_methods)}")
    st.write(f"🎯 クラスタリング: {clustering_method}")
    
    if st.button("🚀 比較解析実行", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 次元削減実行
            status_text.text("🔄 次元削減を実行中...")
            progress_bar.progress(30)
            
            results = search_system.dimensionality_reducer.apply_selected_methods(
                methods=selected_methods
            )
            
            if not results:
                st.error("❌ 次元削減に失敗しました")
                return
            
            progress_bar.progress(60)
            status_text.text("🎯 クラスタリングを実行中...")
            
            # クラスタリング適用
            clusters = search_system.dimensionality_reducer.apply_clustering(
                method=clustering_method, 
                n_clusters=n_clusters, 
                **clustering_kwargs
            )
            
            if clusters is None:
                st.error("❌ クラスタリングに失敗しました")
                return
            
            progress_bar.progress(90)
            status_text.text("🎨 可視化を作成中...")
        
        except Exception as e:
            st.error(f"❌ 処理中にエラー: {e}")
            return
        
        # 可視化作成・表示
        try:
            # 可視化オブジェクトを直接取得
            fig = search_system.dimensionality_reducer.create_comparison_visualization_for_streamlit(
                clusters=clusters,
                show_brand_names=show_brand_names
            )
            
            progress_bar.progress(100)
            status_text.text("✅ 完了!")
            
            # 結果表示
            st.success(f"✅ {len(selected_methods)} 手法の比較可視化が完了しました（全{total_brands}ブランドをデフォルト表示）")
            
            # 可視化を直接表示
            if fig:
                st.subheader("📊 比較可視化結果")
                st.plotly_chart(fig, use_container_width=True)
                
                # 手法説明を追加
                with st.expander("🔬 手法の特徴"):
                    st.write("**PCA**: 線形次元削減、全体構造保持")
                    st.write("**t-SNE**: 非線形、局所構造強調、クラスタ分離に優秀")
                    st.write("**UMAP**: 局所・大域バランス、高速処理")
                    st.write("**MDS**: 距離関係保持、大域構造重視")
                    st.write("**Anchor-UMAP**: アンカー点による安定化")
            else:
                st.error("❌ 可視化の生成に失敗しました")
            
            # クラスター分析
            show_cluster_analysis(clusters, search_system.brand_names, clustering_method)
            
        except Exception as e:
            st.error(f"❌ 可視化作成中にエラー: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()

def show_single_method_analysis(search_system, selected_methods, clustering_method,
                               n_clusters, clustering_kwargs):
    """単一手法詳細解析"""
    if not selected_methods:
        st.warning("詳細解析する手法を1つ選択してください")
        return
    
    target_method = st.selectbox("詳細解析する手法:", selected_methods)
    
    if st.button("🎯 詳細解析実行", type="primary"):
        with st.spinner(f"{target_method} + {clustering_method} 詳細解析中..."):
            # 可視化オブジェクトを直接取得
            fig = search_system.dimensionality_reducer.create_single_method_visualization_for_streamlit(
                method=target_method,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                **clustering_kwargs
            )
        
        st.success(f"✅ {target_method} の詳細解析が完了しました")
        
        # 可視化を直接表示
        if fig:
            st.subheader(f"📊 {target_method} + {clustering_method} 詳細結果")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("可視化の生成に失敗しました")

def show_comprehensive_analysis(search_system, clustering_method, n_clusters, clustering_kwargs):
    """全手法一括解析"""
    # データ情報表示
    total_brands = len(search_system.brand_names)
    st.info(f"📊 全 **{total_brands}** ブランドに対して全手法一括解析を実行（デフォルト表示対象）")
    
    if st.button("📈 全手法一括解析実行", type="primary"):
        with st.spinner("全手法解析実行中..."):
            # 全手法適用
            results = search_system.dimensionality_reducer.apply_all_methods()
            
            if not results:
                st.error("次元削減に失敗しました")
                return
            
            # クラスタリング適用
            clusters = search_system.dimensionality_reducer.apply_clustering(
                method=clustering_method,
                n_clusters=n_clusters,
                **clustering_kwargs
            )
            
            if clusters is None:
                st.error("クラスタリングに失敗しました")
                return
            
            # 可視化作成・表示
            fig = search_system.dimensionality_reducer.create_comparison_visualization_for_streamlit(
                clusters=clusters
            )
        
        st.success(f"✅ 全手法一括解析が完了しました（全{total_brands}ブランドをデフォルト表示）")
        
        # 詳細統計
        st.subheader("📊 解析統計")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("実行手法数", len(results))
        with col2:
            st.metric("クラスター数", len(set(clusters)) if clusters is not None else 0)
        with col3:
            st.metric("総ブランド数（デフォルト表示）", len(search_system.brand_names))
        
        # 可視化を直接表示
        if fig:
            st.subheader("🎨 全手法比較可視化")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("可視化の生成に失敗しました")

def show_cluster_analysis(clusters, brand_names, clustering_method):
    """クラスター分析結果表示"""
    if clusters is None:
        return
    
    st.subheader(f"🎯 {clustering_method} クラスター分析")
    
    # クラスター統計
    unique_clusters = sorted(set(clusters))
    cluster_counts = {cluster: sum(1 for c in clusters if c == cluster) 
                     for cluster in unique_clusters}
    
    # クラスター統計表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**クラスター統計:**")
        for cluster, count in cluster_counts.items():
            st.write(f"クラスター {cluster}: {count} ブランド")
    
    with col2:
        # クラスター分布グラフ
        cluster_df = pd.DataFrame({
            'クラスター': list(cluster_counts.keys()),
            'ブランド数': list(cluster_counts.values())
        })
        fig = px.bar(cluster_df, x='クラスター', y='ブランド数', 
                    title="クラスター分布")
        st.plotly_chart(fig, use_container_width=True)
    
    # 各クラスターの詳細
    with st.expander("🔍 クラスター詳細"):
        for cluster in unique_clusters:
            # 安全にインデックスを取得（範囲チェック付き）
            cluster_brands = [brand_names[i] for i, c in enumerate(clusters) 
                            if c == cluster and i < len(brand_names)]
            st.write(f"**クラスター {cluster}** ({len(cluster_brands)} ブランド):")
            st.write(", ".join(cluster_brands))
            st.write("---")

def show_csv_batch_interface(search_system):
    """CSV一括追加インターフェース"""
    st.header("📂 CSV一括ブランド追加")
    st.markdown("**処理サイクル**: ブランド名検索 → 重複チェック → 説明文生成 → Ruriv3ベクトル化 → 辞書追加")
    
    # 処理フローの可視化
    st.markdown("""
    ```
    📝 CSVブランド名 → 🔍 既存辞書検索 → ❓ 重複チェック
    ↓ (重複なし)
    🤖 GPT-OSS説明文生成 → 🧠 Ruriv3ベクトル化 → 📚 ベクトル辞書追加
    ```
    """)
    
    # CSV形式説明
    with st.expander("📋 CSV形式について"):
        st.markdown("""
        **必要な列:**
        - `ブランド名`: 追加するブランドの名前
        - `説明文`: ブランドの説明文（オプション）
        
        **例:**
        ```csv
        ブランド名,説明文
        Stone Island,イタリアの高級ストリートウェアブランド...
        Fear of God,アメリカ発の高級ファッションブランド...
        ```
        
        **重複ブランドのスキップ機能:**
        - 既に登録済みのブランドは自動的にスキップされます
        - 厳密チェック: 完全一致のみ検出
        - 非厳密チェック: 類似ブランド名も検出（例: "Nike" と "NIKE"）
        
        **注意:**
        - 説明文の処理は選択したモードに応じて決まります
        - 「すべて新規生成」モードでは既存説明文を無視して全て新規生成します
        - ファイルサイズは10MB以下にしてください
        """)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "CSVファイルを選択",
        type=['csv'],
        help="ブランド名,説明文の形式のCSVファイル"
    )
    
    # 設定
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ 処理設定")
        
        api_key = st.text_input(
            "🔑 GPT-OSS設定 (ローカル実行):",
            type="password",
            help="ローカルGPT-OSSサーバーを使用",
            placeholder="ローカル実行のため入力不要"
        )
        
        description_mode = st.radio(
            "説明文生成モード:",
            options=["既存の説明文を優先", "空白のみ生成", "すべて新規生成"],
            index=0,
            help="既存優先: CSVに説明文があれば使用、空白なら生成 / 空白のみ: 空白の場合のみ生成 / すべて新規: 既存説明文を無視して全て新規生成"
        )
        
        strict_csv_check = st.checkbox(
            "🔍 厳密な重複チェック（CSV）",
            value=True,
            help="有効: 完全一致のみチェック / 無効: 類似ブランド名もチェック"
        )
        
    with col2:
        st.subheader("🎯 処理後の自動検索")
        
        auto_search = st.checkbox(
            "追加後に類似度検索を実行", 
            value=False,
            help="各ブランドの類似ブランドを自動検索"
        )
        
        if auto_search:
            search_count = st.slider("検索結果数:", 1, 10, 3)
    
    # CSVプレビュー
    if uploaded_file is not None:
        try:
            # CSVデータ読み込み
            df = pd.read_csv(uploaded_file)
            
            # 列名の正規化
            if 'brand_name' not in df.columns:
                if 'ブランド名' in df.columns:
                    df = df.rename(columns={'ブランド名': 'brand_name'})
                elif 'name' in df.columns:
                    df = df.rename(columns={'name': 'brand_name'})
                else:
                    st.error("❌ 'ブランド名', 'brand_name', または 'name' 列が見つかりません")
                    return
            
            if 'description' not in df.columns:
                if '説明文' in df.columns:
                    df = df.rename(columns={'説明文': 'description'})
                else:
                    df['description'] = ''  # 空の説明文列を追加
            
            st.subheader("📊 CSVプレビュー")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 統計情報
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("総ブランド数", len(df))
            with col2:
                has_desc = sum(1 for desc in df['description'] if desc and str(desc).strip())
                st.metric("説明文あり", has_desc)
            with col3:
                need_generation = len(df) - has_desc
                st.metric("生成が必要", need_generation)
            
            # 処理実行
            if st.button("🚀 一括処理実行", type="primary"):
                
                # データ準備
                csv_data = df.to_dict('records')
                
                # プログレスバーとステータス
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                def update_progress(current, total, message):
                    progress = current / total if total > 0 else 0
                    progress_bar.progress(progress)
                    status_text.text(f"{message} ({current}/{total})")
                
                # 処理モードの設定
                if description_mode == "既存の説明文を優先":
                    use_existing = True
                    force_regenerate = False
                elif description_mode == "空白のみ生成":
                    use_existing = False
                    force_regenerate = False
                else:  # "すべて新規生成"
                    use_existing = False
                    force_regenerate = True
                
                # 一括処理実行
                with st.spinner("CSV一括処理中..."):
                    results = search_system.process_csv_batch(
                        csv_data, 
                        api_key=api_key,
                        progress_callback=update_progress,
                        use_existing_descriptions=use_existing,
                        force_regenerate=force_regenerate,
                        strict_check=strict_csv_check
                    )
                
                # 結果表示
                progress_bar.progress(100)
                status_text.text("✅ 処理完了!")
                
                # 結果統計
                success_count = sum(1 for r in results if r['status'] == 'success')
                skipped_count = sum(1 for r in results if r['status'] == 'skipped')
                failed_count = sum(1 for r in results if r['status'] not in ['success', 'skipped'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("処理済み", len(results))
                with col2:
                    st.metric("成功", success_count, delta=success_count)
                with col3:
                    st.metric("スキップ", skipped_count, delta=skipped_count if skipped_count > 0 else None)
                with col4:
                    st.metric("失敗", failed_count, delta=-failed_count if failed_count > 0 else 0)
                
                # 結果詳細表示
                st.subheader("📊 処理結果")
                
                # 成功・失敗・スキップの分類表示
                if success_count > 0:
                    st.success(f"✅ {success_count} ブランドの処理サイクルが完了しました！")
                    st.markdown("**完了したサイクル**: ブランド名検索 → 重複チェック → 説明文生成 → Ruriv3ベクトル化 → 辞書追加")
                    
                    success_results = [r for r in results if r['status'] == 'success']
                    
                    # 成功結果の詳細表示用
                    display_success = []
                    for r in success_results:
                        display_success.append({
                            'ブランド名': r['brand_name'],
                            'ステータス': '✅ 完了',
                            '説明文生成': '🤖 AI生成' if r.get('generated', False) else '📝 既存使用',
                            'ベクトル形状': f"{r.get('embedding_shape', 'N/A')}",
                            'インデックス': r.get('vector_index', 'N/A'),
                            '説明文': r['description']
                        })
                    
                    success_df = pd.DataFrame(display_success)
                    st.dataframe(success_df, use_container_width=True, hide_index=True)
                
                if skipped_count > 0:
                    st.info(f"⏭️ {skipped_count} ブランドは既にベクトル辞書に存在するためスキップしました")
                    
                    skipped_results = [r for r in results if r['status'] == 'skipped']
                    
                    # スキップ結果の詳細表示用
                    display_skip = []
                    for r in skipped_results:
                        display_skip.append({
                            'ブランド名': r['brand_name'],
                            'ステータス': '⏭️ スキップ',
                            'ステップ': r.get('step', 'N/A'),
                            '理由': r.get('error', 'N/A')
                        })
                    
                    skipped_df = pd.DataFrame(display_skip)
                    st.dataframe(skipped_df, use_container_width=True, hide_index=True)
                
                if failed_count > 0:
                    st.error(f"❌ {failed_count} ブランドの処理サイクルに失敗しました")
                    
                    failed_results = [r for r in results if r['status'] not in ['success', 'skipped']]
                    
                    # 失敗結果の詳細表示用
                    display_failed = []
                    for r in failed_results:
                        display_failed.append({
                            'ブランド名': r['brand_name'],
                            'ステータス': '❌ 失敗',
                            '失敗ステップ': r.get('step', 'unknown'),
                            'エラー詳細': r.get('error', 'N/A'),
                            '説明文生成': '🤖 AI生成' if r.get('generated', False) else '📝 既存使用'
                        })
                    
                    failed_df = pd.DataFrame(display_failed)
                    st.dataframe(failed_df, use_container_width=True, hide_index=True)
                
                # 自動検索実行
                if auto_search and success_count > 0:
                    st.subheader("🔍 追加ブランドの類似度検索")
                    
                    search_results = {}
                    for result in results:
                        if result['status'] == 'success':
                            brand_name = result['brand_name']
                            similar = search_system.search_similar_brands(
                                brand_name, top_k=search_count, min_similarity=0.1
                            )
                            if similar:
                                search_results[brand_name] = similar
                    
                    # 検索結果表示
                    for brand_name, similar_brands in search_results.items():
                        with st.expander(f"🎯 {brand_name} の類似ブランド"):
                            for i, similar in enumerate(similar_brands, 1):
                                st.write(f"{i}. {similar['brand_name']} (類似度: {similar['similarity']:.3f})")
                
                # 清掃
                progress_bar.empty()
                status_text.empty()
                
        except Exception as e:
            st.error(f"❌ CSVファイル処理エラー: {e}")
            st.info("ファイル形式を確認してください。ヘッダーに'ブランド名'列が必要です。")

if __name__ == "__main__":
    main()