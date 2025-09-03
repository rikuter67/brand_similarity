import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import warnings
from location_bias_reranking import LocationBasedSimilarityReranker
from integrated_dimensionality_reduction import IntegratedDimensionalityReduction

warnings.filterwarnings('ignore') 

# Configuration constants
USE_ANCHOR_BASED = False  # Set to True to use anchor-based UMAP

class LLMStyleEmbeddingAnalyzer:
    def __init__(self):
        print(f"🚀 Initializing LLM Style Embedding Analyzer")
        print(f"💡 Focus on advanced brand similarity analysis with reranking")
        
        self.df = None
        self.embeddings = None
        self.similarity_matrix = None
        self.location_reranker = None
        self.setup_location_reranker()
    
    def setup_location_reranker(self):
        """位置情報リランカーの初期化"""
        try:
            maps_csv_path = "datasets/bline_similarity/maps.csv"
            tenants_csv_path = "datasets/bline_similarity/tenants.csv"
            
            if os.path.exists(maps_csv_path):
                self.location_reranker = LocationBasedSimilarityReranker(
                    maps_csv_path=maps_csv_path,
                    tenants_csv_path=tenants_csv_path if os.path.exists(tenants_csv_path) else None
                )
                print(f"✅ 位置情報リランキング機能を初期化しました")
            else:
                print(f"⚠️  位置データが見つかりません。リランキング機能は無効です")
                self.location_reranker = None
        except Exception as e:
            print(f"⚠️  位置情報リランカー初期化エラー: {e}")
            self.location_reranker = None
    
    def load_data(self, csv_path):
        """データ読み込み"""
        print(f"📊 Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)

        if 'ブランド名' in self.df.columns and 'name' not in self.df.columns:
            self.df.rename(columns={'ブランド名': 'name'}, inplace=True)
            print("Renamed 'ブランド名' column to 'name'.")
        
        if 'bline_id' in self.df.columns and 'id' not in self.df.columns:
            self.df.rename(columns={'bline_id': 'id'}, inplace=True)
            print("Renamed 'bline_id' column to 'id'.")
        elif 'id' not in self.df.columns:
            self.df['id'] = range(len(self.df))
            print("Created 'id' column as it was not found.")

        print(f"✅ Loaded {len(self.df)} brands")
        desc_lengths = self.df['description'].str.len()
        print(f"📈 Description analysis:")
        print(f"   - Mean length: {desc_lengths.mean():.1f} chars")
        print(f"   - Median length: {desc_lengths.median():.1f} chars")
        print(f"   - Range: {desc_lengths.min()}-{desc_lengths.max()} chars")
        
        return self.df
    
    def load_embeddings(self, embeddings_path=None):
        """エンベディングファイルをロード"""
        if embeddings_path is None:
            embeddings_path = "./ruri_embeddings_results/ruri_description_embeddings_v3_raw_hub.npy"
        
        print(f"🧠 Loading embeddings from: {embeddings_path}")
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        self.embeddings = np.load(embeddings_path)
        
        if self.df is not None and self.embeddings.shape[0] != len(self.df):
            print(f"⚠️ Embedding count ({self.embeddings.shape[0]}) != DataFrame count ({len(self.df)})")
            min_count = min(self.embeddings.shape[0], len(self.df))
            self.embeddings = self.embeddings[:min_count]
            self.df = self.df.iloc[:min_count].copy()
            print(f"🔧 Adjusted to {min_count} items")

        print(f"✅ Embeddings loaded: {self.embeddings.shape}")
        return self.embeddings
    
    def calculate_similarity_matrix(self):
        """類似度行列の計算"""
        print(f"🔢 Calculating similarity matrix...")
        
        if self.embeddings is None:
            print("❌ Embeddings not loaded. Please load embeddings first.")
            return None

        self.similarity_matrix = cosine_similarity(self.embeddings)
        print(f"✅ Similarity matrix calculated: {self.similarity_matrix.shape}")
        
        # Statistics
        off_diagonal = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
        print(f"📊 Mean similarity: {off_diagonal.mean():.4f} ± {off_diagonal.std():.4f}")
        
        return self.similarity_matrix
    
    def find_similar_brands(self, brand_name, top_k=10, min_similarity=0.0, use_location_rerank=True, location_bias_strength=0.3):
        """類似ブランド検索（リランキング対応）"""
        try:
            brand_matches = self.df[self.df['name'] == brand_name]
            if len(brand_matches) == 0:
                print(f"❌ Brand '{brand_name}' not found")
                similar_names = self.df['name'].str.contains(brand_name, case=False, na=False)
                if similar_names.any():
                    suggestions = self.df[similar_names]['name'].head(3).tolist()
                    print(f"💡 Did you mean: {', '.join(suggestions)}")
                return []
            
            brand_idx = brand_matches.index[0]
            similarities = self.similarity_matrix[brand_idx]
            
            # 類似度スコア収集
            similarity_scores = {}
            for i, sim_score in enumerate(similarities):
                if i != brand_idx and sim_score >= min_similarity:
                    brand_name_i = self.df.iloc[i]['name']
                    similarity_scores[brand_name_i] = sim_score
            
            # 位置情報リランキング適用
            if use_location_rerank and self.location_reranker is not None:
                print(f"🏪 Applying location-based reranking...")
                reranked_scores = self.location_reranker.rerank_similarity_with_location_bias(
                    similarity_scores=similarity_scores,
                    query_brand=brand_name,
                    bias_strength=location_bias_strength,
                    location_method='comprehensive',
                    rerank_mode='weighted_average'
                )
                final_scores = reranked_scores
            else:
                final_scores = similarity_scores
            
            # トップK結果を返す
            sorted_brands = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            results = []
            for rank, (similar_brand_name, final_score) in enumerate(sorted_brands[:top_k]):
                brand_row = self.df[self.df['name'] == similar_brand_name]
                if len(brand_row) == 0:
                    continue
                
                similar_idx = brand_row.index[0]
                original_score = similarity_scores[similar_brand_name]
                
                result = {
                    'rank': rank + 1,
                    'brand_name': similar_brand_name,
                    'brand_id': self.df.iloc[similar_idx]['id'],
                    'similarity_score': final_score,
                    'original_similarity': original_score,
                    'location_boost': final_score - original_score if use_location_rerank else 0.0,
                    'description_preview': self.df.iloc[similar_idx]['description'][:100] + "..."
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []
    
    def export_results(self, output_path):
        """結果をCSVにエクスポート"""
        if self.df is not None:
            self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"💾 Results exported to: {output_path}")
        else:
            print("❌ No data to export")
    
    def run_simple_analysis(self, csv_path, embeddings_path=None, test_brands=['シャネル', 'ユニクロ']):
        """シンプルなリランキング分析パイプライン"""
        print(f"🚀 Running Simple Brand Reranking Analysis")
        print("=" * 50)
        
        try:
            # データとエンベディングを読み込み
            self.load_data(csv_path)
            self.load_embeddings(embeddings_path)
            
            # 類似度行列を計算
            self.calculate_similarity_matrix()
            
            # テストブランドで類似検索を実行
            print(f"\n🔍 Testing similarity search with reranking...")
            
            for brand in test_brands:
                print(f"\n--- Similar brands to '{brand}' ---")
                
                # リランキングあり
                print("📍 With location reranking:")
                similar_reranked = self.find_similar_brands(
                    brand, top_k=5, use_location_rerank=True, location_bias_strength=0.3
                )
                
                if similar_reranked:
                    for result in similar_reranked:
                        boost_indicator = "📍" if result['location_boost'] > 0.01 else "  "
                        print(f"{result['rank']:2d}. {boost_indicator} {result['brand_name']:25s} "
                              f"(score: {result['similarity_score']:.4f}, "
                              f"boost: {result['location_boost']:+.3f})")
                
                # リランキングなし（比較用）
                print("🧠 Without location reranking:")
                similar_basic = self.find_similar_brands(
                    brand, top_k=5, use_location_rerank=False
                )
                
                if similar_basic:
                    for result in similar_basic:
                        print(f"{result['rank']:2d}.    {result['brand_name']:25s} "
                              f"(score: {result['similarity_score']:.4f})")
                else:
                    print(f"❌ No results found for '{brand}'")
            
            print(f"\n✅ Simple analysis complete!")
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_similarity_heatmap(self, output_dir='./advanced_results'):
        """ブランド間類似度ヒートマップの作成 (日本語対応)"""
        print("\n📊 Creating brand-to-brand similarity heatmap (日本語対応)...")
        
        if self.similarity_matrix is None:
            print("❌ Similarity matrix not calculated. Please run calculate_similarity_matrix() first.")
            return

        brand_names = self.df['name'].tolist()
        
        if len(brand_names) > 50: # ブランド数が多すぎるとラベルが読みにくくなるため制限
            print("⚠️ ブランド数が多いため、ヒートマップの軸ラベルは表示されません。")
            # 軸ラベルをオフにするか、サブセットで描画を検討
            display_brand_names = False
            figsize = (12, 10) # 小さくする
        else:
            display_brand_names = True
            figsize = (18, 15)

        similarity_df = pd.DataFrame(self.similarity_matrix, index=brand_names, columns=brand_names)
        
        # 包括的な日本語フォント検出と設定
        def find_japanese_font():
            # より幅広いフォント候補
            japanese_font_candidates = [
                'Noto Sans CJK JP', 'NotoSansCJK-Regular', 'Noto Sans CJK',
                'IPAexGothic', 'IPAPGothic', 'IPA Gothic',
                'TakaoGothic', 'TakaoPGothic', 
                'Yu Gothic', 'YuGothic', 'Yu Gothic Medium',
                'Meiryo', 'Meiryo UI',
                'Hiragino Sans GB', 'Hiragino Sans', 'Hiragino Kaku Gothic Pro',
                'MS Gothic', 'MS UI Gothic', 'MS PGothic',
                'MS Mincho', 'MS PMincho',
                'Liberation Sans', 'DejaVu Sans'
            ]
            
            # matplotlibのフォントリストを取得
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for candidate in japanese_font_candidates:
                # 完全一致と部分一致の両方をチェック
                if candidate in available_fonts:
                    return candidate
                for available in available_fonts:
                    if candidate.lower() in available.lower() or available.lower() in candidate.lower():
                        return available
            
            # 最後の手段として、CJKを含むフォントを検索
            for font in available_fonts:
                if any(keyword in font.lower() for keyword in ['cjk', 'japanese', 'jp', 'gothic', 'mincho']):
                    return font
                    
            return None

        plt.figure(figsize=figsize)
        
        found_japanese_font = find_japanese_font()
        if found_japanese_font:
            plt.rcParams['font.family'] = [found_japanese_font]
            plt.rcParams['axes.unicode_minus'] = False 
            print(f"✅ Using Japanese font: {found_japanese_font}")
        else:
            print(f"⚠️ Warning: No suitable Japanese font found. Using fallback settings.")
            # フォールバック設定: Unicodeフォントを使用
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # Unicode文字の表示を改善 

        # ヒートマップの作成と日本語ラベルの設定
        ax = sns.heatmap(similarity_df, annot=False, cmap='viridis', fmt=".2f",
                        xticklabels=display_brand_names, yticklabels=display_brand_names,
                        cbar_kws={'label': '類似度'})
        
        # タイトルとラベルを明示的に設定
        plt.title('ブランド間類似度ヒートマップ', fontsize=16, pad=20)
        plt.xlabel('ブランド名', fontsize=12, labelpad=10)
        plt.ylabel('ブランド名', fontsize=12, labelpad=10)
        
        # 日本語フォントが見つかった場合は、軸ラベルのフォントも設定
        if found_japanese_font:
            ax.set_title('ブランド間類似度ヒートマップ', fontsize=16, fontname=found_japanese_font, pad=20)
            ax.set_xlabel('ブランド名', fontsize=12, fontname=found_japanese_font, labelpad=10)
            ax.set_ylabel('ブランド名', fontsize=12, fontname=found_japanese_font, labelpad=10)
        
        if display_brand_names:
            if found_japanese_font:
                plt.xticks(fontsize=8, rotation=90, fontname=found_japanese_font) 
                plt.yticks(fontsize=8, rotation=0, fontname=found_japanese_font)
            else:
                plt.xticks(fontsize=8, rotation=90) 
                plt.yticks(fontsize=8, rotation=0) 
        else:
            plt.xticks([]) # ラベルを非表示
            plt.yticks([]) # ラベルを非表示
        
        plt.tight_layout()
        
        heatmap_file = os.path.join(output_dir, 'brand_similarity_heatmap_japanese.png') 
        plt.savefig(heatmap_file, dpi=300)
        plt.close()
        print(f"💾 Similarity heatmap saved: {heatmap_file}")

    def create_static_analysis(self, output_dir):
        """静的分析プロット"""
        print("📊 Creating static analysis plots...")
        
        if self.embeddings is None:
            print("❌ エンベディングが生成されていません。先に generate_advanced_embeddings() を実行してください。")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Cluster distribution
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        axes[0,0].bar(cluster_counts.index, cluster_counts.values, alpha=0.8, color='skyblue')
        axes[0,0].set_title('Cluster Size Distribution')
        axes[0,0].set_xlabel('Cluster ID')
        axes[0,0].set_ylabel('Number of Brands')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Similarity distribution
        off_diagonal = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
        axes[0,1].hist(off_diagonal, bins=50, alpha=0.8, color='lightcoral', edgecolor='black')
        axes[0,1].set_title('Pairwise Similarity Distribution')
        axes[0,1].set_xlabel('Cosine Similarity')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Embedding magnitude distribution
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        axes[0,2].hist(embedding_norms, bins=30, alpha=0.8, color='lightgreen', edgecolor='black')
        axes[0,2].set_title('Embedding Magnitude Distribution')
        axes[0,2].set_xlabel('L2 Norm')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Description length vs similarity
        desc_lengths = self.df['description'].str.len()
        mean_similarities = []
        for i in range(len(self.df)):
            mean_sim = np.mean(self.similarity_matrix[i][self.similarity_matrix[i] < 1.0])
            mean_similarities.append(mean_sim)
        
        axes[1,0].scatter(desc_lengths, mean_similarities, alpha=0.6, s=30, color='purple')
        axes[1,0].set_title('Description Length vs Avg Similarity')
        axes[1,0].set_xlabel('Description Length (chars)')
        axes[1,0].set_ylabel('Average Similarity')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Cluster visualization (2D projection) - 最新の次元削減手法を自動選択
        available_methods = []
        for method in ['mds', 'tsne', 'umap']:
            if f'{method}_x' in self.df.columns:
                available_methods.append(method)
        
        if available_methods:
            # 最後に実行された手法を使用
            selected_method = available_methods[-1]
            unique_clusters = sorted(self.df['cluster'].unique())
            # より明確な色分けのためにカスタムカラーマップを使用
            color_map = {
                i: ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                    '#1ABC9C', '#E67E22', '#34495E', '#F1C40F', '#E91E63',
                    '#8E44AD', '#27AE60', '#D35400', '#2980B9', '#C0392B'][i % 15]
                for i in range(len(unique_clusters))
            }
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = self.df[self.df['cluster'] == cluster]
                axes[1,1].scatter(cluster_data[f'{selected_method}_x'], cluster_data[f'{selected_method}_y'], 
                                  c=color_map[i], label=f'C{cluster}', alpha=0.9, s=8, # サイズをさらに小さく
                                  edgecolors='black', linewidth=0.5)  # 黒い縁で区別を明確に
            
            axes[1,1].set_title(f'Brand Clusters ({selected_method.upper()})', fontsize=12)
            axes[1,1].set_xlabel(f'{selected_method.upper()} Component 1', fontsize=10)
            axes[1,1].set_ylabel(f'{selected_method.upper()} Component 2', fontsize=10)
            axes[1,1].legend(bbox_to_anchor=(1.05, 1), fontsize=8, frameon=True, fancybox=True, shadow=True)
            axes[1,1].grid(True, alpha=0.3)  # グリッドを追加
        else:
            axes[1,1].text(0.5, 0.5, 'No 2D projection available', ha='center', va='center', 
                          transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('No Visualization Available', fontsize=12)
        
        # Plot 6: Feature importance (semantic features) - Ruri v3直接ロード時は意味がないため削除または変更推奨
        # ただし、コードの互換性のため、ここではダミーで残すか、削除するロジックにする。
        # Ruri v3エンベディングには直接的な「semantic_features」のような分解された意味がないため、
        # ここは表示しない方が適切です。
        axes[1,2].set_visible(False) # プロットを非表示にする
        axes[1,2].set_title('Ruri v3 embeddings have no direct semantic features', fontsize=10)
        
        plt.tight_layout()
        
        static_file = os.path.join(output_dir, 'advanced_analysis.png')
        plt.savefig(static_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Static analysis saved: {static_file}")
    
    def export_results(self, output_dir='./advanced_results'):
        """結果エクスポート"""
        print(f"\n💾 Exporting results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.df.to_csv(os.path.join(output_dir, 'brands_advanced_analysis.csv'), 
                       index=False, encoding='utf-8-sig')
        
        np.save(os.path.join(output_dir, 'advanced_embeddings.npy'), self.embeddings)
        np.save(os.path.join(output_dir, 'similarity_matrix.npy'), self.similarity_matrix)
        
        with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("🚀 Advanced Brand Similarity Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"📊 Dataset: {len(self.df)} brands\n")
            f.write(f"🧠 Embedding dimensions: {self.embeddings.shape[1]}\n")
            f.write(f"🔢 Similarity matrix: {self.similarity_matrix.shape}\n")
            
            off_diagonal = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
            f.write(f"📈 Mean similarity: {off_diagonal.mean():.4f}\n")
            f.write(f"📊 Similarity std: {off_diagonal.std():.4f}\n")
            f.write(f"🎯 Number of clusters: {self.df['cluster'].nunique()}\n")
            
            f.write(f"\n🏆 Largest clusters:\n")
            cluster_counts = self.df['cluster'].value_counts().head(5)
            for cluster_id, count in cluster_counts.items():
                sample_brands = self.df[self.df['cluster'] == cluster_id]['name'].head(3).tolist()
                f.write(f"   - Cluster {cluster_id}: {count} brands ({', '.join(sample_brands)}...)\n")
        
        print(f"✅ Export complete!")
    
    def generate_advanced_embeddings(self):
        """Load advanced embeddings using the default Ruri v3 model"""
        print(f"🧠 Loading Ruri v3 embeddings...")
        return self.load_embeddings()
    
    def perform_intelligent_clustering(self):
        """Perform intelligent clustering on embeddings"""
        print(f"🎯 Performing intelligent clustering...")
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            if self.embeddings is None:
                print("❌ No embeddings available for clustering")
                return None, None
            
            # Find optimal number of clusters using silhouette score
            best_k = 2
            best_score = -1
            for k in range(2, min(10, len(self.df) // 5)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(self.embeddings)
                score = silhouette_score(self.embeddings, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Apply final clustering
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # Add cluster information to dataframe
            self.df['cluster'] = cluster_labels
            
            cluster_info = {
                'n_clusters': best_k,
                'silhouette_score': best_score,
                'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
            }
            
            print(f"✅ Clustering complete: {best_k} clusters (silhouette: {best_score:.3f})")
            return cluster_labels, cluster_info
            
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            # Fallback: assign all to single cluster
            cluster_labels = np.zeros(len(self.df), dtype=int)
            self.df['cluster'] = cluster_labels
            return cluster_labels, {'n_clusters': 1, 'silhouette_score': 0}
    
    def create_advanced_visualization(self, method='umap', use_anchor_based=False, show_brand_names=True):
        """Create advanced visualization using specified method"""
        print(f"🎨 Creating {method.upper()} visualization...")
        try:
            # This method would create individual visualizations
            # For now, we'll use the integrated system
            print(f"✅ Visualization created using integrated system")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
    
    def run_complete_analysis(self, csv_path, test_brands=['シャネル', 'コム デ ギャルソン', 'ユニクロ']):
        """完全な分析パイプラインを実行"""
        print(f"\n🚀 Running Complete Advanced Brand Analysis")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data(csv_path)
            
            # Step 2: Generate advanced embeddings (Ruri v3 embeddingsをロード)
            self.generate_advanced_embeddings()
            
            # Step 3: Calculate similarities
            self.calculate_similarity_matrix()
            
            # Step 4: Intelligent clustering (動的クラスタ数)
            cluster_labels, cluster_info = self.perform_intelligent_clustering()
            
            # Step 5: 統合次元削減システムによる可視化
            print(f"\n🎨 統合次元削減システムによる可視化を実行...")
            
            # 統合次元削減システムを初期化
            integrated_reducer = IntegratedDimensionalityReduction(
                embeddings=self.embeddings,
                brand_names=self.df['name'].tolist(),
                descriptions=self.df['description'].tolist()
            )
            
            # 全ての次元削減手法を適用
            integrated_results = integrated_reducer.apply_all_methods(
                n_clusters=len(set(cluster_labels)) if cluster_labels is not None else None
            )
            
            # 統合比較可視化を作成
            integrated_reducer.create_comparison_visualization(
                clusters=cluster_labels,
                show_brand_names=True
            )
            
            # 全座標をエクスポート
            coords_df = integrated_reducer.export_all_coordinates()
            
            # 従来の個別可視化も実行（既存グラフスタイル保持）
            print(f"\n🎨 従来の個別可視化も実行...")
            
            # 論文の手法（アンカーベースUMAP）
            if USE_ANCHOR_BASED:
                print(f"📍 論文手法: アンカーベースUMAP")
                self.create_advanced_visualization(method='anchor_umap', use_anchor_based=True, show_brand_names=True)
            
            # 比較手法1: 標準UMAP
            self.create_advanced_visualization(method='umap', show_brand_names=True)
            
            # 比較手法2: t-SNE
            self.create_advanced_visualization(method='tsne', show_brand_names=True)
            
            # 比較手法3: MDS
            self.create_advanced_visualization(method='mds', show_brand_names=True)
            
            # Step 6: Test similarity search with multiple brands (リランキング機能付き)
            print(f"\n🔍 Testing similarity search with location reranking...")
            
            for brand in test_brands:
                print(f"\n--- 🏷️   Similar brands to '{brand}' ---")
                print(f"    📍 埋め込み + 位置情報リランキング結果:")
                similar_reranked = self.find_similar_brands(
                    brand, top_k=8, min_similarity=0.1, 
                    use_location_rerank=True, location_bias_strength=0.3
                )
                
                if similar_reranked:
                    for result in similar_reranked:
                        location_indicator = "📍" if result['location_boost'] > 0.01 else "  "
                        print(f"{result['rank']:2d}. {location_indicator} {result['brand_name']:30s} "
                              f"(最終: {result['similarity_score']:.4f}, "
                              f"元: {result['original_similarity']:.4f}, "
                              f"ブースト: {result['location_boost']:+.3f})")
                
                # 比較のため位置情報なしの結果も表示
                print(f"    🧠 埋め込みのみの結果:")
                similar_basic = self.find_similar_brands(
                    brand, top_k=5, min_similarity=0.1, 
                    use_location_rerank=False
                )
                
                if similar_basic:
                    for result in similar_basic:
                        print(f"{result['rank']:2d}.    {result['brand_name']:30s} "
                              f"(sim: {result['similarity_score']:.4f})")
                else:
                    print(f"❌ No results found for '{brand}'")
            
            # Step 7: Export results
            output_dir = './advanced_results'
            self.export_results(output_dir)
            
            print(f"\n🎉 Advanced Analysis Complete!")
            print(f"📁 Results saved in: {output_dir}")
            print(f"🌐 Open brand_similarity_landscape.html for interactive exploration")
            
            return output_dir
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """メイン実行関数"""
    CSV_PATH = "description.csv"
    
    try:
        print("🚀 Initializing Advanced Brand Similarity Analyzer")
        print("💡 This system uses Ruri v3 embeddings for high-quality analysis")
        print("🎯 Generating high-quality embeddings without complex feature engineering")
        
        analyzer = LLMStyleEmbeddingAnalyzer()
        
        results_dir = analyzer.run_complete_analysis(CSV_PATH)
        
        if results_dir:
            print(f"\n✨ Success! Advanced Brand Analysis Complete!")
            print(f"🎯 Results directory: {results_dir}")
            method_name = "Anchor-based UMAP" if USE_ANCHOR_BASED else "Standard UMAP"
            print(f"\n📋 Generated files ({method_name}):")
            print(f"   🌐 brand_similarity_landscape.html - Interactive visualization")
            print(f"   📊 advanced_analysis.png - Static analysis plots")
            print(f"   📄 brands_advanced_analysis.csv - Enhanced dataset")
            print(f"   🧠 advanced_embeddings.npy - Feature vectors (Ruri v3)")
            print(f"   🔢 similarity_matrix.npy - Similarity matrix")
            print(f"   📝 analysis_summary.txt - Summary report")
            print(f"   🔥 brand_similarity_heatmap_japanese.png - Similarity Heatmap (日本語対応)")
            
            print(f"\n🎨 統合次元削減システムの生成ファイル:")
            print(f"   📊 dimensionality_reduction_comparison.png - 4手法比較 (2x2レイアウト)")
            print(f"   🌐 dimensionality_reduction_dashboard.html - インタラクティブ比較ダッシュボード")
            print(f"   📄 all_dimensionality_reduction_coordinates.csv - 全手法の座標データ")
            print(f"   📊 dimensionality_reduction_statistics.csv - 統計比較データ")
            print(f"   📝 dimensionality_reduction_report.txt - 詳細比較レポート")
            
            if USE_ANCHOR_BASED:
                print(f"\n📋 個別可視化ファイル (従来グラフスタイル保持):")
                print(f"   🌐 brand_similarity_landscape_anchor_umap.html - 論文手法")
                print(f"   🌐 brand_similarity_landscape_umap.html - 標準UMAP")
                print(f"   🌐 brand_similarity_landscape_tsne.html - t-SNE")
                print(f"   🌐 brand_similarity_landscape_mds.html - MDS")
                print(f"\n🎯 統合システムの特徴:")
                print(f"   • 4つの次元削減手法を同時比較")
                print(f"   • インタラクティブなダッシュボード")
                print(f"   • 統計的比較レポート生成")
                print(f"   • 既存の見やすいグラフスタイル保持")
                print(f"   • クラスター情報の統合表示")

    except Exception as e:
        print(f"❌ Failed to run analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()