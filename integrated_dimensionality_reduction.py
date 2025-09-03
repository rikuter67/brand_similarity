import pandas as pd
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tqdm import tqdm
import warnings
from anchor_based_embedding import AnchorBasedEmbedding

warnings.filterwarnings('ignore')

class IntegratedDimensionalityReduction:
    """
    統合次元削減システム
    MDS・t-SNE・UMAP・アンカーUMAPの4つの手法を同時に比較できるシステム
    """
    
    def __init__(self, embeddings, brand_names=None, descriptions=None):
        """
        初期化
        
        Args:
            embeddings: 高次元埋め込みデータ (N x D)
            brand_names: ブランド名のリスト
            descriptions: 説明文のリスト
        """
        self.embeddings = embeddings
        self.brand_names = brand_names if brand_names is not None else [f"Brand_{i}" for i in range(len(embeddings))]
        self.descriptions = descriptions
        self.results = {}
        self.clustering_results = {}
        
        # 利用可能な次元削減手法
        self.available_methods = {
            'PCA': self._apply_pca,
            'MDS': self._apply_mds,
            't-SNE': self._apply_tsne,
            'UMAP': self._apply_umap,
            'Anchor-UMAP': self._apply_anchor_umap
        }
        
        # 利用可能なクラスタリング手法
        self.available_clustering = {
            'KMeans': self._apply_kmeans,
            'DBSCAN': self._apply_dbscan,
            'Hierarchical': self._apply_hierarchical,
            'GMM': self._apply_gmm
        }
        
        # カラーパレット（視覚的に区別しやすい色）
        self.color_palette = [
            '#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF',
            '#00FFFF', '#FF00FF', '#800000', '#000080', '#008000',
            '#FFFF00', '#FF1493', '#1E90FF', '#32CD32', '#FF69B4',
            '#FFA500', '#9400D3', '#DC143C', '#00CED1', '#FFD700'
        ]
        
    def apply_selected_methods(self, methods=['PCA', 'UMAP', 't-SNE'], n_clusters=None, random_state=42):
        """
        選択した次元削減手法を適用
        
        Args:
            methods: 適用する手法のリスト
            n_clusters: クラスター数（アンカーUMAPで使用）
            random_state: 乱数シード
        """
        print("🚀 選択された次元削減手法を開始...")
        print(f"📊 データ形状: {self.embeddings.shape}")
        print(f"🎯 選択手法: {', '.join(methods)}")
        
        # 各手法を順次適用
        for method_name in methods:
            if method_name not in self.available_methods:
                print(f"⚠️ {method_name} は利用できません。利用可能: {list(self.available_methods.keys())}")
                continue
                
            print(f"\n🔄 {method_name} を実行中...")
            try:
                method_func = self.available_methods[method_name]
                coords_2d = method_func(random_state=random_state, n_clusters=n_clusters)
                self.results[method_name] = {
                    'coordinates': coords_2d,
                    'x': coords_2d[:, 0],
                    'y': coords_2d[:, 1]
                }
                print(f"✅ {method_name} 完了")
            except Exception as e:
                print(f"❌ {method_name} でエラー: {e}")
                continue
        
        print(f"\n🎉 選択手法完了! {len(self.results)} 手法で成功")
        return self.results
    
    def apply_all_methods(self, n_clusters=None, random_state=42):
        """
        全ての次元削減手法を適用（後方互換性のため）
        """
        return self.apply_selected_methods(
            methods=list(self.available_methods.keys()), 
            n_clusters=n_clusters, 
            random_state=random_state
        )
    
    def _apply_pca(self, random_state=42, **kwargs):
        """PCA（主成分分析）を適用"""
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(self.embeddings)
    
    def _apply_mds(self, random_state=42, **kwargs):
        """MDS（多次元尺度構成法）を適用"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # コサイン類似度から距離行列を計算
        similarity_matrix = cosine_similarity(self.embeddings)
        distance_matrix = 1 - similarity_matrix
        
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=random_state,
            max_iter=1000,
            n_init=4
        )
        
        return mds.fit_transform(distance_matrix)
    
    def _apply_tsne(self, random_state=42, **kwargs):
        """t-SNEを適用"""
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, self.embeddings.shape[0] - 1),
            random_state=random_state,
            metric='cosine',
            max_iter=1000,
            n_jobs=-1
        )
        
        return tsne.fit_transform(self.embeddings)
    
    def _apply_umap(self, random_state=42, **kwargs):
        """標準UMAPを適用"""
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, self.embeddings.shape[0] - 1),
            min_dist=0.05,
            spread=1.5,
            metric='cosine',
            random_state=random_state,
            n_epochs=500
        )
        
        return reducer.fit_transform(self.embeddings)
    
    def _apply_anchor_umap(self, random_state=42, n_clusters=None, **kwargs):
        """アンカーベースUMAPを適用"""
        n_anchors = max(5, min(20, self.embeddings.shape[0] // 8)) if n_clusters is None else n_clusters
        
        anchor_embedder = AnchorBasedEmbedding(
            n_anchors=n_anchors,
            lambda_anchor=0.1,
            random_state=random_state
        )
        
        return anchor_embedder.fit_transform(self.embeddings, self.brand_names)
    
    def apply_clustering(self, method='KMeans', n_clusters=5, **kwargs):
        """
        クラスタリングを適用
        
        Args:
            method: クラスタリング手法 ('KMeans', 'DBSCAN', 'Hierarchical', 'GMM')
            n_clusters: クラスター数
            **kwargs: 各手法固有のパラメータ
        """
        if method not in self.available_clustering:
            print(f"❌ {method} は利用できません。利用可能: {list(self.available_clustering.keys())}")
            return None
        
        print(f"🔄 {method} クラスタリングを実行中...")
        try:
            clustering_func = self.available_clustering[method]
            labels = clustering_func(n_clusters=n_clusters, **kwargs)
            self.clustering_results[method] = labels
            print(f"✅ {method} クラスタリング完了: {len(set(labels))} クラスター")
            return labels
        except Exception as e:
            print(f"❌ {method} クラスタリングでエラー: {e}")
            return None
    
    def _apply_kmeans(self, n_clusters=5, random_state=42, **kwargs):
        """K-Meansクラスタリング"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        return kmeans.fit_predict(self.embeddings)
    
    def _apply_dbscan(self, eps=0.5, min_samples=5, n_clusters=None, **kwargs):
        """DBSCANクラスタリング"""
        # DBSCANではn_clustersパラメータは使用しない
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(self.embeddings)
    
    def _apply_hierarchical(self, n_clusters=5, linkage='ward', **kwargs):
        """階層クラスタリング"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
        return hierarchical.fit_predict(self.embeddings)
    
    def _apply_gmm(self, n_clusters=5, random_state=42, **kwargs):
        """ガウス混合モデル"""
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, **kwargs)
        return gmm.fit_predict(self.embeddings)
    
    def create_comparison_visualization(self, output_dir='./integrated_results', 
                                      clusters=None, clustering_method=None, show_brand_names=True):
        """
        複数の手法を同時比較できる可視化を作成
        
        Args:
            output_dir: 出力ディレクトリ
            clusters: クラスターラベル（指定しない場合は自動的にクラスタリング実行）
            clustering_method: 使用するクラスタリング手法
            show_brand_names: ブランド名を表示するか
        """
        print("\n🎨 統合比較可視化を作成中...")
        
        if not self.results:
            print("❌ 先に apply_selected_methods() を実行してください")
            return
        
        # クラスターラベルを自動生成（指定されていない場合）
        if clusters is None and clustering_method:
            clusters = self.apply_clustering(method=clustering_method)
        elif clusters is None:
            # デフォルトでK-Meansを使用
            clusters = self.apply_clustering(method='KMeans', n_clusters=5)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 複数の手法を同時に表示するサブプロット
        self._create_subplot_comparison(output_dir, clusters, show_brand_names)
        
        # 2. インタラクティブな比較ダッシュボード
        self._create_interactive_comparison(output_dir, clusters)
        
        # 3. 統計的比較レポート
        self._create_statistical_comparison(output_dir)
        
        print(f"💾 統合可視化完了: {output_dir}")
    
    def _create_subplot_comparison(self, output_dir, clusters=None, show_brand_names=True):
        """動的サブプロットで複数の手法を比較"""
        methods = list(self.results.keys())
        n_methods = len(methods)
        
        # サブプロット数を縦並びに変更（各手法を縦に並べる）
        cols = 1  # 1列固定で縦並び
        rows = n_methods  # 手法数分の行
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))  # 横幅を最大化
        if rows == 1:
            axes = np.array([axes])  # 1つの手法の場合は配列にする
        # 縦並び（cols=1）の場合はaxesはすでに1次元配列
        
        fig.suptitle(f'次元削減手法の比較 ({", ".join(methods)})', fontsize=16, y=0.95)
        
        # 手法と位置のマッピング（縦並び用）
        method_positions = {}
        for i, method in enumerate(methods):
            row = i  # 縦並びなので行インデックスそのまま
            col = 0  # 1列固定なので常に0
            method_positions[method] = (row, col)
        
        for method_name, (row, col) in method_positions.items():
            if method_name not in self.results:
                ax = axes[row]  # 縦並びなので行インデックスのみ
                ax.text(0.5, 0.5, f'{method_name}\n利用不可', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            ax = axes[row]  # 縦並びなので行インデックスのみ
                
            coords = self.results[method_name]['coordinates']
            
            # クラスターがあれば色分け、なければ統一色
            if clusters is not None:
                unique_clusters = sorted(set(clusters))
                for i, cluster in enumerate(unique_clusters):
                    cluster_mask = np.array(clusters) == cluster
                    cluster_coords = coords[cluster_mask]
                    color = self.color_palette[i % len(self.color_palette)]
                    
                    ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                             c=color, alpha=0.7, s=50, label=f'Cluster {cluster}',
                             edgecolors='black', linewidth=0.5)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], 
                         c='steelblue', alpha=0.7, s=50,
                         edgecolors='black', linewidth=0.5)
            
            # ブランド名表示
            if show_brand_names and len(self.brand_names) <= 50:
                for i, name in enumerate(self.brand_names):
                    if i < len(coords):  # bounds check
                        ax.annotate(name, (coords[i, 0], coords[i, 1]), 
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.8)
            
            ax.set_title(f'{method_name}', fontsize=16, fontweight='bold')
            ax.set_xlabel('次元1', fontsize=12)
            ax.set_ylabel('次元2', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            if clusters is not None and len(set(clusters)) <= 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 縦並びの場合は全ての軸が使用されるので、未使用サブプロットの処理は不要
        # (rows = n_methods なので常に全ての軸が使用される)
        
        plt.tight_layout()
        
        # 保存
        comparison_file = os.path.join(output_dir, 'dimensionality_reduction_comparison.png')
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 サブプロット比較保存: {comparison_file}")
    
    def _create_interactive_comparison(self, output_dir, clusters=None):
        """インタラクティブな比較ダッシュボードを作成"""
        methods = list(self.results.keys())
        n_methods = len(methods)
        
        # 縦並びレイアウトに変更
        cols = 1  # 1列固定で縦並び
        rows = n_methods  # 手法数分の行
        
        subplot_titles = methods
        specs = [[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=specs
        )
        
        # 位置のマッピング（縦並び用）
        positions = []
        for i in range(n_methods):
            row = i + 1  # Plotlyは1始まり、縦並びなので行インデックスそのまま
            col = 1  # 1列固定
            positions.append((row, col))
        
        for i, method_name in enumerate(methods):
            if method_name not in self.results:
                continue
                
            row, col = positions[i]
            coords = self.results[method_name]['coordinates']
            
            # ホバー情報を準備
            hover_text = []
            for j, name in enumerate(self.brand_names):
                if j < len(coords):  # bounds check
                    text = f"<b>{name}</b><br>"
                    text += f"X: {coords[j, 0]:.3f}<br>"
                    text += f"Y: {coords[j, 1]:.3f}"
                    if self.descriptions and j < len(self.descriptions):
                        desc_preview = self.descriptions[j][:100] + "..." if len(self.descriptions[j]) > 100 else self.descriptions[j]
                        text += f"<br>説明: {desc_preview}"
                    hover_text.append(text)
            
            # クラスター別の色分け
            if clusters is not None:
                # クラスター別にトレースを追加
                unique_clusters = sorted(set(clusters))
                for cluster in unique_clusters:
                    cluster_mask = np.array(clusters) == cluster
                    cluster_coords = coords[cluster_mask]
                    cluster_names = [self.brand_names[idx] for idx in np.where(cluster_mask)[0] if idx < len(self.brand_names)]
                    cluster_hover = [hover_text[idx] for idx in np.where(cluster_mask)[0] if idx < len(hover_text)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cluster_coords[:, 0],
                            y=cluster_coords[:, 1],
                            mode='markers',
                            name=f'Cluster {cluster}',
                            text=cluster_names,
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=cluster_hover,
                            marker=dict(
                                size=8,
                                color=self.color_palette[cluster % len(self.color_palette)],
                                opacity=0.8,
                                line=dict(width=1, color='black')
                            ),
                            legendgroup=f'cluster_{cluster}',
                            showlegend=(i == 0)  # 最初のサブプロットでのみ凡例を表示
                        ),
                        row=row, col=col
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode='markers',
                        name=method_name,
                        text=self.brand_names,
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_text,
                        marker=dict(
                            size=8,
                            color='steelblue',
                            opacity=0.8,
                            line=dict(width=1, color='black')
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # レイアウト設定（縦並び用に調整）
        fig.update_layout(
            title={
                'text': '🚀 次元削減手法の統合比較ダッシュボード',
                'x': 0.5,
                'font': {'size': 20}
            },
            width=1600,  # 横幅を最大化
            height=600 * n_methods,  # 手法数に応じて高さを調整
            hovermode='closest'
        )
        
        # 軸ラベル設定（縦並び用）
        for i in range(1, n_methods + 1):
            row = i  # 縦並びなので行インデックスそのまま
            col = 1  # 1列固定
            fig.update_xaxes(title_text="次元1", row=row, col=col)
            fig.update_yaxes(title_text="次元2", row=row, col=col)
        
        # 保存
        interactive_file = os.path.join(output_dir, 'dimensionality_reduction_dashboard.html')
        fig.write_html(interactive_file)
        
        print(f"🌐 インタラクティブダッシュボード保存: {interactive_file}")
    
    def _create_statistical_comparison(self, output_dir):
        """統計的比較レポートを作成"""
        print("📊 統計的比較レポートを作成中...")
        
        # 各手法の統計情報を計算
        stats_data = []
        
        for method_name, result in self.results.items():
            coords = result['coordinates']
            
            # 基本統計
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            center_x = coords[:, 0].mean()
            center_y = coords[:, 1].mean()
            
            # 分散
            x_var = coords[:, 0].var()
            y_var = coords[:, 1].var()
            
            # 点間距離の統計
            from scipy.spatial.distance import pdist
            distances = pdist(coords)
            avg_distance = distances.mean()
            min_distance = distances.min()
            max_distance = distances.max()
            
            stats_data.append({
                '手法': method_name,
                'X範囲': f"{x_range:.3f}",
                'Y範囲': f"{y_range:.3f}",
                'X分散': f"{x_var:.3f}",
                'Y分散': f"{y_var:.3f}",
                '平均点間距離': f"{avg_distance:.3f}",
                '最小点間距離': f"{min_distance:.3f}",
                '最大点間距離': f"{max_distance:.3f}"
            })
        
        # DataFrameに変換
        stats_df = pd.DataFrame(stats_data)
        
        # CSV保存
        stats_file = os.path.join(output_dir, 'dimensionality_reduction_statistics.csv')
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        # テキストレポート作成
        report_file = os.path.join(output_dir, 'dimensionality_reduction_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🚀 次元削減手法統合比較レポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"📊 データセット情報:\n")
            f.write(f"   - サンプル数: {len(self.embeddings)}\n")
            f.write(f"   - 元次元数: {self.embeddings.shape[1]}\n")
            f.write(f"   - 適用手法数: {len(self.results)}\n\n")
            
            f.write("📈 各手法の特徴:\n")
            f.write("   - MDS: 距離関係を最もよく保持、大域構造重視\n")
            f.write("   - t-SNE: 局所構造を強調、クラスタ分離に優秀\n")
            f.write("   - UMAP: 局所・大域のバランス、高速処理\n")
            f.write("   - Anchor-UMAP: アンカー点による安定化、論文手法\n\n")
            
            f.write("📊 統計比較:\n")
            f.write(stats_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("💡 推奨用途:\n")
            f.write("   - 距離保持重視 → MDS\n")
            f.write("   - クラスタ発見 → t-SNE\n")
            f.write("   - バランス型 → UMAP\n")
            f.write("   - 安定性重視 → Anchor-UMAP\n")
        
        print(f"📄 統計レポート保存: {report_file}")
        print(f"📊 統計CSV保存: {stats_file}")
    
    def export_all_coordinates(self, output_dir='./integrated_results'):
        """全手法の座標をエクスポート"""
        if not self.results:
            print("❌ 先に apply_all_methods() を実行してください")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 全座標を統合したDataFrame作成
        export_data = {
            'brand_name': self.brand_names
        }
        
        if self.descriptions:
            export_data['description'] = self.descriptions
        
        # 各手法の座標を追加
        for method_name, result in self.results.items():
            coords = result['coordinates']
            export_data[f'{method_name}_x'] = coords[:, 0]
            export_data[f'{method_name}_y'] = coords[:, 1]
        
        # DataFrame作成・保存
        export_df = pd.DataFrame(export_data)
        
        coords_file = os.path.join(output_dir, 'all_dimensionality_reduction_coordinates.csv')
        export_df.to_csv(coords_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 全座標エクスポート: {coords_file}")
        return export_df

    def create_single_method_visualization(self, method='UMAP', clustering_method='KMeans', 
                                         n_clusters=5, output_dir='./integrated_results'):
        """
        単一手法での詳細可視化を作成
        
        Args:
            method: 次元削減手法
            clustering_method: クラスタリング手法
            n_clusters: クラスター数
            output_dir: 出力ディレクトリ
        """
        if method not in self.available_methods:
            print(f"❌ {method} は利用できません")
            return
        
        # 次元削減とクラスタリングを実行
        print(f"🚀 {method} + {clustering_method} 可視化を作成中...")
        self.apply_selected_methods([method])
        clusters = self.apply_clustering(clustering_method, n_clusters=n_clusters)
        
        if clusters is None:
            print(f"❌ {clustering_method} クラスタリングが失敗しました")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 詳細な可視化
        coords = self.results[method]['coordinates']
        
        # 静的プロット
        plt.figure(figsize=(12, 8))
        unique_clusters = sorted(set(clusters))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_mask = np.array(clusters) == cluster
            cluster_coords = coords[cluster_mask]
            color = self.color_palette[i % len(self.color_palette)]
            
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                       c=color, alpha=0.7, s=60, label=f'Cluster {cluster}',
                       edgecolors='black', linewidth=0.5)
        
        plt.title(f'{method} + {clustering_method} クラスタリング結果', fontsize=16)
        plt.xlabel('次元1', fontsize=12)
        plt.ylabel('次元2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        static_file = os.path.join(output_dir, f'{method}_{clustering_method}_visualization.png')
        plt.savefig(static_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # インタラクティブプロット
        fig = go.Figure()
        
        for cluster in unique_clusters:
            cluster_mask = np.array(clusters) == cluster
            cluster_coords = coords[cluster_mask]
            cluster_names = [self.brand_names[idx] for idx in np.where(cluster_mask)[0]]
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_names,
                marker=dict(
                    size=10,
                    color=self.color_palette[cluster % len(self.color_palette)],
                    opacity=0.8,
                    line=dict(width=1, color='black')
                )
            ))
        
        fig.update_layout(
            title=f'{method} + {clustering_method} インタラクティブ可視化',
            xaxis_title='次元1',
            yaxis_title='次元2',
            width=900,
            height=600
        )
        
        interactive_file = os.path.join(output_dir, f'{method}_{clustering_method}_interactive.html')
        fig.write_html(interactive_file)
        
        print(f"📊 {method} 可視化完了: {static_file}, {interactive_file}")

    def get_available_methods(self):
        """利用可能な次元削減手法を取得"""
        return list(self.available_methods.keys())
    
    def get_available_clustering(self):
        """利用可能なクラスタリング手法を取得"""
        return list(self.available_clustering.keys())
    
    def create_comparison_visualization_for_streamlit(self, clusters=None, show_brand_names=True):
        """
        Streamlit用の比較可視化を作成
        
        Args:
            clusters: クラスターラベル
            show_brand_names: ブランド名を表示するか
            
        Returns:
            plotly.graph_objects.Figure: Streamlit用のplotly図
        """
        if not self.results:
            print("❌ 先に apply_selected_methods() を実行してください")
            return None
        
        methods = list(self.results.keys())
        n_methods = len(methods)
        
        # 縦並びレイアウトに変更
        cols = 1  # 1列固定で縦並び
        rows = n_methods  # 手法数分の行
        
        subplot_titles = methods
        specs = [[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=specs
        )
        
        # 位置のマッピング（縦並び用）
        positions = []
        for i in range(n_methods):
            row = i + 1  # Plotlyは1始まり、縦並びなので行インデックスそのまま
            col = 1  # 1列固定
            positions.append((row, col))
        
        for i, method_name in enumerate(methods):
            if method_name not in self.results:
                continue
                
            row, col = positions[i]
            coords = self.results[method_name]['coordinates']
            
            # ホバー情報を準備
            hover_text = []
            for j, name in enumerate(self.brand_names):
                if j < len(coords):  # bounds check
                    text = f"<b>{name}</b><br>"
                    text += f"X: {coords[j, 0]:.3f}<br>"
                    text += f"Y: {coords[j, 1]:.3f}"
                    if self.descriptions and j < len(self.descriptions):
                        desc_preview = self.descriptions[j][:100] + "..." if len(self.descriptions[j]) > 100 else self.descriptions[j]
                        text += f"<br>説明: {desc_preview}"
                    hover_text.append(text)
            
            # ブランド名表示モード設定
            mode = 'markers+text' if show_brand_names else 'markers'
            textposition = 'top center' if show_brand_names else None
            
            # クラスター別の色分け
            if clusters is not None:
                # クラスター別にトレースを追加
                unique_clusters = sorted(set(clusters))
                for cluster in unique_clusters:
                    cluster_mask = np.array(clusters) == cluster
                    cluster_coords = coords[cluster_mask]
                    cluster_names = [self.brand_names[idx] for idx in np.where(cluster_mask)[0] if idx < len(self.brand_names)]
                    cluster_hover = [hover_text[idx] for idx in np.where(cluster_mask)[0] if idx < len(hover_text)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cluster_coords[:, 0],
                            y=cluster_coords[:, 1],
                            mode=mode,
                            name=f'Cluster {cluster}',
                            text=cluster_names if show_brand_names else cluster_names,
                            textposition=textposition,
                            textfont=dict(size=9) if show_brand_names else None,
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=cluster_hover,
                            marker=dict(
                                size=8,
                                color=self.color_palette[cluster % len(self.color_palette)],
                                opacity=0.8,
                                line=dict(width=1, color='black')
                            ),
                            legendgroup=f'cluster_{cluster}',
                            showlegend=(i == 0)  # 最初のサブプロットでのみ凡例を表示
                        ),
                        row=row, col=col
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode=mode,
                        name=method_name,
                        text=self.brand_names if show_brand_names else self.brand_names,
                        textposition=textposition,
                        textfont=dict(size=9) if show_brand_names else None,
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_text,
                        marker=dict(
                            size=8,
                            color='steelblue',
                            opacity=0.8,
                            line=dict(width=1, color='black')
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # レイアウト設定（縦並び用に調整）
        fig.update_layout(
            title={
                'text': '🚀 次元削減手法の統合比較',
                'x': 0.5,
                'font': {'size': 16}
            },
            width=1400,  # 横幅を最大化
            height=500 * n_methods,  # 手法数に応じて高さを調整
            hovermode='closest'
        )
        
        # 軸ラベル設定（縦並び用）
        for i in range(1, n_methods + 1):
            row = i  # 縦並びなので行インデックスそのまま
            col = 1  # 1列固定
            fig.update_xaxes(title_text="次元1", row=row, col=col)
            fig.update_yaxes(title_text="次元2", row=row, col=col)
        
        return fig

    def create_single_method_visualization_for_streamlit(self, method='UMAP', clustering_method='KMeans', 
                                                       n_clusters=5, **clustering_kwargs):
        """
        Streamlit用の単一手法詳細可視化を作成
        
        Args:
            method: 次元削減手法
            clustering_method: クラスタリング手法
            n_clusters: クラスター数
            **clustering_kwargs: クラスタリング固有パラメータ
            
        Returns:
            plotly.graph_objects.Figure: Streamlit用のplotly図
        """
        if method not in self.available_methods:
            print(f"❌ {method} は利用できません")
            return None
        
        # 次元削減とクラスタリングを実行
        print(f"🚀 {method} + {clustering_method} 可視化を作成中...")
        self.apply_selected_methods([method])
        clusters = self.apply_clustering(clustering_method, n_clusters=n_clusters, **clustering_kwargs)
        
        if clusters is None:
            print(f"❌ {clustering_method} クラスタリングが失敗しました")
            return None
        
        # 詳細な可視化
        coords = self.results[method]['coordinates']
        
        # インタラクティブプロット
        fig = go.Figure()
        
        unique_clusters = sorted(set(clusters))
        for cluster in unique_clusters:
            cluster_mask = np.array(clusters) == cluster
            cluster_coords = coords[cluster_mask]
            cluster_names = [self.brand_names[idx] for idx in np.where(cluster_mask)[0]]
            
            # ホバー情報を準備
            hover_text = []
            for idx in np.where(cluster_mask)[0]:
                if idx < len(self.brand_names) and idx < len(coords):  # bounds check
                    text = f"<b>{self.brand_names[idx]}</b><br>"
                    text += f"クラスター: {cluster}<br>"
                    text += f"X: {coords[idx, 0]:.3f}<br>"
                    text += f"Y: {coords[idx, 1]:.3f}"
                    if self.descriptions and idx < len(self.descriptions):
                        desc_preview = self.descriptions[idx][:100] + "..." if len(self.descriptions[idx]) > 100 else self.descriptions[idx]
                        text += f"<br>説明: {desc_preview}"
                    hover_text.append(text)
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_names,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text,
                marker=dict(
                    size=10,
                    color=self.color_palette[cluster % len(self.color_palette)],
                    opacity=0.8,
                    line=dict(width=1, color='black')
                )
            ))
        
        fig.update_layout(
            title={
                'text': f'{method} + {clustering_method} 詳細解析',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='次元1',
            yaxis_title='次元2',
            width=900,
            height=600,
            hovermode='closest'
        )
        
        return fig


def main():
    """メイン実行関数のデモ"""
    print("🚀 統合次元削減システムのデモを開始...")
    
    # ダミーデータでテスト
    np.random.seed(42)
    n_samples = 50
    n_features = 100
    
    # ランダムな高次元データを生成
    embeddings = np.random.randn(n_samples, n_features)
    brand_names = [f"ブランド_{i+1}" for i in range(n_samples)]
    descriptions = [f"これは{brand_names[i]}の説明文です。" for i in range(n_samples)]
    
    # 統合システムを初期化
    reducer = IntegratedDimensionalityReduction(
        embeddings=embeddings,
        brand_names=brand_names,
        descriptions=descriptions
    )
    
    print(f"📋 利用可能な次元削減手法: {reducer.get_available_methods()}")
    print(f"📋 利用可能なクラスタリング手法: {reducer.get_available_clustering()}")
    
    # 選択した手法を適用
    selected_methods = ['PCA', 'UMAP', 't-SNE']
    results = reducer.apply_selected_methods(methods=selected_methods)
    
    # クラスタリング付き可視化作成
    reducer.create_comparison_visualization(clustering_method='KMeans')
    
    # 単一手法での詳細可視化
    reducer.create_single_method_visualization(method='UMAP', clustering_method='DBSCAN')
    
    # 座標エクスポート
    coords_df = reducer.export_all_coordinates()
    
    print("🎉 統合次元削減システムのデモが完了しました！")
    print("📁 結果は ./integrated_results フォルダに保存されました")
    print("🎯 新機能:")
    print("  - 手法選択可能 (PCA, MDS, t-SNE, UMAP, Anchor-UMAP)")
    print("  - クラスタリング色分け (KMeans, DBSCAN, Hierarchical, GMM)")
    print("  - 動的サブプロット生成")
    print("  - 単一手法詳細可視化")


if __name__ == "__main__":
    main()