import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import difflib

class BrandStoreAnalyzer:
    """
    ブランドと店舗名の分離・マッチング分析を行うクラス
    """
    
    def __init__(self, maps_csv_path: str):
        """
        初期化
        
        Args:
            maps_csv_path: maps.csvのパス
        """
        self.maps_csv_path = maps_csv_path
        self.maps_df = None
        self.brand_store_mapping = defaultdict(set)  # ブランド -> 店舗名セット
        self.store_brand_mapping = defaultdict(set)  # 店舗名 -> ブランドセット
        self.separated_brands = {}  # 元の名前 -> {'brand': ブランド名, 'store': 店舗名}
        
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """データの読み込みと処理"""
        try:
            self.maps_df = pd.read_csv(self.maps_csv_path, encoding='utf-8', low_memory=False)
            print(f"Maps data loaded: {len(self.maps_df)} records")
            
            # 有効な店舗のみフィルタリング
            valid_maps = self.maps_df[
                (self.maps_df['enabled'] == True) & 
                (self.maps_df['closed'] == False)
            ].copy()
            
            print(f"有効な店舗データ: {len(valid_maps)} records")
            
            # ブランド・店舗名の分離処理
            self._separate_brand_and_store(valid_maps)
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            raise
    
    def _separate_brand_and_store(self, df: pd.DataFrame):
        """ブランド名と店舗名の分離"""
        
        # よく見られる店舗・場所のパターン
        store_patterns = [
            # 百貨店・商業施設
            r'(.+?)\s+(三越|高島屋|大丸|松坂屋|そごう|西武|東武|京王|小田急|阪急|阪神|近鉄|名鉄)',
            r'(.+?)\s+(ルミネ|NEWoMan|LUMINE|アトレ|エキュート|ラスカ|シャル)',
            r'(.+?)\s+(ららぽーと|ラゾーナ|マルイ|パルコ|丸井|109|渋谷109)',
            r'(.+?)\s+(イオン|ゆめタウン|フジグラン|イトーヨーカドー)',
            
            # 空港・駅
            r'(.+?)\s+(羽田空港|成田空港|新千歳空港|関西空港)',
            r'(.+?)\s+(新宿|渋谷|池袋|銀座|表参道|青山|六本木|恵比寿|代官山)',
            r'(.+?)\s+(横浜|みなとみらい|川崎|浦和|大宮)',
            r'(.+?)\s+(梅田|心斎橋|なんば|天王寺|京都|神戸)',
            
            # アウトレット・モール
            r'(.+?)\s+(アウトレット|プレミアム・アウトレット|OUTLET)',
            r'(.+?)\s+(店|ショップ|SHOP|Store|store)',
            
            # 地名付きパターン
            r'(.+?)\s+([^0-9]{2,}[店舗館])',
            r'(.+?)\s+([^0-9]{2,}店)',
        ]
        
        for _, row in df.iterrows():
            original_name = str(row['name']).strip()
            if not original_name or original_name == 'nan':
                continue
            
            brand_name = None
            store_name = None
            
            # パターンマッチングで分離を試行
            for pattern in store_patterns:
                match = re.match(pattern, original_name, re.IGNORECASE)
                if match:
                    brand_name = match.group(1).strip()
                    store_name = match.group(2).strip()
                    break
            
            # パターンマッチできない場合のフォールバック処理
            if not brand_name:
                brand_name, store_name = self._fallback_separation(original_name)
            
            # 結果を保存
            self.separated_brands[original_name] = {
                'brand': brand_name,
                'store': store_name,
                'address': row.get('address', ''),
                'building': row.get('building', ''),
                'tenant_id': row.get('tenant_id'),
                'pref_id': row.get('pref_id'),
                'area_id': row.get('area_id')
            }
            
            # マッピング辞書を更新
            if brand_name and store_name:
                self.brand_store_mapping[brand_name].add(store_name)
                self.store_brand_mapping[store_name].add(brand_name)
    
    def _fallback_separation(self, name: str) -> Tuple[str, str]:
        """パターンマッチできない場合の分離処理"""
        
        # スペースで分割して最初の部分をブランド名とする簡易的な方法
        parts = name.split()
        if len(parts) >= 2:
            # 最初の1-2語をブランド名、残りを店舗名とする
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                # 3語以上の場合、最初の2語をブランド名とする傾向
                brand_candidate = ' '.join(parts[:2])
                store_candidate = ' '.join(parts[2:])
                
                # 英語が多い場合は最初の1語のみをブランド名とする
                if self._is_mostly_english(parts[0]):
                    return parts[0], ' '.join(parts[1:])
                else:
                    return brand_candidate, store_candidate
        else:
            # 1語の場合はそのままブランド名とし、店舗名は空
            return name, ""
    
    def _is_mostly_english(self, text: str) -> bool:
        """テキストが主に英語かどうか判定"""
        if not text:
            return False
        
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = len([c for c in text if c.isalpha()])
        
        return english_chars / max(total_chars, 1) > 0.7
    
    def get_brand_store_summary(self) -> pd.DataFrame:
        """ブランドと店舗の対応関係のサマリーを作成"""
        summary_data = []
        
        for brand, stores in self.brand_store_mapping.items():
            summary_data.append({
                'ブランド名': brand,
                '店舗数': len(stores),
                '店舗リスト': ', '.join(sorted(stores)),
                '代表店舗': list(sorted(stores))[0] if stores else ''
            })
        
        return pd.DataFrame(summary_data).sort_values('店舗数', ascending=False)
    
    def calculate_store_overlap_matrix(self) -> pd.DataFrame:
        """ブランド間の店舗一致率行列を計算"""
        brands = list(self.brand_store_mapping.keys())
        n_brands = len(brands)
        
        # 店舗一致率行列を初期化
        overlap_matrix = np.zeros((n_brands, n_brands))
        
        for i, brand1 in enumerate(brands):
            for j, brand2 in enumerate(brands):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    stores1 = self.brand_store_mapping[brand1]
                    stores2 = self.brand_store_mapping[brand2]
                    
                    if not stores1 or not stores2:
                        overlap_matrix[i, j] = 0.0
                    else:
                        # Jaccard係数で店舗一致率を計算
                        intersection = len(stores1 & stores2)
                        union = len(stores1 | stores2)
                        overlap_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        # データフレームとして返す
        return pd.DataFrame(overlap_matrix, index=brands, columns=brands)
    
    def find_high_overlap_brands(self, threshold: float = 0.3) -> List[Dict]:
        """高い店舗一致率を持つブランドペアを検出"""
        overlap_matrix = self.calculate_store_overlap_matrix()
        high_overlap_pairs = []
        
        brands = overlap_matrix.index.tolist()
        
        for i, brand1 in enumerate(brands):
            for j, brand2 in enumerate(brands):
                if i < j:  # 対称行列なので上三角のみ処理
                    overlap_score = overlap_matrix.loc[brand1, brand2]
                    if overlap_score >= threshold:
                        shared_stores = self.brand_store_mapping[brand1] & self.brand_store_mapping[brand2]
                        high_overlap_pairs.append({
                            'ブランド1': brand1,
                            'ブランド2': brand2,
                            '店舗一致率': overlap_score,
                            '共通店舗数': len(shared_stores),
                            '共通店舗': ', '.join(sorted(shared_stores)),
                            'ブランド1店舗数': len(self.brand_store_mapping[brand1]),
                            'ブランド2店舗数': len(self.brand_store_mapping[brand2])
                        })
        
        return sorted(high_overlap_pairs, key=lambda x: x['店舗一致率'], reverse=True)
    
    def get_separation_quality_report(self) -> Dict:
        """分離品質のレポートを生成"""
        total_entries = len(self.separated_brands)
        has_store = sum(1 for v in self.separated_brands.values() if v['store'])
        
        # ブランド名の分布
        brand_counts = defaultdict(int)
        for v in self.separated_brands.values():
            brand_counts[v['brand']] += 1
        
        # 最も多い店舗を持つブランド
        top_brands = sorted(self.brand_store_mapping.items(), 
                           key=lambda x: len(x[1]), reverse=True)[:10]
        
        return {
            '総エントリー数': total_entries,
            '店舗名分離成功': has_store,
            '店舗名分離率': has_store / total_entries if total_entries > 0 else 0,
            '検出ブランド数': len(self.brand_store_mapping),
            '平均店舗数': np.mean([len(stores) for stores in self.brand_store_mapping.values()]) if self.brand_store_mapping else 0,
            'トップブランド': [(brand, len(stores)) for brand, stores in top_brands]
        }
    
    def export_separated_data(self, output_path: str):
        """分離後のデータをCSVに出力"""
        export_data = []
        
        for original_name, info in self.separated_brands.items():
            export_data.append({
                '元の名前': original_name,
                'ブランド名': info['brand'],
                '店舗名': info['store'],
                '住所': info['address'],
                'ビル': info['building'],
                'テナントID': info['tenant_id'],
                '都道府県ID': info['pref_id'],
                'エリアID': info['area_id']
            })
        
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"分離データを出力しました: {output_path}")
        
        return df


if __name__ == "__main__":
    # テスト実行
    try:
        analyzer = BrandStoreAnalyzer("datasets/bline_similarity/maps.csv")
        
        # 品質レポート
        report = analyzer.get_separation_quality_report()
        print("=== 分離品質レポート ===")
        for key, value in report.items():
            print(f"{key}: {value}")
        
        print("\n=== ブランド・店舗サマリー（上位10件) ===")
        summary = analyzer.get_brand_store_summary()
        print(summary.head(10).to_string(index=False))
        
        print("\n=== 高い店舗一致率ペア ===")
        high_overlap = analyzer.find_high_overlap_brands(threshold=0.2)
        for pair in high_overlap[:10]:
            print(f"{pair['ブランド1']} ↔ {pair['ブランド2']}: {pair['店舗一致率']:.3f} ({pair['共通店舗']})")
        
        # 結果をエクスポート
        analyzer.export_separated_data("brand_store_separated.csv")
        
    except Exception as e:
        print(f"エラー: {e}")