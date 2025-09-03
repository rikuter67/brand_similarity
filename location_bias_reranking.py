import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict
import warnings
import unicodedata
warnings.filterwarnings('ignore')

class LocationBasedSimilarityReranker:
    """
    地図データを使用してブランド類似度をリランキングするクラス
    テナント&フロア分析手法をベースに、店舗・場所の一致率で類似度を調整
    """
    
    def __init__(self, maps_csv_path: str, tenants_csv_path: Optional[str] = None):
        """
        初期化
        
        Args:
            maps_csv_path: maps.csvのパス
            tenants_csv_path: tenants.csvのパス（オプション）
        """
        self.maps_csv_path = maps_csv_path
        self.tenants_csv_path = tenants_csv_path
        
        # データ格納用
        self.maps_df = None
        self.tenants_df = None
        
        # 前処理済みデータ
        self.brand_locations = defaultdict(list)  # ブランド -> 店舗リスト
        self.brand_tenants = defaultdict(set)     # ブランド -> テナントセット
        self.brand_buildings = defaultdict(set)   # ブランド -> ビルディングセット
        self.brand_floors = defaultdict(list)     # ブランド -> フロア情報リスト
        self.normalized_brand_mapping = {}        # 正規化ブランド名 -> 元ブランド名のマッピング
        
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """データの読み込み"""
        try:
            # maps.csvの読み込み
            self.maps_df = pd.read_csv(self.maps_csv_path, encoding='utf-8', low_memory=False)
            print(f"Maps data loaded: {len(self.maps_df)} records")
            
            # tenants.csvがあれば読み込み
            if self.tenants_csv_path and pd.io.common.file_exists(self.tenants_csv_path):
                self.tenants_df = pd.read_csv(self.tenants_csv_path, encoding='utf-8')
                print(f"Tenants data loaded: {len(self.tenants_df)} records")
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            raise
    
    def _extract_floor_info(self, building_str: str) -> List[int]:
        """
        建物情報からフロア番号を抽出
        例: "本館 3F(婦人バッグ・財布)" -> [3]
             "B1F 地下1階" -> [-1]
        """
        if pd.isna(building_str) or building_str == '':
            return []
        
        # 正規表現でフロア情報を抽出
        floor_pattern = r'B?[0-9]+[階|F]'
        floors = re.findall(floor_pattern, str(building_str))
        
        floor_numbers = []
        for floor in floors:
            try:
                if floor.startswith('B'):
                    # 地下階はマイナス
                    floor_num = -int(floor[1])
                else:
                    # 地上階
                    if '階' in floor:
                        floor_num = int(floor.replace('階', ''))
                    else:  # F
                        floor_num = int(floor.replace('F', ''))
                floor_numbers.append(floor_num)
            except:
                continue
                
        return floor_numbers
    
    def _normalize_brand_name(self, name: str) -> str:
        """
        ブランド名を正規化し、基本ブランド名を抽出
        
        Args:
            name: 元のブランド名（例: "アンダーアーマー ジ アウトレット北九州"）
            
        Returns:
            正規化された基本ブランド名（例: "アンダーアーマー"）
        """
        if pd.isna(name) or name == '':
            return ''
        
        # 1. Unicode正規化
        normalized = unicodedata.normalize('NFKC', str(name))
        normalized = normalized.strip()
        
        # 2. 店舗固有の情報を除去して基本ブランド名を抽出
        # より具体的なパターンを優先順位順に定義
        patterns_to_remove = [
            r'\s+ジ\s+アウトレット.+$',  # 「ジ アウトレット北九州」等
            r'\s+三井アウトレットパーク.+$',  # 「三井アウトレットパーク 多摩南大沢」等
            r'\s+(アウトレット|OUTLET|outlet).+$',  # その他のアウトレット系
            r'\s+(店|ショップ|SHOP|Store|store)$',  # 末尾の「店」「ショップ」等
            r'\s+[A-Za-zあ-ん一-龯]+店舗?$',  # 「新宿店」「渋谷店舗」等
            r'\s+\w+店$',  # 「○○店」
            r'\s+(ジ|THE)\s+.+$',  # その他の「ジ」「THE」系
            r'\s+[0-9]+F?$',  # 末尾の階数
            r'\s+[A-Za-z0-9\s\-・]+$',  # 英数字の店舗名部分
        ]
        
        # パターンマッチングで店舗固有情報を除去
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized)
        
        # 3. スペースや特殊文字の正規化
        normalized = re.sub(r'\s+', ' ', normalized)  # 複数スペースを1つに
        normalized = re.sub(r'[・･]', '', normalized)  # 中点を削除
        normalized = normalized.strip()
        
        return normalized
    
    def _preprocess_data(self):
        """データの前処理"""
        if self.maps_df is None:
            return
        
        # 有効な店舗のみフィルタリング
        valid_maps = self.maps_df[
            (self.maps_df['enabled'] == True) & 
            (self.maps_df['closed'] == False)
        ].copy()
        
        print(f"有効な店舗データ: {len(valid_maps)} records")
        
        # ブランド名でグループ化してロケーション情報を集約
        for _, row in valid_maps.iterrows():
            brand_name = str(row.get('name', '')).strip()
            if not brand_name or brand_name == 'nan':
                continue
            
            # ブランド名を正規化して基本ブランド名を抽出
            normalized_brand_name = self._normalize_brand_name(brand_name)
            if not normalized_brand_name:
                continue
            
            # 正規化されたブランド名でマッピングを作成
            if normalized_brand_name not in self.normalized_brand_mapping:
                self.normalized_brand_mapping[normalized_brand_name] = []
            self.normalized_brand_mapping[normalized_brand_name].append(brand_name)
            
            # 位置情報
            location_info = {
                'id': row.get('id'),
                'name': brand_name,
                'normalized_name': normalized_brand_name,
                'address': row.get('address', ''),
                'building': row.get('building', ''),
                'lat': row.get('lat'),
                'lng': row.get('lng'),
                'tenant_id': row.get('tenant_id'),
                'shop_id': row.get('shop_id'),
                'pref_id': row.get('pref_id'),
                'area_id': row.get('area_id')
            }
            
            # 正規化されたブランド名でグループ化
            self.brand_locations[normalized_brand_name].append(location_info)
            
            # テナント情報（正規化されたブランド名で保存）
            tenant_id = row.get('tenant_id')
            if pd.notna(tenant_id) and str(tenant_id) != 'NULL':
                self.brand_tenants[normalized_brand_name].add(str(tenant_id))
            
            # ビルディング情報（正規化されたブランド名で保存）
            building = row.get('building', '')
            if pd.notna(building) and building != '':
                self.brand_buildings[normalized_brand_name].add(str(building))
            
            # フロア情報（正規化されたブランド名で保存）
            floor_numbers = self._extract_floor_info(building)
            self.brand_floors[normalized_brand_name].extend(floor_numbers)
        
        print(f"処理済みブランド数: {len(self.brand_locations)}")
    
    def calculate_location_similarity(self, brand1: str, brand2: str, 
                                    method: str = 'comprehensive') -> float:
        """
        2つのブランド間の位置情報類似度を計算
        
        Args:
            brand1, brand2: 比較するブランド名
            method: 計算方法 ('tenant', 'building', 'floor', 'geographic', 'area', 'comprehensive')
            
        Returns:
            類似度スコア (0.0-1.0)
        """
        
        if brand1 not in self.brand_locations or brand2 not in self.brand_locations:
            return 0.0
        
        if method == 'tenant':
            return self._calculate_tenant_similarity(brand1, brand2)
        elif method == 'building':
            return self._calculate_building_similarity(brand1, brand2)
        elif method == 'floor':
            return self._calculate_floor_similarity(brand1, brand2)
        elif method == 'geographic':
            return self._calculate_geographic_similarity(brand1, brand2)
        elif method == 'area':
            return self._calculate_area_similarity(brand1, brand2)
        else:  # comprehensive
            return self._calculate_comprehensive_similarity(brand1, brand2)
    
    def _calculate_tenant_similarity(self, brand1: str, brand2: str) -> float:
        """テナント一致率による類似度"""
        tenants1 = self.brand_tenants[brand1]
        tenants2 = self.brand_tenants[brand2]
        
        if not tenants1 or not tenants2:
            return 0.0
        
        # Jaccard係数
        intersection = len(tenants1 & tenants2)
        union = len(tenants1 | tenants2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_building_similarity(self, brand1: str, brand2: str) -> float:
        """ビルディング一致率による類似度"""
        buildings1 = self.brand_buildings[brand1]
        buildings2 = self.brand_buildings[brand2]
        
        if not buildings1 or not buildings2:
            return 0.0
        
        # Jaccard係数
        intersection = len(buildings1 & buildings2)
        union = len(buildings1 | buildings2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_floor_similarity(self, brand1: str, brand2: str) -> float:
        """フロア一致率による類似度"""
        floors1 = set(self.brand_floors[brand1])
        floors2 = set(self.brand_floors[brand2])
        
        if not floors1 or not floors2:
            return 0.0
        
        # Jaccard係数
        intersection = len(floors1 & floors2)
        union = len(floors1 | floors2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_geographic_similarity(self, brand1: str, brand2: str, max_distance_km: float = 10.0) -> float:
        """地理的距離による類似度計算"""
        locations1 = self.brand_locations[brand1]
        locations2 = self.brand_locations[brand2]
        
        if not locations1 or not locations2:
            return 0.0
        
        min_distance = float('inf')
        
        # 各店舗ペア間の最短距離を計算
        for loc1 in locations1:
            lat1, lng1 = loc1.get('lat'), loc1.get('lng')
            if pd.isna(lat1) or pd.isna(lng1):
                continue
                
            for loc2 in locations2:
                lat2, lng2 = loc2.get('lat'), loc2.get('lng')
                if pd.isna(lat2) or pd.isna(lng2):
                    continue
                
                # ハバーサイン公式で距離計算
                distance = self._haversine_distance(lat1, lng1, lat2, lng2)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        # 距離を類似度に変換（近いほど高い類似度）
        # sigmoid関数で0-1の範囲に正規化
        similarity = 1.0 / (1.0 + (min_distance / max_distance_km))
        return similarity
    
    def _calculate_area_similarity(self, brand1: str, brand2: str) -> float:
        """エリア・都道府県一致度による類似度"""
        locations1 = self.brand_locations[brand1]
        locations2 = self.brand_locations[brand2]
        
        if not locations1 or not locations2:
            return 0.0
        
        # 各ブランドのエリア・都道府県IDを収集
        areas1 = set()
        prefs1 = set()
        for loc in locations1:
            if pd.notna(loc.get('area_id')):
                areas1.add(str(loc['area_id']))
            if pd.notna(loc.get('pref_id')):
                prefs1.add(str(loc['pref_id']))
        
        areas2 = set()
        prefs2 = set()
        for loc in locations2:
            if pd.notna(loc.get('area_id')):
                areas2.add(str(loc['area_id']))
            if pd.notna(loc.get('pref_id')):
                prefs2.add(str(loc['pref_id']))
        
        # エリア一致度計算（Jaccard係数）
        area_similarity = 0.0
        if areas1 and areas2:
            area_intersection = len(areas1 & areas2)
            area_union = len(areas1 | areas2)
            area_similarity = area_intersection / area_union if area_union > 0 else 0.0
        
        # 都道府県一致度計算（Jaccard係数）
        pref_similarity = 0.0
        if prefs1 and prefs2:
            pref_intersection = len(prefs1 & prefs2)
            pref_union = len(prefs1 | prefs2)
            pref_similarity = pref_intersection / pref_union if pref_union > 0 else 0.0
        
        # 都道府県一致に重きを置いた総合スコア
        combined_similarity = 0.6 * pref_similarity + 0.4 * area_similarity
        
        return combined_similarity
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        ハバーサイン公式による2点間の距離計算（km）
        """
        import math
        
        # 度をラジアンに変換
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # 差分計算
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        # ハバーサイン公式
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # 地球の半径（km）
        earth_radius_km = 6371.0
        
        return earth_radius_km * c

    def _calculate_comprehensive_similarity(self, brand1: str, brand2: str) -> float:
        """総合的な類似度（重み付き平均）"""
        tenant_sim = self._calculate_tenant_similarity(brand1, brand2)
        building_sim = self._calculate_building_similarity(brand1, brand2)
        floor_sim = self._calculate_floor_similarity(brand1, brand2)
        geographic_sim = self._calculate_geographic_similarity(brand1, brand2)
        
        # 同一エリア・プリフェクチャボーナス追加
        area_sim = self._calculate_area_similarity(brand1, brand2)
        
        # 重み設定（地理的距離と行政区域を強化）
        weights = {
            'tenant': 0.30,      # テナント一致度を少し下げる
            'building': 0.20,    # ビル一致度を少し下げる
            'floor': 0.10,       # フロア一致度を下げる
            'geographic': 0.25,  # 地理的距離を維持
            'area': 0.15         # エリア・都道府県一致度を追加
        }
        
        comprehensive_sim = (
            weights['tenant'] * tenant_sim +
            weights['building'] * building_sim +
            weights['floor'] * floor_sim +
            weights['geographic'] * geographic_sim +
            weights['area'] * area_sim
        )
        
        return comprehensive_sim
    
    def rerank_similarity_with_location_bias(self, 
                                           similarity_scores: Dict[str, float],
                                           query_brand: str,
                                           bias_strength: float = 0.3,
                                           location_method: str = 'comprehensive',
                                           rerank_mode: str = 'weighted_average',
                                           brand_mapping: Dict[str, str] = None) -> Dict[str, float]:
        """
        位置情報バイアスを適用してブランド類似度をリランキング
        
        Args:
            similarity_scores: {brand_name: similarity_score} の辞書
            query_brand: クエリブランド名
            bias_strength: バイアスの強度 (0.0-1.0)
            location_method: 位置類似度の計算方法
            rerank_mode: リランキング方式 ('weighted_average', 'linear_addition', 'location_rerank')
            brand_mapping: ブランド名マッピング辞書 {integrated_brand: maps_brand}
            
        Returns:
            リランキング後の類似度辞書
        """
        
        # ブランドマッピングを使用してクエリブランドを解決
        mapped_query_brand = query_brand
        if brand_mapping and query_brand in brand_mapping:
            mapped_query_brand = brand_mapping[query_brand]
        
        # クエリブランド名を正規化
        normalized_query_brand = self._normalize_brand_name(mapped_query_brand)
        
        # 店舗がないブランドの場合は元のスコアをそのまま返す
        if normalized_query_brand not in self.brand_locations:
            print(f"警告: {query_brand} の位置情報が見つかりません")
            print(f"正規化後のブランド名: {normalized_query_brand}")
            print(f"利用可能なブランド（一部）: {list(self.brand_locations.keys())[:5]}")
            return similarity_scores
        
        reranked_scores = {}
        location_similarities = {}
        
        # 全ブランドの位置類似度を事前計算
        for brand_name in similarity_scores.keys():
            if brand_name != query_brand:
                # ブランドマッピングを適用
                mapped_brand_name = brand_name
                if brand_mapping and brand_name in brand_mapping:
                    mapped_brand_name = brand_mapping[brand_name]
                
                # 比較対象ブランド名も正規化
                normalized_brand_name = self._normalize_brand_name(mapped_brand_name)
                
                if normalized_brand_name in self.brand_locations:
                    location_similarities[brand_name] = self.calculate_location_similarity(
                        normalized_query_brand, normalized_brand_name, method=location_method
                    )
                else:
                    location_similarities[brand_name] = 0.0
            else:
                location_similarities[brand_name] = 0.0
        
        for brand_name, original_score in similarity_scores.items():
            if brand_name == query_brand:
                reranked_scores[brand_name] = original_score
                continue
            
            location_similarity = location_similarities[brand_name]
            
            if rerank_mode == 'weighted_average':
                # 重み付き平均（従来方式）
                biased_score = (
                    (1 - bias_strength) * original_score + 
                    bias_strength * location_similarity
                )
            elif rerank_mode == 'linear_addition':
                # 一次関数的係数での加算
                # original_score + bias_strength * location_similarity
                biased_score = original_score + bias_strength * location_similarity
            elif rerank_mode == 'location_rerank':
                # 店舗一致度による完全リランキング
                # location_similarity を主要スコアとして使用し、original_score は補正
                # 位置情報がある場合は位置類似度を重視、ない場合は元のスコアを維持
                if location_similarity > 0.0:
                    biased_score = bias_strength * location_similarity + (1 - bias_strength) * original_score * 0.3
                else:
                    # 位置情報がない場合は元のスコアを0.5倍にペナルティ
                    biased_score = original_score * 0.5
            else:
                biased_score = original_score
            
            reranked_scores[brand_name] = biased_score
        
        return reranked_scores
    
    def get_available_brands_for_location_analysis(self) -> List[str]:
        """
        位置情報分析が可能なブランドのリストを取得
        
        Returns:
            位置情報がある全ブランド名のリスト
        """
        return list(self.brand_locations.keys())
    
    def filter_similarity_scores_by_available_brands(self, 
                                                   similarity_scores: Dict[str, float]) -> Dict[str, float]:
        """
        位置情報があるブランドのみでスコアをフィルタリング
        
        Args:
            similarity_scores: 元の類似度スコア辞書
            
        Returns:
            位置情報があるブランドのみの類似度スコア辞書
        """
        available_brands = set(self.brand_locations.keys())
        filtered_scores = {
            brand: score for brand, score in similarity_scores.items()
            if brand in available_brands
        }
        return filtered_scores
    
    def get_brand_location_info(self, brand_name: str) -> Dict:
        """ブランドの位置情報を取得"""
        if brand_name not in self.brand_locations:
            return {}
        
        locations = self.brand_locations[brand_name]
        tenants = list(self.brand_tenants[brand_name])
        buildings = list(self.brand_buildings[brand_name])
        floors = list(set(self.brand_floors[brand_name]))
        
        return {
            'locations': locations,
            'num_locations': len(locations),
            'tenants': tenants,
            'num_tenants': len(tenants),
            'buildings': buildings,
            'num_buildings': len(buildings),
            'floors': sorted(floors),
            'num_floors': len(floors)
        }
    
    def create_location_similarity_matrix(self, brands: List[str], 
                                        method: str = 'comprehensive') -> np.ndarray:
        """
        ブランドリスト間の位置類似度行列を作成
        
        Args:
            brands: ブランド名のリスト
            method: 類似度計算方法
            
        Returns:
            類似度行列 (numpy array)
        """
        n = len(brands)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.calculate_location_similarity(
                        brands[i], brands[j], method=method
                    )
        
        return similarity_matrix


def create_reranker_from_streamlit_data(analyzer) -> LocationBasedSimilarityReranker:
    """
    Streamlitアナライザーのデータから位置情報リランカーを作成
    
    Args:
        analyzer: LLMStyleEmbeddingAnalyzer インスタンス
        
    Returns:
        LocationBasedSimilarityReranker インスタンス
    """
    maps_csv_path = "datasets/bline_similarity/maps.csv"
    tenants_csv_path = "datasets/bline_similarity/tenants.csv"
    
    try:
        reranker = LocationBasedSimilarityReranker(
            maps_csv_path=maps_csv_path,
            tenants_csv_path=tenants_csv_path
        )
        return reranker
    except Exception as e:
        print(f"リランカー作成エラー: {e}")
        return None


if __name__ == "__main__":
    # テスト用の実行
    maps_path = "datasets/bline_similarity/maps.csv"
    
    try:
        reranker = LocationBasedSimilarityReranker(maps_path)
        
        # テスト実行
        brands = list(reranker.brand_locations.keys())[:10]
        print(f"テスト用ブランド: {brands[:5]}")
        
        if len(brands) >= 2:
            # 2ブランド間の類似度テスト
            similarity = reranker.calculate_location_similarity(
                brands[0], brands[1], method='comprehensive'
            )
            print(f"{brands[0]} vs {brands[1]} 位置類似度: {similarity:.3f}")
            
            # ブランド情報表示
            info = reranker.get_brand_location_info(brands[0])
            print(f"{brands[0]} 位置情報: {info['num_locations']} 店舗, {info['num_tenants']} テナント")
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")