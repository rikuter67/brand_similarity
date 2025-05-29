# few_shot.py

# --- Few-shotの高品質な例データ ---
FEW_SHOT_EXAMPLES_DATA = [
    {
        "example_bline_name": "コム デ ギャルソン",
        "example_existing_desc": "コム デ ギャルソン(COMME des GARÇONS)は日本のファッションブランド。創業者川久保玲。黒などモノトーンを多様、孤高の女性を描いた。", 
        "example_generated_style_desc": "コム デ ギャルソンは、川久保玲が手掛ける革新的なファッションブランド。黒を基調としたモノトーンの色使い、左右非対称なカッティング、身体のラインを曖昧にするルーズなシルエットが特徴。代表的なアイテムは独創的なフォルムのジャケットやドレス、穴あきニット。ウールギャバジンなど独自加工の素材を多用し、流行に左右されず個性を表現したいアート志向の層に支持される。"
    },
    {
        "example_bline_name": "ジル サンダー",
        "example_existing_desc": "ジル サンダーはMs.ジル・サンダーが設立。洗練されて繊細、かつ品質にこだわったミニマルなデザインが特徴。", 
        "example_generated_style_desc": "ジル サンダーは、ドイツ発のミニマリズムを代表するブランド。純粋さと品質を追求し、不要な装飾を削ぎ落としたクリーンで洗練されたデザインが特徴。代表的なアイテムは完璧なカッティングのシャツ、構築的シルエットのコート。カシミア、ウール、シルク等の高品質な天然素材を好み、ニュートラルカラーを基調とした落ち着いたパレット。流行に左右されないタイムレスなエレガンスを求める知的な大人に支持される。"
    }
]

# --- 事前要約用のプロンプト ---
PROMPT_PRE_SUMMARY = """以下のブランドの特徴を80字程度で要約してください。

{long_description}

[要約文]:"""

def create_simple_prompt(bline_name, existing_desc=""):
    """シンプルなプロンプトを作成する関数"""
    try:
        if not existing_desc or len(existing_desc) <= 30:
            # 新規生成用
            prompt = f"""あなたはファッションエディターです。「{bline_name}」というブランドの特徴を150文字程度で説明してください。

参考例:
{FEW_SHOT_EXAMPLES_DATA[0]['example_bline_name']}は{FEW_SHOT_EXAMPLES_DATA[0]['example_generated_style_desc'][:100]}...

{bline_name}の特徴:"""
        else:
            # 補完・改善用
            prompt = f"""あなたはファッションエディターです。以下のブランド情報を参考に、「{bline_name}」の特徴を150文字程度で魅力的に説明してください。

参考例:
{FEW_SHOT_EXAMPLES_DATA[0]['example_bline_name']}は{FEW_SHOT_EXAMPLES_DATA[0]['example_generated_style_desc'][:100]}...

ブランド名: {bline_name}
既存情報: {existing_desc[:200]}

{bline_name}の特徴:"""
        
        return prompt
        
    except Exception as e:
        print(f"警告: プロンプト作成中にエラー: {e}")
        return f"{bline_name}の特徴を説明してください。"

# 旧関数（互換性のため残す）
def get_formatted_few_shot_examples(bline_name_for_current_prompt, existing_desc_for_current_prompt):
    """旧関数 - 互換性のため残すが、新機能では使用しない"""
    return ""

def get_formatted_few_shot_examples_for_new_generation(bline_name_for_current_prompt):
    """旧関数 - 互換性のため残すが、新機能では使用しない"""
    return ""

# 旧プロンプトテンプレート（互換性のため残す）
PROMPT_TEMPLATE_SUMMARIZE_RESTRUCTURE = ""
PROMPT_TEMPLATE_GENERATE_COMPLEMENT = ""
PROMPT_TEMPLATE_GENERATE_NEW = ""