import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
import argparse
from bs4 import BeautifulSoup
import re
import few_shot # few_shot.py をインポート

# HTMLタグの除去と基本的なテキストクリーニングを行う関数
def clean_html_and_text(text):
    if pd.isna(text) or text == "":
        return ""
    # BeautifulSoupを使用してHTMLタグを除去
    soup = BeautifulSoup(str(text), 'html.parser')
    cleaned_text = soup.get_text()
    # 改行コードをスペースに置換し、連続する空白を一つにまとめる
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def extract_generated_content(full_text, marker="[生成するブランド紹介文]:"):
    """生成されたテキストから実際の内容部分のみを抽出"""
    try:
        # マーカーで分割
        if marker in full_text:
            parts = full_text.split(marker)
            if len(parts) > 1:
                generated_part = parts[-1].strip()
            else:
                generated_part = full_text.strip()
        else:
            generated_part = full_text.strip()
        
        # Few-shot例の混入を除去
        # "例1:", "例2:" などで始まる部分を除去
        lines = generated_part.split('\n')
        clean_lines = []
        skip_mode = False
        
        for line in lines:
            line = line.strip()
            # Few-shot例の開始を検出
            if re.match(r'^例\d+:', line) or line.startswith('ブランドライン名:') or line.startswith('既存の説明文:'):
                skip_mode = True
                continue
            # Few-shot例の終了を検出（空行または新しい例の開始）
            if skip_mode and (line == '' or re.match(r'^例\d+:', line)):
                if line == '':
                    skip_mode = False
                continue
            # 通常のコンテンツ
            if not skip_mode and line:
                clean_lines.append(line)
        
        result = ' '.join(clean_lines).strip()
        
        # さらなるクリーニング
        # "[生成するブランド紹介文]" が含まれている場合は除去
        result = re.sub(r'\[生成するブランド紹介文\]', '', result).strip()
        
        # 不完全な文の終端を検出して切り詰め
        # 最後の完全な句点で終わらせる
        sentences = result.split('。')
        if len(sentences) > 1 and sentences[-1].strip() == '':
            # 既に句点で終わっている場合
            complete_result = '。'.join(sentences[:-1]) + '。'
        elif len(sentences) > 1 and sentences[-1].strip():
            # 最後の文が不完全な場合、それを除去
            complete_result = '。'.join(sentences[:-1]) + '。'
        else:
            complete_result = result
        
        return complete_result
        
    except Exception as e:
        print(f"    警告: テキスト抽出中にエラー: {e}")
        return ""

# LLMにテキスト生成を指示し、結果を抽出する汎用関数
def llm_call(model, tokenizer, prompt_text, max_new_tokens=200, temperature=0.5, top_p=0.7):
    """
    LLMにプロンプトを渡し、テキストを生成させます。
    """
    try:
        # プロンプトの長さをより厳格に制限
        max_prompt_tokens = min(tokenizer.model_max_length - max_new_tokens - 200, 600)  # さらに厳格に
        
        print(f"    デバッグ: プロンプト文字数={len(prompt_text)}, max_prompt_tokens={max_prompt_tokens}")

        # トークン化時のエラーハンドリング
        try:
            # まずプロンプトの文字数で制限
            if len(prompt_text) > 2000:  # より厳格な文字数制限
                prompt_text = prompt_text[:2000] + "..."
                print(f"    デバッグ: プロンプトを2000文字に切り詰めました")
            
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
            
            print(f"    デバッグ: トークン化後の長さ={input_ids.shape[1]}")
            
            if input_ids.numel() == 0:
                print(f"    警告: トークナイズ後の入力が空です")
                return ""
            
            input_ids = input_ids.to(model.device)
        except Exception as e:
            print(f"    警告: トークナイズ中にエラー: {e}")
            return ""

        # 生成時のパラメータをより安全に制限
        safe_max_new_tokens = min(max_new_tokens, 200)  # 上限200
        safe_temperature = max(0.1, min(temperature, 0.8))
        safe_top_p = max(0.3, min(top_p, 0.9))

        print(f"    デバッグ: 生成パラメータ max_new_tokens={safe_max_new_tokens}, temp={safe_temperature}, top_p={safe_top_p}")

        with torch.no_grad(): 
            try:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=safe_max_new_tokens,
                    do_sample=True,
                    temperature=safe_temperature,
                    top_p=safe_top_p,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1, 
                    early_stopping=False,  # early_stoppingを無効化
                    use_cache=True
                )
                print(f"    デバッグ: 生成完了 出力長={outputs[0].shape[0]}")
            except Exception as e:
                print(f"    警告: 生成中にエラー: {e}")
                return ""
        
        # デコード時のエラーハンドリング
        try:
            # 生成された部分のみを抽出
            input_token_length = input_ids.shape[1]
            if outputs[0].shape[0] > input_token_length:
                generated_tokens = outputs[0][input_token_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""
            
            print(f"    デバッグ: 生成部分デコード完了 テキスト長={len(generated_text)}")
        except Exception as e:
            print(f"    警告: デコード中にエラー: {e}")
            return ""
        
        # テキスト抽出と後処理
        try:
            cleaned_result = extract_generated_content(generated_text)
            
            # 最終的なクリーニング
            cleaned_result = cleaned_result.replace('\n', ' ').replace('\r', '').strip()
            cleaned_result = re.sub(r'\s+', ' ', cleaned_result)
            
        except Exception as e:
            print(f"    警告: テキスト後処理中にエラー: {e}")
            cleaned_result = generated_text if generated_text else ""
            
        print(f"    デバッグ: 最終結果長={len(cleaned_result)}")
        return cleaned_result

    except Exception as e:
        print(f"    重大なエラー: llm_call関数内で予期しないエラー: {e}")
        return ""


def main(args):
    model_path = args.model_path
    data_dir = args.data_dir

    print(f"モード: {args.mode}")
    if args.mode == "validation":
        print(f"検証モードで実行します。処理サンプル数: {args.num_samples}")

    print(f"モデルを {model_path} からロードします。")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) 
        
        model_load_kwargs = {
            "device_map": "auto"
        }
        
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                model_load_kwargs["torch_dtype"] = torch.bfloat16
                print("  torch.bfloat16 を使用します。")
            else:
                model_load_kwargs["torch_dtype"] = torch.float16
                print("  torch.float16 を使用します。")
        else:
            model_load_kwargs["torch_dtype"] = torch.float32
            print("  CPUモードのため torch.float32 を使用します。")

        if args.quantize == "8bit" and torch.cuda.is_available():
            model_load_kwargs["load_in_8bit"] = True
            print("  8bit量子化を有効にします。(bitsandbytesが必要です)")
        elif args.quantize == "4bit" and torch.cuda.is_available():
            model_load_kwargs["load_in_4bit"] = True
            print("  4bit量子化を有効にします。(bitsandbytesが必要です)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_load_kwargs
        )
        model.eval() 
    except Exception as e:
        print(f"エラー: モデルまたはトークナイザーのロードに失敗しました: {e}")
        return

    if torch.cuda.is_available():
        print(f"GPUが利用可能です。モデルは {model.device} にロードされました。")
    else:
        print("GPUが利用できません。CPUでモデルを実行します。")

    try:
        blines_df = pd.read_csv(os.path.join(data_dir, "blines.csv"))
        print(f"blines.csv を読み込みました。({len(blines_df)}行)")
    except FileNotFoundError:
        print(f"エラー: {os.path.join(data_dir, 'blines.csv')} が見つかりません。")
        return
    
    if 'name' not in blines_df.columns or 'description' not in blines_df.columns:
        print("エラー: 'name' または 'description' カラムが見つかりません。")
        return

    print("ディスクリプションの前処理を開始...")
    blines_df['description_cleaned'] = blines_df['description'].apply(clean_html_and_text)
    print("ディスクリプションの前処理が完了しました。")

    if args.mode == "validation":
        target_indices = blines_df.index[:args.num_samples]
        print(f"\n検証モード: {len(target_indices)} 件のブランドラインを処理します。")
    else:
        target_indices = blines_df.index
        print(f"\n本番モード: {len(target_indices)} 件のブランドラインを処理します。")

    # 新しい説明文を格納するための一時カラム
    blines_df['description_new'] = blines_df['description_cleaned'].copy()
    successful_updates = []  # 成功したIDを記録

    print("\n--- スタイル情報特化型ディスクリプション生成を開始 ---")
    for index in tqdm(target_indices, desc="ディスクリプション生成中"):
        row = blines_df.loc[index]
        bline_id = row.get('id', '不明なID') 
        bline_name = row.get('name', f"不明なブランドライン({bline_id})")
        existing_desc_cleaned = str(row['description_cleaned'])
        
        print(f"\n処理中: LID {bline_id} ({bline_name}) - 元の説明文長: {len(existing_desc_cleaned)}文字")
        
        current_existing_desc_for_prompt = existing_desc_cleaned 
        was_presummarized = False

        # 事前要約の閾値をより小さく設定
        if len(existing_desc_cleaned) > 250:  # さらに削減
            print(f"  長文のため事前要約を実行 ({len(existing_desc_cleaned)}文字)")

            # 事前要約用のテキスト長制限をより厳格に
            if len(existing_desc_cleaned) > 800:
                input_for_pre_summary = existing_desc_cleaned[:800] + "..."
                print(f"  入力を800文字に切り詰めました")
            else:
                input_for_pre_summary = existing_desc_cleaned
                
            pre_summary_prompt_text = few_shot.PROMPT_PRE_SUMMARY.format(long_description=input_for_pre_summary)
            
            try:
                summarized_desc = llm_call(model, tokenizer, pre_summary_prompt_text, 
                                           max_new_tokens=60,  # さらに削減
                                           temperature=0.3, top_p=0.6)
                if summarized_desc and len(summarized_desc) > 20: 
                    current_existing_desc_for_prompt = summarized_desc
                    was_presummarized = True
                    print(f"  事前要約成功: {summarized_desc[:50]}...")
                else:
                    print(f"  事前要約失敗、元の説明文を使用")
            except Exception as e_summary:
                print(f"  事前要約中にエラー: {e_summary}")
        
        effective_desc_len_chars = len(current_existing_desc_for_prompt)

        try:
            # より単純なプロンプト選択
            if effective_desc_len_chars <= 30:
                prompt_to_llm = few_shot.create_simple_prompt(bline_name, "")
                chosen_prompt_template_name = "SIMPLE_NEW"
            elif effective_desc_len_chars <= 150:
                prompt_to_llm = few_shot.create_simple_prompt(bline_name, current_existing_desc_for_prompt)
                chosen_prompt_template_name = "SIMPLE_COMPLEMENT"
            else: 
                prompt_to_llm = few_shot.create_simple_prompt(bline_name, current_existing_desc_for_prompt)
                chosen_prompt_template_name = "SIMPLE_RESTRUCTURE"
                
            print(f"  使用プロンプト: {chosen_prompt_template_name}")
                
            try:
                generated_description = llm_call(model, tokenizer, prompt_to_llm, 
                                                 max_new_tokens=args.max_tokens,
                                                 temperature=args.temperature,
                                                 top_p=args.top_p)

                if generated_description and len(generated_description) >= 30:
                    blines_df.loc[index, 'description_new'] = generated_description
                    successful_updates.append(bline_id)
                    print(f"  生成成功: {generated_description[:80]}...")
                else:
                    print(f"  生成失敗または短すぎる結果: '{generated_description}'")

            except Exception as e_generate:
                print(f"  スタイル説明文生成中にエラー: {e_generate}")
                
        except Exception as e_prompt:
            print(f"  プロンプト構築中にエラー: {e_prompt}")

    print(f"\n--- ディスクリプション生成が完了しました ---")
    print(f"成功した更新: {len(successful_updates)} / {len(target_indices)} 件")

    # description カラムを新しい内容で置き換え（成功したもののみ）
    for index in blines_df.index:
        bline_id = blines_df.loc[index, 'id']
        if bline_id in successful_updates:
            blines_df.loc[index, 'description'] = blines_df.loc[index, 'description_new']

    # --- 結果の保存（指定されたカラムのみ） ---
    required_columns = ['id', 'name', 'name_second', 'name_en', 'description', 'map_genre', 'gourmet', 'brand_id', 'enabled', 'created_at', 'updated_at']
    
    # 実際に存在するカラムのみを抽出
    existing_columns_to_save = [col for col in required_columns if col in blines_df.columns]
    
    # 成功したもののみをフィルタリング
    success_mask = blines_df['id'].isin(successful_updates)
    output_df = blines_df[success_mask][existing_columns_to_save].copy()

    output_filename_parts = ["blines_updated_desc"]
    if args.mode == "validation":
        output_filename_parts.append("validation")
    output_filename_parts.append(pd.Timestamp.now().strftime("%Y%m%d%H%M%S"))
    output_filename = "_".join(output_filename_parts) + ".csv"
    output_path = os.path.join(data_dir, output_filename)

    try:
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n更新されたデータを '{output_path}' として保存しました。")
        print(f"保存されたレコード数: {len(output_df)}")
        print(f"保存されたカラム: {existing_columns_to_save}")
    except Exception as e:
        print(f"\nエラー: ファイル保存中に問題が発生しました: {e}")

    # 一時カラムを削除
    blines_df.drop(columns=['description_cleaned', 'description_new'], inplace=True, errors='ignore')

    # メモリ解放
    if 'model' in locals(): del model
    if 'tokenizer' in locals(): del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPUキャッシュを解放しました。")

    print("\n--- スクリプト完了 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ブランドラインの説明文にLLMでスタイル情報を加えます。")
    parser.add_argument("--model_path", type=str, default="/mnt/c/Users/rikuter/Carlin/model/", help="LLMモデルファイルが格納されているディレクトリのパス")
    parser.add_argument("--data_dir", type=str, default="/mnt/c/Users/rikuter/Carlin/datasets/bline_similarity/", help="データファイルが格納されているディレクトリのパス")
    parser.add_argument("--mode", type=str, choices=["validation", "production"], default="validation", help="実行モード (validation/production)")
    parser.add_argument("--num_samples", type=int, default=5, help="検証モードで処理するサンプル数")
    
    parser.add_argument("--max_tokens", type=int, default=180, help="LLMがスタイル説明文を生成する最大トークン数")
    parser.add_argument("--max_summary_tokens", type=int, default=60, help="LLMが事前要約を生成する最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.4, help="LLM生成時のtemperature")
    parser.add_argument("--top_p", type=float, default=0.7, help="LLM生成時のtop_p")

    parser.add_argument("--quantize", type=str, choices=["none", "8bit", "4bit"], default="none", help="モデルの量子化")
    
    args = parser.parse_args()
    
    main(args)