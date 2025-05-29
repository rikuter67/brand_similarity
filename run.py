import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
import argparse
from bs4 import BeautifulSoup
import re
import few_shot # few_shot.py をインポート

def clean_html_and_text(text):
    if pd.isna(text) or text == "":
        return ""
    soup = BeautifulSoup(str(text), 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def llm_call(model, tokenizer, prompt_text, max_new_tokens=400, temperature=0.7, top_p=0.9):
    """ 汎用的なLLM呼び出し関数 """
    inputs = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=1800).to(model.device) # プロンプト全体の長さを考慮
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1
        )
    
    # 生成部分のみをデコード (入力トークン長を利用)
    input_token_length = len(inputs[0])
    generated_tokens = outputs[0][input_token_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    if not generated_text: # 何らかの理由で空の場合、フルテキストでデバッグ (プロンプトがそのまま返るなど)
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # marker = prompt_text.rsplit("\n",1)[-1] # 最後の行のマーカーで分割を試みる
        # if marker in full_generated_text:
        #     generated_text = full_generated_text.rsplit(marker,1)[-1].strip()
        # else:
        print(f"\n警告: 生成テキストの抽出に問題がある可能性があります。フルテキストを確認してください。: {full_generated_text[:200]}")
        # プロンプトがそのまま返ってきている場合は、そこから分離する必要がある。
        # ここでは一旦、デコードされたものをそのまま返す（要改善）
        # 最も単純には、プロンプトの末尾の指示（例："[要約文]"）で分割するなど。
        # しかし、LLMが指示通りに生成しない場合もある。
        # 以下のmarkerは汎用的ではないため、llm_generate_description のように特化させるか、
        # より堅牢な抽出ロジックが必要。
        # last_prompt_line = prompt_text.strip().splitlines()[-1]
        # if last_prompt_line and last_prompt_line in full_generated_text:
        #     generated_text = full_generated_text.split(last_prompt_line,1)[-1].strip()
        # else:
        #     generated_text = full_generated_text # 抽出失敗時はフル
    
    return generated_text.replace('\n', ' ').replace('\r', '').strip()


def llm_generate_styled_description(model, tokenizer, prompt_text, max_new_tokens=500, temperature=0.75, top_p=0.9):
    """ スタイル情報特化型説明文生成用のLLM呼び出し (マーカーベース抽出) """
    inputs = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=1800).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1
        )
    
    full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "[生成するブランド紹介文]:" # few_shot.py のテンプレートの最後の部分
    generated_part = ""
    if marker in full_generated_text:
        generated_part = full_generated_text.rsplit(marker, 1)[-1].strip()
    else:
        # フォールバック: 入力トークン長を使って生成部分のみデコード
        input_token_length = len(inputs[0])
        generated_tokens = outputs[0][input_token_length:]
        generated_part = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if not generated_part:
            print(f"\n警告: スタイル説明文の抽出に失敗しました。フルテキスト: {full_generated_text[:200]}")
            # generated_part = full_generated_text # 必要ならフルで返す

    return generated_part.replace('\n', ' ').replace('\r', '').strip()


def main(args):
    model_path = "/mnt/c/Users/rikuter/Carlin/model/"
    data_dir = "/mnt/c/Users/rikuter/Carlin/datasets/bline_similarity/"

    print(f"モード: {args.mode}")
    if args.mode == "validation":
        print(f"検証モードで実行します。処理サンプル数: {args.num_samples}")

    print(f"モデルを {model_path} からロードします。")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # bfloat16が使えるなら精度向上
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"エラー: モデルまたはトークナイザーのロードに失敗しました: {e}")
        return

    if torch.cuda.is_available():
        print(f"GPUが利用可能です。使用デバイス: {model.device}")
    else:
        print("GPUが利用できません。CPUでモデルを実行します。")

    try:
        blines_df = pd.read_csv(os.path.join(data_dir, "blines.csv"))
        print("blines.csv を読み込みました。")
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
        if len(target_indices) > 100:
            print("多数の処理には時間がかかることがあります。")

    generated_description_column = 'description_styled'
    blines_df[generated_description_column] = "" 
    
    pre_summary_column = 'description_presummarized' # 事前要約格納用
    blines_df[pre_summary_column] = ""


    completed_logs = []

    print("\n--- スタイル情報特化型ディスクリプション生成を開始 ---")
    for index in tqdm(target_indices, desc="ディスクリプション生成中"):
        row = blines_df.loc[index]
        bline_name = row.get('name', f"不明なブランドライン({row.get('id', '不明なID')})")
        existing_desc_cleaned = row['description_cleaned']
        current_existing_desc_for_prompt = existing_desc_cleaned # デフォルト

        chosen_prompt_template_str: str
        formatted_few_shot_examples: str
        prompt_to_llm: str

        desc_len_chars = len(existing_desc_cleaned)
        
        # --- 改善案2: 長文の場合の事前要約ステップ ---
        if desc_len_chars > args.presummary_threshold: # 例: 800文字を超える場合は事前要約
            print(f"\n  LID {row.get('id', 'N/A')} ({bline_name}): 長文のため事前要約を実行します ({desc_len_chars}文字)。")
            pre_summary_prompt = few_shot.PROMPT_PRE_SUMMARY.format(long_description=existing_desc_cleaned)
            try:
                summarized_desc = llm_call(model, tokenizer, pre_summary_prompt, max_new_tokens=args.max_summary_tokens, temperature=0.5) # 要約はより事実に忠実に
                if summarized_desc and len(summarized_desc) > 20: # ある程度の長さがあるか
                    current_existing_desc_for_prompt = summarized_desc
                    blines_df.loc[index, pre_summary_column] = summarized_desc
                    print(f"    事前要約結果 (ID {row.get('id', 'N/A')}): {summarized_desc[:100]}...")
                else:
                    print(f"    警告: ID {row.get('id', 'N/A')} の事前要約が短すぎるか空です。元の説明文を使用します。")
                    # current_existing_desc_for_prompt は元のまま
            except Exception as e_summary:
                print(f"\n  エラー: ID {row.get('id', 'N/A')} ({bline_name}) の事前要約中にエラー: {e_summary}")
                # エラー時は元の説明文を使用 (current_existing_desc_for_prompt は元のまま)
        
        # 実際のプロンプト選択は、事前要約後の長さか、元の長さか、どちらを基準にするか明確に
        # ここでは、事前要約後の current_existing_desc_for_prompt を基準とする
        effective_desc_len = len(current_existing_desc_for_prompt)

        if effective_desc_len > args.summarize_restructure_threshold: # 例: 事前要約後でもまだ長い、または元々中程度の長さ
            chosen_prompt_template_str = few_shot.PROMPT_TEMPLATE_SUMMARIZE_RESTRUCTURE
            formatted_few_shot_examples = few_shot.get_formatted_few_shot_examples(bline_name, current_existing_desc_for_prompt)
            prompt_to_llm = chosen_prompt_template_str.format(
                bline_name=bline_name,
                existing_description_cleaned=current_existing_desc_for_prompt, # 事前要約されたもの or 元の短いもの
                few_shot_examples_formatted=formatted_few_shot_examples
            )
        elif effective_desc_len > args.generate_complement_threshold: # ある程度情報がある場合
            chosen_prompt_template_str = few_shot.PROMPT_TEMPLATE_GENERATE_COMPLEMENT
            formatted_few_shot_examples = few_shot.get_formatted_few_shot_examples(bline_name, current_existing_desc_for_prompt)
            prompt_to_llm = chosen_prompt_template_str.format(
                bline_name=bline_name,
                existing_description_cleaned=current_existing_desc_for_prompt,
                few_shot_examples_formatted=formatted_few_shot_examples
            )
        else: # 既存情報がほぼない場合
            chosen_prompt_template_str = few_shot.PROMPT_TEMPLATE_GENERATE_NEW
            formatted_few_shot_examples = few_shot.get_formatted_few_shot_examples_for_new_generation(bline_name)
            prompt_to_llm = chosen_prompt_template_str.format(
                bline_name=bline_name,
                few_shot_examples_formatted=formatted_few_shot_examples
            )

        try:
            # print(f"\n--- Final Prompt for {bline_name} ---\n{prompt_to_llm[:500]}...\n") # デバッグ用
            generated_description = llm_generate_styled_description(model, tokenizer, prompt_to_llm, max_new_tokens=args.max_tokens, temperature=args.temperature)

            if not generated_description or len(generated_description) < 50: # 最低限の文字数チェック
                print(f"\n警告: ID {row.get('id', 'N/A')} ({bline_name}) の最終生成結果が短すぎるか空です。元のcleaned説明文を維持します。 生成文: '{generated_description}'")
                blines_df.loc[index, generated_description_column] = existing_desc_cleaned # クリーニング済みの元の説明を維持
            else:
                blines_df.loc[index, generated_description_column] = generated_description
            
            completed_logs.append({
                'bline_id': row.get('id', 'N/A'),
                'bline_name': bline_name,
                'original_cleaned_desc_len': desc_len_chars,
                'used_desc_for_prompt_len': effective_desc_len,
                'pre_summarized': blines_df.loc[index, pre_summary_column] != "",
                'chosen_prompt_type': chosen_prompt_template_str.splitlines()[0],
                'generated_description': blines_df.loc[index, generated_description_column]
            })

        except Exception as e_generate:
            print(f"\nエラー: ID {row.get('id', 'N/A')} ({bline_name}) のスタイル説明文生成中にエラー: {e_generate}")
            blines_df.loc[index, generated_description_column] = existing_desc_cleaned


    print("\n--- ディスクリプション生成が完了しました ---")
    if completed_logs:
        results_df = pd.DataFrame(completed_logs)
        print("\n生成されたディスクリプションの例 (最大5件):")
        print(results_df[['bline_name', 'generated_description']].head())
        
        # ログファイルも保存
        log_filename = f"generation_log_{pd.Timestamp.now().strftime('%Y%m%d%H%M')}.csv"
        log_output_path = os.path.join(data_dir, log_filename)
        try:
            results_df.to_csv(log_output_path, index=False, encoding='utf-8-sig')
            print(f"生成ログを '{log_output_path}' として保存しました。")
        except Exception as e_log:
            print(f"エラー: ログファイル保存中に問題が発生しました: {e_log}")


    output_filename_parts = ["blines_with_AI_styled_descriptions"]
    if args.mode == "validation":
        output_filename_parts.append("validation")
    output_filename_parts.append(pd.Timestamp.now().strftime("%Y%m%d%H%M"))
    output_filename = "_".join(output_filename_parts) + ".csv"
    output_path = os.path.join(data_dir, output_filename)

    try:
        blines_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n処理後のデータフレームを '{output_path}' として保存しました。")
    except Exception as e:
        print(f"\nエラー: ファイル保存中に問題が発生しました: {e}")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPUキャッシュを解放しました。")

    print("\n--- スクリプト完了 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ブランドラインの説明文にLLMでスタイル情報を加えます。")
    parser.add_argument("--mode", type=str, choices=["validation", "production"], default="validation", help="実行モード (validation/production)")
    parser.add_argument("--num_samples", type=int, default=3, help="検証モードで処理するサンプル数")
    parser.add_argument("--max_tokens", type=int, default=500, help="LLMがスタイル説明文を生成する最大トークン数")
    parser.add_argument("--max_summary_tokens", type=int, default=150, help="LLMが事前要約を生成する最大トークン数 (400字程度を想定)")
    parser.add_argument("--temperature", type=float, default=0.75, help="LLM生成時のtemperature")
    parser.add_argument("--presummary_threshold", type=int, default=800, help="この文字数以上の既存説明文は事前要約対象 (元の文字数)")
    parser.add_argument("--summarize_restructure_threshold", type=int, default=200, help="この文字数以上の入力説明文(事前要約後含む)は要約・再構成プロンプトを使用")
    parser.add_argument("--generate_complement_threshold", type=int, default=30, help="この文字数以上の入力説明文は補完・生成プロンプトを使用 (これ未満は新規生成)")
    
    args = parser.parse_args()
    main(args)