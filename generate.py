import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # 正しいローカルモデルのパスを指定
    model_path = "/home/www_kyoko_ogawa/simpo_train_project/output/simpo-trained-model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # pad_token_id を設定 (もしトークナイザーに設定されていない場合)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # テキストを生成するための入力
    input_text = """
与えられた検索クエリと文章に基づいて、15字以内でユーザーにアピールするキャッチコピーを作成してください。

検索クエリ:おすすめ 医療保険
文章:医療保険の人気おすすめランキングを発表！ナビナビ保険で資料請求が多い保険商品を比較して、ランキング形式で公開いたします。実際にユーザーに選ばれた上位の保険商品順になっているので、ぜひ参考にしてください！
出力:
"""
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # 文章生成
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,  # 明示的に pad_token_id を設定
    )

    # 結果をデコードして表示
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    main()
