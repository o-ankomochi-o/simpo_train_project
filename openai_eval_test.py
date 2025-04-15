import os

import openai

# OPENAI_API_KEY の環境変数を読み込む
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY が設定されていません。環境変数を確認してください。"
    )

# OpenAI API クライアントを設定
client = openai.OpenAI(api_key=api_key)

# テスト用のプロンプトと生成テキスト
prompt = "健康的な朝食について説明してください。"
generated_text = (
    "朝食には、オートミールやフルーツ、ヨーグルトを取り入れるとよいでしょう。"
)

# 評価プロンプト
evaluation_prompt = f"""
評価してください:

プロンプト:
{prompt}

生成されたテキスト:
{generated_text}

以下の項目について0〜5の数値で評価し、それぞれの理由も説明してください：
1. 関連性: プロンプトの内容に関連しているか
2. 多様性: 表現や語彙の多様性があるか
3. 訴求点: 異なる観点や主張を含んでいるか
4. 読みやすさ: 文章構造や流れの自然さ
5. 全体評価: 総合的な質

結果は以下のJSON形式で返してください:
```json
{{
  "relevance": {{"score": 数値, "reason": "理由"}},
  "diversity": {{"score": 数値, "reason": "理由"}},
  "appeals": {{"score": 数値, "reason": "理由"}},
  "readability": {{"score": 数値, "reason": "理由"}},
  "overall": {{"score": 数値, "reason": "理由"}}
}}
"""
# API呼び出し
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "あなたは文章評価の専門家です。指示に従って生成された文章を評価してください。",
        },
        {"role": "user", "content": evaluation_prompt},
    ],
    temperature=0,
)

# 結果の表示
print("\n=== 評価結果 ===")
print(response.choices[0].message.content.strip())
