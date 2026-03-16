import json
from openai import OpenAI

client = OpenAI()

with open("dataset.json") as f:
    data = json.load(f)

for item in data:
    prompt = item["prompt"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print("PROMPT:", prompt)
    print("MODEL:", answer)
    print("EXPECTED:", item["expected"])
    print("-" * 40)
    print("Japanese LLM evaluation toolkit")
