#!/usr/bin/env python3
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("eval.jsonl") as f:
    examples = [json.loads(line) for line in f]

examples = examples[:10]

results = []
correct = 0

for i, example in enumerate(examples, 1):
    system_msg = example["messages"][0]
    user_msg = example["messages"][1]
    expected = example["messages"][2]["content"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg, user_msg],
        temperature=0,
        max_tokens=20,
    )

    predicted = response.choices[0].message.content.strip()
    is_correct = predicted == expected

    if is_correct:
        correct += 1

    result = {
        "example": i,
        "user_input": user_msg["content"][:80],
        "expected": expected,
        "predicted": predicted,
        "correct": is_correct,
    }
    results.append(result)
    print(f"{i}/10  expected={expected:<20} predicted={predicted:<20} {'OK' if is_correct else 'WRONG'}")

print(f"\nAccuracy: {correct}/10 ({correct * 10}%)")

with open("baseline_results.json", "w") as f:
    json.dump({"accuracy": f"{correct}/10", "results": results}, f, indent=2)

print("Saved to baseline_results.json")
