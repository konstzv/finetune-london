#!/usr/bin/env python3
import json
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VALID_CATEGORIES = {"FOOD_AND_DRINKS", "ACTIVITIES", "PLACES", "UNCATEGORIZED"}
CONFIDENCE_THRESHOLD = 85

CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"

COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}

SYSTEM_PROMPT = """You classify London Reddit posts into one of: FOOD_AND_DRINKS, ACTIVITIES, PLACES, UNCATEGORIZED

Rules:
- If there's a scheduled activity (match, meetup, concert, party) → ACTIVITIES
- If it's about a place to visit (park, museum, landmark) → PLACES
- Bars, pubs, restaurants, cafes, food questions → FOOD_AND_DRINKS
- Everything else → UNCATEGORIZED

Respond with ONLY valid JSON, no other text:
{"category": "<CATEGORY>", "confidence": <0-100>}"""


def call_model(model, user_input):
    """Call a model and return parsed result."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=50,
    )
    latency = time.time() - start

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens / 1000) * COST_PER_1K[model]["input"] + \
           (output_tokens / 1000) * COST_PER_1K[model]["output"]

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        category = parsed.get("category", "")
        confidence = parsed.get("confidence", 0)
    except json.JSONDecodeError:
        category = None
        confidence = 0

    if category not in VALID_CATEGORIES:
        category = None
        confidence = 0

    return {
        "model": model,
        "category": category,
        "confidence": confidence,
        "latency": round(latency, 2),
        "cost": round(cost, 6),
        "raw": raw,
    }


def classify(user_input):
    """Route: cheap model first, escalate if unsure."""

    # Step 1: Try cheap model
    cheap = call_model(CHEAP_MODEL, user_input)

    if cheap["category"] and cheap["confidence"] >= CONFIDENCE_THRESHOLD:
        # Confident — accept cheap model's answer
        return {
            "input": user_input[:80],
            "category": cheap["category"],
            "routed_to": CHEAP_MODEL,
            "escalated": False,
            "cheap": cheap,
            "strong": None,
            "answer_changed": False,
            "latency": cheap["latency"],
            "cost": cheap["cost"],
        }

    # Step 2: Not confident — escalate to strong model
    strong = call_model(STRONG_MODEL, user_input)

    return {
        "input": user_input[:80],
        "category": strong["category"],
        "routed_to": STRONG_MODEL,
        "escalated": True,
        "cheap": cheap,
        "strong": strong,
        "answer_changed": cheap["category"] != strong["category"],
        "latency": round(cheap["latency"] + strong["latency"], 2),
        "cost": round(cheap["cost"] + strong["cost"], 6),
    }


# === Test Cases ===

test_inputs = [
    # Clean — should stay on cheap model
    ("Best pizza in Soho?", "Subreddit: r/london\nTitle: Best pizza in Soho?\nBody: Want proper Neapolitan pizza, not Dominos\nFlair: Food"),
    ("Pub quiz at The Lamb", "Subreddit: r/LondonSocialClub\nTitle: [03/05/26] Pub quiz @ The Lamb\nBody: Weekly pub quiz, teams of 6, starts 7:30pm\nFlair: Going Ahead"),
    ("Hampstead Heath views", "Subreddit: r/london\nTitle: Hampstead Heath in autumn\nBody: The views from Parliament Hill are incredible this time of year\nFlair: None"),
    ("Jubilee line packed", "Subreddit: r/london\nTitle: Is the Jubilee line always this packed?\nBody: Moved here from Leeds, the commute is killing me\nFlair: None"),

    # Borderline — might escalate
    ("Pub with board games", "Subreddit: r/london\nTitle: Pub with board games?\nBody: Want somewhere to play games and have a pint\nFlair: None"),
    ("Brewery tour + live music", "Subreddit: r/london\nTitle: Brewery tour with live music\nBody: This place does tours of their brewery and has a band playing in the taproom after\nFlair: None"),
    ("Rooftop cinema + cocktails", "Subreddit: r/london\nTitle: Rooftop cinema and cocktails\nBody: They show old movies on a rooftop in Peckham and serve drinks. Is it worth the price?\nFlair: None"),
    ("Yoga in Regent's Park", "Subreddit: r/london\nTitle: Yoga in the park\nBody: Free yoga sessions at Regent's Park every Saturday morning. Bring your own mat.\nFlair: None"),

    # Noisy — should probably escalate
    ("gibberish", "asdkjh askjdh 2k3j4h london food maybe???"),
    ("just 'london'", "london"),
    ("spam", "buy cheap viagra online best prices london pharmacy"),
    ("moving to London", "Subreddit: r/london\nTitle: Moving to London from NYC\nBody: Any tips? I work in finance. Where should I live, where should I eat, what should I do on weekends?\nFlair: None"),
]

print("=" * 90)
print(f"{'INPUT':<30} {'MODEL':<15} {'CATEGORY':<20} {'CONF':>4}  {'COST':>8}  {'TIME':>5}  {'CHANGED'}")
print("=" * 90)

results = []
for label, inp in test_inputs:
    r = classify(inp)
    results.append(r)
    changed = "YES" if r["answer_changed"] else ""
    print(f"  {label:<28} {r['routed_to']:<15} {str(r['category']):<20} {r['cheap']['confidence']:>4}  ${r['cost']:<7.5f}  {r['latency']:>4.1f}s  {changed}")

# === Summary ===
cheap_count = sum(1 for r in results if not r["escalated"])
escalated_count = sum(1 for r in results if r["escalated"])
changed_count = sum(1 for r in results if r["answer_changed"])
total_cost = sum(r["cost"] for r in results)
total_latency = sum(r["latency"] for r in results)

cheap_only_cost = sum(r["cheap"]["cost"] for r in results)
strong_only_cost = sum(call_model(STRONG_MODEL, inp)["cost"] for _, inp in []) # hypothetical

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"  Total requests:          {len(results)}")
print(f"  Stayed on {CHEAP_MODEL}:    {cheap_count}")
print(f"  Escalated to {STRONG_MODEL}:      {escalated_count}")
print(f"  Answer changed on escalation: {changed_count}")
print(f"  Total cost:              ${total_cost:.5f}")
print(f"  Total latency:           {total_latency:.1f}s")
print(f"  Avg cost (cheap only):   ${sum(r['cost'] for r in results if not r['escalated']) / max(cheap_count,1):.5f}")
print(f"  Avg cost (escalated):    ${sum(r['cost'] for r in results if r['escalated']) / max(escalated_count,1):.5f}")
print(f"  Avg latency (cheap):     {sum(r['latency'] for r in results if not r['escalated']) / max(cheap_count,1):.2f}s")
print(f"  Avg latency (escalated): {sum(r['latency'] for r in results if r['escalated']) / max(escalated_count,1):.2f}s")

# Save results
with open("routing_results.json", "w") as f:
    json.dump({
        "config": {
            "cheap_model": CHEAP_MODEL,
            "strong_model": STRONG_MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        "summary": {
            "total": len(results),
            "stayed_cheap": cheap_count,
            "escalated": escalated_count,
            "answer_changed": changed_count,
            "total_cost": round(total_cost, 6),
            "total_latency": round(total_latency, 2),
        },
        "results": results,
    }, f, indent=2)

print(f"\nSaved to routing_results.json")
