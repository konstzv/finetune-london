#!/usr/bin/env python3
import json
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VALID_CATEGORIES = {"FOOD_AND_DRINKS", "ACTIVITIES", "PLACES", "UNCATEGORIZED"}
CONFIDENCE_THRESHOLD = 85
REDUNDANCY_RUNS = 3

SYSTEM_PROMPT_SCORING = """You classify London Reddit posts into one of: FOOD_AND_DRINKS, ACTIVITIES, PLACES, UNCATEGORIZED

Rules:
- If there's a scheduled activity (match, meetup, concert, party) → ACTIVITIES
- If it's about a place to visit (park, museum, landmark) → PLACES
- Bars, pubs, restaurants, cafes, food questions → FOOD_AND_DRINKS
- Everything else → UNCATEGORIZED

Respond with ONLY valid JSON, no other text:
{"category": "<CATEGORY>", "confidence": <0-100>}"""


def call_with_scoring(user_input):
    """Approach D: model returns category + confidence score."""
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SCORING},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=50,
    )
    latency = time.time() - start
    tokens = response.usage.total_tokens
    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        category = parsed.get("category", "")
        confidence = parsed.get("confidence", 0)
    except json.JSONDecodeError:
        return {"category": None, "confidence": 0, "raw": raw, "latency": latency, "tokens": tokens}

    if category not in VALID_CATEGORIES:
        return {"category": None, "confidence": 0, "raw": raw, "latency": latency, "tokens": tokens}

    return {"category": category, "confidence": confidence, "raw": raw, "latency": latency, "tokens": tokens}


def call_with_redundancy(user_input):
    """Approach B: run 3 times, compare answers."""
    system_msg = "You classify London Reddit posts into one of: FOOD_AND_DRINKS, ACTIVITIES, PLACES, UNCATEGORIZED\n\nRespond with ONLY the category name, nothing else."
    answers = []
    total_latency = 0
    total_tokens = 0

    for _ in range(REDUNDANCY_RUNS):
        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7,
            max_tokens=20,
        )
        total_latency += time.time() - start
        total_tokens += response.usage.total_tokens
        answers.append(response.choices[0].message.content.strip())

    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1

    best = max(counts, key=counts.get)
    agreement = counts[best] / REDUNDANCY_RUNS

    return {
        "answers": answers,
        "best": best,
        "agreement": agreement,
        "latency": total_latency,
        "tokens": total_tokens,
    }


def classify(user_input):
    """Combined pipeline: scoring + redundancy."""

    # Step 1: Scoring
    scoring = call_with_scoring(user_input)

    # Step 2: Decide if we need redundancy
    if scoring["category"] is None:
        status = "FAIL"
        final_category = None
        redundancy = None
    elif scoring["confidence"] >= CONFIDENCE_THRESHOLD:
        # High confidence — trust scoring, skip redundancy
        status = "OK"
        final_category = scoring["category"]
        redundancy = None
    else:
        # Low confidence — run redundancy check
        redundancy = call_with_redundancy(user_input)

        if redundancy["agreement"] == 1.0 and redundancy["best"] in VALID_CATEGORIES:
            status = "OK"
            final_category = redundancy["best"]
        elif redundancy["agreement"] >= 0.66 and redundancy["best"] in VALID_CATEGORIES:
            status = "UNSURE"
            final_category = redundancy["best"]
        else:
            status = "FAIL"
            final_category = None

    total_latency = scoring["latency"] + (redundancy["latency"] if redundancy else 0)
    total_tokens = scoring["tokens"] + (redundancy["tokens"] if redundancy else 0)

    return {
        "input": user_input[:80],
        "category": final_category,
        "status": status,
        "confidence": scoring["confidence"],
        "scoring_category": scoring["category"],
        "redundancy": redundancy,
        "latency_s": round(total_latency, 2),
        "tokens": total_tokens,
    }


# === Test Cases ===

clean_inputs = [
    "Subreddit: r/london\nTitle: Best pizza in Soho?\nBody: Want proper Neapolitan pizza, not Dominos\nFlair: Food",
    "Subreddit: r/LondonSocialClub\nTitle: [03/05/26] Pub quiz @ The Lamb\nBody: Weekly pub quiz, teams of 6, starts 7:30pm\nFlair: Going Ahead",
    "Subreddit: r/london\nTitle: Hampstead Heath in autumn\nBody: The views from Parliament Hill are incredible this time of year\nFlair: None",
    "Subreddit: r/london\nTitle: Is the Jubilee line always this packed?\nBody: Moved here from Leeds, the commute is killing me\nFlair: None",
]

borderline_inputs = [
    "Subreddit: r/london\nTitle: Pub with board games?\nBody: Want somewhere to play games and have a pint\nFlair: None",
    "Subreddit: r/london\nTitle: Street food at Victoria Park\nBody: The Sunday market there has amazing dumplings\nFlair: None",
    "Subreddit: r/london\nTitle: Open mic night at a cafe\nBody: This cafe does poetry readings and serves great coffee\nFlair: None",
    "Subreddit: r/london\nTitle: Walking tour of old pubs\nBody: Did a tour of historic pubs in the City. Ye Olde Cheshire Cheese was the highlight\nFlair: None",
    "Subreddit: r/london\nTitle: Yoga in the park\nBody: Free yoga sessions at Regent's Park every Saturday morning. Bring your own mat.\nFlair: None",
    "Subreddit: r/london\nTitle: Brewery tour with live music\nBody: This place does tours of their brewery and has a band playing in the taproom after\nFlair: None",
    "Subreddit: r/london\nTitle: Rooftop cinema and cocktails\nBody: They show old movies on a rooftop in Peckham and serve drinks. Is it worth the price?\nFlair: None",
]

noisy_inputs = [
    "asdkjh askjdh 2k3j4h london food maybe???",
    "Title: \nBody: \nFlair: ",
    "Subreddit: r/cats\nTitle: My cat knocked over the Christmas tree again\nBody: Third year in a row. Why do I even bother.\nFlair: None",
    "buy cheap viagra online best prices london pharmacy",
    "london",
    "🔥🔥🔥 BEST DEALS 50% OFF click here now!!!",
    "Subreddit: r/london\nTitle: yes\nBody: no\nFlair: None",
    "Subreddit: r/london\nTitle: Moving to London from NYC\nBody: Any tips? I work in finance and my office is in Canary Wharf. Where should I live, where should I eat, what should I do on weekends?\nFlair: None",
]

print("=" * 70)
print("CLEAN INPUTS")
print("=" * 70)
clean_results = []
for inp in clean_inputs:
    r = classify(inp)
    clean_results.append(r)
    print(f"  [{r['status']}] {r['category']:<20} conf={r['confidence']:>3}  latency={r['latency_s']}s  tokens={r['tokens']}")

print("\n" + "=" * 70)
print("BORDERLINE INPUTS")
print("=" * 70)
border_results = []
for inp in borderline_inputs:
    r = classify(inp)
    border_results.append(r)
    print(f"  [{r['status']}] {r['category']:<20} conf={r['confidence']:>3}  latency={r['latency_s']}s  tokens={r['tokens']}")
    if r["redundancy"]:
        print(f"         redundancy: {r['redundancy']['answers']}  agreement={r['redundancy']['agreement']}")

print("\n" + "=" * 70)
print("NOISY INPUTS")
print("=" * 70)
noisy_results = []
for inp in noisy_inputs:
    r = classify(inp)
    noisy_results.append(r)
    print(f"  [{r['status']}] {str(r['category']):<20} conf={r['confidence']:>3}  latency={r['latency_s']}s  tokens={r['tokens']}")
    if r["redundancy"]:
        print(f"         redundancy: {r['redundancy']['answers']}  agreement={r['redundancy']['agreement']}")

# === Summary ===
all_results = clean_results + border_results + noisy_results

ok_count = sum(1 for r in all_results if r["status"] == "OK")
unsure_count = sum(1 for r in all_results if r["status"] == "UNSURE")
fail_count = sum(1 for r in all_results if r["status"] == "FAIL")
redundancy_count = sum(1 for r in all_results if r["redundancy"] is not None)
total_latency = sum(r["latency_s"] for r in all_results)
total_tokens = sum(r["tokens"] for r in all_results)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total requests:        {len(all_results)}")
print(f"  Accepted (OK):         {ok_count}")
print(f"  Uncertain (UNSURE):    {unsure_count}")
print(f"  Rejected (FAIL):       {fail_count}")
print(f"  Needed redundancy:     {redundancy_count}")
print(f"  Total latency:         {total_latency:.1f}s")
print(f"  Total tokens:          {total_tokens}")
print(f"  Avg latency (OK):      {sum(r['latency_s'] for r in all_results if r['status']=='OK') / max(ok_count,1):.2f}s")
print(f"  Avg latency (UNSURE):  {sum(r['latency_s'] for r in all_results if r['status']=='UNSURE') / max(unsure_count,1):.2f}s")

# Save results
with open("inference_results.json", "w") as f:
    json.dump({
        "summary": {
            "total": len(all_results),
            "ok": ok_count,
            "unsure": unsure_count,
            "fail": fail_count,
            "redundancy_needed": redundancy_count,
            "total_latency_s": round(total_latency, 2),
            "total_tokens": total_tokens,
        },
        "clean": clean_results,
        "borderline": border_results,
        "noisy": noisy_results,
    }, f, indent=2)

print("\nSaved to inference_results.json")
