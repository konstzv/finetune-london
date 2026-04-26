#!/usr/bin/env python3
import json
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"

VALID_CATEGORIES = ["FOOD_AND_DRINKS", "ACTIVITIES", "PLACES", "UNCATEGORIZED"]
VALID_PRICE = ["FREE", "BUDGET", "MID", "PREMIUM", "UNKNOWN"]
VALID_BEST_FOR = ["SOLO", "COUPLES", "FRIENDS", "FAMILIES", "ANYONE"]
VALID_NEIGHBORHOODS = [
    "SOHO", "CAMDEN", "SHOREDITCH", "BRIXTON", "PECKHAM", "HACKNEY",
    "WESTMINSTER", "SOUTH_BANK", "KENSINGTON", "GREENWICH", "ISLINGTON",
    "CLAPHAM", "DALSTON", "BERMONDSEY", "CITY", "FITZROVIA", "COVENT_GARDEN",
    "NOTTING_HILL", "FULHAM", "BLOOMSBURY", "OTHER", "UNKNOWN",
]


# =====================================================================
# VARIANT A: MONOLITHIC — one prompt, one call
# =====================================================================

MONOLITHIC_PROMPT = """You extract structured data from London Reddit posts.

Return ONLY valid JSON with these fields:
{
  "category": one of FOOD_AND_DRINKS | ACTIVITIES | PLACES | UNCATEGORIZED,
  "neighborhood": one of SOHO | CAMDEN | SHOREDITCH | BRIXTON | PECKHAM | HACKNEY | WESTMINSTER | SOUTH_BANK | KENSINGTON | GREENWICH | ISLINGTON | CLAPHAM | DALSTON | BERMONDSEY | CITY | FITZROVIA | COVENT_GARDEN | NOTTING_HILL | FULHAM | BLOOMSBURY | OTHER | UNKNOWN,
  "price_range": one of FREE | BUDGET | MID | PREMIUM | UNKNOWN,
  "best_for": one of SOLO | COUPLES | FRIENDS | FAMILIES | ANYONE,
  "summary": 1-2 sentence recommendation
}"""


def monolithic(user_input):
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MONOLITHIC_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=200,
    )
    latency = time.time() - start
    tokens = response.usage.total_tokens
    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"error": "invalid JSON", "raw": raw}

    return {"result": result, "latency": round(latency, 2), "tokens": tokens, "calls": 1}


# =====================================================================
# VARIANT B: MULTI-STAGE — 3 steps
# =====================================================================

STAGE1_PROMPT = """You normalize Reddit posts for processing.

Extract and return ONLY valid JSON:
{
  "title": the post title (cleaned, no special chars),
  "body": the post body (first 200 chars, cleaned),
  "subreddit": subreddit name without r/,
  "has_date": true if post mentions a specific date,
  "has_location": true if post mentions a specific place/area,
  "language": "en" or detected language code
}"""

STAGE2_PROMPT = """You classify and extract fields from a normalized London Reddit post.

Return ONLY valid JSON:
{
  "category": one of FOOD_AND_DRINKS | ACTIVITIES | PLACES | UNCATEGORIZED,
  "neighborhood": one of SOHO | CAMDEN | SHOREDITCH | BRIXTON | PECKHAM | HACKNEY | WESTMINSTER | SOUTH_BANK | KENSINGTON | GREENWICH | ISLINGTON | CLAPHAM | DALSTON | BERMONDSEY | CITY | FITZROVIA | COVENT_GARDEN | NOTTING_HILL | FULHAM | BLOOMSBURY | OTHER | UNKNOWN,
  "price_range": one of FREE | BUDGET | MID | PREMIUM | UNKNOWN,
  "best_for": one of SOLO | COUPLES | FRIENDS | FAMILIES | ANYONE
}

Rules:
- Scheduled activity (meetup, concert, match, quiz) → ACTIVITIES
- Place to visit (park, museum, landmark) → PLACES
- Bar, pub, restaurant, cafe, food question → FOOD_AND_DRINKS
- Everything else → UNCATEGORIZED
- If no neighborhood mentioned → UNKNOWN
- If no price info → UNKNOWN
- If audience unclear → ANYONE"""

STAGE3_PROMPT = """You write short recommendation summaries for a London recommendations app.

Given the original post and extracted fields, write a 1-2 sentence summary.
Be specific and helpful. Mention the neighborhood if known.

Return ONLY valid JSON:
{
  "summary": "your 1-2 sentence summary"
}"""


def call_stage(prompt, user_content):
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=200,
    )
    latency = time.time() - start
    tokens = response.usage.total_tokens
    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"error": "invalid JSON", "raw": raw}

    return result, latency, tokens


def multistage(user_input):
    total_latency = 0
    total_tokens = 0

    # Stage 1: Normalize
    stage1, lat, tok = call_stage(STAGE1_PROMPT, user_input)
    total_latency += lat
    total_tokens += tok

    # Stage 2: Extract & Classify (feed stage 1 output)
    stage2_input = json.dumps(stage1)
    stage2, lat, tok = call_stage(STAGE2_PROMPT, stage2_input)
    total_latency += lat
    total_tokens += tok

    # Stage 3: Summarize (feed original + stage 2 fields)
    stage3_input = f"Original post:\n{user_input}\n\nExtracted fields:\n{json.dumps(stage2)}"
    stage3, lat, tok = call_stage(STAGE3_PROMPT, stage3_input)
    total_latency += lat
    total_tokens += tok

    result = {**stage2, **stage3}

    return {
        "result": result,
        "stages": {"normalize": stage1, "classify": stage2, "summarize": stage3},
        "latency": round(total_latency, 2),
        "tokens": total_tokens,
        "calls": 3,
    }


# =====================================================================
# VALIDATION
# =====================================================================

def validate_result(result):
    """Check if all fields exist and have valid enum values."""
    issues = []
    data = result.get("result", {})

    if "error" in data:
        return ["invalid JSON response"]

    if data.get("category") not in VALID_CATEGORIES:
        issues.append(f"bad category: {data.get('category')}")
    if data.get("neighborhood") not in VALID_NEIGHBORHOODS:
        issues.append(f"bad neighborhood: {data.get('neighborhood')}")
    if data.get("price_range") not in VALID_PRICE:
        issues.append(f"bad price_range: {data.get('price_range')}")
    if data.get("best_for") not in VALID_BEST_FOR:
        issues.append(f"bad best_for: {data.get('best_for')}")
    if not data.get("summary") or len(data.get("summary", "")) < 10:
        issues.append("missing or too short summary")

    return issues


# =====================================================================
# TEST CASES
# =====================================================================

test_inputs = [
    ("Clean: pizza in Soho",
     "Subreddit: r/london\nTitle: Best pizza in Soho?\nBody: Want proper Neapolitan pizza, not Dominos. Somewhere sit-down with wine.\nFlair: Food"),

    ("Clean: pub quiz",
     "Subreddit: r/LondonSocialClub\nTitle: [03/05/26] Pub quiz @ The Lamb, Bloomsbury\nBody: Running our weekly pub quiz again. Teams of up to 6, starts 7:30pm. No booking needed. Free entry.\nFlair: Going Ahead"),

    ("Clean: park visit",
     "Subreddit: r/london\nTitle: Hampstead Heath in autumn\nBody: The views from Parliament Hill are incredible this time of year. Went with my partner and it was perfect for a weekend walk.\nFlair: None"),

    ("Borderline: brewery + music",
     "Subreddit: r/london\nTitle: Brewery tour with live music in Bermondsey\nBody: This place does tours of their brewery for £15 and has a band playing in the taproom after. Great for a group night out.\nFlair: None"),

    ("Borderline: rooftop cinema",
     "Subreddit: r/london\nTitle: Rooftop cinema and cocktails in Peckham\nBody: They show old movies on a rooftop and serve expensive cocktails. Is it worth £30?\nFlair: None"),

    ("Messy: multi-topic",
     "Subreddit: r/london\nTitle: Moving to London from NYC\nBody: Any tips? I work in finance and my office is in Canary Wharf. Where should I live, where should I eat, what should I do on weekends? Budget around £2000/month for rent.\nFlair: None"),

    ("Noisy: gibberish",
     "asdkjh askjdh 2k3j4h london food maybe???"),

    ("Noisy: minimal",
     "london pubs"),
]


# =====================================================================
# RUN COMPARISON
# =====================================================================

print("=" * 90)
print(f"{'TEST CASE':<30} {'APPROACH':<12} {'CAT':<18} {'AREA':<12} {'PRICE':<8} {'ISSUES':<6} {'COST':>5} {'TIME':>5}")
print("=" * 90)

all_results = []

for label, inp in test_inputs:
    mono = monolithic(inp)
    multi = multistage(inp)

    mono_issues = validate_result(mono)
    multi_issues = validate_result(multi)

    mono_cat = mono["result"].get("category", "?")
    mono_area = mono["result"].get("neighborhood", "?")
    mono_price = mono["result"].get("price_range", "?")
    multi_cat = multi["result"].get("category", "?")
    multi_area = multi["result"].get("neighborhood", "?")
    multi_price = multi["result"].get("price_range", "?")

    print(f"  {label:<28} {'MONO':<12} {mono_cat:<18} {mono_area:<12} {mono_price:<8} {len(mono_issues):<6} {mono['tokens']:>5} {mono['latency']:>4.1f}s")
    print(f"  {'':<28} {'MULTI':<12} {multi_cat:<18} {multi_area:<12} {multi_price:<8} {len(multi_issues):<6} {multi['tokens']:>5} {multi['latency']:>4.1f}s")

    if mono_issues:
        print(f"  {'':>28}   MONO issues: {mono_issues}")
    if multi_issues:
        print(f"  {'':>28}   MULTI issues: {multi_issues}")
    print()

    all_results.append({
        "label": label,
        "input": inp[:100],
        "monolithic": {**mono, "issues": mono_issues},
        "multistage": {**multi, "issues": multi_issues},
    })

# === Summary ===
mono_total_issues = sum(len(r["monolithic"]["issues"]) for r in all_results)
multi_total_issues = sum(len(r["multistage"]["issues"]) for r in all_results)
mono_total_tokens = sum(r["monolithic"]["tokens"] for r in all_results)
multi_total_tokens = sum(r["multistage"]["tokens"] for r in all_results)
mono_total_latency = sum(r["monolithic"]["latency"] for r in all_results)
multi_total_latency = sum(r["multistage"]["latency"] for r in all_results)
mono_valid = sum(1 for r in all_results if not r["monolithic"]["issues"])
multi_valid = sum(1 for r in all_results if not r["multistage"]["issues"])

print("=" * 90)
print("COMPARISON SUMMARY")
print("=" * 90)
print(f"  {'Metric':<30} {'Monolithic':>15} {'Multi-stage':>15}")
print(f"  {'-'*30} {'-'*15} {'-'*15}")
print(f"  {'API calls per request':<30} {'1':>15} {'3':>15}")
print(f"  {'Total tokens':<30} {mono_total_tokens:>15} {multi_total_tokens:>15}")
print(f"  {'Total latency':<30} {f'{mono_total_latency:.1f}s':>15} {f'{multi_total_latency:.1f}s':>15}")
print(f"  {'Valid results':<30} {f'{mono_valid}/{len(all_results)}':>15} {f'{multi_valid}/{len(all_results)}':>15}")
print(f"  {'Total field issues':<30} {mono_total_issues:>15} {multi_total_issues:>15}")

with open("multistage_results.json", "w") as f:
    json.dump({
        "summary": {
            "monolithic": {
                "calls_per_request": 1,
                "total_tokens": mono_total_tokens,
                "total_latency": round(mono_total_latency, 2),
                "valid_results": mono_valid,
                "total_issues": mono_total_issues,
            },
            "multistage": {
                "calls_per_request": 3,
                "total_tokens": multi_total_tokens,
                "total_latency": round(multi_total_latency, 2),
                "valid_results": multi_valid,
                "total_issues": multi_total_issues,
            },
        },
        "results": all_results,
    }, f, indent=2, default=str)

print(f"\nSaved to multistage_results.json")
