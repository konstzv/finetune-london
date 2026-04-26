#!/usr/bin/env python3
import json
import os
import time
import math
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CONFIDENCE_THRESHOLD = 0.08
VALID_CATEGORIES = {"FOOD_AND_DRINKS", "ACTIVITIES", "PLACES", "UNCATEGORIZED"}

LLM_PROMPT = """You classify London Reddit posts into one of: FOOD_AND_DRINKS, ACTIVITIES, PLACES, UNCATEGORIZED

Rules:
- Scheduled activity (meetup, concert, match, quiz) → ACTIVITIES
- Place to visit (park, museum, landmark) → PLACES
- Bar, pub, restaurant, cafe, food question → FOOD_AND_DRINKS
- Everything else → UNCATEGORIZED

Respond with ONLY valid JSON:
{"category": "<CATEGORY>", "confidence": <0-100>}"""


# =====================================================================
# EMBEDDING HELPERS
# =====================================================================

def get_embedding(text):
    """Get embedding vector for a text string."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =====================================================================
# STEP 1: BUILD REFERENCE VECTORS FROM TRAINING DATA
# =====================================================================

def build_reference_vectors():
    """Average the embeddings of all training examples per category."""
    print("Building reference vectors from train.jsonl...")

    categories = {}
    with open("train.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            cat = obj["messages"][2]["content"]
            user_text = obj["messages"][1]["content"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(user_text)

    reference = {}
    for cat, texts in categories.items():
        print(f"  Embedding {len(texts)} examples for {cat}...")
        embeddings = []
        for text in texts:
            emb = get_embedding(text)
            embeddings.append(emb)

        dim = len(embeddings[0])
        avg = [0.0] * dim
        for emb in embeddings:
            for i in range(dim):
                avg[i] += emb[i]
        for i in range(dim):
            avg[i] /= len(embeddings)

        reference[cat] = avg

    with open("reference_vectors.json", "w") as f:
        json.dump(reference, f)

    print(f"  Saved reference vectors for {len(reference)} categories\n")
    return reference


def load_reference_vectors():
    """Load cached reference vectors if available."""
    if os.path.exists("reference_vectors.json"):
        with open("reference_vectors.json") as f:
            return json.load(f)
    return build_reference_vectors()


# =====================================================================
# LEVEL 1: MICRO-MODEL (embedding-based)
# =====================================================================

def micro_classify(user_input, reference):
    """Classify using embedding similarity to reference vectors."""
    start = time.time()
    emb = get_embedding(user_input)
    latency = time.time() - start

    scores = {}
    for cat, ref_vec in reference.items():
        scores[cat] = cosine_similarity(emb, ref_vec)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_cat = sorted_scores[0][0]
    best_score = sorted_scores[0][1]
    second_score = sorted_scores[1][1]

    gap = best_score - second_score

    if gap >= CONFIDENCE_THRESHOLD:
        status = "OK"
    else:
        status = "UNSURE"

    return {
        "category": best_cat,
        "status": status,
        "best_score": round(best_score, 4),
        "gap": round(gap, 4),
        "all_scores": {k: round(v, 4) for k, v in sorted_scores},
        "latency": round(latency, 2),
    }


# =====================================================================
# LEVEL 2: LLM FALLBACK
# =====================================================================

def llm_classify(user_input):
    """Fall back to LLM when micro-model is unsure."""
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": LLM_PROMPT},
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
        category = parsed.get("category", "UNCATEGORIZED")
        confidence = parsed.get("confidence", 0)
    except json.JSONDecodeError:
        category = "UNCATEGORIZED"
        confidence = 0

    return {
        "category": category,
        "confidence": confidence,
        "latency": round(latency, 2),
        "tokens": tokens,
    }


# =====================================================================
# COMBINED PIPELINE
# =====================================================================

def classify(user_input, reference):
    """Micro-model first, LLM fallback if unsure."""

    micro = micro_classify(user_input, reference)

    if micro["status"] == "OK":
        return {
            "input": user_input[:80],
            "category": micro["category"],
            "handled_by": "MICRO",
            "micro": micro,
            "llm": None,
            "latency": micro["latency"],
        }

    llm = llm_classify(user_input)

    return {
        "input": user_input[:80],
        "category": llm["category"],
        "handled_by": "LLM",
        "micro": micro,
        "llm": llm,
        "latency": round(micro["latency"] + llm["latency"], 2),
    }


# =====================================================================
# TEST CASES (30 inputs)
# =====================================================================

test_inputs = [
    # Clean — obvious category (10)
    ("Best pizza in Soho", "Subreddit: r/london\nTitle: Best pizza in Soho?\nBody: Want proper Neapolitan pizza, not Dominos\nFlair: Food"),
    ("Sunday roast under £20", "Subreddit: r/london\nTitle: Best Sunday roast under £20\nBody: Looking for a solid Sunday roast that won't break the bank\nFlair: None"),
    ("Fish and chips", "Subreddit: r/london\nTitle: Underrated fish and chips?\nBody: Where do locals actually go for proper fish and chips?\nFlair: Food"),
    ("Pub quiz Bloomsbury", "Subreddit: r/LondonSocialClub\nTitle: [03/05/26] Pub quiz @ The Lamb, Bloomsbury\nBody: Weekly pub quiz, teams of 6, starts 7:30pm\nFlair: Going Ahead"),
    ("Marathon 2026", "Subreddit: r/london\nTitle: London Marathon 2026\nBody: Record breaking marathon in London sunshine\nFlair: image"),
    ("Football match", "Subreddit: r/LondonSocialClub\nTitle: [10/05/26] 5-a-side football @ Paddington Rec\nBody: Need 3 more players for Saturday morning kickabout\nFlair: Going Ahead"),
    ("Hampstead Heath", "Subreddit: r/london\nTitle: Hampstead Heath in autumn\nBody: The views from Parliament Hill are incredible this time of year\nFlair: None"),
    ("Tate Modern", "Subreddit: r/london\nTitle: Tate Modern vs Tate Britain - which one?\nBody: Only have time for one museum\nFlair: None"),
    ("Postman's Park", "Subreddit: r/london\nTitle: Postman's Park - the memorial wall\nBody: Found this incredible memorial. How did I not know about this?\nFlair: None"),
    ("Council tax bill", "Subreddit: r/london\nTitle: Cost of living reality check\nBody: Just got my council tax bill. Is £2100 for a 1-bed normal?\nFlair: None"),

    # Borderline — could be multiple categories (10)
    ("Pub with board games", "Subreddit: r/london\nTitle: Pub with board games?\nBody: Want somewhere to play games and have a pint\nFlair: None"),
    ("Street food Victoria Park", "Subreddit: r/london\nTitle: Street food at Victoria Park\nBody: The Sunday market there has amazing dumplings\nFlair: None"),
    ("Brewery tour + music", "Subreddit: r/london\nTitle: Brewery tour with live music\nBody: This place does tours and has a band playing in the taproom after\nFlair: None"),
    ("Rooftop cinema cocktails", "Subreddit: r/london\nTitle: Rooftop cinema and cocktails in Peckham\nBody: They show old movies on a rooftop and serve drinks\nFlair: None"),
    ("Walking tour of pubs", "Subreddit: r/london\nTitle: Walking tour of old pubs\nBody: Did a tour of historic pubs in the City. Ye Olde Cheshire Cheese highlight\nFlair: None"),
    ("Yoga in the park", "Subreddit: r/london\nTitle: Yoga in the park\nBody: Free yoga sessions at Regent's Park every Saturday\nFlair: None"),
    ("Jazz bar Soho", "Subreddit: r/london\nTitle: Jazz bars in Soho?\nBody: Looking for proper live jazz, somewhere you can sit and listen\nFlair: None"),
    ("V&A late opening", "Subreddit: r/LondonSocialClub\nTitle: [01/05/26] V&A late night opening\nBody: Late openings on Fridays till 10pm. Going this week\nFlair: None"),
    ("Cafe with poetry", "Subreddit: r/london\nTitle: Open mic night at a cafe\nBody: This cafe does poetry readings and serves great coffee\nFlair: None"),
    ("Canal walk + pub", "Subreddit: r/london\nTitle: Regent's Canal walk ending at a pub\nBody: Want to walk the canal and finish at a nice pub in Camden\nFlair: None"),

    # Noisy / difficult (10)
    ("Gibberish", "asdkjh askjdh 2k3j4h london food maybe???"),
    ("Just 'london'", "london"),
    ("Spam", "buy cheap viagra online best prices london pharmacy"),
    ("Cat post", "Subreddit: r/cats\nTitle: My cat knocked over the Christmas tree\nBody: Third year in a row\nFlair: None"),
    ("Empty-ish", "Title: \nBody: \nFlair: "),
    ("Emoji spam", "🔥🔥🔥 BEST DEALS 50% OFF click here now!!!"),
    ("Moving to London", "Subreddit: r/london\nTitle: Moving from NYC\nBody: Where should I live, eat, and what should I do on weekends?\nFlair: None"),
    ("Yes/no post", "Subreddit: r/london\nTitle: yes\nBody: no\nFlair: None"),
    ("French post", "Subreddit: r/london\nTitle: Meilleurs restaurants français à Londres?\nBody: Je cherche un bon restaurant français pas trop cher\nFlair: None"),
    ("Mixed languages", "Subreddit: r/london\nTitle: Лучшие парки в Лондоне?\nBody: Looking for nice parks near central\nFlair: None"),
]


# =====================================================================
# RUN
# =====================================================================

reference = load_reference_vectors()

print("=" * 95)
print(f"{'INPUT':<28} {'HANDLER':<8} {'CATEGORY':<18} {'SCORE':>6} {'GAP':>6} {'TIME':>6}")
print("=" * 95)

results = []
for label, inp in test_inputs:
    r = classify(inp, reference)
    results.append(r)

    score = r["micro"]["best_score"]
    gap = r["micro"]["gap"]
    print(f"  {label:<26} {r['handled_by']:<8} {r['category']:<18} {score:>6.3f} {gap:>6.3f} {r['latency']:>5.2f}s")

# === Summary ===
micro_count = sum(1 for r in results if r["handled_by"] == "MICRO")
llm_count = sum(1 for r in results if r["handled_by"] == "LLM")
total_latency = sum(r["latency"] for r in results)
micro_latency = [r["latency"] for r in results if r["handled_by"] == "MICRO"]
llm_latency = [r["latency"] for r in results if r["handled_by"] == "LLM"]

# Check agreement: for LLM fallbacks, did micro and LLM agree?
agree_count = sum(1 for r in results if r["llm"] and r["micro"]["category"] == r["llm"]["category"])
disagree_count = sum(1 for r in results if r["llm"] and r["micro"]["category"] != r["llm"]["category"])

print("\n" + "=" * 95)
print("SUMMARY")
print("=" * 95)
print(f"  Total requests:           {len(results)}")
print(f"  Handled by MICRO:         {micro_count} ({micro_count*100//len(results)}%)")
print(f"  Escalated to LLM:         {llm_count} ({llm_count*100//len(results)}%)")
print(f"  LLM agreed with micro:    {agree_count}")
print(f"  LLM disagreed with micro: {disagree_count}")
print(f"  Total latency:            {total_latency:.1f}s")
print(f"  Avg latency (MICRO):      {sum(micro_latency)/max(len(micro_latency),1):.2f}s")
print(f"  Avg latency (LLM):        {sum(llm_latency)/max(len(llm_latency),1):.2f}s")
print(f"  Confidence threshold:     {CONFIDENCE_THRESHOLD} (gap between top 2 scores)")

with open("micromodel_results.json", "w") as f:
    json.dump({
        "config": {
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        "summary": {
            "total": len(results),
            "micro_handled": micro_count,
            "llm_fallback": llm_count,
            "llm_agreed": agree_count,
            "llm_disagreed": disagree_count,
            "total_latency": round(total_latency, 2),
            "avg_micro_latency": round(sum(micro_latency)/max(len(micro_latency),1), 2),
            "avg_llm_latency": round(sum(llm_latency)/max(len(llm_latency),1), 2),
        },
        "results": results,
    }, f, indent=2)

print(f"\nSaved to micromodel_results.json")
