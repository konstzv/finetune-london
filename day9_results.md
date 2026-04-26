# Day 9: Multi-Stage Inference Decomposition

## Task
Compare monolithic (1 prompt) vs multi-stage (3 prompts) inference for complex extraction.

## What We Extract
From each Reddit post, extract 5 fields:
- **category**: FOOD_AND_DRINKS | ACTIVITIES | PLACES | UNCATEGORIZED
- **neighborhood**: SOHO | CAMDEN | SHOREDITCH | ... | UNKNOWN (22 options)
- **price_range**: FREE | BUDGET | MID | PREMIUM | UNKNOWN
- **best_for**: SOLO | COUPLES | FRIENDS | FAMILIES | ANYONE
- **summary**: 1-2 sentence recommendation

## Approaches

### Variant A: Monolithic
One prompt asks for all 5 fields at once. Single API call.

### Variant B: Multi-stage (3 steps)
1. **Normalize** — clean the input, detect if it has a date/location
2. **Extract & Classify** — extract category, neighborhood, price, best_for from clean input
3. **Summarize** — write a recommendation using original post + extracted fields

Each stage has a focused prompt with strict enum values.

## Results (8 test inputs)

| Test Case              | Approach | Category        | Neighborhood | Price   | Issues |
|------------------------|----------|-----------------|-------------|---------|--------|
| Pizza in Soho          | MONO     | FOOD_AND_DRINKS | SOHO        | MID     | 0      |
|                        | MULTI    | FOOD_AND_DRINKS | SOHO        | UNKNOWN | 0      |
| Pub quiz Bloomsbury    | MONO     | ACTIVITIES      | BLOOMSBURY  | FREE    | 0      |
|                        | MULTI    | ACTIVITIES      | BLOOMSBURY  | FREE    | 0      |
| Hampstead Heath        | MONO     | ACTIVITIES      | HAMPSTEAD*  | FREE    | 1      |
|                        | MULTI    | PLACES          | UNKNOWN     | FREE    | 0      |
| Brewery + music        | MONO     | ACTIVITIES      | BERMONDSEY  | MID     | 0      |
|                        | MULTI    | ACTIVITIES      | BERMONDSEY  | MID     | 0      |
| Rooftop cinema         | MONO     | ACTIVITIES      | PECKHAM     | MID     | 0      |
|                        | MULTI    | ACTIVITIES      | PECKHAM     | PREMIUM | 0      |
| Moving to London       | MONO     | UNCATEGORIZED   | UNKNOWN     | UNKNOWN | 0      |
|                        | MULTI    | UNCATEGORIZED   | UNKNOWN     | BUDGET  | 0      |
| Gibberish              | MONO     | FOOD_AND_DRINKS | UNKNOWN     | UNKNOWN | 0      |
|                        | MULTI    | FOOD_AND_DRINKS | UNKNOWN     | UNKNOWN | 0      |
| "london pubs"          | MONO     | FOOD_AND_DRINKS | UNKNOWN     | BUDGET  | 0      |
|                        | MULTI    | FOOD_AND_DRINKS | UNKNOWN     | UNKNOWN | 0      |

*HAMPSTEAD is not in the allowed neighborhood list — validation failure.

## Comparison Summary

| Metric                 | Monolithic | Multi-stage |
|------------------------|-----------|-------------|
| API calls per request  | 1         | 3           |
| Total tokens           | 2483      | 6287        |
| Total latency          | 11.5s     | 29.4s       |
| Valid results           | 7/8       | 8/8         |
| Total field issues     | 1         | 0           |

## Key Findings
1. **Multi-stage is more accurate** — 8/8 valid vs 7/8. Monolithic invented a neighborhood value (HAMPSTEAD) not in the allowed list.
2. **Multi-stage classifies better** — correctly labeled Hampstead Heath as PLACES; monolithic said ACTIVITIES.
3. **Multi-stage extracts more detail** — caught the £2000 rent mention as BUDGET, £30 cocktails as PREMIUM.
4. **Monolithic is 2.5x cheaper and faster** — fewer tokens, fewer API calls.
5. **Tradeoff**: monolithic for speed/cost, multi-stage for accuracy/robustness.

## Deliverables
- `multistage.py` — both approaches with validation and comparison
- `multistage_results.json` — full test results
