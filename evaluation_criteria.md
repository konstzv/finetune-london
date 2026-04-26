# Evaluation Criteria

## Baseline
- Model: gpt-4o-mini (no fine-tuning)
- Accuracy: 7/10 (70%)
- Main failure: model classifies event-related posts as UNCATEGORIZED

## Categories
FOOD_AND_DRINKS, ACTIVITIES, PLACES, UNCATEGORIZED

## Classification Rule
- If there's a scheduled activity (match, meetup, concert, party) → ACTIVITIES
- If it's about a place to visit (park, museum, landmark, bar) → PLACES or FOOD_AND_DRINKS
- Bars, pubs, restaurants, cafes → FOOD_AND_DRINKS
- Everything else → UNCATEGORIZED

## Success Criteria (post fine-tuning)
1. **Accuracy**: >85% on eval set (baseline is 70%)
2. **Format**: model responds with exactly one category name, no extra text
3. **Consistency**: model follows our rule — events are ACTIVITIES, not UNCATEGORIZED
