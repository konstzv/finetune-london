# Day 6: Dataset for Fine-Tuning

## Task
Build a JSONL dataset to fine-tune gpt-4o-mini for classifying London Reddit posts.

## Approach
- Chose **classification** task: Reddit post → category
- Fetched 28 real posts from r/london and r/LondonSocialClub via Reddit JSON API
- Generated 39 more realistic posts to balance categories
- Manually labeled all examples (human = ground truth)

## Classification Rule
- Scheduled activity (meetup, concert, match) → **ACTIVITIES**
- Place to visit (park, museum, landmark) → **PLACES**
- Bar, pub, restaurant, cafe → **FOOD_AND_DRINKS**
- Everything else → **UNCATEGORIZED**

## Dataset

| Split | Total | ACTIVITIES | FOOD_AND_DRINKS | PLACES | UNCATEGORIZED |
|-------|-------|------------|-----------------|--------|---------------|
| Train | 48    | 15         | 11              | 11     | 11            |
| Eval  | 19    | 9          | 3               | 3      | 4             |

## Baseline (gpt-4o-mini without fine-tuning)
- **Accuracy: 7/10 (70%)**
- Main failure: model classifies event-related posts as UNCATEGORIZED
- Examples: marathon photos, Bach concert → model said UNCATEGORIZED, we said ACTIVITIES

## Deliverables
- `train.jsonl` — 48 training examples
- `eval.jsonl` — 19 evaluation examples
- `validate.py` — checks JSON validity, required roles, empty content
- `baseline.py` — runs 10 eval examples through base model
- `baseline_results.json` — baseline answers
- `evaluation_criteria.md` — success criteria (target >85% accuracy)
- `finetune_client.py` — uploads file + starts fine-tuning job (ready, not executed)
