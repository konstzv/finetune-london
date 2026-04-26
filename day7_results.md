# Day 7: Confidence Evaluation and Quality Control

## Task
Implement confidence evaluation for inference without fine-tuning. Minimum 2 approaches.

## Approaches Implemented

### Approach D: Scoring
Model returns JSON with category + confidence score (0-100).
If confidence < 85 → flag as uncertain.

### Approach B: Redundancy
Same request sent 3 times with temperature=0.7.
- All 3 agree → OK
- 2 of 3 agree → UNSURE
- All different → FAIL

### Combined Pipeline
1. Run scoring (1 API call)
2. If confidence >= 85 → accept (cheap path)
3. If confidence < 85 → run redundancy (3 more API calls)
4. Use redundancy agreement to decide OK / UNSURE / FAIL

## Test Results (19 inputs)

| Input Type   | Count | Accepted (OK) | Uncertain (UNSURE) | Rejected (FAIL) |
|-------------|-------|---------------|-------------------|----------------|
| Clean        | 4     | 4             | 0                 | 0              |
| Borderline   | 7     | 7             | 0                 | 0              |
| Noisy        | 8     | 8             | 0                 | 0              |

- **3 requests triggered redundancy** (all noisy inputs with confidence < 85)
- **0 requests rejected** — model was consistent even on garbage input

## Cost and Latency

| Path          | Avg Latency | Avg Tokens | API Calls |
|--------------|-------------|------------|-----------|
| Confident    | ~0.75s      | ~175       | 1         |
| Redundancy   | ~2.8s       | ~380       | 4         |

## Key Finding
The model is **overconfident** — it returns confidence 70+ even for gibberish input like "asdkjh askjdh". Redundancy catches uncertainty (model disagrees with itself) but not confident-and-wrong cases. This limitation motivated Day 8's routing approach.

## Deliverables
- `inference.py` — combined scoring + redundancy pipeline
- `inference_results.json` — full test results
