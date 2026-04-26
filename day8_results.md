# Day 8: Routing Between Models

## Task
Implement request routing between models with fallback logic.

## Strategy
1. Send request to **gpt-4o-mini** (cheap, fast) with confidence scoring
2. If confidence >= 85 → accept answer
3. If confidence < 85 → escalate to **gpt-4o** (expensive, smarter)

## Test Results (12 inputs)

| Input                      | Model Used   | Category        | Confidence | Cost      | Latency |
|---------------------------|-------------|-----------------|------------|-----------|---------|
| Best pizza in Soho?        | gpt-4o-mini | FOOD_AND_DRINKS | 95         | $0.00003  | 1.0s    |
| Pub quiz at The Lamb       | gpt-4o-mini | ACTIVITIES      | 95         | $0.00003  | 1.0s    |
| Hampstead Heath views      | gpt-4o-mini | PLACES          | 90         | $0.00003  | 0.7s    |
| Jubilee line packed        | gpt-4o-mini | UNCATEGORIZED   | 85         | $0.00003  | 0.5s    |
| Pub with board games       | gpt-4o-mini | FOOD_AND_DRINKS | 90         | $0.00003  | 0.7s    |
| Brewery tour + live music  | gpt-4o-mini | ACTIVITIES      | 95         | $0.00003  | 0.6s    |
| Rooftop cinema + cocktails | gpt-4o-mini | ACTIVITIES      | 85         | $0.00003  | 0.6s    |
| Yoga in Regent's Park      | gpt-4o-mini | ACTIVITIES      | 95         | $0.00003  | 0.8s    |
| Gibberish                  | gpt-4o      | FOOD_AND_DRINKS | 70         | $0.00056  | 1.2s    |
| Just "london"              | gpt-4o      | UNCATEGORIZED   | 50         | $0.00049  | 1.2s    |
| Spam                       | gpt-4o-mini | UNCATEGORIZED   | 85         | $0.00003  | 0.6s    |
| Moving to London           | gpt-4o      | UNCATEGORIZED   | 75         | $0.00060  | 1.2s    |

## Routing Summary

| Metric                          | Value      |
|--------------------------------|------------|
| Stayed on gpt-4o-mini          | 9 (75%)    |
| Escalated to gpt-4o            | 3 (25%)    |
| Answer changed on escalation   | 0          |
| Total cost (with routing)      | $0.00195   |
| Total cost (all gpt-4o)        | ~$0.006    |
| **Savings from routing**       | **~3x**    |

## Cost Per Request

| Path       | Avg Cost   | Avg Latency |
|-----------|------------|-------------|
| Cheap only | $0.00003   | 0.72s       |
| Escalated  | $0.00055   | 1.21s       |

Escalated requests cost **18x more** per call, but only 25% of requests needed escalation.

## Key Finding
Routing saves significant cost by keeping easy requests on the cheap model. 75% of requests were handled by gpt-4o-mini at $0.00003 each. Only ambiguous or noisy inputs triggered the expensive model.

## Deliverables
- `routing.py` — routing pipeline with cost tracking
- `routing_results.json` — full test results
