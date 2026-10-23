# Tuning the decision model

Fine-tuning Laya on your own traffic is the right eventual move, but it is the
last lever. Do these in order; each step produces what the next one needs.

## 1. Log everything

Pass `log_path` to `create_routed_agent` (or a `RoutingLogger` to the
middleware). Every invocation appends one line to a JSONL file: the condensed
request, the full probability vector, the chosen tier and why, and the final
answer.

```python
agent = create_routed_agent(tools, log_path="data/routing.jsonl")
```

This is free, and nothing below works without it. Decide retention and privacy
up front: this file contains user requests.

## 2. Build a labeled set

A request's label is **the cheapest tier whose answer passes a quality bar**.
The router's own choice is not a label; training on it teaches the model to
agree with itself.

`systemone-label` answers each logged request on every tier, has a judge model
grade each answer against the bar, and labels the request with the first
passing tier.

```bash
systemone-label data/routing.jsonl --limit 200            # prints the call count and stops
systemone-label data/routing.jsonl --limit 200 --yes      # runs
```

It writes:

| File | Format | Consumer |
|------|--------|----------|
| `data/train.jsonl` | `{"text", "label", "router_tier"}` | `laya-train --text-column text --label-column label` |
| `data/evals.jsonl` | `{"state", "questions", "expected"}` using the router's own question | `laya-evals run` |
| `data/train.details.jsonl` | every answer and judgment | your review |

It also prints router agreement with the labels, split into "too cheap" (the
expensive mistake) and "too expensive". Spot-check the details file before
trusting the numbers: the judge is Opus 5.5 by default, and `--quality-bar`
sets what "passes" means. Answers are generated single-turn without tools, so
tool-heavy requests are judged on the answer alone.

## 3. Measure, then tune the cheap knobs

```bash
pip install "systemone-router[local]"
laya-evals run data/evals.jsonl --json reports/baseline.json
```

Then, in this order, re-running the eval after each:

1. **Tier descriptions.** Laya reads them as the criteria. Rewrite the
   `description` of a tier using the kinds of requests it misroutes, taken
   from the details file.
2. **`min_confidence`.** Raise it if the router is too cheap too often; the
   escalation only ever moves one tier up.
3. **Temperature calibration.** The shipped checkpoints are over-confident.
   `laya.calibrate` refits per-bucket temperatures on your logits and makes
   the confidence threshold mean what it says, with no change to the weights.

## 4. Fine-tune only if that plateaus

```bash
laya-train --data data/train.jsonl --text-column text --label-column label \
           --base convaiinnovations/laya --out ./checkpoints/router-v1 \
           --eval data/evals.jsonl
laya-evals run data/evals.jsonl --model ./checkpoints/router-v1 --json reports/v1.json
laya-evals compare reports/v1.json --baseline reports/baseline.json
```

Keep a holdout split out of training and gate every retrain on `compare`, so a
new checkpoint cannot silently regress.

Two costs to accept first:

- **Self-hosting.** The Vercel gateway serves the stock checkpoint only. A
  tuned model runs through the `server` backend (`laya-serve`) or `local`.
- **Data.** A few hundred well-labeled examples is the floor. Below that,
  tier descriptions and calibration move the numbers more than training does.

Fine-tuning pays off once the generic model misreads something specific to
your traffic, such as short-looking requests that are hard in your product,
and the cheap knobs have stopped helping. Retrain periodically as traffic drifts.
