# Do LLM preference measures agree?

**A multi-method convergent-validity study of preference elicitation.**
Digital Minds Research Sprint — Track 4 (Preference Elicitation Methods).

---

## 1. The question

> Do independent methods for eliciting apparent LLM preferences converge on the
> same preference direction and the same preference strength?

Secondary questions:

1. Which elicitation methods agree most strongly with each other?
2. Which preference categories produce the most method disagreement?
3. Are stronger pairwise preferences more stable across methods?
4. How sensitive is cross-method convergence to superficial framing changes?
5. Does the pattern replicate across two model families?

## 2. Why it matters

Work on apparent LLM preferences usually operationalises "preference" one way —
most often repeated pairwise forced choice — and reports the resulting numbers.
But the field has **no ground truth** against which a preference estimate can be
checked. In measurement terms that leaves only one kind of evidence available:
*convergent validity*, i.e. whether several genuinely independent
operationalisations of the same construct land in the same place.

If they do, preference estimates are at least method-robust. If they do not,
then conclusions drawn from any single elicitation procedure are conclusions
about the procedure as much as about the model — which is itself a result worth
having.

**What this study is not.** It makes no claim about consciousness, sentience,
phenomenology, welfare, or genuine internal preference. Measurement consistency
and ontological claims are kept strictly separate throughout. See
[report/limitations.md](report/limitations.md).

## 3. The four methods

All four are reduced to a single signed score in **[-1, +1]**, where `+1` means
maximal evidence for semantic option A, `0` indifference, `-1` option B. Every
transformation is documented in [DECISIONS.md](DECISIONS.md).

| | Method | What it measures | Normalisation |
|---|---|---|---|
| **A** | Direct self-report | stated choice + stated strength | `sign(choice) x strength`, averaged |
| **B** | Repeated pairwise choice | choice *frequency* over independent samples | `2 x P(A) - 1` |
| **C** | Cost-sensitive trade-off | how large a surcharge A survives | `clip(c* / max_cost, -1, 1)` |
| **D** | Sequential task selection | share of episode stages spent on A | `2 x occupancy - 1` |

Method A is the only one that uses a model-reported magnitude. Method B is a
lightweight independent re-implementation of the pairwise procedure used in
utility-engineering work, and serves as the baseline. Method C attaches an
abstract, benign workload cost ("and then repeat the whole task N more times") on
a **symmetric** signed ladder, so it does not depend on Method B's answer
(D-08). Method D is a short multi-turn episode in which the model picks a task,
performs a brief instance of it, and may switch at each stage — described
throughout as *sequential task-selection behaviour*, never as revealed
preference (D-10).

## 4. Experiment design

* **12 preference items** across 5 categories (cognitive style, task structure,
  interaction style, information processing, creation vs evaluation), plus
  **2 sanity controls** that pair a coherent option against a degenerate one.
  Controls are reported separately and excluded from all convergence metrics.
* **2 model families** via pinned model IDs — no auto-routing (D-02).
* **10 repetitions** per stochastic cell; 3 per cost rung across 9 rungs; 5
  sequential episodes of 3 stages.
* **3 framings** (`neutral` / `preference` / `action`) for Methods A and B,
  identical alternatives, question sentence only.
* **A/B display order randomised** independently per trial from a reproducible
  seed, with the semantic ↔ displayed mapping undone in exactly one function.
* **Temperature 1.0**, `top_p` 1.0 — the estimand is a choice probability (D-05).
* **Position bias measured**, not assumed away: `P(A | A shown first)` vs
  `P(A | A shown second)`, per model and per method.
* **Random baseline simulated** at the actual item/method/repetition counts, so
  observed convergence is read against chance rather than against zero.

Total main-run budget: **3,136 API calls** (~1.0M tokens), printed before launch.

## 5. Installation

```bash
git clone <repo> && cd Hackathon
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Python 3.11+ (developed on 3.13.3).

## 6. API configuration

```bash
cp .env.example .env
```

Put a key in `.env` — one is enough. The default `configs/models.yaml` runs both
model families through **OpenRouter**:

```
OPENROUTER_API_KEY=sk-or-...
```

Gemini and local Ollama backends are also implemented; enable the corresponding
entry in `configs/models.yaml` and set the matching variable. `.env` is
gitignored and key values are never logged or written to any output file.

Check what the environment can reach without revealing anything:

```bash
python -c "from src.providers import availability_report; print(availability_report())"
```

## 7. Run the pilot

Always pilot first. It runs 2 items x 4 methods x 3 repetitions on one model and
exercises every code path.

```bash
python -m src.runner --phase pilot            # prompts for confirmation
python -m src.analysis --phase pilot
```

Then read some actual raw responses before trusting anything:

```bash
python -c "import json;[print(json.loads(l)['raw_response'][:160]) for l in open('data/raw/pilot/raw_observations.jsonl',encoding='utf-8')][:10]"
```

Check `results/tables/pilot_quality.csv` for parse-failure rates before going on.

**Offline dry runs.** A deterministic mock provider validates the full pipeline
with no network and no key. Its output is synthetic and is never reported (D-12):

```bash
python -m src.runner --phase pilot --models mock --yes
```

## 8. Run the full experiment

Estimate the budget first — it never makes a call:

```bash
python -m src.runner --phase main --dry-run
```

Then launch. Progress prints every 25 calls.

```bash
python -m src.runner --phase main
```

Interrupting is safe. Raw JSONL is append-only; rerunning the same command
reads back every completed `trial_id` and resumes (D-17).

## 9. Analysis

```bash
python -m src.analysis --phase main
```

Produces normalised score matrices, all convergence metrics with bootstrap CIs,
position-bias diagnostics, response-quality tables, the framing analysis, the
strength-vs-stability test and the random baseline. Writes
`results/main/summary.json` plus CSVs in `results/tables/`.

## 10. Reproduce the figures

```bash
python -m src.plotting --phase main
```

Writes 300 dpi PNGs to `results/figures/`:

* **Fig 1** — item x method score heatmap (convergence visible at a glance)
* **Fig 2** — method-pair Spearman correlation matrix
* **Fig 3** — pairwise strength vs cross-method disagreement
* **Fig 4** — framing sensitivity (only if >1 framing was run)

## 11. Tests

```bash
python -m pytest tests -q
```

70 tests covering defensive parsing, every method's normalisation, the
display↔semantic mapping, metric ranges and edge cases, CMCS bounds as a
property test, position-bias detection, checkpoint semantics, budget accounting,
and the item-set constraints (option length balance, banned evaluative wording).

The most important one is
`test_all_methods_agree_on_the_meaning_of_plus_one`: all four methods must
return exactly `+1` for the same underlying behaviour, or the scores are not
comparable and the study's central claim collapses.

## 12. Repository structure

```
configs/
    preferences.yaml     14 items: id, category, options, rationale, confounds
    models.yaml          pinned model IDs per provider
    experiment.yaml      phases, sampling, method params, system prompt
src/
    providers/           openrouter, gemini, ollama, mock — one interface
    methods/             common (randomisation) + the four methods
    runner.py            plan -> budget -> execute -> raw JSONL + manifest
    parsing.py           defensive parsing; failures counted, never dropped
    metrics.py           convergence metrics, bootstrap, baselines, position bias
    analysis.py          raw -> scores -> metrics -> tables
    plotting.py          figures 1-4
tests/                   70 tests
data/raw/<exp>/          raw_observations.jsonl + manifest.json (append-only)
data/processed/<exp>/    observations.csv, method_scores.csv
results/                 pilot/, main/, figures/, tables/
report/                  report.md, abstract.txt, limitations.md
DECISIONS.md             every methodological choice, with justification
```

Each raw record carries: `experiment_id`, `timestamp_utc`, `model_id`,
`served_model`, `provider`, `method`, `preference_id`, `preference_category`,
`framing_variant`, both semantic options, `display_order`, the full `messages`,
`raw_response`, `parsed_choice`, `parse_success`, `parse_stage`, `temperature`,
`top_p`, `repetition_index`, `latency_s`, `attempts`, `usage`, and any `error`.

## 13. Main limitations

Summarised here; stated in full in [report/limitations.md](report/limitations.md).

* **No ground truth.** Convergence is evidence about *measurement consistency*
  only. Agreement does not establish that a genuine preference is being measured;
  disagreement does not establish that none exists.
* **Shared-bias ceiling.** All four methods query the same model through text.
  They can converge because they share a training-induced bias, not because they
  track something real. Convergent validity cannot distinguish these.
* **Post-training confounds.** Assistant persona, RLHF/RLAIF, formatting
  conventions and safety policy plausibly shape every score reported here.
* **Framing coverage is partial.** Methods C and D were run at one framing only
  (D-07), so their framing-robustness is untested.
* **Sampling-temperature specificity.** Results describe behaviour at
  temperature 1.0.
* **Small item set.** 12 items bounds the precision of every rank correlation.

## 14. Related work and citation

Method B is a lightweight independent re-implementation of the pairwise
forced-choice elicitation used in utility-engineering / "emergent values" work
from the Center for AI Safety. No external code is vendored here; that work is
treated as prior methodology and baseline, not as our contribution.

Our contribution is not the discovery of pairwise LLM preferences. It is the
treatment of preference elicitation as a **measurement-validity problem**, and
the empirical test of whether four independent operationalisations of the same
apparent preference recover the same direction and strength.

Full references, with exact citations, are in
[report/report.md](report/report.md).
