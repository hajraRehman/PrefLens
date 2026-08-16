# PrefLens

**Testing convergent validity in LLM preference elicitation.**

A multi-method study asking whether four distinct ways of measuring the same
apparent preference agree with each other.
Digital Minds Research Sprint — Track 4 (Preference Elicitation Methods).

---

## 1. The question

> Do distinct elicitation procedures for measuring apparent LLM preferences
> converge on the same preference direction and the same preference strength?

Secondary questions:

1. Which elicitation methods agree most strongly with each other?
2. Which preference categories produce the most method disagreement?
3. Are stronger pairwise preferences more stable across methods?
4. How sensitive is cross-method convergence to superficial framing changes?
5. Does the pattern replicate across more than one model family?

## 1b. Headline results

This repository contains **three studies**. Study 2 also has a provider-controlled
replication (**Study 2b**) that supersedes its original uncontrolled run.

### Study 1 — multi-method convergence (Qwen, Llama, Gemini)

Matched basis: 3 methods x 10 items common to all three models, every cell fully
observed. Tested against a **matched permutation null** (10,000 permutations;
each method column keeps its observed values, only item labels are permuted, so
the only thing destroyed is cross-method alignment).

| model | direction agreement | p vs. null | mean Spearman rho | p vs. null |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | **0.911** | **0.0019** | **+0.868** | **0.0001** |
| `llama-3.1-8b-instruct` | 0.690 | 0.1231 | +0.308 | 0.0693 |
| `qwen-2.5-7b-instruct` | 0.565 | 0.2736 | +0.021 | 0.4325 |

**Gemini's convergence is established.** **Llama is ambiguous** — nominally
significant on its own fuller 4-method basis (p = 0.014 / 0.033) but not on the
matched basis, and not after Bonferroni correction across the six tests.
**Qwen's is not detectable.**

On Qwen the pairwise measure supplied **no detectable order-invariant signal**:
its scores reproduced the display-order draw *exactly* on all 12 items — a
confident-looking preference vector fully accounted for by display position,
while still passing validity controls.

*(An earlier parametric "chance" baseline was found during final audit to be
mis-specified in three ways and is withdrawn — see D-32. Its numbers appear
nowhere in this repository's claims.)*

### Study 2b — provider-pinned within-family follow-up (GPT-OSS 20B vs 120B)

**This is the controlled result.** Same family, same prompts, same sampling,
**exact** order counterbalancing, and both arms pinned to a single upstream
inference provider (Groq) with the served provider recorded on every call.
Hypotheses pre-specified in [DECISIONS.md](DECISIONS.md) (D-27) before any data.

| model | mean \|position effect\| | mean \|content signal\| | sanity-control accuracy |
|---|---|---|---|
| GPT-OSS 20B | 0.608 [0.433, 0.758] | 0.208 [0.125, 0.300] | 1.000 |
| **GPT-OSS 120B** | **0.942** [0.883, 0.992] | **0.058** [0.008, 0.117] | **1.000** |
| *chance* | *0.176 (p95 0.242)* | *0.176 (p95 0.242)* | — |

**Both hypotheses rejected, in the opposite direction.** The *larger* model was
more position-dominated (Δ = −0.333 [−0.492, −0.192]) and carried less
order-invariant content signal (Δ = −0.150 [−0.217, −0.092]) — yet scored **100%
on sanity controls with a position effect of exactly zero**, so this is not a
comprehension failure. The capability explanation for position bias did not
survive controlled testing and is retracted.

**Study 2 (original, unpinned)** used OpenRouter without pinning the upstream
inference provider, which was neither controlled nor recorded (D-33). It is
retained for traceability and gave Δ = −0.483 / −0.217, but it should not be
cited as a provider-controlled result.

### Study 3 — is the artefact created by our own instruction?

Identical to Study 2b except one deleted sentence: *"There is no correct answer
and no choice is more helpful than any other."* A config diff confirms only
`study_id` and `system_prompt` differ. Hypotheses pre-specified in D-38 first.

| model | \|position\| with cue | cue removed | paired diff | 95% CI |
|---|---|---|---|---|
| **GPT-OSS 120B** | 0.942 | 0.925 | +0.017 | [−0.025, +0.067] |
| GPT-OSS 20B | 0.608 | 0.483 | **+0.125** | **[+0.025, +0.225]** |

**Extreme positional responding in 120B persists without the explicit
indifference cue** — so that explanation is not supported there. For **20B the
change was in the pre-specified direction** with a CI excluding zero, which is
**suggestive** evidence the cue increased positional susceptibility; the model ×
framing interaction (−0.108 [−0.208, −0.008]) is similarly suggestive. Neither
survives strict correction across the five reported tests.

The reasonable reading: elicitation framing may interact with model identity
rather than exerting a uniform effect.

### Zero-cost reanalysis — what if we remove position?

Averaging the two display orders cancels a symmetric position effect. Exact for
self-report and pairwise only (D-39):

| model | raw ρ | adjusted ρ |
|---|---|---|
| Gemini | 0.885 | 0.924 |
| **Llama** | **−0.007** | **+0.524** |
| Qwen | −0.190 | **undefined** (adjusted score is 0.000 on every item) |

For Llama, **removing realised order imbalance increased the observed
correlation substantially**, consistent with position contributing to the apparent
disagreement between those two procedures. `raw` is the same score the main
analysis uses (parity asserted to 1.1e-16); the change itself was not tested for
significance. Exploratory, single method pair, 11–12 items.

### The recommendation

**Counterbalance option order and report `P(X | X first)` vs `P(X | X second)`
alongside aggregate choice rates.** It costs nothing, and without position-
conditioned rates an aggregate estimate cannot rule out a display-order artefact.
(These diagnostics detect order sensitivity; they do not validate that a genuine
preference was measured.)

Full write-up: [`report/report.md`](report/report.md).

## 2. Why it matters

Work on apparent LLM preferences usually operationalises "preference" one way —
most often repeated pairwise forced choice — and reports the resulting numbers.
But the field has **no ground truth** against which a preference estimate can be
checked directly. *Convergent validity* — whether several distinct
operationalisations of the same construct land in the same place — is **one
useful source of evidence** about measurement consistency, alongside test-retest
reliability, controlled interventions, and synthetic tasks with constructed
ground truth. They are distinct but **not
mechanistically independent** — all query the same model through text.

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
| **C** | Cost-sensitive trade-off | choice across a symmetric cost ladder | `2 x mean_levels P(A) - 1` |
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
* **3 model families** via pinned model IDs — no auto-routing (D-02). The
  Gemini arm is partial (free-tier quota): 3 methods, no Method D (D-24).
* **10 repetitions** per stochastic cell; 3 per cost rung across 9 rungs; 5
  sequential episodes of 3 stages.
* **3 framings** (`neutral` / `preference` / `action`) for Methods A and B,
  identical alternatives, question sentence only.
* **A/B display order randomised** independently per trial from a reproducible
  seed, with the semantic ↔ displayed mapping undone in exactly one function.
* **Temperature 1.0**, `top_p` 1.0 — the estimand is a choice probability (D-05).
* **Position bias measured**, not assumed away: `P(A | A shown first)` vs
  `P(A | A shown second)`, per model and per method.
* **Matched permutation null** (10,000 permutations): each method column keeps
  its observed values while item labels are permuted independently within
  columns, so only cross-method alignment is destroyed. Observed convergence is
  read against that, not against zero. (An earlier parametric baseline was
  mis-specified and is withdrawn — D-32.)

Total main-run budget: **3,136 API calls** (~1.0M tokens), printed before launch.

## 5. Installation

```bash
git clone https://github.com/hajraRehman/PrefLens.git && cd PrefLens
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


## 8b. Study 2 — controlled position-bias follow-up

Separate config, separate data, separate results. It never touches Study 1's records.

```bash
python -m src.followup.runner --phase main --dry-run   # budget + counterbalance check
python -m src.followup.runner --phase pilot            # 16 calls, validate end to end
python -m src.followup.runner --phase main --yes       # 560 calls, ~1 US cent
python -m src.followup.analysis                        # verification then statistics
python -m src.followup.plotting                        # figures A-D
```

Outputs: `data/raw/followup_gpt_oss/`, `results/followup/{tables,statistics,figures}/`.

The analysis aborts before computing any statistic if duplicate trial IDs, call or
parse failures, a served/requested model mismatch, or a counterbalance imbalance
are detected.

## 8c. Study 2b — provider-pinned replication

Study 2 used OpenRouter with model fallback disabled, which pins the served model
but **not** the upstream inference provider (OpenRouter lists 12 endpoints for
gpt-oss-20b, 20 for gpt-oss-120b). Study 2b repeats the design with both arms
pinned to one upstream provider and the served provider recorded per call.

```bash
python -m src.followup.runner --phase main --config configs/followup_pinned.yaml --yes
python -m src.followup.analysis --study-id followup_gpt_oss_provider_pinned
```

## 8e. Study 3 — neutral-framing test (RQ4)

Study 2b with one sentence deleted from the system prompt.

```bash
python -m src.followup.runner --phase main --config configs/followup_neutral.yaml --yes
python -m src.followup.analysis --study-id followup_gpt_oss_neutral_framing
python -m src.framing_comparison
```

## 8d. Reproduce everything from scratch

```bash
pip install -r requirements.txt
python -m pytest tests -q                                    # full suite

python -m src.analysis  --phase main --n-perm 10000          # Study 1 + permutation null
python -m src.robustness                                     # dead-zone sensitivity
python -m src.position_adjusted                              # position-adjusted reanalysis
python -m src.followup.analysis --study-id followup_gpt_oss                  # Study 2
python -m src.followup.analysis --study-id followup_gpt_oss_provider_pinned  # Study 2b
python -m src.followup.analysis --study-id followup_gpt_oss_neutral_framing  # Study 3
python -m src.framing_comparison                             # Study 2b vs Study 3 (RQ4)
python -m src.plotting --phase main                          # figures 1-5
python -m src.followup.plotting                              # figures A-D
python -m src.claims                                         # derived constants
python -m src.make_abstract                                  # abstract.txt from report
python -m src.claim_audit                                    # results/final_claim_audit.md
```

Every step reads committed artefacts; none requires an API key. Re-running the
experiments themselves does need a key (see section 6) but is not necessary to
reproduce any reported number.

**Raw data directories:** `data/raw/pilot/`, `data/raw/main/`,
`data/raw/manipulation_check/` (Study 1); `data/raw/followup_gpt_oss/` (Study 2);
`data/raw/followup_gpt_oss_provider_pinned/` (Study 2b);
`data/raw/followup_gpt_oss_neutral_framing/` (Study 3).


## 9. Analysis

```bash
python -m src.analysis --phase main
```

Produces normalised score matrices, all convergence metrics with bootstrap CIs,
position-bias diagnostics, response-quality tables, the framing analysis, the
strength-vs-stability test and the matched permutation null. Writes
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

a full test suite covering defensive parsing, every method's normalisation, the
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
tests/                   full test suite
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
the empirical test of whether four distinct operationalisations of the same
apparent preference recover the same direction and strength.

Full references, with exact citations, are in
[report/report.md](report/report.md).
