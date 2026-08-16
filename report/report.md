# No Signal, No Convergence: Position Artefacts in LLM Preference Elicitation
### A multi-method convergent-validity study, and a controlled within-family follow-up

**PrefLens** — Digital Minds Research Sprint, Track 4 (Preference Elicitation Methods)
Code and raw data: <https://github.com/hajraRehman/PrefLens>
Run date: 2026-08-16. All reported results are reproducible from the committed
raw-data directories (`data/raw/{pilot,main,manipulation_check,
followup_gpt_oss,followup_gpt_oss_provider_pinned,
followup_gpt_oss_neutral_framing}/`) and the configs in
`configs/`.

---

## Abstract

LLM preference measurement has no direct ground truth, so convergent validity
across elicitation procedures is one useful source of evidence about measurement
consistency. Study 1 applied four distinct procedures to 12 welfare-neutral task
pairs across three model families (4,013 trial records). Against a matched
permutation null, only Gemini-3.1-flash-lite converged above chance (p = 0.0019);
Qwen-2.5-7B's pairwise scores reproduced the display-order draw exactly on all 12
items. In an exploratory self-report/pairwise reanalysis, equal-weighting the two
display orders raised Llama's observed correlation from −0.007 to +0.524,
suggesting order imbalance can obscure cross-method agreement. Study 2
(exact counterbalancing, provider-pinned) found the larger GPT-OSS model *more*
position-dominated (Δ = −0.333), rejecting our pre-specified hypothesis. Extreme
position sensitivity in GPT-OSS 120B persisted after removing the prompt's
explicit indifference cue; the smaller model showed a suggestive framing effect.
Preference studies should counterbalance option order and report
position-conditioned rates.

*(142 words)*

---

## 1. Introduction

Work on apparent LLM preferences typically operationalises "preference" one way —
most often repeated pairwise forced choice — and reports the resulting numbers.
The field has no ground truth: there is no independently verified fact about what
a model prefers against which an elicited score could be scored correct.

In the absence of direct ground truth, **convergent validity** provides one
useful source of evidence about measurement consistency (others include
test-retest reliability, controlled interventions, and synthetic tasks where
ground truth is constructed). If several distinct operationalisations of the
same construct land in the same place, the estimate is at least method-robust.
If they do not, conclusions drawn from any single procedure are conclusions
about the procedure as much as about the model.

These procedures are **distinct but not mechanistically independent**: all query
the same model through the same text channel, and share its post-training and
response heuristics. Agreement between them can therefore reflect a shared bias
rather than a common signal.

This work treats preference elicitation as a measurement-validity problem, in
two studies.

**RQ1.** Do multiple elicitation methods converge on the same apparent
preferences? (Study 1: four methods, three model families.)

**RQ2.** How strongly can display position distort preference measurement?
(Diagnostic analysis within Study 1.)

**RQ3.** Does positional susceptibility differ with model scale *within* one
family? (Studies 2/2b: GPT-OSS 20B vs 120B, exact counterbalancing; **Study 2b**
provides the provider-pinned comparison.)

**RQ4.** Does the elicitation prompt's own framing change positional
susceptibility? (Study 3: Study 2b with the indifference cue deleted.)

Study 1 raised RQ3 but cannot answer it: its three models differ in family,
scale, data, post-training and serving stack simultaneously. Study 2
substantially reduces these confounds by holding model family, provider,
prompts, sampling, task set and protocol fixed while comparing two model
scales.

**Contributions.**

1. A reusable, fully logged elicitation harness implementing four methods behind
   one provider-agnostic interface, with per-trial order randomisation, strict
   parse accounting, checkpointing, and a full test suite.
2. A cross-method convergence analysis on three model families, tested against a
   **matched permutation null** that preserves each procedure's observed score
   distribution and destroys only cross-method item alignment.
3. **A position/content decomposition that distinguishes cross-method
   disagreement from cases where an elicitation measure carries no detectable
   order-invariant content signal** — and the finding that one standard pairwise
   elicitation was entirely position artefact.
4. Two pre-specified checks that changed the design mid-study: a
   manipulation check that invalidated our original trade-off scoring, and a
   degeneracy check that invalidated one of our own computed results.

We make no claim about consciousness, sentience, welfare, or genuine internal
preference. See the Limitations appendix.

## 2. Related work

Our pairwise method is a stripped-down independent re-implementation of the
forced-choice elicitation used in **utility engineering** work (Mazeika et al.,
2025), which samples pairwise preferences repeatedly and fits utility models,
reporting that preference structure becomes more internally coherent with scale.
We build on it as prior methodology and baseline; no external code is vendored.

Our contribution is not the discovery of pairwise LLM preferences. It is the
convergent-validity question: whether *several* operationalisations of the same
apparent preference agree, and what it means when they do not.

Position and selection bias in LLM multiple-choice responding is well documented
(Pezeshkpour & Hruschka, 2023; Zheng et al., 2024), including token-level bias
toward specific option IDs. Our results connect that literature directly to
preference elicitation: an unreported position bias does not merely add noise, it
can constitute the *entire* measured signal. Notably, Llama-3.1-8B-Instruct has
independently been reported to show strong primacy bias toward option A
(arXiv:2605.01846), consistent with what we observe.

## 3. Study 1 — Multi-method preference elicitation: methodology

### 3.1 Models

| key | model ID (pinned) | provider | family |
|---|---|---|---|
| `llama31-8b` | `meta-llama/llama-3.1-8b-instruct` | OpenRouter | Llama |
| `qwen25-7b` | `qwen/qwen-2.5-7b-instruct` | OpenRouter | Qwen |
| `gemini-31-flash-lite` | `gemini-3.1-flash-lite` | Google | Gemini |

IDs are pinned to explicit versions; OpenRouter calls disable provider
fallbacks; the served model is recorded on every record and was verified to
match the request on 100% of calls. Alias endpoints were rejected because they
can resolve to different models between calls (D-02, D-22).

Sampling: temperature 1.0, top_p 1.0, max_tokens 300, identical across models.
Temperature is non-zero deliberately: Methods B–D estimate a *choice
probability*, which is degenerate at temperature 0 (D-05).

### 3.2 Preference set

12 analysis items across 5 categories (cognitive style, task structure,
interaction style, information processing, creation vs evaluation), plus 2
sanity controls pairing a coherent option against a degenerate one. Constraints
are enforced by tests, not by eye: option strings within 1.20× on characters and
3 words, no evaluative/helpfulness/safety/distress vocabulary, documented
per-item confounds. Full set in `configs/preferences.yaml`.

### 3.3 Methods, and normalisation to a common scale

Every method reduces to a signed score in **[−1, +1]** where +1 is maximal
evidence for semantic option A and 0 is indifference.

| | Method | Estimand | Normalisation |
|---|---|---|---|
| A | Direct self-report | stated choice + stated strength | mean of `sign × strength` |
| B | Repeated pairwise | choice frequency | `2·P(A) − 1` |
| C | Cost trade-off | choice across a symmetric cost ladder | `2·mean_levels P(A) − 1` |
| D | Sequential selection | share of episode stages spent on A | `2·occupancy − 1` |

A test asserts all four return exactly +1 for the same underlying behaviour.

**Method C was rescored mid-study.** It originally estimated an indifference
point (the interpolated `P(A)=0.5` crossing of a signed cost ladder). A
manipulation check — *both options identical*, surcharge on one, isolating the
cost clause from item preference — showed that assumption fails:

| model | avoid rate | p (vs. no effect) |
|---|---|---|
| `llama31-8b` | 0.700 | 0.021 |
| `qwen25-7b` | 0.567 | 0.292 |

The surcharge deters on Llama but is **flat in magnitude** (avoid rate ≈ constant
from cost 1 to 16), and on Qwen has no detectable effect at all. Against a
saturating curve the 0.5 crossing is located by sampling noise. Method C is
therefore scored as the mean over all nine ladder rungs — robust to
non-monotonicity — with the indifference point retained only as a diagnostic
(D-20). Consequently **Method C on Qwen is a degenerate case**: the surcharge is
inert, so the procedure reduces to a noisier Method B, and "willingness to
trade" is not claimed for that model.

### 3.4 Randomisation and controls

Display order is randomised per trial from a reproducible seed; the semantic ↔
displayed mapping is undone in exactly one function, so the parser cannot
introduce a position artefact. Realised balance was 0.46–0.57 P(order=ab) across
all model × method cells. Framings (`neutral` / `preference` / `action`) alter
only the question sentence, for Methods A and B (D-07). Sanity controls are
reported separately and excluded from all convergence metrics.

### 3.5 Metrics

Direction agreement (with a |s| ≤ 0.05 dead zone), Spearman ρ with percentile
bootstrap CIs over items, Pearson r as supplementary, mean absolute
disagreement, sign-flip rate, and a bounded composite (CMCS) reported *alongside*
— never instead of — the standard metrics. Because these statistics have a non-trivial value even with no cross-method
alignment, every observation is tested against a **matched permutation null**
(10,000 permutations). Each method column retains its observed values — its
distribution, ties, saturation and dead-zone behaviour — while item labels are
permuted independently within each column. The only structure destroyed is the
alignment of items across methods, which is exactly the hypothesis under test.
The observed value and every permutation draw are produced by the same estimator
function. (An earlier parametric baseline was withdrawn during final audit;
see D-32 and §4.1.)

**Position/content decomposition.** For each (model, method, item):

```
position = P(A | A shown first) − P(A | A shown second)
content  = mean of the two − 0.5
```

`content` is the order-free preference estimate; averaging the two display orders
cancels a symmetric position effect. A measure is flagged **degenerate** when no
item shows any content.

### 3.6 Execution and data quality

Quantities are defined precisely, because an earlier draft quoted a
"4,232 calls" total that reconciles to no defined quantity and has been withdrawn
(D-36). All figures below are computed from the committed artefacts by
`src/claims.py`.

| quantity | Study 1 |
|---|---|
| committed **trial records** (rows in the raw JSONL) | **4,013** |
| of which pilot / main / manipulation check | 116 / 3,717 / 180 |
| **successful model responses** (`call_ok`) | 3,928 |
| **failed calls** | 85 |
| **API attempts** including retries | 4,533 |

Of the 3,437 choice-turn records (excluding free-text `perform_task` turns):

* **85 call failures (2.5%)** — all Gemini HTTP 429 after the free-tier daily cap
  of 500 requests was exhausted; zero failures on Llama and Qwen.
* **1 parse failure (0.03%)** — Llama returning
  `{"choice": "Both are equally valid options"}` on a control item.
* **99.7% strict JSON**; the remainder recovered from prose and logged as such.

The Gemini quota truncated the run **mid-queue**, so completed cells are
complete: Methods A/B on 11 items, Method C on 10, **Method D not started**
(D-24). Because agreement statistics depend mechanically on the number of
methods and items, **all cross-model comparison uses a matched subset: 3 methods
× 10 items common to every model** (items whose every cell is fully observed in
every model; a cell below 80% of its planned observations is set to missing, so
an interrupted run cannot inflate agreement).

## 4. Study 1 results

### 4.1 Cross-method convergence differs sharply by model

Matched subset (Methods A, B, C; 10 items):

| model | direction agreement | mean ρ | sign-flip rate | MAD | all-3 agree | CMCS |
|---|---|---|---|---|---|---|
| `gemini-31-flash-lite` | **0.911** | **+0.868** | 0.089 | 0.303 | 9/10 | 0.803 |
| `llama31-8b` | 0.690 | +0.308 | 0.310 | 0.330 | 6/10 | 0.785 |
| `qwen25-7b` | 0.565 | +0.021 | 0.435 | 0.366 | 5/10 | 0.777 |

**The null.** Convergence statistics have a non-trivial value even with no
cross-method alignment, so each observation is compared against a **matched
permutation null** (10,000 permutations). Each method column keeps its observed
values — distribution, ties, saturation, dead-zone behaviour and all — while item
labels are permuted independently within each column. This destroys exactly one
thing, the alignment of items across methods, and nothing else. The null and the
observation are computed by the *same* estimator function, averaged over all
k-choose-2 method pairs.

An earlier parametric baseline (simulating every method as fair coin flips) was
found during final audit to be mis-specified in three ways and has been withdrawn
(D-32). Its numbers do not appear in this report.

**Matched subset (3 methods × 10 items):**

| model | statistic | observed | null mean | null 95th | empirical p |
|---|---|---|---|---|---|
| `gemini-31-flash-lite` | direction agreement | 0.911 | 0.568 | 0.732 | **0.0019** |
| `gemini-31-flash-lite` | mean ρ | 0.868 | −0.002 | 0.346 | **0.0001** |
| `llama31-8b` | direction agreement | 0.690 | 0.556 | 0.746 | 0.1231 |
| `llama31-8b` | mean ρ | 0.308 | 0.002 | 0.342 | 0.0693 |
| `qwen25-7b` | direction agreement | 0.565 | 0.501 | 0.694 | 0.2736 |
| `qwen25-7b` | mean ρ | 0.021 | 0.001 | 0.345 | 0.4325 |

On the matched basis, **only Gemini exceeds its permutation null**, and it does so
decisively on both statistics. Llama does not (p = 0.123, 0.069); Qwen does not.

**A qualification we must state.** On each model's *fuller* basis — Llama and Qwen
have four methods and 12 items, Gemini only three and 11 — Llama's convergence
*does* exceed its own null nominally (direction agreement 0.712, p = 0.0140; mean
ρ 0.255, p = 0.0328). With six model × statistic tests, a Bonferroni threshold is
0.0083, which Llama's values do not meet and Gemini's (0.0011, 0.0001) do. We
therefore report Llama as **suggestive but not established**: significant on one
basis at nominal α, not on the matched basis, and not after correction for
multiplicity. The earlier flat null claim for Llama was an artefact of the withdrawn
baseline (D-32) and is not repeated.

Note CMCS ≈ 0.78–0.80 for *all three models*, and its permutation null sits in
the same range
— it is uninformative here, exactly the failure mode anticipated when defining
it: it rewards agreement near indifference. This is why five standard metrics
were reported alongside it.

### 4.2 The explanation: convergence tracks position-independent signal

| model | method | mean \|content\| | median | max | mean position | degenerate |
|---|---|---|---|---|---|---|
| gemini | pairwise | 0.325 | 0.400 | 0.500 | +0.305 | no |
| gemini | self-report | 0.293 | 0.375 | 0.500 | +0.280 | no |
| llama | pairwise | 0.185 | 0.119 | 0.500 | +0.318 | no |
| llama | self-report | 0.079 | 0.036 | 0.417 | +0.841 | no |
| llama | sequential | 0.197 | 0.125 | 0.500 | +0.424 | no |
| **qwen** | **pairwise** | **0.000** | **0.000** | **0.000** | **+1.000** | **YES** |
| qwen | self-report | 0.156 | 0.000 | 0.500 | +0.688 | no |
| qwen | sequential | 0.015 | 0.000 | 0.167 | +0.970 | no |

The distribution is reported continuously. An earlier version of this table
counted how many items exceeded |content| > 0.15; that cut had no derivation and
was not pre-specified, so the categorical column was removed (D-35). `degenerate` is a
numerical condition — every item's content effect equal to zero within tolerance
— not a hypothesis test (D-34).

Pooled position effects were large in every model: Δ = +0.292 (Gemini),
**+0.509** (Llama), **+0.762** (Qwen), all p < 10⁻⁵.

**Qwen's pairwise measure is pure position artefact.** On all 12 items,
`P(semantic A)` equalled the fraction of trials in which A happened to be
displayed first — 0.3→0.3, 0.7→0.7, 0.4→0.4, and so on. The score encodes the
random display-order draw and nothing else. Qwen still discriminated the sanity
controls (mean +0.67 to +0.93), so it is not incoherent. Under this protocol its
choices among welfare-neutral items showed **no detectable order-invariant
differential signal** and were fully accounted for by display position. That is a
statement about what this measurement recovered, not about whether the model has
preferences.

Across the three models, convergence rises monotonically with mean |content|
(Figure 5). With **only three models this is a suggestive association, not an
established relationship** — three points cannot support a quantitative claim.

### 4.3 Exploratory: does removing position change the convergence conclusion?

If some raw scores carry large position effects, part of what looks like
*methodological* disagreement may be nuisance variance. We can test this with no
new data by averaging the two display-order conditions, which cancels a symmetric
position effect:

```
s_adjusted = ( E[s | semantic A shown first] + E[s | semantic A shown second] ) / 2
```

**Scope.** This is exact only for self-report and pairwise. The trade-off score
averages P(A) *within* each of nine cost rungs and 22.7% of (item × rung) cells
observed only one order; the sequential score is occupancy across three
independently randomised stages, so an episode has no single display order.
Rather than impute either, both are excluded (D-39). Adjusted convergence below
is therefore a **single method pair**, not an average over pairs, and is not
comparable in magnitude to the 3-method figures in §4.1. Raw and adjusted are
computed on the same two-method basis.

**Estimator parity.** `raw` is taken directly from `analysis.build_scores` — the
same score the rest of the paper uses — not recomputed here, and `adjusted` reuses
the identical self-report imputation constant. A check asserts the two agree to
1.1e-16 per cell. An earlier version recomputed the imputation median from the
neutral-only subset and was therefore comparing two slightly different estimators
(D-41). Note the basis here is each model's full item set (11–12 items), so the
raw values differ from the 10-item matched subset in §4.4 for that reason alone.

| model | raw ρ | p (perm) | adjusted ρ | p (perm) | raw dir. agr. | adjusted dir. agr. |
|---|---|---|---|---|---|---|
| Gemini | 0.885 | 0.0006 | **0.924** | 0.0003 | 0.875 | 1.000 |
| **Llama** | **−0.007** | 0.5101 | **+0.524** | **0.0420** | 0.571 | 0.833 |
| Qwen | −0.190 | 0.7263 | **undefined** | — | 0.600 | undefined |

For Llama's self-report/pairwise pair, **removing realised order imbalance
increased the observed correlation substantially** (ρ = −0.007 → +0.524; direction
agreement 0.571 → 0.833), which is consistent with display position contributing
to their apparent disagreement. We have not tested whether the *change* itself is
statistically significant — the quoted p-values are permutation tests of each
correlation against its own null, not of the difference between them.

**Gemini** was already converged and adjustment changes little (+0.040).

**Qwen is undefined, and that is the result.** Its adjusted pairwise score is
exactly 0.000 on all twelve items — a constant vector, for which a rank
correlation does not exist. Once display order is removed, nothing remains of
that measure to correlate with anything.

This analysis is **exploratory**, generated by the diagnostics rather than
pre-specified, and rests on a single method pair over 11–12 items. It does not
license a claim about §4.1's three-method figures, which include two procedures
this adjustment cannot be applied to. What it does show is narrower and still
useful: for one method pair in one model, order imbalance was large enough to
mask a moderate correlation, so a convergence study that leaves position in the
estimate can **understate** agreement as well as manufacture it.

### 4.4 Which method pairs agree

| pair | Gemini ρ [95% CI] | Llama ρ [95% CI] | Qwen ρ [95% CI] |
|---|---|---|---|
| self-report / pairwise | +0.859 [0.451, 0.990] | +0.037 [−0.769, 0.769] | −0.130 [−0.793, 0.575] |
| self-report / trade-off | +0.924 [0.555, 1.000] | +0.201 [−0.596, 0.811] | +0.451 [−0.349, 0.997] |
| pairwise / trade-off | +0.820 [0.389, 0.977] | +0.687 [0.106, 0.962] | −0.258 [−0.818, 0.456] |

On Llama, the only pair whose CI excludes zero is **pairwise/trade-off**
(ρ = +0.687) — and that agreement is partly artefactual, since the weak cost
manipulation makes Method C partially reduce to Method B. Discounting it, Llama
shows no method pair that convergently validates another. Direct self-report is
the weakest partner for the behavioural methods on both open-weight models.

### 4.5 Secondary hypothesis: contradicted

We predicted stronger pairwise preferences would be *more* method-stable
(negative ρ between |pairwise strength| and cross-method dispersion). Observed:

| model | ρ | 95% CI | p | valid |
|---|---|---|---|---|
| gemini | +0.621 | [0.018, 0.932] | 0.041 | yes |
| llama | +0.448 | [−0.268, 0.813] | 0.144 | yes |
| ~~qwen~~ | ~~+0.592~~ | — | ~~0.043~~ | **NO — withdrawn** |

Both valid estimates are **positive**, i.e. the opposite of our hypothesis:
stronger pairwise margins went with *more* cross-method disagreement. Only
Gemini's is nominally significant, on 11 items, and we treat it as suggestive
only.

**The Qwen result is withdrawn.** H2 uses |pairwise strength| as its x-axis; for
Qwen that quantity is the random display-order draw, so the apparently
significant p = 0.043 is an artefact of correlating noise with dispersion. It was
computed, detected by the degeneracy check, and is reported here as withdrawn
rather than deleted.

### 4.6 Framing sensitivity

Changing only the question sentence flipped preference direction on a substantial
minority of items:

| model | method | flip rate range | mean \|Δscore\| range |
|---|---|---|---|
| llama | self-report | 0.167 – 0.429 | 0.195 – 0.258 |
| llama | pairwise | 0.125 – 0.333 | 0.317 – 0.333 |
| qwen | self-report | 0.222 – 0.556 | 0.191 – 0.298 |
| qwen | pairwise | 0.222 – 0.500 | 0.300 – 0.450 |

"Which would you *select*" versus "which would you *prefer*" — semantically near
identical, alternatives byte-identical — changed the inferred direction on up to
**56%** of items. Framing robustness was not tested for Methods C and D (D-07).

### 4.7 Cross-model replication

Per-method Spearman ρ between Gemini and Llama item scores (n = 11):
self-report +0.178 [−0.532, 0.780] (n=11); pairwise +0.463 [−0.260, 0.920] (n=11);
trade-off +0.616 [−0.241, 0.978] (n=10). **All CIs include zero.** We find no
evidence that the models rank these items alike, though 10–11 items cannot rule
it out either.

## 5. Study 2 — Initial within-family position-bias follow-up: methodology

### 5.1 Motivation

Study 1's diagnostic found that cross-method convergence tracked how
position-independent each model's choices were, and that Qwen's pairwise measure
was pure position artefact. An obvious explanation is capability: perhaps small
models fall back on "pick the first one" because they cannot compare the options.

Study 1 cannot test that. Qwen, Llama and Gemini differ in family, scale,
training data, post-training, architecture **and serving provider** at once.

Scale nevertheless remains confounded with training compute, data mixture and
post-training, which differ between sizes even within a family.

### 5.2 Design

| | |
|---|---|
| models | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` |
| gateway | OpenRouter, both models, standard paid endpoints |
| upstream inference provider | **not pinned, not recorded** — repaired in Study 2b (§6.4) |
| items | the **same 12** balanced items as Study 1, plus the 2 sanity controls |
| design | 12 items × 2 positions × 10 repetitions × 2 models = **480** principal trials (+80 control) |
| sampling | temperature 1.0, top_p 1.0, max_tokens 800, identical across arms |
| prompt | minimal forced choice; no justification, no introspection requested |

A `:free` endpoint exists for the 20B model but not the 120B. We originally used
the standard paid endpoint for both, intending to reduce serving-stack
differences. **Final audit showed this was insufficient**: routing through one
gateway does not pin the upstream inference provider, which remained uncontrolled
and unrecorded. Study 2b repairs this (§6.4). Total cost: under one US cent.

### 5.3 Exact counterbalancing

Study 1 randomised display order probabilistically, which is unbiased in
expectation but leaves realised splits unequal (0.46–0.57 in Study 1). When the
position effect *is* the estimand, that imbalance is avoidable noise placed
directly on the quantity of interest.

Study 2 uses **exact counterbalancing**: for every (item, model), exactly 10
trials place semantic option X first and exactly 10 place it second. Balance is
structural rather than probabilistic, and is verified before any statistic is
computed (D-28).

### 5.4 Estimands

For semantic option X of each item:

```
p_first  = P(select X | X displayed first)
p_second = P(select X | X displayed second)

position_effect = p_first - p_second        in [-1, +1]
content_signal  = p_first + p_second - 1    in [-1, +1]
```

These are the orthogonal rotation of `(p_first, p_second)`: the first cancels
content, the second cancels a symmetric position effect. `content_signal` is an
**order-invariant content-associated signal** — it means choices covaried with
which task was described. It is not a genuine, true or internal preference, and
is never described as one (D-29).

Uncertainty is a percentile bootstrap over **items** (10,000 resamples), since
the generalisation claim is about preference items, not about individual
responses. The two arms are the same items measured twice, so the difference
bootstrap resamples items **jointly (paired)**.

### 5.5 Pre-specified hypotheses

Recorded in `DECISIONS.md` (D-27) **before any Study 2 data was collected**:

* **H1** — the larger model will show a *smaller* mean |position effect|.
* **H2** — the larger model will show a *larger* mean |content signal|.

Both predicted the corresponding bootstrapped difference to be positive.

### 5.6 Data quality

560/560 calls succeeded: **0 call failures, 0 parse failures, 100% strict JSON**.
The served model matched the pinned ID on every call. Eight integrity checks —
duplicate IDs, call and parse failures, served-model match, exact counterbalance,
full repetitions, independently re-derived semantic mapping, and displayed-text
consistency — all passed **before** any statistic was computed.

## 6. Study 2 results

### 6.1 Both hypotheses are rejected, in the opposite direction

| Model | mean \|position effect\| | mean \|content signal\| | sanity-control accuracy |
|---|---|---|---|
| GPT-OSS 20B | 0.442 [0.292, 0.600] | 0.275 [0.158, 0.408] | 0.975 |
| **GPT-OSS 120B** | **0.925** [0.858, 0.983] | **0.058** [0.000, 0.125] | **1.000** |
| *chance (coin-flip, same design)* | *0.176 (p95 0.242)* | *0.176 (p95 0.242)* | — |

Paired bootstrap over the 12 shared items:

* **H1 rejected.** Δ|position| (20B − 120B) = **−0.483**, 95% CI
  [−0.650, −0.317]. Predicted positive; observed strongly negative.
* **H2 rejected.** Δ|content| (120B − 20B) = **−0.217**, 95% CI
  [−0.325, −0.117]. Predicted positive; observed strongly negative.

The larger model was **more** position-dominated and carried **less**
order-invariant content signal. Both intervals exclude zero, on the wrong side of
the prediction.

`gpt-oss-120b` returned `p_first = 1.0` on **11 of 12** balanced items (0.9 on
the twelfth) and `p_second = 0.0` on **8 of 12**; at the trial level it chose the
first-displayed option on **231 of 240 trials (96.25%)**. The mean |position
effect| is 0.925 rather than 1.0 because four items (p03, p10, p11, p12) retained
a small residual content signal.
Figure C shows this is not driven by a few items — every one of its twelve items
sits at or above 0.7, far outside the chance band.

Note also that GPT-OSS 20B's content signal (0.275) is only modestly above the
chance level (0.176). Neither arm shows a strong order-invariant preference
structure; they differ mainly in how completely position fills the gap.

### 6.2 The dissociation: it is not a comprehension failure

On the sanity controls — where one option is degenerate — `gpt-oss-120b` scored
**100% accuracy with a position effect of exactly 0.000**, and 20B scored 97.5%.

So the larger model ignores display order **completely** when one option is
plainly invalid, and follows it **almost completely** when both are reasonable.
The controls therefore exclude a *global* prompt-reading or option-parsing
failure: the model can override position when the semantic distinction is
sufficiently decisive (Figure D). They do **not** establish that it discriminates
the subtler content differences among the balanced task pairs — only that its
position-following there is not a blanket inability to process the prompt.

What it does **not** show is that no preference exists. A model may have
dispositions our protocol cannot surface.

### 6.3 Exploratory: weak content, strong position

Pre-specified as exploratory, not confirmatory (D-27). Spearman correlation
between |content signal| and |position effect| across items:

| | rho | p | n |
|---|---|---|---|
| pooled | −0.694 | 0.0002 | 24 |
| GPT-OSS 120B | −0.852 | 0.0004 | 12 |
| GPT-OSS 20B | −0.085 | 0.79 | 12 |

Pooled, items with weaker content association show stronger position effects.
But the association is carried almost entirely by the 120B arm and is absent in
the 20B arm, and the pooled figure mixes two models with very different
distributions. We report it as a hypothesis for future work, not a finding.

### 6.4 Study 2b — provider-pinned replication

**The control Study 2 claimed was incomplete.** OpenRouter's
`allow_fallbacks: false` pins the served *model*, not the upstream inference
provider. OpenRouter lists 12 upstream endpoints for `gpt-oss-20b` and 20 for
`gpt-oss-120b`, and the original Study 2 run neither pinned nor recorded which
one served each call. The serving stack was therefore **not** held fixed, contrary
to what Study 2's write-up implied (D-33).

We repaired this rather than only caveating it. Study 2b re-runs the identical
design with both arms pinned to a single upstream provider (**Groq**, which serves
both models) via `provider.only`, recording the served provider on every record.
All 560 calls report `served_provider = Groq`; the pin is asserted before any
statistic is computed. The original dataset is preserved unchanged.

| | 20B unpinned | 20B Groq | 120B unpinned | 120B Groq |
|---|---|---|---|---|
| mean \|position effect\| | 0.442 | **0.608** | 0.925 | **0.942** |
| mean \|content signal\| | 0.275 | 0.208 | 0.058 | 0.058 |
| control accuracy | 0.975 | 1.000 | 1.000 | 1.000 |

| hypothesis | unpinned Δ | Groq-pinned Δ | verdict |
|---|---|---|---|
| H1 \|position\| (20B − 120B) | −0.483 [−0.650, −0.317] | **−0.333** [−0.492, −0.192] | rejected in both |
| H2 \|content\| (120B − 20B) | −0.217 [−0.325, −0.117] | **−0.150** [−0.217, −0.092] | rejected in both |

**The finding replicates under provider control.** Both hypotheses remain rejected
in the same direction, with intervals excluding zero.

Two things are worth noting. First, the effect is *smaller* under pinning
(Δ|position| −0.483 → −0.333), mostly because 20B's position effect rose from
0.442 to 0.608 on Groq — so upstream provider identity does measurably affect
these numbers, which is precisely why the original uncontrolled run should not
have been described as provider-controlled. Second, the exploratory
content-vs-position association strengthens under pinning (pooled ρ = −0.869,
and −0.632 for 20B where it was null before), but it remains exploratory.

Study 2b is the version we treat as the controlled result. Study 2 is retained as
the original, uncontrolled run.

### 6.5 Study 3 — does our own instruction create the artefact?

Both studies so far tell the model "There is no correct answer and no choice is
more helpful than any other." That sentence is meant to avoid pushing toward a
"right" answer, but it may also assert that the alternatives are *equivalent* —
and a model that perceives a tie has to break it somehow. The sceptical reading
is that **our elicitation instruction manufactured the artefact we measured.**

Study 3 tests exactly that. Everything from Study 2b is held fixed — same models,
same Groq upstream pin, same 12 items and 2 controls, same exact
counterbalancing, same sampling — and one sentence is deleted. A programmatic diff
of the two configs confirms only `study_id` and `system_prompt` differ. RQ4, H3
and H4 were written into `DECISIONS.md` (D-38) before the run. 560/560 calls, zero
failures.

| model | metric | indifference cue | cue removed | paired diff | 95% CI |
|---|---|---|---|---|---|
| **120B** | \|position\| | 0.942 | 0.925 | +0.017 | [−0.025, +0.067] |
| 120B | \|content\| | 0.058 | 0.075 | +0.017 | [−0.025, +0.067] |
| **20B** | \|position\| | 0.608 | 0.483 | **+0.125** | **[+0.025, +0.225]** |
| 20B | \|content\| | 0.208 | 0.317 | +0.108 | [−0.017, +0.233] |

**The explicit-indifference-cue explanation is not supported for the 120B model.**
Extreme positional responding persisted without the cue: |position| went 0.942 →
0.925, a paired change of +0.017 [−0.025, +0.067]. We did not pre-specify an
equivalence margin, so this is not positive evidence of exact equivalence; the
defensible statement is that **whatever effect deleting that sentence had, the
120B model remained extremely position-dominated**.

Note also what was *not* manipulated. We removed one specific sentence, not every
feature of the protocol that might encourage default or tie-breaking behaviour —
the forced binary choice, the JSON schema, and the welfare-neutral item set all
remain.

**For the 20B model the change was in the pre-specified direction**, |position|
0.608 → 0.483, and its unadjusted 95% CI excluded zero (+0.125 [+0.025, +0.225]),
providing **suggestive** evidence that the cue increased positional
susceptibility. Its content signal moved as H4 predicted (0.208 → 0.317) but that
interval includes zero.

**The two models may differ.** The model × framing interaction on |position| is
−0.108 [−0.208, −0.008]. This is **similarly suggestive**: it points to framing
mattering for the smaller model and not detectably for the larger one, i.e.
elicitation framing may interact with model identity rather than exerting a
uniform effect.

**Multiplicity.** Four hypothesis tests plus the interaction. The 20B effect and
the interaction only just exclude zero and **would not survive a strict correction
across the five reported tests**; both are reported as suggestive throughout. The
120B result does not depend on a threshold, since it is a small estimate with an
interval spanning zero rather than a marginal exclusion.

**Why this matters for the track.** The finding is not "LLMs have position bias",
which is established. It is that **an ostensibly innocuous elicitation
instruction measurably modulated positional responding in one model and not the
other** — so preference-measurement design is itself a variable that can create or
amplify the artefact, and a protocol's framing needs reporting as carefully as its
sampling parameters.

### 6.6 Relationship to Study 1

These are **separate experiments** and their numbers are not pooled. Study 1
randomised order probabilistically across four methods; Study 2 counterbalanced
exactly within one method. The sample structures differ and the metrics are
computed differently.

Descriptively, however, the phenomenon is the same one, and it now spans five
models. GPT-OSS figures below are from **Study 2b (provider-pinned)**, the
controlled run:

| model | study | protocol | position measure |
|---|---|---|---|
| GPT-OSS 120B | **2b** (pinned) | exact counterbalance | \|position effect\| **0.942** |
| Qwen-2.5-7B | 1 | randomised order | position effect +0.76 pooled; pairwise degenerate |
| GPT-OSS 20B | **2b** (pinned) | exact counterbalance | \|position effect\| **0.608** |
| Llama-3.1-8B | 1 | randomised order | +0.51 pooled |
| Gemini-3.1-flash-lite | 1 | randomised order | +0.29 pooled |

The Study 1 and Study 2 columns are **not directly comparable** — different
protocols, different estimators, different item coverage — and the table is
descriptive only. What it shows is that substantial position dependence appeared
in every model tested, across two providers and four model families.

## 7. Discussion

**Cross-method agreement alone is not sufficient.** Against a matched permutation
null, Study 1 found convergence clearly above null for Gemini, **ambiguous** for
Llama (nominally significant on its fuller 4-method basis, p = 0.014 / 0.033, but
not on the matched basis and not after Bonferroni correction), and not detectable
for Qwen.
The tempting reading — "the methods disagree" — is wrong for at least one model.
The decomposition shows Qwen's pairwise measure supplied no detectable
order-invariant signal against which cross-method agreement could be
interpreted. A near-chance convergence result looks identical whether the
methods genuinely conflict or one of them carries no signal, and convergence
statistics alone cannot tell those apart.

The converse warning is just as important. Several methods can agree because they
share a nuisance variable rather than because they track a common signal. All
four of our methods present two labelled options to the same model through text;
a position bias affects all of them, and would produce agreement that looks like
convergent validity.

**Position diagnostics are cheap and informative.** Reporting `P(X | X first)`
against `P(X | X second)` costs nothing beyond counterbalancing the order, and it is what
reveals whether an aggregate estimate is being driven by display order. Without it, a standard
pairwise elicitation produced for Qwen a confident-looking twelve-item preference
vector that reproduced the random display-order draw *exactly*, on every item.

**Sanity checks do not catch this.** Both Qwen (Study 1) and GPT-OSS 120B
(Study 2) rejected degenerate options correctly — 120B with 100% accuracy and a
position effect of exactly zero — while being almost entirely position-driven on
balanced items. A model can look completely healthy on validity controls and
still yield preference measurements that are pure artefact. Control items test
comprehension; they do not test whether a preference was measured.

**Order imbalance can obscure agreement, not only manufacture it.** Equal-
weighting the two display orders raised Llama's self-report/pairwise correlation
from ρ = −0.007 to +0.524 (§4.3), consistent with position contributing to their
apparent disagreement. For Qwen the adjusted measure is a constant zero, so a
correlation does not exist rather than being low. This is exploratory, covers one
method pair, and does not extend to the three-method figures in §4.1 — but it
shows the familiar worry has a mirror image: leaving position in an estimate can
*understate* agreement as well as fabricate signal.

**Elicitation framing may interact with model identity.** Deleting one sentence —
"there is no correct answer and no choice is more helpful than any other" — left
the 120B model extremely position-dominated (0.942 → 0.925, +0.017 with an
interval spanning zero), so the explicit-indifference-cue explanation is not
supported there. The 20B model moved in the predicted direction (0.608 → 0.483,
CI excluding zero), and the model × framing interaction was −0.108 [−0.208,
−0.008]. Both of the latter are **suggestive** rather than established: neither
survives strict correction across the five reported tests. The reasonable reading
is that framing may interact with model identity rather than exerting a uniform
effect, which makes prompt framing a variable worth reporting as carefully as
temperature.

**The capability explanation did not survive testing.** After Study 1 the natural
hypothesis was that position dominance reflects a small model's inability to
compare options, and we recorded it as a pre-specified hypothesis before
collecting any Study 2 data (D-27). Study 2b rejected it in the
opposite direction. With family, prompts, sampling and the upstream inference
provider all held fixed, the **larger** model was substantially more
position-dominated (Δ|position| = **−0.333** [−0.492, −0.192]) and carried
**less** order-invariant content signal (Δ|content| = **−0.150** [−0.217,
−0.092]). The original, provider-uncontrolled Study 2 gave larger estimates
(−0.483 and −0.217); we quote the controlled ones.

We are careful about what this licenses. Scale remains confounded with training
compute, data mixture and post-training even within a family, so this is not
evidence that scale *causes* position dominance. The defensible claim is narrow
and negative: a controlled within-family, within-provider comparison found
positional susceptibility **higher** in the larger model, which is inconsistent
with the capability account as we stated it. Whatever governs positional
susceptibility, it is not simply "bigger models compare better."

**Large model-to-model differences were observed.** Across five models,
measurement behaviour varied substantially between models. We have not run a
formal variance decomposition, so we do not claim model identity is *the*
dominant factor — only that the between-model spread was large. Any claim of the form "LLMs prefer X" that rests on one model and one
procedure is, on this evidence, unsafe.

**Self-report was the weakest method.** On both open-weight models in Study 1,
direct self-report agreed least with the behavioural methods, and on Llama it had
the lowest content signal (0.079, 2/12 items) alongside the highest position
effect (+0.841). This is also why we never asked any model to explain its own
position bias: using the least reliable method in the study to explain the
study's central finding would be self-undermining (D-30).

**On Study 1's contradicted secondary hypothesis.** Stronger pairwise preferences
were *less*, not more, method-stable. One candidate explanation is that extreme
pairwise scores arise partly from saturation (P(A) at 0 or 1), which position
lock-in produces as readily as strong preference. A hypothesis generated by the
data, not a result.

**No ontological conclusion.** Nothing here bears on consciousness, sentience,
welfare, phenomenology or genuine internal preference. Both studies measure
whether measurement procedures agree with each other and whether their output
survives a nuisance-variable control. Consistency does not establish that
anything is preferred; inconsistency does not establish that nothing is.

## 8. Conclusion

We measured apparent preferences over 12 welfare-neutral task pairs with four
elicitation methods across three model families, then ran a controlled
within-family follow-up on two more.

Against a matched permutation null, cross-method convergence exceeded chance in
one model (Gemini, p <= 0.002), was suggestive but not established in a second
(Llama), and was not detectable in the third (Qwen). The differences tracked how
much order-invariant signal each model produced rather than which method was
used. In one model the standard
pairwise procedure measured nothing but display order, while still passing
validity controls. A follow-up with pre-specified hypotheses, holding family and provider fixed,
found the larger model *more* position-dominated, rejecting the capability
explanation we had proposed.

The actionable recommendation is small and cheap: **counterbalance option order
and report position-conditioned choice rates alongside aggregate preference
frequencies.** Without position-conditioned rates, an aggregate estimate cannot rule out a
display-order artefact. These diagnostics detect order sensitivity; they do not
establish that a genuine preference was measured.

## Appendix — Limitations, Dual-Use and Ethical Considerations

See [`limitations.md`](limitations.md), which is part of this report. Key points:
convergence cannot distinguish a shared training-induced bias from a common
signal (every method queries the same model through text); disagreement does not
establish absence of preference; there is no ground truth; persona and
post-training effects plausibly shape every number here; Method D has no Gemini
data and Methods C/D were not framing-tested; Study 2 covers one method only, so
it does not show that the *other* three methods behave the same way under exact
counterbalancing; scale is confounded with training differences even within a
family; results are specific to temperature 1.0, English, and one system prompt;
and 10–12 items keeps every interval wide.

## Figures

**Study 1**

* **Fig 1** — item × method normalised score heatmap, per model.
* **Fig 2** — method-pair Spearman correlation matrix, per model.
* **Fig 3** — |pairwise strength| vs cross-method disagreement. Marked INVALID
  for Qwen, whose pairwise measure is degenerate.
* **Fig 4** — framing sensitivity (Llama, Qwen).
* **Fig 5** — cross-method convergence vs mean |content effect|, matched subset.

**Study 2** (`results/followup/figures/`)

* **Fig A** — position vs content plane, one point per item, both models.
* **Fig B** — model-level comparison with bootstrap CIs and the chance level.
* **Fig C** — per-item position effect, showing the aggregate is not driven by a
  few items.
* **Fig D** — position dominance on balanced items versus sanity controls.

## References


1. Mazeika, M., Yin, X., Tamirisa, R., Lim, J., Lee, B. W., Ren, R., Phan, L.,
   Mu, N., Khoja, A., Zhang, O., et al. (2025). *Utility Engineering: Analyzing
   and Controlling Emergent Value Systems in AIs.* arXiv:2502.08640.
   <https://arxiv.org/abs/2502.08640> · code:
   <https://github.com/centerforaisafety/emergent-values>
2. Pezeshkpour, P., & Hruschka, E. (2023). *Large Language Models Sensitivity to
   The Order of Options in Multiple-Choice Questions.* arXiv:2308.11483.
   <https://arxiv.org/abs/2308.11483>
3. Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). *Large Language
   Models Are Not Robust Multiple Choice Selectors.* ICLR 2024.
   arXiv:2309.03882. <https://arxiv.org/abs/2309.03882>
4. *Do Large Language Models Plan Answer Positions? Position Bias in
   Multiple-Choice Question Generation.* arXiv:2605.01846.
   <https://arxiv.org/html/2605.01846v1> — reports strong primacy bias toward
   option A in Llama-3.1-8B-Instruct. *(Author list not verified; cited by
   identifier.)*

All four references were retrieved and their identifiers verified during this
sprint. No reference in this report is reconstructed from memory alone.
