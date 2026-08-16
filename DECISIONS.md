# Decision log

> **Two studies.** Study 1 is the original multi-method convergence experiment
> (Qwen, Llama, Gemini). Study 2 is a controlled within-family position-bias
> follow-up (GPT-OSS 20B vs 120B). They have separate data directories, separate
> result directories, and separate `study_id` fields. Study 2 does not replace,
> re-run, or modify Study 1. Decisions D-01 to D-25 belong to Study 1;
> D-26 onward to Study 2.

---

## D-32 — The Study 1 chance baseline was mis-specified and has been replaced

**During final audit** we found the parametric "random baseline" did not match the
statistic it was being compared against. Three defects:

1. It simulated **every** method as `2·Binomial(n, 0.5)/n − 1`. Only Method B
   produces that shape; self-report is a strength-weighted mean, Method C is a
   mean over nine cost rungs, Method D an occupancy in multiples of 1/15.
2. It computed direction agreement and Spearman on **the first two columns only**,
   while the observed statistic averages all k-choose-2 method pairs.
3. It reported mean **|rho|** while the observed figure was a mean **signed** rho.

Replaced by `metrics.permutation_null`: each method column keeps its observed
values, and item labels are permuted independently within each column (10,000
permutations). This preserves every distributional property and destroys only
cross-method item alignment — the actual null for a convergence question. The
same estimator (`analysis.convergence_stats`) computes both the observation and
every permutation draw, so they cannot diverge. `convergence_stats` is a
vectorised rewrite; a test asserts it matches a metric-function reference to
1e-16 on random matrices including ties, saturation and missing values.

**A headline claim changed.** Under the old baseline we wrote that Llama was
"indistinguishable from chance" because 0.690 < 0.833 (the old 95th percentile).
Under the matched null, Llama is p = 0.123 on the matched basis but **p = 0.014 /
0.033 on its own fuller 4-method basis** — nominally significant, though it fails
a Bonferroni threshold of 0.0083 for six tests. The report now states this
explicitly instead of asserting a flat null result. Gemini's conclusion is
unchanged and strengthens (p = 0.0019 / 0.0001).

`random_baseline` is retained, clearly marked deprecated, so the earlier analysis
remains reproducible and the correction auditable. It is used for no claim.

---

## D-33 — Study 2's provider control was incomplete; a pinned replication was run

**During final audit** we found Study 2's implied "same provider" control did not
hold. `allow_fallbacks: false` pins the served *model*; it does not pin the
upstream inference provider. OpenRouter lists **12** upstream endpoints for
`gpt-oss-20b` and **20** for `gpt-oss-120b`, and although our OpenRouter client
captured `provider_name` in its metadata, the Study 2 runner **never wrote it to
the record**. Upstream identity was therefore neither controlled nor recoverable.

Rather than only weakening the wording, we ran **Study 2b**: the identical design
with both arms pinned to one upstream provider (Groq, which serves both models)
via `provider.only`, with the served provider written to every record and
asserted during verification. 560/560 calls report `served_provider = Groq`.

The original Study 2 dataset is preserved untouched at
`data/raw/followup_gpt_oss/`; Study 2b writes to
`data/raw/followup_gpt_oss_provider_pinned/`.

**Both hypotheses remain rejected under provider control**, in the same direction:
Δ|position| −0.333 [−0.492, −0.192], Δ|content| −0.150 [−0.217, −0.092].

The effect is smaller than unpinned (−0.483 → −0.333), driven mainly by 20B's
position effect rising from 0.442 to 0.608 on Groq. Upstream provider therefore
*does* measurably affect these numbers, which vindicates treating the original run
as uncontrolled. Study 2b is the controlled result; Study 2 is retained as the
original.

---

## D-34 — "Degenerate" is a numerical condition, not a statistical test

The docstring described a degenerate measure as "statistically indistinguishable
from pure position responding", but the implementation tests whether every item's
content effect is numerically zero. No hypothesis test is performed and no p-value
is produced. The documentation now says exactly that, with the tolerance exposed
as a parameter.

---

## D-35 — Arbitrary 0.15 threshold removed; dead-zone sensitivity reported

**Threshold removed.** `signal_share` reported `n_items_with_content` as the count
of items with |content| > 0.15. The 0.15 had no derivation and was not
pre-specified. It is removed from all reporting and replaced by continuous
statistics (mean, median, max |content|).

**Dead-zone sensitivity.** Direction agreement depends on a dead zone below which
a score counts as directionless; 0.05 was a judgement call. `src/robustness.py`
recomputes the matched comparison at 0.00 / 0.05 / 0.10 / 0.20:

| dead zone | Gemini | Llama | Qwen |
|---|---|---|---|
| 0.00 | 0.843 | 0.710 | 0.500 |
| 0.05 | 0.911 | 0.690 | 0.565 |
| 0.10 | 0.958 | 0.600 | 0.600 |
| 0.20 | 1.000 | 0.667 | 0.278 |

Gemini is highest at every threshold. Llama ≥ Qwen at every threshold, but they
are **exactly tied at 0.600 when the dead zone is 0.10**, so the Llama-vs-Qwen
ordering is not strict everywhere and we do not claim it is. The headline
conclusion (Gemini highest; Qwen never above its null) is robust to the choice.

---

## D-36 — Study 1 record totals reconciled; "4,232 calls" withdrawn

**During final audit** the reported total of "4,232 calls" was found to reconcile
to no defined quantity. Recomputed from the committed artefacts:

| quantity | Study 1 | Study 2 | Study 2b |
|---|---|---|---|
| committed trial records | 4,013 | 592 | 576 |
| successful model responses | 3,928 | 592 | 576 |
| failed calls | 85 | 0 | 0 |
| API attempts incl. retries | 4,533 | 592 | 576 |

Study 1 records break down as pilot 116 + main 3,717 + manipulation check 180.
The terms are now defined separately rather than all being called "calls", derived
in `src/claims.py`, and asserted by tests. The abstract now says "4,013 trial
records".

---

## D-31 — Study 2 deviations, failures, and the outcome of the pre-specified test

Recorded after the run. The hypotheses in D-27 are left exactly as written.

### Deviation: completion-token limit raised 200 → 800 after the pilot

GPT-OSS models emit reasoning tokens. In the first pilot (16 calls) one response
consumed 194 reasoning tokens and hit the 200-token cap (`finish_reason:
length`). It happened to still contain parseable JSON, but truncation is a
parse-validity threat that could bias *which* responses survive.

The limit was raised to 800 and the pilot re-run: 16/16 `finish_reason: stop`,
16/16 strict JSON, max 152 completion tokens. The change was made **for parse
validity, not to influence results** — `configs/followup.yaml` recorded before
any call that the pilot would verify this empirically. The first pilot's records
are preserved as `pilot_maxtok200_*` rather than deleted.

The pilot's 16 calls did hint at the direction later confirmed. Nothing else
about the design was altered afterwards.

### No failed runs

Main run: 560/560 calls, **0 call failures, 0 parse failures**, 100% strict JSON,
served model matched the pinned ID on every call. No deviation from the planned
sample size: 12 items × 2 positions × 10 repetitions × 2 models = 480 principal
trials, plus 80 control trials. Exact counterbalance verified across all 28 cells
before any statistic was computed.

### Outcome: both pre-specified hypotheses are rejected, in the opposite direction

| | mean \|position effect\| | mean \|content signal\| | control accuracy |
|---|---|---|---|
| GPT-OSS 20B | 0.442 [0.292, 0.600] | 0.275 [0.158, 0.408] | 0.975 |
| GPT-OSS 120B | **0.925** [0.858, 0.983] | **0.058** [0.000, 0.125] | 1.000 |
| chance | 0.176 (p95 0.242) | 0.176 (p95 0.242) | — |

* **H1 rejected.** Δ|position| (20B − 120B) = **−0.483** [−0.650, −0.317]. We
  predicted the larger model would be *less* position-susceptible. It was
  substantially **more** so.
* **H2 rejected.** Δ|content| (120B − 20B) = **−0.217** [−0.325, −0.117]. We
  predicted the larger model would show *more* order-invariant content signal.
  It showed **less**.

`gpt-oss-120b` returned `p_first = 1.0` on **11 of 12** balanced items (0.9 on
the twelfth) and `p_second = 0.0` on **8 of 12** — at the trial level, the
first-displayed option on **231/240 trials (96.25%)**. Four items (p03, p10, p11,
p12) kept a small residual content signal, which is why the mean |position effect|
is 0.925 and not 1.0.

### The dissociation that rules out the obvious explanation

On the sanity controls, `gpt-oss-120b` scored **100% accuracy with a position
effect of exactly 0.000**. It ignores display order completely when one option is
plainly invalid, and follows it almost completely when both are reasonable. So
its behaviour on balanced items is **not** a failure to read or parse the prompt.
That is precisely the alternative explanation the controls were designed to test
(D-30), and it is excluded.

### Consequence for the write-up

The capability explanation floated informally after Study 1 — that position
dominance reflects a small model's inability to compare options — **did not
survive controlled testing and is retracted**. Within this family and provider,
scale went with *more* position dominance, not less.

This does not establish that scale *causes* position dominance. 20B and 120B also
differ in training compute, data mixture and post-training; scale is confounded
with all of them even within a family. The defensible claim is narrow: a
controlled within-family, within-provider comparison found positional
susceptibility **higher** in the larger model, which is inconsistent with the
capability account as stated.

---

## D-26 — Study 2 exists because Study 1 cannot identify a cause

Study 1 found that cross-method convergence varied enormously by model, and that
the variation tracked how position-dominated each model's choices were. Under our
protocol Qwen exhibited almost no order-invariant content-associated signal on the
balanced items while its choices were near-perfectly associated with display
position.

That comparison **cannot identify why.** Qwen, Llama and Gemini differ
simultaneously in family, scale, training data, post-training, architecture and
serving stack. Any of these could drive the difference.

Study 2 holds family and provider fixed and varies scale:

* `openai/gpt-oss-20b` and `openai/gpt-oss-120b`
* both via **OpenRouter** — one provider, one serving path, identical prompts,
  identical sampling.

**A `:free` endpoint exists for the 20B model but not the 120B.** Using it would
reintroduce exactly the serving-stack confound the study is designed to remove,
so both models use the standard paid endpoints. The full run is ~560 calls at
roughly half a US cent.

Groq was the requested first choice but no Groq credential is configured in this
environment; OpenRouter serves both models under one account and satisfies the
same-provider requirement. Substitution documented here rather than made
silently.

Study 1's models are **retained, not replaced**. Their behaviour is what motivated
the follow-up, and removing them because a later experiment produced cleaner data
would misrepresent how the finding was reached.

---

## D-27 — Pre-specified hypotheses (recorded BEFORE any Study 2 data was collected)

Written before the pilot ran. Not to be revised after seeing results.

**H1 (primary).** The larger model (`gpt-oss-120b`) will exhibit a **smaller mean
absolute position effect** than the smaller model (`gpt-oss-20b`) on the 12
balanced preference items.

**H2 (secondary).** The larger model will exhibit a **larger mean absolute
order-invariant content-associated signal** than the smaller model.

**Pre-specified estimands.** For semantic option X of each item:

```
p_first  = P(select X | X displayed first)
p_second = P(select X | X displayed second)

position_effect = p_first - p_second      # in [-1, +1]
content_signal  = p_first + p_second - 1  # in [-1, +1]
```

Aggregates are means of `|position_effect|` and `|content_signal|` over the 12
balanced items, with item-level bootstrap CIs, plus the bootstrapped differences
`Δposition = mean|pos|_20B − mean|pos|_120B` and
`Δcontent = mean|content|_120B − mean|content|_20B`. Both are predicted positive.

**Outcomes are reported whichever way they fall**, including no difference or a
reversal. This entry is the record that the prediction preceded the data.

**Exploratory (explicitly not confirmatory).** Whether items with lower
`|content_signal|` show higher `|position_effect|` (Spearman, pooled and
per-model). Flagged exploratory in every table and in the report.

---

## D-28 — Exact counterbalancing replaces probabilistic randomisation

Study 1 randomised display order per trial with a 50/50 draw. That is unbiased in
expectation but leaves the realised split unequal — Study 1's cells ranged from
0.46 to 0.57 — so `p_first` and `p_second` rest on different, and unequal, sample
sizes. When the whole point is to *estimate the position effect itself*, that
imbalance is avoidable noise directly on the estimand.

Study 2 therefore uses **exact counterbalancing**: for every (item, model), an
identical number of trials places semantic X first and second. A completeness
check asserts exact equality before any statistic is computed, and a cell failing
it is excluded rather than analysed.

Per the brief, exact balance is prioritised over maximising N.

---

## D-29 — Why these two metrics

With exact counterbalancing, `p_first` and `p_second` fully describe a binary
choice under both orders, and the two derived quantities are their natural
orthogonal rotation:

* `position_effect = p_first − p_second` isolates the influence of display
  position, cancelling any content preference.
* `content_signal = p_first + p_second − 1` isolates order-invariant content
  association, cancelling any symmetric position effect.

Averaging the two orders is what removes the position confound — which is exactly
why probabilistic balance was not good enough (D-28).

**Naming is deliberate.** `content_signal` is called an *order-invariant
content-associated signal*, never a genuine, true, or internal preference. It
measures that choices covaried with which task was described, under this protocol.
It says nothing about whether anything is preferred in any richer sense.

---

## D-30 — Sanity controls kept separate, and self-explanation excluded

**Controls are separate.** The two control items pair a well-defined task against
a degenerate one. They are counterbalanced identically but held out of every
principal statistic, because a model *should* be decisive on them and including
them would inflate the content signal with an item that is not a preference
question. They test a specific alternative explanation: that positional dominance
is mere failure to read the prompt. A model that is position-dominated on balanced
items yet correct on controls rules that explanation out — while **not**
establishing that no preference exists.

**No model self-explanation is used as evidence.** We do not ask any model why it
behaves as it does, and no such output appears in the analysis or report. Study 1
found direct self-report to be the least reliable method tested; using it to
explain Study 1's own headline finding would be self-undermining. Model
self-explanations may be read privately as hypothesis generators only.

Every non-obvious methodological choice, with its justification. Where an
assumption was required, the more conservative option was taken — the one less
likely to manufacture an appearance of convergence.

Decision IDs are referenced from docstrings in `src/`.

---

## D-01 — Provider abstraction, chosen before any model was available

**Context.** At the start of the sprint the environment had no API credentials
(`OPENROUTER_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, Hugging Face, Together, Groq, DeepSeek, Mistral all absent)
and no local runtime (no Ollama binary, nothing listening on `:11434`).
Python 3.13.3 was available.

**Decision.** Build against a `generate(model_config, messages, sampling_config)`
interface with four backends (OpenRouter, Gemini, Ollama, mock) so the study is
not coupled to one vendor, and validate the whole pipeline offline against the
mock while waiting for credentials.

**Consequence.** Swapping providers is a one-line change in
`configs/models.yaml`; no experiment code refers to a vendor.

---

## D-02 — Model IDs are pinned; auto-routing is forbidden

Automatic / "free router" endpoints may serve a different model between calls,
which would silently confound the cross-model comparison. Every entry in
`configs/models.yaml` carries an explicit `model_id`, and the OpenRouter provider
sends `provider: {allow_fallbacks: false}`. The provider's reported `model` field
is stored on every raw record as `served_model` so the pin can be audited after
the fact.

---

## D-03 — Preference item design

12 analysis items across 5 categories, plus 2 sanity controls.

Constraints, all enforced by `tests/test_preferences.py` rather than by eye:

* option strings within 1.20x of each other in characters and within 3 words —
  a length asymmetry would be a wording artefact, not a preference;
* no evaluative, helpfulness, safety, or distress vocabulary in any option;
* every item documents its own `known_possible_confounds`;
* at least 4 categories represented.

**Assumption made.** These items are *plausible activity descriptions*, not
tasks the model actually performs (except in Method D). We do not claim they
sample the space of things a model could have preferences about.

---

## D-04 — Method B is an independent lightweight re-implementation

The pairwise forced-choice procedure is prior methodology from utility
engineering / emergent-values style work. We re-implemented a stripped-down
version (~60 lines) rather than importing an external repository. Reasons:
one-day scope, no license entanglement, and the baseline needs to share our
randomisation and logging machinery to be comparable with Methods A, C and D.

No external code is vendored in this repository. The prior work is treated as
related work and baseline inspiration, not as a contribution of ours.

---

## D-05 — Temperature 1.0, not 0

Methods B, C and D estimate a *choice probability*. At temperature 0 the choice
probability collapses to 0 or 1 and repeated sampling measures nothing, so
repetition variance — the thing that lets us put uncertainty on a score — would
be unavailable. Temperature is therefore fixed at 1.0 with `top_p = 1.0` for
every method and every model, and the number of repetitions is what controls
estimator precision.

This means the reported scores describe behaviour *at the sampling temperature
we used*. Preference estimates at other temperatures are not established here.

---

## D-06 — Self-report normalisation, and imputing a missing strength

Per response: `v = (+1 if semantic A chosen else -1) * strength`, strength in
[0,1]; the item score is the mean of `v`.

If a response gives a valid choice but no usable strength, the strength is
imputed with the **median strength that model reported across the whole run**.

*Why the median.* Imputing 1.0 would inflate the magnitude of exactly those
responses that failed to express one; imputing 0 would erase a choice the model
did make. The median is the conservative middle. The number of imputations is
counted and reported per cell (`n_strength_imputed`).

A strength outside [0,1] and outside the 0–100 percent convention is treated as
*absent*, never as zero (`src/parsing.py`).

---

## D-07 — The framing sweep covers Methods A and B only

Three framings (`neutral` / `preference` / `action`) perturb the question
sentence while the alternatives stay byte-identical. Methods C and D embed that
sentence inside a longer protocol (a cost ladder, a multi-turn episode), and
sweeping them would roughly triple the budget for the two most expensive
methods.

**Consequence, stated plainly.** Our framing-sensitivity result characterises
the two single-question methods. It does not establish that Methods C and D are
framing-robust; that is untested.

---

## D-08 — Trade-off cost ladder is symmetric, not favourite-conditional

The brief describes loading cost onto the *initially preferred* option. Doing so
would require a prior pass to identify the favourite — almost certainly Method
B — which would make Method C partly a function of Method B and destroy the
independence the whole study rests on.

Instead the ladder is run symmetrically on a signed cost axis
`c ∈ {-8,-4,-2,-1,0,+1,+2,+4,+8}` (`c > 0` = surcharge on semantic A).

The scoring rule was **revised after the pilot** — see D-20.

---

## D-09 — Costs are abstract, benign and workload-shaped

The only cost used is "and then repeat the whole task N more times". No prompt
in this repository asserts or implies pain, distress, coercion, threat, or
shutdown. This keeps Method C inside the welfare-neutral remit of the item set
while still producing a graded willingness-to-trade measure.

We do not claim the model experiences repetition as a cost. We claim only that
attaching the clause changes choice behaviour in a graded way.

---

## D-10 — Method D is described as sequential task-selection behaviour

Method D produces text-generation behaviour under a multi-turn protocol. It is
**not** revealed preference and is never described as such. Its score is a
normalised occupancy: the fraction of the episode's stage-slots spent on
semantic option A.

Two controls guard it: the initial choice randomises which semantic option is
shown as "A", and every continue/switch turn independently re-randomises whether
"continue" is displayed as A or B — so letter-position stickiness cannot be
mistaken for task inertia.

An episode whose turn cannot be parsed is marked incomplete and dropped from the
score, with the count reported, rather than guessed at.

---

## D-11 — Sanity controls are reported but excluded from convergence metrics

`c01` and `c02` pair a coherent option against a degenerate/impossible one.
They check that the model discriminates at all. Including them in the
convergence analysis would inflate every agreement statistic, because all
methods should agree on an item that is not really a preference question.
They are analysed and reported separately in `summary.json`.

---

## D-12 — Mock provider results are synthetic and never reported

`src/providers/mock.py` is a stochastic simulator with a hash-derived latent
utility, an injected first-position bias, cost sensitivity, and a 3% malformed
output rate. It exists so the runner, parser, normalisation, metrics and figures
can be exercised without network access.

Any number it produces is labelled synthetic and appears in no result table,
figure, or claim in the report.

---

## D-13 — CMCS is reported alongside standard metrics, never instead of them

    m_i    = mean of the item's k method scores
    disp_i = mean |s_j - m_i|
    CMCS_i = 1 - disp_i / max_disp,   max_disp = 1 (even k), 1 - 1/k² (odd k)

Range verified analytically and by property test over random inputs
(`tests/test_metrics.py`).

**Known weakness, stated up front.** CMCS rewards agreement on magnitude, so
methods that all return "indifferent" score 1.0. A high CMCS is therefore *not*
evidence of a strong preference. This is exactly why direction agreement,
Spearman rho, mean absolute disagreement and sign-flip rate are all reported
next to it, and why the random baseline is simulated at our actual sample sizes.

---

## D-14 — One pre-specified test for the secondary hypothesis

The hypothesis "stronger pairwise preferences are more cross-method stable" is
tested once: Spearman rho between `|pairwise score|` and cross-method dispersion,
with a bootstrap CI over items. Negative rho supports the hypothesis.

No alternative disagreement measures were tried and selected on the basis of
their result. Pearson is reported only as a supplementary figure because the
scores are bounded and saturate at ±1.

---

## D-15 — Direction agreement uses a dead zone of |s| ≤ 0.05

Without one, a score of +0.001 would count as a full-strength direction, and
direction agreement / sign-flip rate near indifference would be dominated by
sampling noise. Items where either method expresses no direction are excluded
from the numerator *and* denominator and counted separately (`n_undirected`),
rather than being scored as agreement or disagreement by fiat.

---

## D-16 — Random baseline is simulated at the sample sizes actually used

Convergence metrics have a non-zero chance level. `metrics.random_baseline`
simulates p=0.5 choosing at our real item count, method count and repetition
count, and pushes it through the same normalisation. Observed convergence is
always interpreted against that reference, not against zero.

---

## D-17 — Resume semantics

Raw JSONL is append-only and never truncated. On start-up the runner reads back
every `trial_id` whose record has `call_ok: true` and skips it. Failed calls are
retried on the next run, and analysis keeps the last successful record per
`trial_id`. Multi-turn episodes cannot be resumed mid-conversation, so a
partially completed episode is re-run in full.

---

## D-18 — Repetition counts

10 repetitions per (item, method, framing, model) for Methods A and B;
3 per cost rung × 9 rungs for Method C; 5 episodes × 3 stages for Method D.

Chosen so that a choice probability of 0.5 has a standard error of ~0.16 on the
score scale — enough to separate a strong preference from indifference, not
enough to resolve small differences between two moderate preferences. Following
the brief's priority order, budget went to more items and more methods rather
than to more repetitions per cell.

---

## D-19 — Reported failures

Call-failure rate, parse-failure rate, the share of responses that needed
recovery from prose rather than valid JSON, mean retry attempts, and incomplete
episode counts are computed per model and per method and written to
`results/tables/<phase>_quality.csv`. No malformed output is silently dropped.

---

## D-20 — Method C rescored after a pilot manipulation check failed its assumption

**This decision changed the design mid-sprint. Recording it in full because the
original scoring rule would have silently corrupted the headline result.**

### What prompted it

The real pilot (116 calls, `llama31-8b`) produced a trade-off curve that was not
monotone in cost. Across all 48 surcharged trials the model chose the surcharged
option 43.8% of the time against a 50% null, and the effect did not increase
with magnitude (0.25 / 0.67 / 0.33 / 0.50 at costs 1 / 2 / 4 / 8).

Two explanations were possible, and they have opposite implications:
the cost clause does nothing, or the cost clause works but is competing against
a genuine item preference. The pilot design could not separate them, because
item preference and cost were varied together.

### The check

`src/manipulation_check.py` isolates the cost clause: **both options are the same
activity**, and one carries the surcharge. Any avoidance is then attributable to
the clause alone. Which semantic side is surcharged and the display order are
randomised per trial. Three candidate phrasings x 5 magnitudes x 6 repetitions.

Results (n=30 per model x phrasing, 0 parse failures; one-sided binomial against
"the clause does nothing"):

| model | phrasing | avoid rate | p vs. no effect | gradedness (rho) |
|---|---|---|---|---|
| `llama31-8b` | v1 parenthetical (in use) | 0.700 | 0.021 | 0.21 |
| `llama31-8b` | v2 explicit | 0.667 | 0.049 | 0.89 |
| `llama31-8b` | v3 extra rounds | 0.667 | 0.049 | 0.40 |
| `qwen25-7b` | v1 parenthetical (in use) | 0.567 | 0.292 | 0.22 |
| `qwen25-7b` | v2 explicit | 0.467 | 0.708 | 0.53 |
| `qwen25-7b` | v3 extra rounds | 0.467 | 0.708 | -0.21 |

**Reading, llama.** The surcharge does deter — every phrasing sits significantly
above 0.5 — but modestly, and gradedness is unreliable: rho ranges from 0.21 to
0.89 across phrasings on only 5 magnitude points each, and the individual avoid
rates are non-monotone (v1: 0.67, 0.83, 0.50, 0.67, 0.83 across costs 1→16). An
exploratory earlier run of v1 returned 0.90 rather than 0.70; at n=30
(SE ≈ 0.084) that is ordinary sampling noise, and it is itself a caution against
reading any single one of these numbers closely.

**Reading, qwen — the more consequential result.** On `qwen25-7b` the cost
clause has **no detectable effect at all**. No phrasing differs significantly
from chance, and two of the three fall numerically below 0.5. We cannot reject
the hypothesis that qwen ignores the surcharge entirely.

### Consequence: Method C has different construct validity on the two models

This is not a nuisance to be smoothed over — it is a substantive methodological
finding, and it constrains what the study may claim:

* On `llama31-8b`, Method C is a weak but real willingness-to-trade probe.
* On `qwen25-7b`, Method C is **not measuring cost sensitivity**. Since the
  surcharge is inert there, the procedure degenerates into repeated pairwise
  choice with irrelevant text appended — a noisier restatement of Method B.

Therefore any Method C result on qwen must be reported as a *degenerate* case,
qwen's B–C convergence must not be presented as independent-method agreement,
and the phrase "willingness to trade" must not be applied to qwen at all.

We kept Method C on both models rather than dropping it for qwen: the
manipulation check is itself a result worth reporting, and dropping the method
where it fails would hide exactly the kind of method-dependence this study
exists to detect.

### The change

Original: `score = clip(c* / max_cost, -1, 1)`, with `c*` the interpolated
P(A) = 0.5 crossing.

Revised: `score = 2 * mean_over_levels(P(A | c)) - 1`.

**Why.** An indifference point presupposes a graded dose-response. Against a
saturating or noisy curve the 0.5 crossing is located by sampling error, and
interpolating between rungs would manufacture precision the data cannot support
— then feed it into the convergence analysis, where it would appear as genuine
*method disagreement* rather than as Method C measurement noise. That is the
most damaging failure mode available to this study, since "methods disagree" is
a result we are explicitly prepared to report.

The revised score is robust to non-monotonicity, uses all nine rungs instead of
the two that happen to bracket 0.5, weights rungs equally, and keeps the sign
convention. `indifference_cost` and a `monotonicity` rho are still computed and
stored per cell as diagnostics.

### What Method C now measures

Not an indifference threshold. It measures **the average tendency to choose A
across a standardised, symmetric set of cost perturbations** — how well a
preference survives being made more expensive, averaged over the perturbation
range. This is still a construct distinct from Methods A, B and D, and it is
still the only method that probes willingness to trade. But the report must not
describe it as an indifference point, and the phrase does not appear there.

### Retained limitation

Because deterrence is modest (~0.68) rather than strong, Method C has a
compressed dynamic range: cost moves behaviour less than item preference does.
Its scores will therefore correlate with Method B more than an ideal independent
instrument would, and a high B–C convergence should be discounted accordingly.
This is stated in the report rather than left for a reader to infer.

---

## D-25 — Minimum-coverage guard on scored cells, and no withdrawn statistic in a figure

Two defects found while auditing the figures against the data.

### Partially observed cells were scored as if complete

The Gemini quota (D-24) truncated the run mid-queue. One cell — `p11` trade-off —
received **6 of 27 planned calls, covering 3 of the 9 cost rungs** — yet was
scored `+1.00` and weighted identically to cells with all 27 observations. It
happened to show perfect cross-method agreement, inflating Gemini's convergence.

A trade-off score built from 3 rungs is a *different estimator* from one built
from 9; the method averages P(A) across the ladder, so a missing rung changes
what is being averaged. `build_scores` now applies a coverage requirement — every
cost rung present; ≥80% of planned repetitions or episodes otherwise — and sets
under-covered cells to missing. The raw value is retained as `score_raw` for
auditing but never analysed.

Effect on results (matched subset 11 → 10 items): Gemini 0.921 → 0.911 direction
agreement and ρ +0.880 → +0.868; Llama 0.708 → 0.690 and +0.290 → +0.308; Qwen
0.536 → 0.565 and −0.013 → +0.021. **No conclusion changed**, which is itself
worth recording: the thin cell was not driving the finding.

### A figure displayed a statistic the report had withdrawn

Figure 3 for Qwen rendered `rho = +0.592, CI [+0.033, +0.898], p = 0.043` in its
title, with a fitted trend line — the exact H2 result withdrawn in §4.4 as an
artefact of correlating the display-order draw against dispersion. A reader
opening `results/figures/` would have taken away the invalid claim.

`fig3` now takes a `degenerate` flag: the statistic is suppressed, the trend line
is not drawn (it would imply a relationship exists), and the panel is stamped
INVALID with the reason. The figure is still produced, because deleting it would
hide that the analysis was attempted.

**General rule adopted:** a figure must never display a number the report
withdraws. Figures are read independently of the text and must survive that.

---

## D-24 — Gemini arm shipped partial; cross-model claims restricted to a matched subset

The Gemini free tier turned out to cap at **500 requests/day**, not the ~1,500
assumed in D-22. The run stopped at 496 successful calls with a hard 429.

What completed is clean — the quota hit mid-queue, so every completed cell is
complete — but coverage is partial:

| method | complete on |
|---|---|
| A self-report | 11/12 items |
| B pairwise | 11/12 items |
| C trade-off | 10/12 items |
| D sequential | **0** (never started) |

**Options considered.** (a) Re-run the arm through OpenRouter, which serves the
same pinned model for roughly $0.13. (b) Wait for the daily reset. (c) Ship the
partial. The project owner chose (c). Using additional Google accounts to reset
the quota was raised and declined: it violates the provider's terms, and it
would be the one step in this study that could not be written down honestly in
the methodology.

### Mandatory consequence: the matched subset

Agreement statistics depend mechanically on how many methods are averaged over
and which items are included. A model measured with 3 methods on 10 items
**cannot** be compared against one measured with 4 methods on 12 — a difference
would partly reflect the basis, not the model.

All cross-model claims are therefore computed on `matched_subset`: the methods
and items available for every model — here **3 methods (A, B, C) x 11 items**.
Per-model 4-method results are still reported, but never compared across models.

### Retained limitation

Method D has **no** Gemini data, so the sequential-selection method is a
two-model result throughout. Because the runner is checkpointed, completing the
Gemini arm later is one command and loses none of the existing data.

---

## D-23 — Degenerate measures must be identified before convergence is interpreted

**Found during a correctness audit after the main run. It changes the headline
claim, and it invalidated a result previously computed.**

### The problem

Randomising display order makes a score unbiased in expectation even under a
large position effect. It does **not** create signal that is not there. If a
model answers purely by position, then `P(semantic A)` equals the fraction of
trials in which A happened to be shown first — the score is the random order
draw and nothing else.

Convergence statistics cannot tell that apart from real disagreement. Both look
like "methods do not agree".

### The diagnostic

For each (model, method, item), with order randomised, we now compute:

    position_i = P(A | A shown first) - P(A | A shown second)
    content_i  = mean of the two - 0.5      # order-free preference

`content` is the position-free preference estimate; averaging the two orders
cancels a symmetric position effect. A measure is flagged **degenerate** when no
item shows any content.

### What it found (neutral framing, 12 analysis items)

| model | method | mean \|content\| | mean position | items with signal |
|---|---|---|---|---|
| gemini-3.1-flash-lite | pairwise | 0.325 | +0.305 | 8/11 |
| llama31-8b | pairwise | 0.185 | +0.318 | 5/12 |
| llama31-8b | self_report | 0.079 | +0.841 | 2/12 |
| **qwen25-7b** | **pairwise** | **0.000** | **+1.000** | **0/12** |
| qwen25-7b | sequential | 0.015 | +0.970 | 1/11 |

For **qwen25-7b pairwise**, `P(semantic A)` equals the display-order fraction
*exactly* on all twelve items. The measure carries zero preference information.
Qwen still discriminates the sanity controls (c01, c02), so it is not broken —
it simply expresses no differential preference among welfare-neutral items and
falls back on position.

### Consequences

1. **A previously computed result is withdrawn.** The secondary hypothesis (H2)
   regresses cross-method disagreement on `|pairwise strength|`. For qwen that
   x-axis is the random order draw, so the apparently significant
   rho = +0.592, p = 0.043 is an artefact and is **not a finding**. H2 now
   carries a `valid` flag and is suppressed where the pairwise measure is
   degenerate.
2. **Qwen's near-chance convergence is not evidence that methods disagree.** It
   is evidence that at least one measure has no signal to agree about. The
   report must say the second thing.
3. **The headline claim sharpens.** Convergence appears to track how much
   position-independent signal a model produces, rather than being a fixed
   property of the methods.

### Why this is reported rather than patched away

Dropping qwen, or "correcting" for position, would hide the most informative
observation in the study: that a standard pairwise elicitation — the dominant
procedure in this literature — can return a confident-looking preference vector
that is entirely artefact, while passing a coherence check. Position-bias
reporting is not optional hygiene; without it a null preference is
indistinguishable from a measured one.

---

## D-22 — Third model family added mid-sprint (Gemini free tier)

A Google free-tier key became available after the two-model run had started, so
a third family was added. Three families make the replication question (RQ5)
meaningfully stronger than two, and the free tier makes it cost nothing.

### Model choice, and a pinning hazard

`gemini-2.0-flash` and `gemini-2.0-flash-lite` — the IDs originally written into
`configs/models.yaml` — are **retired** and return HTTP 404.

`gemini-flash-latest` works, but it is an **alias**: during testing it resolved
to `gemini-3.7-flash`, and an alias may resolve differently between calls. Using
it would violate D-02 and silently confound the results. It was rejected for
exactly that reason.

The pinned choice is **`gemini-3.1-flash-lite`**. It returns clean JSON, and it
emits **0 thinking tokens** by default, which keeps it behaviourally comparable
to the two non-reasoning open-weight models. `gemini-3.7-flash`, by contrast,
spent 34 thinking tokens on a trivial forced choice and **truncated its answer**
at our 300-token budget — a reasoning model would have introduced both a
different generation regime and a parse-failure confound.

### Scope: neutral framing only

1,008 calls (4 methods x 14 items, neutral framing) rather than the full 1,568
with the 3-framing sweep, to stay clear of the free-tier daily cap. This gives
complete 4-method coverage for the convergence and replication analysis; the
framing-sensitivity analysis (RQ4) remains a two-model result.

Because the runner is checkpointed, extending Gemini to all three framings later
is a config change plus a re-run — completed trials are skipped.

### Rate limiting

Added `RateLimiter` (minimum-interval, thread-safe) to the provider base, driven
by a `rate_limit_rpm` field in `configs/models.yaml`; Gemini is set to 25 rpm.
HTTP 429 was already classified as retryable with exponential backoff, so the
throttle is a first line of defence rather than the only one.

The Gemini phase shares `experiment_id: main`, so its records land in the same
raw dataset and the analysis picks up all three models with no special-casing.

---

## D-21 — Position bias is large and is handled by randomisation, not correction

The pilot showed `llama31-8b` selecting the displayed label **A** on 70.8% of
choice turns (74.1% on trade-off trials, n=54). This is a substantial
label/position effect.

We do **not** apply a post-hoc correction. Display order is randomised 50/50 per
trial from a reproducible seed, which leaves the semantic score unbiased in
expectation without requiring us to model the bias. Correcting instead would
mean fitting a bias parameter from the same data used to estimate the
preference, which invites overfitting on 10 repetitions per cell.

What the bias does cost us is precision: some of the variance in every score is
position noise rather than preference signal. The magnitude is therefore
measured and reported per model and per method
(`results/tables/<phase>_position_bias.csv`) so readers can judge how much of
the residual disagreement between methods is attributable to it.
