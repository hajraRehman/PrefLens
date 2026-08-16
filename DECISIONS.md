# Decision log

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
