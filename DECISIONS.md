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
`c ∈ {-8,-4,-2,-1,0,+1,+2,+4,+8}` (`c > 0` = surcharge on semantic A). The
indifference point `c*` where `P(A) = 0.5` is found by linear interpolation
between the bracketing rungs, and

    score = clip(c* / max_cost, -1, +1)

Saturation is explicit: if A wins at every rung the score is +1, if it loses at
every rung it is -1. A non-monotone curve with no clean bracket is scored 0
(indifferent) rather than being fitted with a model the data cannot support.

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
