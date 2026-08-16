# No Signal, No Convergence: A Multi-Method Convergent-Validity Study of LLM Preference Elicitation

**PrefLens** — Digital Minds Research Sprint, Track 4 (Preference Elicitation Methods)
Code and raw data: <https://github.com/hajraRehman/PrefLens>
Run date: 2026-08-16. All results reproducible from `data/raw/main/`.

---

## Abstract

Apparent LLM preferences are increasingly measured, but no ground truth exists
against which any measurement can be checked, leaving convergent validity as the
main available evidence. We implemented four independent elicitation methods —
direct self-report, repeated pairwise choice, a cost trade-off, and sequential
task selection — over 12 welfare-neutral task pairs, on three model families
(4,232 logged calls, temperature 1.0, display order randomised per trial). On a
matched 3-method × 10-item basis, cross-method convergence differed sharply by
model: Spearman ρ = +0.868 (Gemini-3.1-flash-lite), +0.308 (Llama-3.1-8B),
+0.021 (Qwen-2.5-7B); only Gemini exceeded a simulated chance ceiling.
Decomposing choices into position and content components explains this:
Qwen's pairwise measure was **pure position artefact**, reproducing the random
display-order draw exactly on all 12 items while still passing coherence checks.
Convergence tracked available signal, not method quality. Measurement
consistency alone licenses no claim about genuine preference.

*(149 words)*

---

## 1. Introduction

Work on apparent LLM preferences typically operationalises "preference" one way —
most often repeated pairwise forced choice — and reports the resulting numbers.
The field has no ground truth: there is no independently verified fact about what
a model prefers against which an elicited score could be scored correct.

In measurement terms that leaves one kind of evidence available: **convergent
validity**. If several genuinely independent operationalisations of the same
construct land in the same place, the estimate is at least method-robust. If they
do not, conclusions drawn from any single procedure are conclusions about the
procedure as much as about the model.

This study treats preference elicitation as a measurement-validity problem. We
ask whether four independent methods recover the same preference direction and
strength, which methods agree, and whether apparent agreement survives framing
perturbation and replicates across model families.

**Contributions.**

1. A reusable, fully logged elicitation harness implementing four methods behind
   one provider-agnostic interface, with per-trial order randomisation, strict
   parse accounting, checkpointing, and 74 tests.
2. A cross-method convergence analysis on three model families, reported against
   a chance baseline simulated at the actual sample sizes.
3. **A position/content decomposition that distinguishes genuine method
   disagreement from a measure that carries no preference signal at all** — and
   the finding that one standard pairwise elicitation was entirely artefact.
4. Two pre-registered-style checks that changed the design mid-study: a
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

## 3. Methodology

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
— never instead of — the standard metrics. A **random baseline** is simulated at
the actual item, method and repetition counts, because these statistics have a
non-trivial chance level on 11–12 items.

**Position/content decomposition.** For each (model, method, item):

```
position = P(A | A shown first) − P(A | A shown second)
content  = mean of the two − 0.5
```

`content` is the order-free preference estimate; averaging the two display orders
cancels a symmetric position effect. A measure is flagged **degenerate** when no
item shows any content.

### 3.6 Execution and data quality

4,232 logged calls total (pilot 116; manipulation check 180; main 3,136 for
Llama+Qwen; 579 attempted for Gemini). Of 3,437 analysed choice calls:

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

## 4. Results

### 4.1 Cross-method convergence differs sharply by model

Matched subset (Methods A, B, C; 10 items):

| model | direction agreement | mean ρ | sign-flip rate | MAD | all-3 agree | CMCS |
|---|---|---|---|---|---|---|
| `gemini-31-flash-lite` | **0.911** | **+0.868** | 0.089 | 0.303 | 9/10 | 0.803 |
| `llama31-8b` | 0.690 | +0.308 | 0.310 | 0.330 | 6/10 | 0.785 |
| `qwen25-7b` | 0.565 | +0.021 | 0.435 | 0.366 | 5/10 | 0.777 |
| *simulated chance* | *0.50 (95th pct **0.833**)* | *0.25 (95th pct **0.58**)* | — | — | — | *0.77–0.78* |

**Only Gemini's convergence exceeds the chance ceiling.** Llama's direction
agreement (0.690) sits below the 95th percentile of chance (0.833) and its mean
ρ (+0.308) below the chance 95th percentile (0.580). Qwen is at chance
throughout. We therefore do **not** claim that Llama's methods converged.

Note CMCS ≈ 0.78–0.80 for *all three models* against a chance level of 0.77–0.78
— it is uninformative here, exactly the failure mode anticipated when defining
it: it rewards agreement near indifference. This is why five standard metrics
were reported alongside it.

### 4.2 The explanation: convergence tracks position-independent signal

| model | method | mean \|content\| | mean position | items with signal | degenerate |
|---|---|---|---|---|---|
| gemini | pairwise | 0.325 | +0.305 | 8/11 | no |
| gemini | self-report | 0.293 | +0.280 | 6/11 | no |
| llama | pairwise | 0.185 | +0.318 | 5/12 | no |
| llama | self-report | 0.079 | +0.841 | 2/12 | no |
| llama | sequential | 0.197 | +0.424 | 5/11 | no |
| **qwen** | **pairwise** | **0.000** | **+1.000** | **0/12** | **YES** |
| qwen | self-report | 0.156 | +0.688 | 4/12 | no |
| qwen | sequential | 0.015 | +0.970 | 1/11 | no |

Pooled position effects were large in every model: Δ = +0.292 (Gemini),
**+0.509** (Llama), **+0.762** (Qwen), all p < 10⁻⁵.

**Qwen's pairwise measure is pure position artefact.** On all 12 items,
`P(semantic A)` equalled the fraction of trials in which A happened to be
displayed first — 0.3→0.3, 0.7→0.7, 0.4→0.4, and so on. The score encodes the
random display-order draw and nothing else. Qwen still discriminated the sanity
controls (mean +0.67 to +0.93), so it is not incoherent; it simply expressed no
differential preference among welfare-neutral items and defaulted to position.

Across the three models, convergence rises monotonically with mean |content|
(Figure 5). With **only three models this is a suggestive association, not an
established relationship** — three points cannot support a quantitative claim.

### 4.3 Which method pairs agree

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

### 4.4 Secondary hypothesis: contradicted

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

### 4.5 Framing sensitivity

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

### 4.6 Cross-model replication

Per-method Spearman ρ between Gemini and Llama item scores (n = 11):
self-report +0.178 [−0.532, 0.780] (n=11); pairwise +0.463 [−0.260, 0.920] (n=11);
trade-off +0.616 [−0.241, 0.978] (n=10). **All CIs include zero.** We find no
evidence that the models rank these items alike, though 10–11 items cannot rule
it out either.

## 5. Discussion

**Where methods agree.** Only on Gemini, and there strongly and consistently
(all three pairs ρ ≈ 0.82–0.92, CIs excluding zero). Where a model produces a
stable order-free preference, independent operationalisations recover it.

**Where they diverge, and why.** On the two open-weight models, convergence was
indistinguishable from chance. The decomposition shows this is not primarily
"methods measuring different constructs" — it is that little position-independent
signal existed to be measured. This distinction is the study's main methodological
point, and it is invisible to convergence statistics alone: a near-chance
convergence result looks identical whether methods genuinely disagree or one
measure is empty.

**Self-report vs behaviour.** On both open-weight models, direct self-report was
the weakest partner for the behavioural methods, and had the lowest content
signal on Llama (0.079, only 2/12 items) despite the highest position effect
(+0.841). Asking a model what it prefers, and observing what it selects, did not
recover the same thing.

**The practical warning.** A standard pairwise elicitation returned a
confident-looking 12-item preference vector for Qwen that was entirely position
artefact — while passing a coherence check on degenerate controls. Published
preference findings that do not report a position decomposition cannot be
distinguished from this case. We recommend that any preference-elicitation result
report `P(A | A first)` versus `P(A | A second)` per item as a minimum.

**On the contradicted hypothesis.** That stronger pairwise preferences were
*less* method-stable is unexpected. One candidate explanation is that extreme
pairwise scores are produced partly by saturation (P(A) at 0 or 1), which can
arise from position lock-in as readily as from strong preference, and such items
need not behave extremely under other methods. We flag this as a hypothesis
generated by the data, not a result.

## 6. Conclusion

Across three model families and four elicitation methods on 12 welfare-neutral
task pairs, cross-method convergence was strong in one model, and
indistinguishable from chance in the other two. A position/content decomposition
indicates the difference is largely about how much order-independent signal each
model produced, not about which method is best. In one model, the standard
pairwise procedure measured nothing but display order.

Preference conclusions in this setting are method-dependent and framing-dependent,
and can be artefactual without any outward sign. Measurement consistency —
present or absent — licenses no conclusion about genuine internal preference.

## Appendix — Limitations, Dual-Use and Ethical Considerations

See [`limitations.md`](limitations.md), which is part of this report. Key points:
convergence cannot distinguish a shared training-induced bias from a common
signal (all four methods query the same model through text); disagreement does
not establish absence of preference; there is no ground truth; persona and
post-training effects plausibly shape every number here; Method D has no Gemini
data and Methods C/D were not framing-tested; results are specific to temperature
1.0, English, and one system prompt; and 11–12 items keeps every interval wide.

## Figures

* **Fig 1** — item × method normalised score heatmap, per model.
* **Fig 2** — method-pair Spearman correlation matrix, per model.
* **Fig 3** — |pairwise strength| vs cross-method disagreement (H2).
* **Fig 4** — framing sensitivity (Llama, Qwen).
* **Fig 5** — cross-method convergence vs mean |content effect|, matched subset.

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
