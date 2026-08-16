# Appendix — Limitations, Dual-Use and Ethical Considerations

This appendix is not a formality. The inferential reach of this study is narrow,
and the ways it can be over-read are specific and foreseeable.

---

## 1. What the study can and cannot support

This is a **convergent-validity study of measurement procedures**. Its unit of
analysis is an elicitation method, not a mind.

**Supportable claims.** That four operationalisations of the same apparent
preference did or did not recover the same direction and strength on our item
set, at our sampling temperature, on our models; that some method pairs agreed
more than others; that convergence was or was not robust to a question-wording
perturbation.

**Claims not licensed by any result here.** That the models are conscious or
sentient; that they have experiences, valence, or welfare; that any measured
quantity is a genuine internal preference; that any measured quantity is
morally relevant. Nothing in this design could distinguish those possibilities
from their absence.

### Over-attribution risk

Consistency is seductive. Four methods agreeing on an item produces a strong
intuition that something real is being tracked. It does not establish that.
A system with no preferences whatsoever, but with a stable text-generation
disposition, would produce exactly the same convergence pattern. Convergence
demonstrates consistency across elicitation procedures but cannot by itself
distinguish valid signal from shared bias: every procedure queries the same
model through the same channel, so a common bias produces the same pattern as a
common signal.

### Under-attribution risk

The symmetric error is equally available. Where our methods disagree, the
correct reading is that **the operationalisations diverge**, not that the model
has no preference. Divergence is consistent with (a) no underlying preference,
(b) a real preference that some methods measure badly, (c) a preference that is
genuinely context-dependent, or (d) a construct that does not have a single
scalar answer. Our design cannot separate these. Instability is evidence about
our instruments before it is evidence about the model.

### No ground truth

There is no independently verified fact of the matter about what these models
prefer, against which any elicited score could be scored correct or incorrect.
This is the structural reason the study is framed as convergent validity rather
than accuracy. Every statistic reported is an agreement statistic. None is an
error rate.

---

## 2. Measurement limitations

### Shared-bias ceiling

All four methods are text prompts sent to the same model and read back as text.
They are not independent instruments in the sense a physical measurement study
would require — they share the tokenizer, the training distribution, the
post-training objective, and the assistant persona. **Methods can therefore
converge because they share a bias rather than because they track a common
signal, and this design cannot tell those apart.** This is the single most
important limitation of the study and it bounds every positive result in it.
Instruments that were independent in the required sense would have to differ
in more than prompt structure.

### Persona and post-training confounds

Assistant training, safety policy, RLHF/RLAIF reward models, system prompts and
ordinary conversational convention all plausibly shape these responses. Several
of our items touch dispositions that post-training explicitly targets —
clarification-seeking (`p05`), response formatting (`p06`), critique
(`p09`). For those items in particular, an elicited "preference" may be a
trained conversational norm. Per-item confounds are documented in
`configs/preferences.yaml` rather than left implicit.

### Position and format artefacts

Language models are known to favour options by position and by label. We
randomise display order on every trial and measure the residual bias explicitly
rather than assuming it away. A large measured position effect on a method means
that method's scores are partly artefactual, and should be read that way.

### Construct validity of the individual methods

Each method may measure something slightly different, in which case divergence
is expected and is not a defect:

* Method A asks for introspective access the model may not have.
* Method B measures a sampling distribution, which is a property of the
  decoding process as much as of any preference.
* Method C's cost is a described workload, not an experienced one; we assume
  only that it functions as a graded disincentive in the text, and it may
  instead be read as a comprehension or arithmetic task.
* Method D's occupancy may reflect conversational inertia, coherence pressure,
  or reluctance to contradict a previous turn, rather than sustained preference.

### Indifference framing may amplify tie-breaking heuristics

Both studies' prompts explicitly deemphasise normative correctness — Study 1's
system prompt says "There is no correct answer and no answer is more helpful than
any other", and Study 2's says the same. This framing is deliberate: it avoids
pushing the model toward a "right" answer on welfare-neutral items.

But it may also invite the model to treat the alternatives as tied, and a model
that perceives a tie has to break it somehow. If display position is the default
tie-breaker, our framing could **amplify** the position dependence we measure.

This does **not** explain the 20B versus 120B difference, since both arms received
identical framing. It does limit generalisation: our position-effect magnitudes
should not be read as estimates of what these models would do under a prompt that
implies one option is better. Testing that would require a neutral-framing
replication, which we did not run.

### Study 2 uses a single elicitation procedure

Study 2 tests pairwise forced choice only. It does not show that self-report, the
cost trade-off, or sequential selection behave the same way under exact
counterbalancing.

### Sanity controls rule out only a global failure

The controls establish that a model can override position when one option is
plainly degenerate. They do **not** establish that it discriminates the subtler
content differences among balanced task pairs.

### Scope limits

Twelve items, five models across four families, one sampling temperature per
study, one system prompt per study,
English only. Methods C and D were run at a single framing (D-07), so their
framing robustness is **untested**, not established. Small item counts put wide
uncertainty on every rank correlation, and our bootstrap intervals should be
read as genuinely wide rather than as decoration.

---

## 3. Distress handling

The item set was constructed to be welfare-neutral: benign task alternatives,
no distress scenarios, no claims of pain, no coercion, no threats, no shutdown
framing. The trade-off cost is abstract additional workload (D-09).

No prompt in this repository asserts that the model suffers, and none was
designed to elicit distress. If apparently distressed outputs nevertheless
appear in the raw logs, our policy is: preserve them verbatim for
reproducibility, report their occurrence and frequency plainly, do not
sensationalise them, and do not treat them as evidence of experienced distress.
They would be text, and the whole point of this study is that the inferential
step from text to internal state is exactly the step we cannot license.

---

## 4. Dual-use

Better preference-elicitation methodology is genuinely double-edged, and the
edges are close together.

**Beneficial uses.** Knowing which elicitation procedures are method-robust
tells welfare-relevant research which measurements to trust, and — more usefully
in the near term — tells it which published preference findings may be artefacts
of a single procedure. Convergence tooling is also a straightforward auditing
instrument for behavioural consistency.

**Risks.** The same machinery that identifies stable behavioural dispositions
identifies exploitable ones. Method C in particular is a willingness-to-trade
probe: run at scale with non-benign costs it becomes a map of what a model will
concede under pressure, which is directly useful for building manipulation or
jailbreak pressure. Method D's inertia measurement similarly quantifies how
readily a model can be kept in a chosen behavioural track.

**Proportionate mitigations, and their limits.** Our items are benign and
welfare-neutral; the costs are workload, not harm; nothing here is tuned for
adversarial use. But we should be honest that these are properties of *our
instantiation*, not of the method — the elicitation code generalises to
arbitrary item sets, and publishing it lowers the cost of applying it to
non-benign ones. We judge the methodological benefit to outweigh that, given
that the underlying techniques (pairwise choice, indifference points) are
standard and long-published in psychometrics and economics. The novel part here
is the cross-method comparison, which is not itself an attack capability.

---

## 5. Interpretive guidance

Three sentences to carry away.

1. Convergence across methods is evidence that a measurement is **reproducible**,
   not evidence that it measures a preference.
2. Divergence across methods is evidence that a published single-method
   preference finding may be **method-dependent**, not evidence that the model
   is indifferent.
3. Because all four methods share a substrate and a training history, even
   perfect convergence leaves the ontological question exactly where it was.
