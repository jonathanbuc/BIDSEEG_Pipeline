# 20 — Hierarchical Gaussian Filters for trialwise belief learning

## What changed

This research pass added two companion design documents:

- [HGF theory primer](HGF_RESEARCH_AND_INTEGRATION.md): beginner-first
  intuition, equations, model comparisons, classic/generalized HGF, and
  evidential limits.
- [HGF research and repository plan](HGF_REPOSITORY_INTEGRATION_PLAN.md):
  current papers/software, data audit, scientific models, event timing,
  architecture, outputs, tests, and phased go/no-go gates.

No HGF code was added, no package was installed, and no participant was fitted.
This is deliberate: the audit found two data-contract blockers that should be
repaired before modeling.

## Computer-science concept: state, identity, and causal data flow

An HGF is a **state-space model**. Its state after trial `t` becomes the prior
state for trial `t + 1`. Consequently, rows cannot be treated as an unordered
table and exclusions cannot simply delete past experience.

The current `thisN` field restarts within every 20-trial block. The apparent key
`(participant, block_cond, block_order, thisN)` is therefore many-to-many: the
current 18,795-row EEG IAF table has only 1,720 unique values of that
combination. The required repair is to create a chronological `trial_seq` and
deterministic `trial_uid` at the raw source boundary, before block, RT, or EEG
filtering.

The model also needs two validity masks:

- `state_update_valid`: whether the participant observed the environmental
  outcome;
- `response_likelihood_valid`: whether choice/RT can be used to fit the response
  model.

A participant can see a trial yet give no usable response. That trial should
update subsequent beliefs while contributing no choice/RT likelihood. EEG
artifact rejection is a third downstream mask and must never alter the fitted
behavioral history.

This is the broader CS lesson: **identity and event order belong to the data
contract, not to an accidental DataFrame row number**.

## Psychology and neuroscience concept: adaptive learning under uncertainty

The HGF estimates nested beliefs:

1. the observed outcome;
2. its changing probability or tendency;
3. how volatile that tendency is.

Prediction errors are weighted by relative precision, so effective learning rate
changes across trials. This differs from a fixed Rescorla-Wagner learning rate.
It also differs from the repository's DDM/HSSM: HGF describes learning **across
trials**, while the DDM describes evidence accumulation **within a trial**. They
can be combined, for example by letting pre-outcome HGF prior belief shift DDM
starting point and current coherence drive drift.

For this experiment, a useful binary outcome is:

$$
u_t=\mathbb 1(\text{motion direction}_t=\text{prior direction}_t).
$$

The audited low-level test stream is approximately 74.94% prior-congruent. The
high-level test stream is approximately 50.06%, while demo learning is 75%.
Low-level trials therefore suit a standard cue-validity learner; high-level
trials pose a different question about persistence of an instructed belief when
physical test evidence is uninformative.

Timing determines neural interpretation:

- cue/pre-RDK EEG can be regressed on **predicted** belief and precision;
- RDK-locked EEG can be regressed on current-trial surprise and prediction
  errors;
- a posterior updated with the current RDK cannot explain EEG or choice that
  occurred before that update.

## Measurable improvement from the research pass

- Audited 43 processed behavioral tables containing about 20,620 rows.
- Established that those full-sample derivatives contain zero learning-block
  rows.
- Distinguished the 43-subject behavioral sample from the 42-subject/18,795-row
  EEG IAF derivative.
- Quantified low-level test congruence at about 74.94% and high-level at about
  50.06%.
- Demonstrated that the former merge key collapses 18,795 EEG rows to only 1,720
  unique keys.
- Defined a six-model comparison ladder rather than assuming an HGF will win.
- Defined separate cue-, RDK-, and response-locked HGF regressors to prevent
  temporal leakage.
- Defined five implementation phases with explicit recovery, predictive,
  identity, and convergence gates.

The measurable scientific improvement is currently **risk reduction and testable
design**, not a claimed behavioral or EEG effect. The next implementation
milestone is a unique full-history behavioral table for all subjects, including
recovered learning trials.
