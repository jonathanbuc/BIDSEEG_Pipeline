# 02 — Six correctness bugs (Workstream B)

*Teaching note for the round of correctness fixes after the PI cleared the "must reproduce old
results" constraint. Three of these silently changed the science; three were latent crashes.*

These split into two groups:
- **Silent science bugs** (1–3): the pipeline ran fine and produced numbers, but the numbers were
  not what the config said they should be. These are the dangerous ones — no error, just wrong.
- **Latent crashes** (4–6): code that works on the happy path but blows up (or hides a failure) on
  a branch the demo config doesn't currently hit.

---

## Bug 1 — Average re-reference was computed, then thrown away

**File:** `b_ArtifactCorrection_module.py`

```python
# before
if perform_rereferencing:
    raw_step, log_df = rereferencing(epochs, rereference, log_df)   # result → raw_step
    ...
# ...later steps all operate on `epochs`, never on raw_step

# after
if perform_rereferencing:
    epochs, log_df = rereferencing(epochs, rereference, log_df)     # result flows forward
```

The re-referenced data was stored in a variable (`raw_step`) that nothing downstream ever used —
channel interpolation, autoreject, and the saved file all kept using the *old* `epochs`. So
`perform.rereferencing = true` did nothing.

- **CS concept — dead stores / variable aliasing.** A "dead store" is a value you compute and
  assign but never read. The function did real work, but its output was bound to a name that fell
  out of use one line later. Languages with linters flag this; plain Python won't. The fix is to
  thread the result into the variable the rest of the pipeline actually reads.
- **Neuro concept — what re-referencing *is*.** Every EEG number is a *voltage difference* between
  an electrode and some reference point. The choice of reference biases every channel. **Average
  reference** subtracts the mean across all electrodes from each one, removing the arbitrary
  reference's fingerprint — it's a standard prerequisite for topographies and spectral/alpha
  analysis. Skipping it (which is what the bug did) leaves a reference bias baked into every result
  module `c` produces.
- **Why it helped:** the alpha-power and TFR results now reflect properly average-referenced data,
  matching what the config declared all along.

---

## Bug 2 — FOOOF ran no matter what the flag said

**File:** `c_EEGAnalysis_module.py`

```python
# before — the flag only guards the log line, not the work
if compute_fooof:
    utils.log_msg('Computing FOOOF')
fm, df_fooofsum, df_fooofalpha = run_fooof_analysis(...)   # runs unconditionally

# after — the work is inside the branch, with downstream guards
if compute_fooof:
    utils.log_msg('Computing FOOOF')
    fm, df_fooofsum, df_fooofalpha = run_fooof_analysis(...)
    ...
else:
    utils.log_msg(' -- FOOOF not performed')
```

In Python, **indentation is the control flow.** The expensive `run_fooof_analysis(...)` call sat at
the same indent level as the `if`, so it was a sibling of the `if`, not a child — it always ran.
I also had to guard the *downstream* code (`pd.concat(fooof_alpha_dfs)`, the merge, and the
plotting), because turning FOOOF off leaves those lists empty and those columns missing.

- **CS concept — significant whitespace & dead configuration.** A config flag that doesn't actually
  gate anything is "dead config": it lies to the user about what the run did. Making the flag real
  meant also handling the *absence* of FOOOF outputs everywhere they're consumed (empty-list guards,
  and filtering the plot to columns that exist).
- **Neuro concept — FOOOF / specparam.** An EEG power spectrum is a mix of an **aperiodic 1/f
  background** (power falling off with frequency) and **periodic bumps** (true oscillations like the
  alpha peak). FOOOF fits and separates the two, so you can ask "is there a real alpha *peak*" rather
  than being fooled by the sloping background. It's a distinct, optional analysis from the
  wavelet/TFR alpha power — which is exactly why it deserves its own working on/off switch.
- **Why it helped:** `compute_fooof` is now a real switch, and the module no longer crashes if you
  flip it off.

---

## Bug 3 — `tfr_method` in the config was ignored (morlet hardcoded)

**File:** `c_EEGAnalysis_module.py` · **Config:** `inputs.json` (`Analysis.tfr_method`,
new `Analysis.time_bandwidth`)

```python
# before — config said "multitaper", code always did morlet
tfr_cond_avg = epochs[q].compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles, ...)
utils.log_update(log_df, 'tfr_method', 'morlet')   # also hardcoded — the log lied too

# after — dispatch on the configured method
tfr_kwargs = dict(freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, decim=3, ...)
match tfr_method:
    case "multitaper":
        ... compute_tfr(method="multitaper", time_bandwidth=time_bandwidth, **tfr_kwargs)
    case _:  # morlet
        ... compute_tfr(method="morlet", **tfr_kwargs)
utils.log_update(log_df, 'tfr_method', tfr_method)
```

> ⚠️ **This one changes the numbers.** With the demo config (`tfr_method: "multitaper"`), the TFR is
> now computed with multitaper instead of morlet. Worth flagging to the PI even though old results
> don't need to reproduce.

- **CS concept — single source of truth & dispatch tables.** A value should be declared in exactly
  one place and read from there. Here the truth lived in `inputs.json` but the code shipped a
  *second*, contradictory copy as a hardcoded string. I replaced the constant with a `match`
  dispatch (the idiom this codebase already uses for filter/reref/save type selection), so the knob
  in the config is the only place the method is decided.
- **Neuro concept — Morlet vs. multitaper time-frequency analysis.** Both turn a signal into a
  power-over-time-and-frequency picture, but they trade off differently against the
  **time-frequency uncertainty principle** (you can't have perfect time *and* frequency resolution
  at once). A **Morlet wavelet** is a single windowed sinusoid per frequency — clean, intuitive.
  **Multitaper** averages several orthogonal tapers (DPSS/Slepian sequences); `time_bandwidth`
  controls how much spectral smoothing they apply. Multitaper generally gives better
  signal-to-noise for *sustained* oscillations (like task alpha) at the cost of some frequency
  sharpness. The config asking for multitaper suggests that was the intended trade.
- **Why it helped:** the config is now truthful — change `tfr_method` and the analysis actually
  changes, and the log records what really ran.

---

## Bug 4 — the saver failed *silently*

**File:** `utils_module.py` (`save_preprocessing_step`)

```python
# before
case _ :
    print(f"{filename} NOT SAVED ... is type {type(file)}")   # prints, then continues as if fine

# after
case _ :
    raise TypeError(f"save_preprocessing_step: no serialization case for '{filename}' ...")
```

The saver is a big `match str(type(file))` that picks how to serialize each object type. The
catch-all `case _` just *printed* a message and let the program continue — so an unhandled object
type meant a file that was supposed to exist silently didn't, and a later stage would fail far away
from the real cause (or worse, quietly use stale data).

- **CS concept — fail-fast vs. silent failure.** A `print` is not error handling; the program keeps
  going in a corrupted state. Raising an exception is **failing fast**: stop at the exact line where
  the assumption broke, with a message naming the type that wasn't handled. The cost of a loud
  failure now is tiny compared to debugging a missing-file mystery three stages downstream.
- **Neuro/pipeline concept — provenance integrity.** This pipeline is a *chain* of saved artifacts
  (`proc-01rawfilter → 02ICA → 03chInterp → …`). Each stage loads the previous stage's file. A
  silently-skipped save breaks that chain invisibly, so a "finished" run could be missing a step.
- **Why it helped:** an unhandled save now stops the run with a clear, type-named error instead of
  corrupting the artifact chain. (The end-to-end smoke test confirms the normal path never trips
  this — if it ever does, that's a real new type that genuinely needs a `case`.)

---

## Bug 5 — crash if downsampling is turned off

**File:** `a_preprocessing_module.py`

```python
# before
if perform_downsampling:
    raw_step, log_df = down_sample(raw, samplingrate_down, log_df)
    ...
else:
    utils.log_msg(f"... remains at {raw_step.info['sfreq']}")   # raw_step never assigned here!

# after
else:
    raw_step = raw
    utils.log_msg(f"... remains at {raw_step.info['sfreq']}")
```

`raw_step` is only created inside the `if` branch. If you set `perform.downsampling = false`, the
`else` branch referenced `raw_step` before it existed → `NameError`, and every step after it also
expects `raw_step`. The demo config has downsampling *on*, so this never fired — a latent landmine.

- **CS concept — variable initialization across control-flow paths (branch coverage).** Every
  branch that *reads* a variable must guarantee it was *written* on every path that reaches it. The
  happy path initialized `raw_step`; the alternate path didn't. The fix makes the `else` carry the
  unmodified `raw` forward under the same name, so all later steps have something to work on.
- **Neuro concept — downsampling.** See note 01: dropping the sample rate (e.g. 1000→250 Hz) is
  optional. The pipeline must still run when it's skipped, just at the original rate.
- **Why it helped:** the "don't downsample" configuration no longer crashes the whole module.

---

## Bug 6 — epoching returned the wrong shape and crashed before its own guard

**File:** `b_ArtifactCorrection_module.py`

```python
# before — function returns a bare None on the no-events path
def epoching(...):
    ...
    return epochs, log_df      # success: a 2-tuple
    return None                # failure: a single value

# caller
epochs, log_df = epoching(...)          # unpacking None → TypeError, crashes HERE
diagnostic_plots(epochs, ...)           # uses epochs before checking it
if epochs is None or not bool(epochs):  # guard can never be reached
    continue
```

Two problems stacked: (a) the function returned **inconsistent shapes** — a 2-tuple on success, a
bare `None` on failure — so the caller's `epochs, log_df = ...` unpack crashed before the existing
"is it None?" guard could run; and (b) even if it hadn't, `diagnostic_plots(epochs, ...)` *used*
`epochs` one line *before* the guard checked it. Fixed both:

```python
# function
return None, log_df            # always a 2-tuple

# caller — check before use
epochs, log_df = epoching(...)
if epochs is None or not bool(epochs):
    utils.log_msg('... events not found ... skipping subject')
    continue
diagnostic_plots(epochs, ...)  # only runs once epochs is known good
```

- **CS concept — return-type contracts & guard-before-use.** A function should return the **same
  shape on every path** so callers can rely on a single unpacking pattern; mixing `(a, b)` and
  `None` is a contract violation waiting to throw. And a safety check is only useful if it runs
  *before* the thing it guards — order matters.
- **Neuro concept — epoching.** Continuous EEG is one long recording; **epoching** cuts it into
  short windows time-locked to event triggers (e.g. −0.6 s to +1.5 s around a stimulus marker). If
  a subject's recording is missing those markers, there are no epochs to make — a real, expected
  case that should *skip the subject*, not crash the batch.
- **Why it helped:** a subject with missing event markers is now skipped cleanly, and the whole
  batch keeps going instead of dying on one bad file.

---

## Net effect

| # | Bug | Type | Changes numbers? |
|---|-----|------|------------------|
| 1 | Re-reference discarded | silent science | **yes** (now actually re-referenced) |
| 2 | FOOOF ignored its flag | silent science | only if you turn the flag off |
| 3 | `tfr_method` ignored (morlet hardcoded) | silent science | **yes** (now multitaper per config) |
| 4 | Saver failed silently | latent / integrity | no (makes hidden failures loud) |
| 5 | Crash if downsampling off | latent crash | no (default path unchanged) |
| 6 | Epoching None-unpack crash | latent crash | no (default path unchanged) |

The default demo run produces the same *kind* of outputs as before, but bugs **1 and 3** genuinely
change the EEG numbers — these are the two to mention to the PI.
