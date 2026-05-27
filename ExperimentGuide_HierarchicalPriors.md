# Experiment Guide — Hierarchical Priors RDK EEG Task

This file provides a experimental guide linking the analysis of *Buchholz & Hesselmann (in review) - Hierarchical Priors Shape Dynamic Evidence Accumulation and Aperiodic EEG Activity* with the here provided data and code. This can be used as an assisting tool for first running the semi-automated EEGpipeline.

This EEG-BIDS dataset contains cue-locked EEG recordings and event annotations from a random-dot kinematogram (RDK) task designed to dissociate low-level and high-level priors during perceptual decision-making. The accompanying analysis tests whether hierarchical priors bias behavior through signal detection theory (SDT) and generalized drift-diffusion modeling (gDDM), and whether they modulate occipital oscillatory or aperiodic EEG activity.

The trial-level variables needed for analysis are stored in `epochs.metadata` and are illustrated by the accompanying `metadata.csv` file. Each row corresponds to one cue-locked epoch/trial retained in the MNE `Epochs` object.

## Task summary

Each trial followed this sequence:

1. auditory cue/tone onset;
2. fixation-only interstimulus interval (ISI; approximately 1000 ms);
3. RDK onset with leftward or rightward net motion at individualized coherence;
4. left/right response, followed by a jittered intertrial interval (ITI; approximately 1000–1500 ms).

Motion coherence was individualized with two QUEST staircases targeting 75% discrimination accuracy. The resulting threshold defined the medium coherence level. Low coherence was 50% of this threshold; high coherence was 200% of this threshold.

The experiment contained three within-subject conditions:

- **Baseline / no prior** (`exp = base`): a neutral, nonpredictive 750 Hz tone preceded the RDK.
- **Low-level prior** (`exp = lowlevel`): two tones (600/900 Hz) predicted leftward or rightward motion with 75% contingency. Participants were not informed about the tone-motion association.
- **High-level prior** (`exp = highlevel`): the tone was neutral/nonpredictive, but participants wore transparent tinted glasses and were led to believe that the glasses enhanced perception of either leftward or rightward motion. This belief was reinforced during learning blocks before the analyzed test blocks.

The main analyses use baseline trials and prior-condition test trials. Learning blocks served to establish the tone-motion associations or glass beliefs and should not be included unless explicitly intended.

## Raw BIDS events

Each `*_events.tsv` file contains raw trigger-level events, and the `*_events.json` sidecar defines the BIDS event columns. Events are stored under `sub-0XX/ses-01/eeg/`.

**BIDS naming note:** the provided example sidecar is named `task-RDK_events.json`, whereas the provided example events file is named `task-HierPrior_events.tsv`. For strict BIDS compatibility, task labels should be identical across the EEG data file, events file, and JSON sidecar.

### Event columns

| Column | Meaning |
|---|---|
| `onset` | Event onset in seconds from the first stored data point. |
| `duration` | Event duration in seconds. Impulse events are coded as `0.001`. |
| `sample` | Event onset in sampling points; first sample is 0. |
| `value` | Numeric trigger/event code. |
| `trial_type` | Human-readable trigger label, for example `S_1`, `S_8`, or `R_1`. |

### Trigger codes

| `value` | `trial_type` | Meaning |
|---:|---|---|
| 1 | `S_1` | Neutral 750 Hz tone cue onset; used in baseline and high-level prior trials. |
| 2 | `S_2` | Low-level prior tone cue predicting rightward motion. |
| 4 | `S_4` | Low-level prior tone cue predicting leftward motion. |
| 8 | `S_8` | RDK onset with leftward net motion. |
| 16 | `S_16` | RDK onset with rightward net motion. |
| 101 | `R_1` | Correct response. |
| 102 | `R_2` | Incorrect response. |

A raw trial can be reconstructed as:

`cue event (S_1/S_2/S_4) → RDK event (S_8/S_16) → response event (R_1/R_2, if present)`

For manuscript-level reproduction, use `epochs.metadata` rather than only the raw triggers, because the metadata contains condition labels, prior direction, motion coherence, counterbalancing, response direction, and behavioral exclusion flags.

## `epochs.metadata` column dictionary

The table below explains the columns in `metadata.csv`, which are attached to each cue-locked MNE `Epochs` object as `epochs.metadata`. Values shown are examples from the supplied file; levels may vary across participants.

| Column | Example values / type | Meaning | Use for reproduction |
|---|---|---|---|
| `trigger` | `1` | Technical event identifier used during cue-locked epoch creation. In the provided metadata this is constant because epochs are anchored to cue onset. | Bookkeeping only. Use `exp`, `cueHz`, and `prior` for analysis coding. |
| `participant` | `sub-001` | BIDS participant label. | Grouping variable for participant-level SDT, gDDM, EEG, and spectral summaries; merge key for questionnaire data. |
| `exp` | `base`, `lowlevel`, `highlevel` | Experimental condition: baseline/no prior, implicit low-level prior, or explicit high-level prior. | Primary within-subject factor `prior condition` in behavioral and EEG analyses. |
| `block_cond` | `base`, `test` | Analysis phase/block type. Baseline blocks are coded `base`; analyzed prior-condition test blocks are coded `test`. | Keep `base` and `test` rows for the main analyses; exclude learning blocks if present in other metadata files. |
| `block_order` | e.g., `bhl` | Counterbalanced order of the three conditions. Letters denote baseline (`b`), high-level (`h`), and low-level (`l`). | Optional check for order effects; not a primary manuscript variable. |
| `thisN` | `0`–`19` | Trial index within the current 20-trial block as generated by the task script. | Trial-order diagnostics and within-block checks. |
| `cueAss` | e.g., `highleft` | Low-level tone-motion counterbalancing. `highleft` means the high tone predicted leftward motion and the low tone predicted rightward motion. | Reconstruct or validate `prior` in low-level prior trials from `cueHz`. |
| `cueHz` | `600`, `750`, `900` | Auditory cue frequency in Hz. `750` is neutral/nonpredictive; `600` and `900` are predictive low-level prior cues. | Validate cue condition; reconstruct low-level prior direction together with `cueAss`. |
| `motion_direction` | `left`, `right` | Physical net direction of RDK motion on the trial. | Stimulus direction for accuracy, SDT signal coding, and prior congruency. |
| `prior` | `noprior`, `left`, `right` | Direction predicted by the current prior. `noprior` is used in baseline trials; `left`/`right` indicate the tone-predicted or glasses-predicted direction. | Defines prior-congruent versus prior-incongruent stimuli and responses. |
| `response` | `left`, `right`, missing | Participant's reported perceived motion direction. | Behavioral choice variable for SDT and gDDM. Exclude missing responses for behavioral modeling. |
| `corr` | `0`, `1` | Accuracy relative to physical RDK direction: `1` = response matches `motion_direction`; `0` = incorrect. | Performance checks and inclusion diagnostics. |
| `rt` | seconds | Reaction time from RDK onset to button response. | gDDM response-time variable and RT outlier screening. |
| `thresh75` | participant-specific proportion | Individual QUEST threshold targeting 75% correct discrimination. This is the participant's medium coherence. | Defines individualized sensory uncertainty; `medium = thresh75`, `low = 0.5 × thresh75`, `high = 2 × thresh75`. |
| `BiasFirst` | e.g., `rightfirst`, `leftfirst` | Counterbalancing/order variable for the high-level glasses manipulation, indicating which glasses/prior direction was introduced or tested first. | Optional check for high-level prior order effects. |
| `coh` | proportion, e.g., `0.0562`, `0.1124` | Trial-wise RDK motion coherence. | Continuous sensory-evidence strength; drift rate was modeled as a function of coherence in the gDDM. |
| `coh_level` | `low`, `medium`, `high` | Categorical coherence level derived from `thresh75`. In analyzed test trials, low and medium coherence are expected. | Descriptive performance checks and model validation by coherence level. |
| `filename` | e.g., `sub-001_RDKdeutsch_highleft` | Source behavioral/task-log identifier. Usually contains participant and counterbalancing information. | Provenance and debugging. |
| `rt_flag` | `False`, `True` | Trial-level behavioral exclusion flag. `True` marks trials excluded because of missing responses or RT outlier criteria. | For behavioral SDT/gDDM reproduction, use `rt_flag == False` with nonmissing `rt` and `response`. |
| `lowCoh_performance` | proportion correct | Participant-level accuracy at low coherence, repeated across rows for that participant. | Inclusion/performance diagnostics; used to verify monotonic performance across coherence levels. |
| `medCoh_performance` | proportion correct | Participant-level accuracy at medium coherence, repeated across rows for that participant. | Inclusion/performance diagnostics. |
| `highCoh_performance` | proportion correct | Participant-level accuracy at high coherence, repeated across rows for that participant. | Inclusion/performance diagnostics; should exceed medium and low coherence performance. |
| `prior_dic` | `noprior`, `prior` | Binary prior grouping: baseline/no-prior versus any prior condition. | Convenience grouping only. Use `exp` to distinguish low-level from high-level priors. |
| `response_prior` | `0`, `1`, missing | Response recoded relative to the prior-defined decision bound. In prior conditions, `1` = response in the prior-congruent direction and `0` = response in the prior-incongruent direction. In baseline trials, `1` = left response and `0` = right response. | Primary binary response variable for SDT and gDDM boundary coding. |

## Reproducing the manuscript analyses from metadata

### 1. Recommended trial filters

For behavioral SDT and gDDM analyses, start from the metadata table and retain analyzed trials:

```python
meta_beh = metadata.query("exp in ['base', 'lowlevel', 'highlevel']").copy()
meta_beh = meta_beh.query("block_cond in ['base', 'test']").copy()
meta_beh = meta_beh[meta_beh['rt_flag'] == False]
meta_beh = meta_beh.dropna(subset=['response', 'rt', 'response_prior'])
```

When using preprocessed MNE `Epochs`, the rows in `epochs.metadata` remain aligned to the retained EEG epochs. If the exact manuscript behavioral analyses were run before EEG epoch rejection, a complete behavioral log may be needed to reproduce those behavioral trial counts exactly.

### 2. Manuscript variables and metadata columns

| Manuscript variable / analysis concept | Metadata column(s) | Coding / transformation |
|---|---|---|
| Participant | `participant` | Group all first-level estimates by participant. |
| Prior condition | `exp` | `base` = baseline/no prior; `lowlevel` = low-level prior; `highlevel` = high-level prior. |
| Baseline/no-prior trials | `exp == 'base'`, `prior == 'noprior'`, `cueHz == 750` | Neutral tone, no directional prediction. |
| Low-level prior trials | `exp == 'lowlevel'`, `cueHz in [600, 900]` | Tone direction mapping is encoded in `cueAss`; final predicted direction is already stored in `prior`. |
| High-level prior trials | `exp == 'highlevel'`, `cueHz == 750`, `prior in ['left','right']` | Directional prediction comes from the alleged motion-enhancing glasses, not from the tone. |
| Motion direction | `motion_direction` | Physical RDK direction: left or right. |
| Prior direction | `prior` | Direction predicted by tone or glasses; `noprior` for baseline. |
| Response direction | `response` | Participant's left/right report. |
| Accuracy | `corr` | `1` if `response == motion_direction`, else `0`. |
| RT | `rt` | Seconds from RDK onset to response. |
| RT/response exclusion | `rt_flag`, `response`, `rt` | Exclude `rt_flag == True` and missing values. |
| Sensory precision / motion coherence | `coh`, `coh_level`, `thresh75` | Use continuous `coh` for gDDM drift modulation; use `coh_level` for descriptive checks. |
| Prior-congruent stimulus | `motion_direction`, `prior`, `exp` | Prior conditions: `motion_direction == prior`; baseline: `motion_direction == 'left'` for SDT coding. |
| Prior-congruent response | `response_prior` | Prior conditions: `1` = response matches `prior`; baseline: `1` = left response. |
| SDT signal-present trial | `motion_direction`, `prior`, `exp` | Prior conditions: prior-congruent motion; baseline: leftward motion. |
| SDT signal response | `response_prior` | `response_prior == 1`. |
| DDM response/boundary | `response_prior` | Prior conditions: prior-congruent vs. prior-incongruent response; baseline: left vs. right response. |
| EEG condition contrast | `exp` | Compare `lowlevel` vs `base` and `highlevel` vs `base`. |
| Spectral parameterization condition | `exp` | Estimate PSD/specparam measures separately for `base`, `lowlevel`, and `highlevel`. |
| Psychosis proneness | external questionnaire data | Not in metadata. Merge PDI/CAPS composite scores by `participant`. |
| High-level belief strength | external questionnaire data | Not in metadata. Merge VAS belief ratings by `participant`. |

### 3. Signal detection theory coding

For SDT, the manuscript defined the signal differently for baseline and prior conditions. In prior conditions, signal-present trials are trials in which physical motion is congruent with the prior. In baseline trials, leftward motion is treated as the signal.

```python
import numpy as np

is_prior = meta_beh['exp'].isin(['lowlevel', 'highlevel'])

meta_beh['sdt_signal'] = np.where(
    is_prior,
    meta_beh['motion_direction'] == meta_beh['prior'],
    meta_beh['motion_direction'] == 'left'
)

meta_beh['sdt_signal_response'] = meta_beh['response_prior'] == 1
```

For each participant and condition, compute hit and false-alarm rates from `sdt_signal` and `sdt_signal_response`, then calculate criterion:

```python
c = -0.5 * (z_hit_rate + z_false_alarm_rate)
```

Negative criterion values in the prior conditions indicate a bias toward the prior-congruent direction.

### 4. Generalized drift-diffusion modeling coding

For each participant and condition, fit a gDDM using:

| gDDM input | Metadata source |
|---|---|
| Response time | `rt` |
| Binary response / bound | `response_prior` |
| Condition | `exp` |
| Motion coherence | `coh` or `coh_level` |
| Trial exclusion | `rt_flag == False` |

In the prior conditions, the decision bounds are prior-congruent (`response_prior = 1`) versus prior-incongruent (`response_prior = 0`). In the baseline condition, the same column codes left (`1`) versus right (`0`) responses. The manuscript estimated drift rate (`d`), boundary separation (`a`), and starting point (`z`), with noise fixed to 1, nondecision time fixed to 200 ms, maximum RT set to 3000 ms, and drift modeled as a function of motion coherence.

### 5. EEG time-frequency and spectral analyses

EEG analyses are cue-locked. Use the MNE epoch time axis rather than metadata onsets for time-resolved analyses.

| EEG analysis step | Metadata / data source |
|---|---|
| Epoch anchor | cue onset; rows in `epochs.metadata` correspond to cue-locked epochs |
| Epoch window | EEG time axis, -600 to 1500 ms relative to cue onset |
| Baseline interval | EEG time axis, -600 to -100 ms |
| Condition factor | `exp` |
| Occipital channel cluster | O1, Oz, O2 from EEG data |
| TFR contrasts | `lowlevel` vs `base`; `highlevel` vs `base` |
| TFR frequency range | 5–40 Hz |
| Spectral-parameterization window | 0–1500 ms relative to cue onset |
| Spectral-parameterization grouping | average/fit spectra separately by `participant` and `exp` |

Spectral parameterization outputs are not stored in `epochs.metadata`. They should be computed from the EEG signal and then summarized by participant and condition as aperiodic offset, aperiodic exponent, and periodic alpha power.

### 6. Psychosis-proneness correlations

Psychosis-proneness measures are not stored in `epochs.metadata`. To reproduce the associative analyses:

1. compute participant-level behavioral prior metrics, preferably drift rates from the gDDM, separately for `lowlevel` and `highlevel`;
2. merge these metrics with questionnaire-derived psychosis proneness scores by `participant`;
3. compute product-moment correlations between psychosis proneness and low-level/high-level drift-rate metrics.

## Variables not contained in `epochs.metadata`

The metadata table does not contain PDI scores, CAPS scores, the PDI/CAPS psychosis-proneness composite, VAS belief strength ratings, raw ICA labels, bad-channel interpolation logs, rejected-epoch annotations, SDT estimates, gDDM estimates, TFR outputs, or spectral-parameterization outputs. These variables must be obtained from questionnaire files, preprocessing logs, or analysis derivatives.
