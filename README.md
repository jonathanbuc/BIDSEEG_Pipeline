# BIDS-EEG Processing Pipeline for Computational Psychiatry/Neuroscience - J.Buchholz

BIDSEEG Pipeline is a semi-automated Python workflow for reproducible EEG and behavioral data analysis in computational neuroscience and psychiatry. It supports end-to-end processing of BIDS-formatted EEG data, from **preprocessing** and **artifact correction (including optimized ICA)** to **conventional spectral analysis**, **spectral parameterization**, and trial-wise **event-related potentials (ERP)** and **instantaneous alpha frequency (IAF)** estimation. Integrated generalized and **hierarchical drift-diffusion modelling** (Hierarchical Sequential Sampling Modeling (HSSM)) enables researchers to investigate how experimental manipulations, trial-level neural measures, such as ERP-amplitudes, oscillatory activity, or psychiatric metrics such as symptom seveery, relate to modelled latent decision-making processes. The pipeline is configured through a single JSON file, providing an accessible framework for multimodal EEG–behavior analyses *without requiring extensive programming expertise*.


This repository contains the analysis pipeline for the BIDS EEG dataset hosted on [OpenNeuro](https://openneuro.org/datasets/ds008083/versions/1.0.0):

- OpenNeuro accession: `ds008083`
- Dataset page: https://openneuro.org/datasets/ds008083/versions/1.0.0
- Dataset version used in this pipeline: `v1.0.0`
- Dataset DOI: `10.18112/openneuro.ds008083.v1.0.0`

The raw BIDS dataset is not stored in this GitHub repository. It should be downloaded from [here](https://openneuro.org/datasets/ds008083/versions/1.0.0) OpenNeuro before running the pipeline.


The pipeline implements **6 analysis modules** (plus optional BIDSification):

1 *Preprocessing* (`a_preprocessing_module.py`):
    - Preprocessing BIDSified EEG data including downsampling, rereferencing, linenoise removal and filtering.

1.5 *ICA Preprocessing* (`x_ica_preprocessing_module.py`, optional):
    - Separate preprocessing stream dedicated to ICA decomposition (see [Two-stage ICA](#two-stage-ica) below). Fits ICA on ICA-optimized data, classifies components with ICLabel, and saves weights for module 2.

2 *Artifact Correction/Rejection* (`b_ArtifactCorrection_module.py`):
    - Data epoching, ICA application (pretrained weights from module 1.5 **or** fresh fit on analysis data), bad channel interpolation (RANSAC), bad epoch deletion (autoreject).

3 *EEG Analysis*:
    - Computation of time-frequency representations, consecutive cluster-based permutation tests and spectral parameterization analysis (SpecParam, former FOOOF).

4 *Behavioral Analysis*:
    - Signal detection theory analysis and generalized drift diffusion modelling (+ predictive checks and parameter recovery study for parameter validation).

5 *Hierarchical Bayesian Modelling (HSSM)*:
    - Hierarchical Bayesian drift-diffusion modelling across all subjects at once. Drift rate varies by experimental condition and motion coherence with a per-participant random intercept (partial pooling). Returns full posterior distributions with convergence diagnostics (R-hat, ESS) instead of per-subject point estimates.

0.1 *BIDSification*:
    - If your data is not in BIDS-format yet, using `z_BIDSification_module.py` you can convert data from any output format (e.g. BrainVision) into BIDS (Brain Imaging Data Structure).

### Two-stage ICA

ICA quality depends strongly on how the data is filtered, epoched, and cleaned **before** decomposition. Fitting ICA on the same filtered, analysis-ready epochs used for ERP/TFR can blur or remove the very artifacts you want to separate. For instance, an analysis epoch could contain systematically half a blink, which might remain in the data as artifactual component. 

This pipeline therefore supports a **two-stage ICA workflow** (see [Dimigen, 2020](https://doi.org/10.1016/j.neuroimage.2019.116329) and [Hülsemann et al., 2025](https://doi.org/10.1016/j.neuroimage.2024.120876)):

| Stage | Module | What it does |
|-------|--------|--------------|
| **1 — Train ICA** | `x_ica_preprocessing_module.py` | Reads **raw BIDS** (`bids_root_in`) and runs ICA-specific preprocessing: band-pass (default 1–60 Hz), downsampling, epoching, optional bad-channel masking, amplitude rejection, extended infomax ICA + ICLabel. Saves weights and exclusion metadata to `derivatives/BIDSprocessed/sub-*/ses-*/eeg/*_proc-00ICAweights_*`. Independent of module 1 filter settings. |
| **2 — Analysis preprocessing** | `a_preprocessing_module.py` | Your analysis filters (e.g. 0.1–40 Hz), rereferencing, etc. Writes `01rawfilter` and later steps to `derivatives/BIDSprocessed/`. |
| **3 — Apply ICA** | `b_ArtifactCorrection_module.py` | Loads analysis-preprocessed data (`01rawfilter`), epochs it, and **applies the saved ICA weights** (no refit). Then continues with interpolation and autoreject as usual. |

**Why separate streams?** ICA benefits from a wider passband and longer epochs than typical ERP analysis. Training on dedicated ICA-preprocessed data and applying the weights to analysis-preprocessed epochs gives cleaner component separation without changing your downstream filter settings.

**When to use it:** Enable when automatic ICA on analysis data leaves residual eye/muscle artifacts, or when you want reproducible, inspectable ICA weights per subject. With well-tuned ICA preprocessing you may be able to rely less on aggressive RANSAC/autoreject downstream and reduce general data preprocessing, because: [**"EEG is better left alone" - Delorme (2023)**](https://www.nature.com/articles/s41598-023-27528-0)

**When to skip it:** Set `ICA_preprocessing.perform` to `false` and `ICA_preprocessing.apply_pretrained_weights` to `false` in `inputs.json` — module 2 will fit ICA directly on analysis epochs ('standard', single-stage behaviour).

## 1. Conda Setup
### 1.1 Prerequisites
The pipeline is best executed within a conda environment. Follow the subsequent for successful setup.
Using the appropriate commands for your OS, e.g. Windows, MacOS, install `conda` from [Miniconda Install](https://docs.anaconda.com/miniconda/install/) and follow the installation guide. 

Testing the installation:
    When the installation finishes, open your terminal application (find this by typing "terminal" into your computer's launchpad). Test your installation by running:

```bash
conda list
```

If conda has been installed correctly, a list of installed packages will appear.

### 1.2 Install Interpreter (Visual Studio Code)
We recommend using a interpreter like `Visual Studio Code` or `cursor`. You may install `VS Code` from [here](https://code.visualstudio.com/download) and follow the installation guide. Pipeline input can edited via .json-file and executed via the terminal within VS Code.

**For Windows-Users:** We recommend using  open VS Code from out your `Anaconda Prompt`. Open your Anaconda Prompt and run:
```bash
code
```
that way you have access to your conda environments in the terminal within VS Code

**For Mac-Users:** Just open `VS Code` from your applications.

### 1.3 Clone the Repository and the Data (OpenNeuro)
After creating and activating the environment, clone the pipeline repository:
```bash
git clone https://github.com/jonathanbuc/BIDSEEG_Pipeline.git
```

Next, install the OpenNeuro CLI, handling data down- and upload in OpenNeuro from out your terminal.
```bash
deno install -A --global jsr:@openneuro/cli -n openneuro
```

executing the `download_openneuro.sh` file like the following will download the raw BIDS data to `./data/BIDShierPriors` of your repository clone.
```bash
cd /path/to/BIDSEEG_Pipeline
./scripts/download_openneuro.sh
```


-> Now simply open a new terminal `Terminal/New Terminal` and proceed with steps 1.3 and 1.4

### 1.4 Setup conda in Anaconda Prompt (windows) or terminal (Mac)
In your terminal, check if you are in your repository **BIDSEEG_pipeline*
If not, navigate to the eeg pipeline via `cd path/to/BIDSEEG_pipeline` i.e., `cd C:\Users\YourName\Research\EEGresearch\BIDSEEG_pipeline`

### 1.5 Create a conda environment in your terminal (*anaconda prompt* = windows / *terminal* = Mac)

Install the provided conda environment via the .yml file. Be sure to execute the command within the *EEGpipeline* directory.

```bash
conda env create -f environment_setup.yml
```
--> Note, 'mne-env' is the default name of the conda environment you are creating within the *EEGpipeline* directory. You can define a different name by changing *name: mne-env* within the `environment_setup.yml` file

To **activate** your conda environment, run:

```bash
conda activate 'mne-env'
```

To **deactivate** your conda environment, run:

```bash
conda deactivate
```

### 1.6 Explore inputs.json
--> **IMPORTANT:** Consult the [ExperimentGuide_HierarchicalPriors.md](ExperimentGuide_HierarchicalPriors.md) file to gain a better understanding of the input structure.
- *inputs.json* is the dictionary that provides values for each of the subsequent steps (preprocessing, artifact correction, analysis)
- while exploring the pipeline, you may change the values in this file to see differences across different changes (i.e., change "samplingrate_down": 250 --> 500).
- please note, to use the updated inputs.json file, you must save the file to execute new commands.  Save = *ctl + s*


## 2. Usage - Quick Start

The `data/BIDShierPriors` contains 3 subjects from *Buchholz & Hesselmann (in review) - Hierarchical Priors Shape Dynamic Evidence Accumulation and Aperiodic EEG Activity* as testing data for an initial run (steps 2.1–2.5). 

> **Note on the demo data.** The shipped EEG is a **250 Hz downsampled** version of the 3-subject demo (the rate the preprocessing module downsamples to anyway), kept small enough for Git LFS. It is sufficient to run the whole pipeline end-to-end from a clean clone. The exact provenance — raw BrainVision → `z_BIDSification_module.py` → resample to 250 Hz — is documented in `scripts/make_demo_data.py`. The small per-subject behavioral logs and trait questionnaire live under `data/sourcedata/`; the large raw EEG is not tracked.

**IMPORTANT**: Consider the [ExperimentGuide_HierarchicalPriors.md](ExperimentGuide_HierarchicalPriors.md), which provides a thorough linkage of the experiment and this pipeline. We recommend using this file as guideline when proceeding with steps `2.1–2.4`. This file can also be used for a replication of the respective study.

### 2.1 - Module 1: Preprocessing

To preprocess the BIDSified data, run:

```bash
python a_preprocessing_module.py inputs.json
```

### 2.2 - Module 1.5: ICA Preprocessing (two-stage ICA, recommended)

Run this **before** artifact correction when using pretrained ICA weights (the default in `inputs.json`). It reads directly from raw BIDS — **not** from module 1 output — so it can run before or after `a_preprocessing_module.py`, but both module 1 and module 1.5 must finish before module 2.

```bash
python x_ica_preprocessing_module.py inputs.json
```

This fits ICA per subject on ICA-dedicated preprocessing and writes:

- `*_proc-00ICAweights_ica.fif` — ICA weight matrix
- `*_proc-00ICAweights_metadata.json` — excluded component indices, ICLabel probabilities, bad channels
- diagnostic IC plots under `derivatives/BIDSprocessed/ICA/`

**Enable / configure** in `inputs.json → ICA_preprocessing`:

```json
"ICA_preprocessing": {
    "perform": true,
    "apply_pretrained_weights": true,
    "low_cutoff": 1,
    "high_cutoff": 60,
    "filter_method": "fir",
    "line_method": "band_stop",
    "downsample_to": 250,
    "rereference": "average",
    "drop_ref_channel": null,
    "reject_amplitude_uv": 1000,
    "ICA_method": "infomax",
    "n_components": 20,
    "max_iter": 200,
    "random_state_seed": 1943,
    "ch_interpolation_method": "none",
    "epoch_dict": {
        "S_1": 1,
        "S_2": 2,
        "S_4": 4
    },
    "tmin": -1.0,
    "tmax": 2.5,
    "rej_thresholds": {
        "muscle artifact": 0.6,
        "eye blink": 0.25,
        "heart beat": 0.6,
        "line noise": 0.75,
        "channel noise": 0.75
    },
}
```

| Key | Meaning |
|-----|---------|
| `perform` | Run `x_ica_preprocessing_module.py` (set `false` to skip training) |
| `apply_pretrained_weights` | Module 2 loads saved weights instead of fitting ICA anew |
| `low_cutoff` / `high_cutoff` | Band-pass for ICA training only (not analysis filters) |
| `epoch_dict` | dictionairy storing event `trial_type` (key) and `values` (value) |
| `tmin` / `tmax` | max and min time (sec) for epochs relative to trigger in `epoch_dict`|
| `ch_interpolation_method` | Bad channels before ICA: `"none"`, `"ransac"`, or `"neighbor_corr"` |
| `rej_thresholds` | Per-label ICLabel probability thresholds for IC exclusion |

> **Order matters:** Run `a_preprocessing` and `x_ica_preprocessing` before `b_ArtifactCorrection`. Module 2 requires `00ICAweights` files when `apply_pretrained_weights` is `true`, and loads analysis-preprocessed data from module 1 (`01rawfilter`).

### 2.3 - Module 2: Artifact Correction

To perform artifact correction/rejection, run:

```bash
python b_ArtifactCorrection_module.py inputs.json
```

With `apply_pretrained_weights: true`, ICA is **applied** (not refit) using weights from step 2.2. Set `perform.ICA` to `true` in `inputs.json` so ICA correction runs at all.

### 2.4 - Module 3: EEG Analysis

To perform EEG analysis, run:

```bash
python c_EEGAnalysis_module.py inputs.json
```

### 2.5 - Module 4: Behavioral Analysis

To perform behavioral analysis, run:

```bash
python d_BehavAnalysis_module.py inputs.json
```

### 2.6 - Module 5: Hierarchical Bayesian Modelling (HSSM)

Where module 4 fits a separate drift-diffusion model **per subject**, this module fits **one
hierarchical Bayesian DDM across all subjects at once**. Drift rate is modelled as a function of
experimental condition and motion coherence, with each participant getting their own random
intercept (*partial pooling*), and the output is a full posterior distribution per parameter — so
you can make direct probability statements ("95% credible that high-level priors raise drift rate")
rather than relying on a t-test over point estimates.

**Prerequisites:**
- **Run module 4 first.** HSSM reads the group behavioral table
  `.../results/groupBehavioral/behavioraldata_hierprior.csv` that `d_BehavAnalysis_module.py`
  writes. Without it the module stops with a `FileNotFoundError`.
- The `hssm` package must be installed (it ships with the `environment_setup.yml` environment; if
  you built your env before HSSM was added, run `pip install hssm`).

**Enable it.** HSSM is off by default because Markov-Chain-Monte-Carlo sampling is slow (expect
~15–60 min for the full dataset). Switch it on in `inputs.json`:

```json
"perform": {
    "compute_hssm": true
}
```

**Run it:**

```bash
python e_HSSM_module.py inputs.json
```

**Configure it.** All knobs live in `inputs.json → Analysis.hssm`:

```json
"hssm": {
    "model_type":     "ddm",                                  // DDM variant to fit
    "sampler":        "nuts_numpyro",                          // "mcmc" (PyMC) or "nuts_numpyro" (faster, JAX)
    "draws":          1000,                                    // posterior samples to keep per chain
    "tune":           1000,                                    // warm-up steps (discarded)
    "chains":         2,                                       // independent chains (for R-hat convergence check)
    "cores":          2,                                       // CPU cores for parallel chains
    "target_accept":  0.9,                                     // NUTS acceptance target; raise toward 0.99 if you see divergences
    "prior_settings": "safe",                                  // HSSM's regularising default priors
    "link_settings":  "log_logit",
    "formula_v":      "v ~ 1 + exp * coh_level + (1|participant)"  // drift-rate regression formula
}
```

The `formula_v` is the scientific heart of the model. `exp` is the condition column
(`base`/`lowlevel`/`highlevel`), `coh_level` is motion coherence, `exp * coh_level` includes their
interaction, and `(1|participant)` adds the per-subject random intercept. To test a direct
EEG–behavior link you would point drift rate at a trial-level EEG predictor instead, e.g.
`"v ~ alpha_power + (1|participant)"` — just make sure that column is present in the behavioral
table.

**Outputs** (written to `.../results/groupBehavioral/`):
- `hssm_posterior_summary.csv` — mean, SD, 95% HDI, R-hat and ESS for every parameter.
- `hssm_trace.png` — diagnostic trace plots; converged chains look like overlapping "hairy
  caterpillars" and R-hat should be ≈ 1.0.

> For the statistics behind this module (Bayesian vs. MLE, NUTS, partial pooling, and what each DDM
> parameter means), see `docs/learning/03-hssm-integration.md` and `04-hssm-explained.md`.
