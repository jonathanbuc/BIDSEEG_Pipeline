# BIDS-EEG Processing Pipeline - J.Buchholz

This pipeline enables easy-use, semi-automated processing of EEG-data from **BIDSraw** up to statistical analysis, including **preprocessing**, **artifact correction/rejection** and **conventional** and **parameterized frequency analysis**. The behavioral module further provides code for **generlized drift diffusion modelling**. All input can be provided via a single .json file, wherefore *no extensive coding expertise is required!*

The pipeline implements 5 modules:

1 *Preprocessing*:
    - Preprocessing BIDSified EEG data including downsampling, rereferencing, linenoise removal and filtering.

2 *Artifact Correction/Rejection*:
    - Data epoching, ICA (+ automatic OR manual artifactual IC deletion), bad channel interpolation (RANSAC), bad epoch deletion (autoreject).

3 *EEG Analysis*:
    - Computation of time-frequency representations, consecutive cluster-based permutation tests and spectral parameterization analysis (SpecParam, former FOOOF).

4 *Behavioral Analysis*:
    - Signal detection theory analysis and generalized drift diffusion modelling (+ predictive checks and parameter recovery study for parameter validation).

5 *Hierarchical Bayesian Modelling (HSSM)*:
    - Hierarchical Bayesian drift-diffusion modelling across all subjects at once. Drift rate varies by experimental condition and motion coherence with a per-participant random intercept (partial pooling). Returns full posterior distributions with convergence diagnostics (R-hat, ESS) instead of per-subject point estimates.

0.1 *BIDSification*:
    - If your data is not in BIDS-format yet, using `z_BIDSification_module.py` you can convert data from any output format (e.g. BrainVision) into BIDS (Brain Imaging Data Structure).

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

**Clone the Repository**
After creating and activating the environment, clone the pipeline repository:
```bash
git clone https://github.com/jonathanbuc/BIDSEEG_Pipeline.git
```

-> Now simply open a new terminal `Terminal/New Terminal` and proceed with steps 1.3 and 1.4

### 1.3 Setup conda in Anaconda Prompt (windows) or terminal (Mac)
In your terminal, check if you are in your repository **BIDSEEG_pipeline*
If not, navigate to the eeg pipeline via `cd path/to/BIDSEEG_pipeline` i.e., `cd C:\Users\YourName\Research\EEGresearch\BIDSEEG_pipeline`

### 1.4 Create a conda environment in your terminal (*anaconda prompt* = windows / *terminal* = Mac)

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

### 1.5 Explore inputs.json
--> **IMPORTANT:** Consult the [ExperimentGuide_HierarchicalPriors.md](ExperimentGuide_HierarchicalPriors.md) file to gain a better understanding of the input structure.
- *inputs.json* is the dictionary that provides values for each of the subsequent steps (preprocessing, artifact correction, analysis)
- while exploring the pipeline, you may change the values in this file to see differences across different changes (i.e., change "samplingrate_down": 250 --> 500).
- please note, to use the updated inputs.json file, you must save the file to execute new commands.  Save = *ctl + s*


## 2. Usage - Quick Start

The `data/BIDShierPriors` contains 3 subjects from *Buchholz & Hesselmann (in review) - Hierarchical Priors Shape Dynamic Evidence Accumulation and Aperiodic EEG Activity* as testing data for an initial run (steps 2.1-2.4). 

> **Note on the demo data.** The shipped EEG is a **250 Hz downsampled** version of the 3-subject demo (the rate the preprocessing module downsamples to anyway), kept small enough for Git LFS. It is sufficient to run the whole pipeline end-to-end from a clean clone. The exact provenance — raw BrainVision → `z_BIDSification_module.py` → resample to 250 Hz — is documented in `scripts/make_demo_data.py`. The small per-subject behavioral logs and trait questionnaire live under `data/sourcedata/`; the large raw EEG is not tracked.

**IMPORTANT**: Consider the [ExperimentGuide_HierarchicalPriors.md](ExperimentGuide_HierarchicalPriors.md), which provides a thorough linkage of the experiment and this pipeline. We recommend using this file as guideline when proceeding with steps `2.1-2.3` This file can also be used for a replication of the respective study.

### 2.1 - Module 1: Preprocessing

To preprocess the BIDSified data, run:

```bash
python a_preprocessing_module.py inputs.json
```

### 2.2 - Module 2: Artifact Correction

To perform artifact correction/rejection, run:

```bash
python b_ArtifactCorrection_module.py inputs.json
```

### 2.3 - Module 3: EEG Analysis

To perform EEG analysis, run:

```bash
python c_EEGAnalysis_module.py inputs.json
```

### 2.4 - Module 4: Behavioral Analysis

To perform behavioral analysis, run:

```bash
python d_BehavAnalysis_module.py inputs.json
```

### 2.5 - Module 5: Hierarchical Bayesian Modelling (HSSM)

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