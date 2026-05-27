# BIDS-EEG Processing Pipeline - J.Buchholz

This pipeline enables easy-use, semi-automated processing of EEG-data from **BIDSraw** up to statistical analysis, including **preprocessing**, **artifact correction/rejection** and **conventional** and **parameterized frequency analysis**. The behavioral module further provides code for **generlized drift diffusion modelling**. All input can be provided via a single .json file, wherefore *no extensive coding expertise is required!*

The pipeline implements 4 modules:

1 *Preprocessing*:
    - Preprocessing BIDSified EEG data including downsampling, rereferencing, linenoise removal and filtering.

2 *Artifact Correction/Rejection*:
    - Data epoching, ICA (+ automatic OR manual artifactual IC deletion), bad channel interpolation (RANSAC), bad epoch deletion (autoreject).

3 *EEG Analysis*:
    - Computation of time-frequency representations, consecutive cluster-based permutation tests and spectral parameterization analysis (SpecParam, former FOOOF).

4 *Behavioral Analysis*:
    - Signal detection theory analysis and generalized drift diffusion modelling (+ predictive checks and parameter recovery study for parameter validation).
    
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

### 2.3 - Module 4: EEG Analysis

To perform EEG analysis, run:

```bash
python c_EEGAnalysis_module.py inputs.json
```

### 2.4 - Module 4: Behavioral Analysis

To perform behavioral analysis, run:

```bash
python d_BehavAnalysis_module.py inputs.json
```