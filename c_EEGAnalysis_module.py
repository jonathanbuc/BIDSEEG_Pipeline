# _____________________________Module_name_______________________________________
# run with
# python c_EEGAnalysis_module.py inputs.json
#
#
# * Allgemeine und Biologische Psychologie - AG Hesselman
# * Psychologische Hochschule
#
# ## Author(s)
# * Buchholz, Jonathan; Psychologische Hochschule Berlin, AG Hesselmann
# * Hofstetter, Lily; Psychologische Hochschule Berlin, AG Hesselmann
# * Rowan, Dowd; Psychologische Hochschule Berlin, AG Hesselmann
#
# * last update: 2025.23.06
#
# * this script provides an array of functions for EEG data analysis, including time-frequency analyses 
# * using cluster-based permutation tests (CBPT), ERP analyses using ANOVAs for statistical comparisons between conditions or analyses
# * of periodic and aperiodic components of the EEG power spectra
# * ...to be continued
# _______________________________________________________________________________



# _____________________________Imports___________________________________________
# basic packages
from configparser import NoSectionError #QUESTION: is this used again? 
import sys
import io
import utils_module as utils
import mne
import scipy.stats
import numpy as np
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
#from utils_module import ttest_rel_wrapper



# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fooof import FOOOF
from fooof.plts.annotate import plot_annotated_model
from plotting_module import (
    plot_and_save_cbpt_results,
    plot_full_fooof_model_detailed,
    raincloud_plot,
    paired_plot,
    tfr_plots_subjects,
    grand_avg_tfr_plots
)


# statistical analysis
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
import csv
import pandas as pd
from statsmodels.formula.api import mixedlm
import pingouin as pg
import math
from joblib import Parallel, delayed #parallel processing 
import json
from pathlib import Path

# _______________________________________________________________________________


# _____________________________Functions_________________________________________

def plot_epochs_timeseries(epochs, electrodes, subject, save_dir,
                           tmin=None, tmax=None, n_epochs=15, scale_factor=2.5,
                           fmt="svg", condition_dict=None):
    """
    Plots continuous-style EEG timeseries for selected electrodes and saves as vector graphic.
    Epochs are concatenated along the time axis (like the MNE browser), with vertical offset per channel.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs object for a single subject.
    electrodes : list of str
        Electrode names to plot (e.g. ['O1', 'Oz', 'O2']).
    subject : str
        Subject identifier (used in filename).
    save_dir : str
        Directory to save the output figure.
    tmin : float or None
        Crop start time in seconds. None keeps the original epoch start.
    tmax : float or None
        Crop end time in seconds. None keeps the original epoch end.
    n_epochs : int
        Number of consecutive epochs to display.
    scale_factor : float
        Controls vertical spacing between channels (higher = more spacing).
    fmt : str
        File format for the saved figure ('svg', 'png', 'pdf').
    condition_dict : dict or None
        If provided, plots a separate figure per condition.
        Format: {'column_name': ['cond1', 'cond2', ...]}.
        If None, plots all epochs regardless of condition.

    Returns
    -------
    list of str
        Paths of saved figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    if condition_dict is not None:
        cond_col, conditions = list(condition_dict.items())[0]
        cond_epochs = {cond: epochs[f"{cond_col} == '{cond}'"] for cond in conditions}
    else:
        cond_epochs = {"all": epochs}

    for cond, ep_cond in cond_epochs.items():
        ep = ep_cond.copy().pick(electrodes)
        if tmin is not None or tmax is not None:
            ep = ep.crop(tmin=tmin, tmax=tmax)

        data = ep.get_data()[:n_epochs] * 1e6  # (n_ep, n_ch, n_times), µV
        n_ep, n_ch, n_t = data.shape
        if n_ep == 0:
            continue

        ch_names = ep.ch_names
        epoch_dur = ep.times[-1] - ep.times[0]
        total_times = np.concatenate([ep.times + i * epoch_dur for i in range(n_ep)]) * 1000
        data_concat = data.transpose(1, 0, 2).reshape(n_ch, -1)

        scale = np.percentile(np.abs(data_concat), 95) * scale_factor
        offsets = np.arange(n_ch)[::-1] * scale

        fig, ax = plt.subplots(figsize=(18, max(0.9 * n_ch + 1, 3)))

        for ch_idx in range(n_ch):
            ax.plot(total_times, data_concat[ch_idx] + offsets[ch_idx],
                    linewidth=0.5, color="black")

        for i in range(1, n_ep):
            ax.axvline(i * epoch_dur * 1000, color="red", linewidth=0.6,
                       linestyle="--", alpha=0.5)

        for i in range(n_ep):
            t0 = (i * epoch_dur + abs(ep.times[0])) * 1000
            ax.axvline(t0, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.6)

        ax.set_yticks(offsets)
        ax.set_yticklabels(ch_names, fontsize=11)
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_xlim(total_times[0], total_times[-1])
        ax.set_ylim(offsets[-1] - scale, offsets[0] + scale)
        ax.spines[["top", "right"]].set_visible(False)

        t_lo = int(ep.times[0] * 1000)
        t_hi = int(ep.times[-1] * 1000)
        cond_label = f" — {cond}" if cond != "all" else ""
        ax.set_title(
            f"sub-{subject}{cond_label}  |  {n_ep} epochs  [{t_lo} to {t_hi} ms]  |  {ch_names}",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout()

        fname = f"epochs_timeseries_sub-{subject}_{cond}.{fmt}"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, format=fmt, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(fpath)

    return saved_paths


## extract ERP features
def extract_erp_features(evoked, elec, subject, cond, tmin, tmax, trial=None):
    """
    Helper function for extracting ERP features: peak latency, peak amplitude, and mean amplitude.

    Parameters
    ----------
    evoked: mne.Evoked
        <br> The evoked response to analyze.    
    elec: str
        <br> The electrode to analyze.
    subject: str
        <br> The subject identifier.
    cond: str, condition label
        <br> The experimental condition label.
    tmin, tmax: float
        <br> Time window.
    trial: int or None
        <br> Trial index (optional).

    Returns
    -------
    dict:
        <br> Extracted ERP features.
    """
    ch, peak_latency, peak_amplitude = evoked.get_peak(
        ch_type="eeg", tmin=tmin, tmax=tmax, mode="abs", return_amplitude=True
    )
    mean_amplitude = evoked.data[evoked.ch_names.index(elec), (evoked.times >= tmin) & (evoked.times <= tmax)].mean()

    result = {
        "Subject": subject,
        "Electrode": elec,
        "Condition": cond,
        "Peak_Latency": peak_latency,
        "Peak_Amplitude": peak_amplitude,
        "Mean_Amplitude": mean_amplitude
    }
    if trial is not None:
        result["Trial"] = trial  # add trial if LMM
    return result

# ERP Analysis
def erp_analysis(epochs_list, subjects, conditions, electrodes, tmin, tmax, result_dir):
    """
    Analyze ERP data for given subjects, electrodes, and conditions.

    Parameters:
    -----------
    epochs_list: list of str
        <br> List of file paths to epochs for each subject.
    subjects: list of str   
        <br> List of subject identifiers.
    conditions: dict
        <br> Experimental conditions.   
    electrodes: list of str
        <br> List of electrode names.
    tmin, tmax: float
        <br> Time window for ERP analysis.
    result_dir: str
        <br> Directory to save results.

    Returns:   
    ------- 
    pandas.DataFrame
        <br> DataFrame containing extracted ERP features.

    Notes:
    ------
    This function iterates through subjects and epochs, extracting ERP features for each condition and electrode.
    It supports two statistical models: "LMM" for Linear Mixed Models and "ANOVA" for repeated measures ANOVA.
=======
    - epochs_list: list of str, paths to the epochs files
    - subjects: list of str, subject identifiers
    - conditions: list of str, experimental conditions
    - electrodes: list of str, electrode names
    - tmin: float, start time for peak analysis
    - tmax: float, end time for peak analysis
    - result_dir: str, directory to save results
    """

    results = [] 
    erp_result_dir = os.path.join(result_dir, 'erp_analysis_results')
    os.makedirs(erp_result_dir, exist_ok=True)   
    
    # extract experimental conditions
    cond_col, conditions = list(conditions.items())[0]

    # Prepare to collect evoked responses for grand average
    evokeds = {cond: {elec: [] for elec in electrodes} for cond in conditions}

    # iterate through subjects and epochs
    for file, subject in zip(epochs_list, subjects):
        # Load epochs once per subject
        epochs = mne.read_epochs(file, preload=True, verbose=False)
        
        for elec in electrodes:
            for cond in conditions:
                # Select epochs for the current condition
                epochs_cond = epochs[f"{cond_col} == '{cond}'"]

                # Suppress all output from plot_image
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(os.devnull, 'w') as devnull:
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            fig = epochs_cond.plot_image(picks=elec, combine='median', show=False) # FIXME currently saving plot for every sub, cond, elec... group by condition? or by elec?
                # Save the image plot
                save_path = os.path.join(erp_result_dir, f'epochs_image_{subject}_{cond}.png')
                fig[0].savefig(save_path)
                plt.close(fig[0])

                match stat_model:
                    case "ANOVA":
                        # calculate average across epochs for the condition
                        evoked = epochs_cond.average().pick(elec)
                        results.append(
                            extract_erp_features(evoked, elec, subject, cond, tmin, tmax)
                        )
                        evokeds[cond][elec].append(evoked) # Save by condition and electrode
                    case "LMM": # trial-wise analysis
                        for trial in range(len(epochs_cond)):
                            evoked = epochs_cond[trial].average().pick(elec)
                            results.append(
                                extract_erp_features(evoked, elec, subject, cond, tmin, tmax, trial=trial)
                            )
                            evokeds[cond][elec].append(evoked) # Save by condition and electrode

    # Plot ROI or per electrode
    if plot_roi:
        # Collect evoked responses for ROI
        evokeds_roi = collect_evokeds_roi(epochs_list, conditions, electrodes, cond_col)
        # Plot comparison of evoked responses across conditions for the ROI
        plot_erp_comparison(evokeds_roi, conditions, electrodes, result_dir)
        # Plot joint ERP (one plot per condition) for the ROI
        plot_joint_erp(evokeds_roi, conditions, result_dir, times) # can delete "times" parameter to plot default "peaks" in topomap
    else:
        # Plot comparison of evoked responses across conditions for each electrode
        plot_erp_comparison(evokeds, conditions, electrodes, result_dir)
        # plot_joint_erp(evokeds, conditions, electrodes, result_dir) # topography requires at least 2 electrodes

    # save results to a dataframe and csv file
    df = pd.DataFrame(results)
    
    # add trial column for LMM
    if "Trial" in df.columns:
        cols = df.columns.tolist()
        cols.insert(1, cols.pop(cols.index("Trial")))  # move "Trial" to second column in df
        df = df[cols]
    
    # Save to csv
    df.to_csv(os.path.join(erp_result_dir, 'erp_results.csv'), index=False)
    utils.log_msg(f'        ERP analysis completed.')   
    return df

def plot_erp_comparison(evokeds, conditions, electrodes, result_dir):
    """
    Plot ERP comparisons either per electrode or over a region of interest (ROI)

    Parameters:
    - evokeds: dict
        If plot_roi=False, structure should be: evokeds[cond][elec] = list of Evoked
        If plot_roi=True, structure should be: evokeds[cond] = list of Evoked (already averaged over ROI)
    - conditions: list of str, experimental conditions
    - electrodes: list of str, electrode names (for ROI or for looping per-electrode)
    - result_dir: str, directory to save plots
    """

    erp_result_dir = os.path.join(result_dir, 'erp_analysis_results')
    os.makedirs(erp_result_dir, exist_ok=True)

    if plot_roi:
        # Grand averages across ROI (already averaged Evoked objects)
        grand_averages = {
            cond: mne.grand_average(evokeds[cond])
            for cond in conditions
        }

        # Plot comparison of grand averages across conditions for the ROI
        fig = mne.viz.plot_compare_evokeds(
            grand_averages,
            picks=electrodes,
            combine="mean",
            show_sensors='upper right',
            ci=True,
            show=False
        )

        # Save the figure
        fig = plt.gcf()
        roi_name = "+".join(electrodes)
        fig.savefig(os.path.join(erp_result_dir, f'erp_comparison_roi_{roi_name}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        utils.log_msg(f'        ERP comparison plot saved for ROI: {roi_name}')

    else:
        # Per-electrode plotting
        for elec in electrodes:
            grand_averages = {
                cond: mne.grand_average(evokeds[cond][elec])
                for cond in conditions
            }

            # Plot comparison of grand averages across conditions for the electrode
            fig = mne.viz.plot_compare_evokeds(
                grand_averages,
                picks=elec,
                show_sensors='upper right',
                ci=True,
                show=False
            )

            # Save the figure
            fig = plt.gcf()
            fig.savefig(os.path.join(erp_result_dir, f'erp_comparison_{elec}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            utils.log_msg(f'        ERP comparison plot saved for electrode: {elec}')

def plot_joint_erp(evokeds, conditions, result_dir, times='peaks'):
    """ 
    Plot joint ERP for each condition, showing all electrodes in ROI.  
    
    Parameters:
    - evokeds: dict, structure should be: evokeds[cond] = list of Evoked objects for that condition
    - conditions: list of str, experimental conditions
    - result_dir: str, directory to save plots
    """

    erp_result_dir = os.path.join(result_dir, 'erp_analysis_results')
    os.makedirs(erp_result_dir, exist_ok=True)

    # Plot joint ERP for each condition
    for cond in conditions:
        grand = mne.grand_average(evokeds[cond])
        fig = grand.plot_joint(
            title=f"ERP: {cond}",
            times=times,
            show=False
        )
        
        fig.savefig(os.path.join(erp_result_dir, f"joint_erp_{cond}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        utils.log_msg(f'        Joint ERP plot saved for condition: {cond}')

def collect_evokeds_roi(epochs_list, conditions, electrodes, cond_col):
    """ 
    Helper function: Collect evoked responses for a region of interest (ROI) across multiple epochs files.

    Parameters:
    - epochs_list: list of str, paths to the epochs files
    - conditions: list of str, experimental conditions
    - electrodes: list of str, electrode names for the ROI
    - cond_col: str, name of the condition column in the epochs metadata

    Returns:
    dict: evoked responses for the ROI
    """
    # Initialize a dictionary to hold evoked responses for each condition
    evokeds_roi = {cond: [] for cond in conditions}
    # Iterate through each epochs file and extract evoked responses for the specified conditions and electrodes
    for file in epochs_list:
        epochs = mne.read_epochs(file, preload=True, verbose=False)
        for cond in conditions:
            epochs_cond = epochs[f"{cond_col} == '{cond}'"].pick(electrodes)
            evoked = epochs_cond.average()
            evokeds_roi[cond].append(evoked)
    return evokeds_roi

def statistical_analysis(df, result_dir):
    """
    Perform statistical analysis on ERP data.

    Parameters
    ----------
    df: pandas.DataFrame
        <br> DataFrame containing extracted ERP features
    result_dir: str
        <br> Directory to save statistical results.
    
    Returns
    -------
    None

    Notes
    -----
    This function performs statistical analysis on the ERP data using either repeated measures ANOVA or Linear Mixed Models (LMM).
=======
    Parameters:
    - df: pandas.DataFrame, DataFrame containing extracted ERP features
    - result_dir: str, directory to save statistical results
    """

    # make plot path 
    erp_result_dir = os.path.join(result_dir, 'erp_analysis_results')
    os.makedirs(erp_result_dir, exist_ok=True)
    
    match stat_model:
        case "ANOVA":
            utils.log_msg(f'        Performing repeated measures ANOVA with {metric}...')
            anova_model = pg.rm_anova(
                data=df,
                dv=metric,
                within=['Condition', 'Electrode'],
                subject='Subject',
                detailed=True
            )
            # Save statistical results to a CSV file
            anova_model.to_csv(os.path.join(erp_result_dir, f"{metric}_anova_results.csv"), index=False)
            utils.log_msg(f'        Results for {metric} saved to {metric}_anova_results.csv')
        case "LMM":
            utils.log_msg(f'        Performing Linear Mixed Model (LMM) with {metric}...')
            # Fit the LMM model
            lm_model = mixedlm(
                formula=f'{metric} ~ C(Condition) * C(Electrode)',
                data=df,
                groups=df['Subject']
            ).fit()
            # Save the summary results
            lm_results = pd.DataFrame({
                'Predictor': lm_model.params.index,
                'Coef.': lm_model.params.values,
                'Std.Err.': lm_model.bse.values,
                'z': lm_model.tvalues.values,
                'P>|z|': lm_model.pvalues.values
            })
            lm_results.to_csv(os.path.join(erp_result_dir, f"{metric}_lmm_results.csv"), index=False)
            utils.log_msg(f'        Results for {metric} saved to {metric}_lmm_results.csv')

# Time-Frequency Representation (TFR) Analysis

    # Plot Peak Amplitude by Condition and Electrode
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
     # Peak Amplitude
    df.pivot_table(values='Peak_Amplitude', index='Condition', columns='Electrode', aggfunc='mean').plot(kind='bar', ax=axes[0])
    axes[0].set_title('Peak Amplitude by Condition')
    axes[0].set_ylabel('Peak Amplitude (µV)')
    
    # Mean Amplitude  
    df.pivot_table(values='Mean_Amplitude', index='Condition', columns='Electrode', aggfunc='mean').plot(kind='bar', ax=axes[1])
    axes[1].set_title('Mean Amplitude by Condition')
    axes[1].set_ylabel('Mean Amplitude (µV)')
    
    # Peak Latency
    df.pivot_table(values='Peak_Latency', index='Condition', columns='Electrode', aggfunc='mean').plot(kind='bar', ax=axes[2])
    axes[2].set_title('Peak Latency by Condition')
    axes[2].set_ylabel('Peak Latency (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(erp_result_dir, 'erp_summary.png'), dpi=300, bbox_inches='tight')

## TFR FUNCTIONS (+ CBPT)
## def function A
## def function B

## FOOOF
## def function A
## def function B


def subject_tfr(epochs_clean, baseline, tmin, tmax, fmin, fmax, time_res, freq_res, alpha_freq_range, condition_dict, roi, bidspath_out, subject, log_df):
    """
    Computes and plots time-frequency representations (TFR) and frequency-specific topographies
    for specified EEG experimental conditions.

    Parameters
    ----------
    epochs_clean : mne.Epochs
        <br> Cleaned EEG epochs.
    baseline : tuple
        <br> Baseline period for TFR baseline correction (e.g., (None, 0)).
    tmin : float
        <br> Start time (in seconds) for TFR plotting.
    tmax : float
        <br> End time (in seconds) for TFR plotting.
    fmin : float
        <br> Minimum frequency of interest (Hz).
    fmax : float
        <br> Maximum frequency of interest (Hz).
    condition_dict : dict
        <br> Dictionary with a single key-value pair where the key is the name of the condition
        column and the value is a list of condition labels to analyze.
    bidspath_out : object
        <br> Object containing the output directory path via `bidspath_out.directory`.
    roi : list
        <br> List of electrode/channel names to include in TFR computation and plots.
    log_df : pandas.DataFrame
        <br> Log dataframe.
    Returns
    -------
    mne.time_frequency.EpochsTFR
        <br> The computed TFR object (non-averaged across epochs).
    log_df : pandas.DataFrame
        <br> Log dataframe.
    """
   
    # define frequency range and number of cycles
    freqs = np.logspace(*np.log10([fmin, fmax]), num=freq_res)# define frequency range
    n_cycles = freqs / time_res # time_res

    # extract conditions
    cond_col, conditions = list(condition_dict.items())[0]
    tfr_dir = os.path.join(bidspath_out.directory, 'TFR')
    os.makedirs(tfr_dir, exist_ok=True)

    # tmin = epochs_clean.times.min()
    # tmax = epochs_clean.times.max()

    baseline = [tmin, -0.1]

    # make epochs object with virtual ROI channel
    epochs_roi = utils.make_roi_channel(epochs_clean, roi)
    

    # _________________Compute epochs-level TFR_________________
    #permutation cluster needs multiple epochs cannot do average. 
    tfr_dict_epochs = {}
    tfr_epochs = epochs_roi.compute_tfr(
         method="morlet",
         picks='roi',
         freqs=freqs,
         n_cycles=n_cycles,
         average=False,
         return_itc=False,
         decim=3,
         verbose=False
    )

    #_________________Compute condition-level TFR (averaged across epochs)_________________
    tfr_dict_avg = {}
    power_dict = {}
    power_df_list = []
    for condition in conditions:
        # compute TFR
        utils.log_msg(f'        Computing TFR over electrode cluster: {roi}... Dimensions - spectral ({fmin}-{fmax}Hz); temporal ({tmin}-{tmax}s)')
        condition_query = f"{cond_col} == '{condition}'"
        tfr_cond_avg = epochs_clean[condition_query].compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc = False,
                average=True,
                decim=3,
                verbose = False
                )
        
       
        ### DECIBEL CONVERSION
        tfr_db = tfr_cond_avg.copy()
        # Get baseline indices
        baseline_mask = (tfr_db.times >= baseline[0]) & (tfr_db.times <= baseline[1])
        # Calculate baseline power (mean across time for each frequency and channel)
        baseline_power = np.mean(tfr_db.data[:, :, baseline_mask], axis=2, keepdims=True)
        # Apply decibel conversion: 10 * log10(power / baseline_power)
        tfr_db.data = 10 * np.log10(tfr_db.data / baseline_power)
        # Save the averaged subject-level TFR
        avg_tfr_path = os.path.join(tfr_dir, f"sub-{subject}_cond-{condition}_avg-tfr.fif")
        tfr_db.save(avg_tfr_path, overwrite=True, verbose=False)

        ### average across channels
        # pick channels
        tfr_db_averaged = tfr_db.copy().pick(roi)
        # average across channels
        tfr_db_averaged.data = tfr_db_averaged.data.mean(axis=0, keepdims=True)
        # Update channel info to reflect single virtual channel
        info_avg = mne.create_info(['roi'], tfr_db.info['sfreq'], ch_types='eeg', verbose=False)
        tfr_db_averaged.info = info_avg


        # open dictionaries to store TFR and power data 
        tfr_dict_avg[condition] = tfr_db_averaged
        power_dict[cond_col] = condition
        # save epochs-level TFR to condition dict
        tfr_dict_epochs[condition] = tfr_epochs[condition_query]

        ### Plot TFR by condition
        fig = tfr_db_averaged.plot(
            baseline=None, fmin=fmin, fmax=fmax, # mode='mean',
            tmin=tmin, tmax=tmax-0.1, cmap="RdBu_r",
            title=f'Time-Frequency Representation - {condition}; chn: {roi}',
            show=False, colorbar=True, verbose=False)[0]
        fig.savefig(os.path.join(tfr_dir, f"tfr_ch{roi}_{condition}.png"), format="png", dpi=300)
        plt.close(fig)


        ### COMPUTE ALPHA POWER
        freq_mask = (tfr_db_averaged.freqs >= alpha_freq_range[0]) & (tfr_db_averaged.freqs <= alpha_freq_range[1])
        # Select the time indices for prediction window
        time_mask = (tfr_db_averaged.times >= 0) & (tfr_db_averaged.times <= tmax)
        # Extract the power data within the desired time and frequency window
        selected_power = tfr_db_averaged.data[:, freq_mask][:, :, time_mask]
        # Compute absolute alpha power
        total_alpha_power = np.mean(selected_power, axis=(1, 2))  # shape: (n_channels,)
        power_dict['total_alpha_dB'] = total_alpha_power.item()
        # Compute relative alpha power
        total_power = np.mean(tfr_db_averaged.data[:, :, time_mask], axis=(1, 2))  # total power across all frequencies
        relative_alpha_power = total_alpha_power / total_power
        power_dict['relative_alpha_dB'] = relative_alpha_power.item()

        power_df = pd.DataFrame([power_dict])
        power_df_list.append(power_df)
        
    
    # logging
    utils.log_update(log_df, 'tfr_method', 'morlet')
    utils.log_update(log_df, 'frequency_range', [fmin, fmax])
    utils.log_update(log_df, 'temporal_range', [tmin, tmax])
    utils.log_update(log_df, 'cycles_denominator', time_res)
    utils.log_update(log_df, 'alpha_frequency_range', alpha_freq_range)
    utils.log_update(log_df, 'tfr_baseline', baseline)
    utils.log_update(log_df, 'tfr_roi', roi)
    
    # concatenate condition dfs
    power_df = pd.concat(power_df_list, ignore_index=True, sort=False)     
       
    return tfr_dict_avg, tfr_dict_epochs, power_df, log_df
    #return tfr_dict_avg, power_df, log_df #tfr_dict_epochs,

# Group TFR Analysis
def group_tfr(tfr_dict_subjects, condition_dict, eeg_dir):
    """
    Compute grand-average time-frequency representations (TFR) across multiple subjects,
    separately for each condition.

    Parameters
    ----------
    tfr_dict_subjects : dict
        <br> Dictionary of TFR objects for each subject and condition. 
        <br> Structure: {subject_id: {condition1: tfr_object, condition2: tfr_object, ...}}
    condition_dict : dict
        <br> Dictionary with one key (condition column) and a list of condition labels.
    eeg_dir : str
        <br> Directory to save the grand-averaged TFRs.

    Returns
    -------
    dict
        <br> Dictionary mapping each condition to its grand-averaged (across subjects) TFR object.
    
    Notes
    -----
    This function computes the grand-average TFR across subjects for each condition
    and saves the results to the specified output directory.
    """

    # Paths to output CSVs
    # subject_csv = os.path.join(tfr_dir, f"{datatype}_summary.csv")
    # master_csv = os.path.join(tfr_dir, "all_subjects_cluster_summary.csv")

    # # List to accumulate cluster-level summary rows for CSV
    # rows = []

    # make plot path 
    tfr_dir = os.path.join(eeg_dir, 'groupTFR')
    os.makedirs(tfr_dir, exist_ok=True)

    # extract conditions
    cond_col, conditions = list(condition_dict.items())[0]

    #_________________Compute grand average TFR for each condition_________________
    grand_avg_tfr = {}
    for condition in conditions:
        # Collect all TFR objects for this condition across subjects and store in list
        condition_tfrs = []
        utils.log_msg(f'     Computing grand average TFR for condition: {condition}')

        for subject_id, subject_tfr_dict in tfr_dict_subjects.items():
            if condition in subject_tfr_dict:
                condition_tfrs.append(subject_tfr_dict[condition])
        
        if condition_tfrs:
            # Compute grand average 
            grand_avg = mne.grand_average(condition_tfrs)
            grand_avg_tfr[condition] = grand_avg
            
            # Save if result_dir is provided
            if eeg_dir:
                save_path = os.path.join(tfr_dir, f'grand_avg_cond-{condition}_tfr.fif')
                grand_avg.save(save_path, overwrite=True, verbose=False)
                utils.log_msg(f'     Saved grand average TFR for condition {condition} to {save_path}')
        else:
            utils.log_msg(f'     Warning: No TFR data found for condition {condition}')

    #____________________________________________________________

    return grand_avg_tfr
    

# Data Extraction from TFR Function for CBPT
def cbpt_tfr_prep(tfr_dict, cond1, cond2, datatype, epochs_min, log_df):
    """
    Clean utility to extract channel-specific power data from TFRs for all conditions.

    This function extracts the correct TFR data arrays from the larger MNE TFR structure so they can be used in the cbpt analysis
    <br> selects the correct conditions (1/2)
    <br> selects the correct channel (Oz)
    <br> Extracts data in the correct shape (n_obs, n_freqs, n_times; n_obs = epochs at subject level or subjects at group level)
    <br> Handling group vs. subject level data extraction via datatype parameter

    Parameters
    ----------
    tfr_dict : dict
        <br> Dictionary of TFRs, with keys as condition names and values as MNE TFR objects.
    
    channel : str
        <br> Channel name to extract (e.g., 'Oz').

    Returns
    -------
    data_dict : dict
        <br> Dictionary where keys are conditions and values are power data arrays of shape
        <br> (n_epochs, n_freqs, n_times) for the specified channel.
    log_df : pandas.DataFrame
        <br> Log dataframe.
    """

    estimated_cue_onset = abs(epochs_min)
    # Extract data ready for CBPT-function for subject and group TFRs 
    match datatype:
        case "group":
            # dict[subject][condition] = tfr

            # Extract TFR data for condition comparisons
            # First, crop TFR objects to remove baseline (cue onset onwards) before extracting data
            first_subject = list(tfr_dict.keys())[0]
            sample_tfr = tfr_dict[first_subject][cond1]
            
            # Determine cue onset: if times include negative values, cue onset is at 0
            # If times start at 0 or positive, they're shifted and we need to find cue onset
            if sample_tfr.times.min() < 0:
                # TFR times preserve epoch time axis, cue onset is at time 0
                cue_onset_time = 0.0
                print(f'Cue onset time: {cue_onset_time}')
            else:
                # TFR times are shifted (start at 0 instead of epoch tmin)
                # In subject_tfr, baseline ends at abs(tmin) - 0.1
                # If epochs are -0.6 to 1.5, baseline ends at 0.5, so cue onset is just after
                # Estimate: find time point closest to where baseline would end + small buffer
                # Based on typical baseline of ~0.5s before cue, cue onset is around 0.5-0.6s in shifted time
                # Use the time point closest to 0.6 (typical abs(tmin) for -0.6 epoch start)
                # Find closest time point to estimated cue onset
                cue_onset_time = sample_tfr.times[np.argmin(np.abs(sample_tfr.times - estimated_cue_onset))]
                # Ensure we don't go before the first time point
                if cue_onset_time < sample_tfr.times.min():
                    cue_onset_time = sample_tfr.times.min()
            
            # Crop TFRs to cue onset onwards using MNE crop method (more reliable)
            # Crop each TFR object to remove baseline
            for subject in tfr_dict.keys():
                tfr_dict[subject][cond1] = tfr_dict[subject][cond1].copy().crop(tmin=cue_onset_time, tmax=None)
                tfr_dict[subject][cond2] = tfr_dict[subject][cond2].copy().crop(tmin=cue_onset_time, tmax=None)
            
            # Now extract data after cropping
            data_1 = np.array([tfr_dict[subject][cond1].data[0] for subject in tfr_dict.keys()]) # shape = (data(n_subjects), n_freqs, n_times)
            data_2 = np.array([tfr_dict[subject][cond2].data[0] for subject in tfr_dict.keys()])

            # extract freq and times for plotting
            freqs = tfr_dict[first_subject][cond1].freqs
            times = tfr_dict[first_subject][cond1].times
            times_ms = 1e3 * times
            
        case _:
            # Find the index of the desired channel once (assumes same channel list across all conditions)
            ch_idx = list(tfr_dict.values())[0].ch_names.index('roi')
            
            # Determine cue onset from TFR times
            sample_tfr = tfr_dict[cond1]
            if sample_tfr.times.min() < 0:
                # TFR times preserve epoch time axis, cue onset is at time 0
                cue_onset_time = 0.0
            else:
                # TFR times are shifted (start at 0 instead of epoch tmin)
                # In subject_tfr, baseline ends at abs(tmin) - 0.1
                # If epochs are -0.6 to 1.5, baseline ends at 0.5, so cue onset is just after
                # Estimate: find time point closest to where baseline would end + small buffer
                # Find closest time point to estimated cue onset
                cue_onset_time = sample_tfr.times[np.argmin(np.abs(sample_tfr.times - estimated_cue_onset))]
                # Ensure we don't go before the first time point
                if cue_onset_time < sample_tfr.times.min():
                    cue_onset_time = sample_tfr.times.min()
            
            # Crop TFR objects to remove baseline before extracting data
            tfr_dict[cond1] = tfr_dict[cond1].copy().crop(tmin=cue_onset_time, tmax=None)
            tfr_dict[cond2] = tfr_dict[cond2].copy().crop(tmin=cue_onset_time, tmax=None)
            
            # extractdata_1 and data_2 for subject level
            data_1 = tfr_dict[cond1].data[:, ch_idx, :, :]  # shape = (n_epochs, n_freqs, n_times)
            data_2 = tfr_dict[cond2].data[:, ch_idx, :, :]

            # extract freq and times for plotting
            freqs = tfr_dict[cond1].freqs
            times = tfr_dict[cond1].times
            times_ms = 1e3 * times

    # logging
    utils.log_update(log_df, 'cbpt_window_start', cue_onset_time)
    utils.log_update(log_df, 'cbpt_window_end', times.max())

    return data_1, data_2, freqs, times_ms, log_df

# Cluster Based Permutation Test (CBPT)
def CBPT(tfr_dict, datatype, comparisons, roi, bidspath_out, log_df, epochs_min, threshold=None, n_permutations=10000, alpha=0.05, seed=42):
    """
    Perform subject-level cluster-based permutation tests on time-frequency data (TFRs),
    comparing power differences across three conditions (low vs base, high vs base, high vs low).

    Parameters
    ----------
    tfr_dict : dict
        <br> Dictionary of averaged MNE TFR objects, keyed by condition labels.
    times : ndarray
        <br> 1D array of time points (in seconds) from TFR.
    datatype : str
        <br> Subject ID (e.g., '001') used for filenames.
    condition_dict : dict
        <br> Format: {'condition': ['base', 'lowlevel', 'highlevel']}. Determines test conditions.
    channel : str
        <br> EEG channel to analyze (e.g., 'Oz'). Must exist in all TFRs.
    threshold : float
        <br> Threshold for clustering (T-values above this form clusters).
    n_permutations : int
        <br> Number of random permutations for non-parametric testing.
    alpha : float
        <br> Significance level for identifying significant clusters.
    seed : int
        <br> Random seed for reproducibility.
    save_dir : str
        <br> Directory where all plots and CSVs will be saved.
    log_df : pandas.DataFrame
        <br> Log dataframe.
    """

    # Logging CBPT parameters
    utils.log_update(log_df, 'cbpt_threshold', threshold)
    utils.log_update(log_df, 'cbpt_n_permutations', n_permutations)
    utils.log_update(log_df, 'cbpt_alpha', alpha)
    utils.log_update(log_df, 'cbpt_seed', seed)
    utils.log_update(log_df, 'cbpt_comparisons', comparisons)
    utils.log_update(log_df, 'cbpt_roi', roi)

    # create output dir for group or subject level
    match datatype:
        case "group":
            tfr_dir = os.path.join(bidspath_out, 'groupCBPT')
            os.makedirs(tfr_dir, exist_ok=True)
        case _:
            # Ensure output directory exists
            tfr_dir = os.path.join(bidspath_out.directory, 'TFR/CBPT')
            os.makedirs(tfr_dir, exist_ok=True)
            datatype = "sub-" + datatype


    # Creates a loop of comparisons via cbpt_tfr_prep function
    for cond1, cond2 in comparisons:
        utils.log_msg(f"        === Running Cluster Test: {cond1} vs {cond2} (datatype: {datatype}) ===")

        data_1, data_2, freqs, times_ms, log_df = cbpt_tfr_prep(tfr_dict, cond1, cond2, datatype, epochs_min, log_df)

        # Run nonparametric CBPT
        stdout_buffer = io.StringIO() 
        with redirect_stdout(stdout_buffer):    
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test( #permutation_cluster_1samp_test
                [data_1, data_2], 
                out_type="mask",
                n_permutations=n_permutations, 
                threshold=threshold, 
                stat_fun = None, #F-statistic mne.stats.f_oneway, #t-statistic: utils.ttest_rel_wrapper, # paired (repeated measures) t-test - returns t-statistic array
                tail=1, # 0 = two-tailed (undirected) test, 1 = right-tailed test, -1 = left-tailed test
                seed=seed 
            )
            # Save raw cluster data for external plotting
            np.savez_compressed(
                os.path.join(tfr_dir, f"{datatype}_{cond1}vs{cond2}_cbpt_raw.npz"),
                data_1=data_1, data_2=data_2,
                freqs=freqs, times_ms=times_ms,
                T_obs=T_obs, clusters=clusters, cluster_p_values=cluster_p_values
            )

            # logging
            for line in ((stdout_buffer.getvalue()).splitlines()):
                utils.log_msg(f"        {line}") 
  
            # Plotting
            plot_and_save_cbpt_results(
                data_1, data_2, freqs, times_ms, T_obs, clusters, cluster_p_values,
                cond1, cond2, roi, datatype, alpha, tfr_dir
            )

                
            

    return log_df


alpha = 0.05
def plot_cbpt_results(cluster_p_values, clusters, T_obs, alpha, component, comparison, epochs_info, save_dir):


    sig_clusters = np.where(cluster_p_values < alpha)[0]

    print(f"Found {len(sig_clusters)} significant clusters")

    for i in sig_clusters:
        print(
            f"Cluster {i}: "
            f"p = {cluster_p_values[i]:.4f}, "
            f"n_channels = {clusters[i].sum()}"
        )

    for i_clu, clu_idx in enumerate(np.where(cluster_p_values < alpha)[0]):

        mask = clusters[clu_idx]  # Boolean mask for channels

        fig, ax = plt.subplots(figsize=(6, 6))

        im, _ = mne.viz.plot_topomap(
            T_obs,
            epochs_info,
            mask=mask,
            axes=ax,
            show=False,
            extrapolate = 'head',
            cmap = 'PiYG'
        )

        ax.set_title(
            f"Cluster {clu_idx}\n"
            f"p = {cluster_p_values[clu_idx]:.4f}"
        )

        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(f'{component} {comparison} T Values')
        #save_dir = "/Users/cenjingwang/Desktop/Berlin_Summer_Research_2026_DAAD_RISE/BIDSEEG_Pipeline/data/BIDShierPriors/derivatives/BIDSprocesse/results"
        fig.savefig(os.path.join(save_dir, f"{component}_AperiodicOffset_global_cluster{clu_idx}_cbpt_topoplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return im


def test_arrays(data, component, cond_names):
    """
    Reorganizes fooof_bands_total sample into arrays for cluster analysis for repeated measures.

    Parameters
    ----------
    data = pandas.DataFrame 
        with columns ['participant', 'exp', 'channels', component]
    component : str
        Name of the component to analyze, e.g. 'Aperiodic_Exponent'
    cond_names : list of str
        List of condition names.

    Returns
    -------
    f_test_array : numpy.ndarray
        Array for the f test across all conditions.
    t_test_arrays : dict
        Dictionary containing arrays for each t-test comparison.
    """
    average_rows = data[data["channels"] == "average"].index
    data = data.drop(average_rows, inplace=False)

    participants = sorted(data["participant"].unique())

    condition_arrays = {}

    for cond in cond_names:
        pivot = (
            data.loc[data["exp"] == cond, ["participant", "channels", component]]
                .groupby(["participant", "channels"])[component]
                .mean()
                .unstack("channels")
                .reindex(index=participants, columns=sorted(data["channels"].unique()))
        )

        if pivot.isna().any().any():
            missing = pivot.isna().sum()
            #raise ValueError(f"Missing values in condition {cond}:\n{missing}")
            print(f"Missing values in condition {cond}:\n{missing}")

        condition_arrays[cond] = pivot.to_numpy()

    f_test_array = np.stack([condition_arrays[cond] for cond in cond_names], axis=1)

    print(f"f_test_array.shape: {f_test_array.shape}")

    t_test_arrays = {}

    t_test_arrays["base_lowlevel"] = condition_arrays["base"] - condition_arrays["lowlevel"]
    t_test_arrays["base_highlevel"] = condition_arrays["base"] - condition_arrays["highlevel"]
    t_test_arrays["low_highlevel"] = condition_arrays["lowlevel"] - condition_arrays["highlevel"]

    print(f"t_test_arrays['base_lowlevel'].shape: {t_test_arrays['base_lowlevel'].shape}")

    return f_test_array, t_test_arrays

def cbpt_global(f_test_array, n_permutations, alpha, seed, chn_adjacency):
    """Performs a cluster-based permutation test for the global F-test across all conditions.
    
    Parameters
    ----------
    f_test_array : numpy.ndarray
        Array for the f test across all conditions.
    n_permutations : int
        Number of permutations to perform.
    alpha : float
        Significance level for cluster formation.
    seed : int
        Random seed for reproducibility.
    chn_adjacency : scipy.sparse.csr_matrix or csr_array
        Adjacency matrix for the channels.

    Returns
    -------
    F_obs : array, shape (p[, q][, r])
        Statistic (F by default) observed for all variables.
    clusters : list of boolean arrays, each with the same shape as the input data
        True Values indicate positions that are part of a cluster
    cluster_p_values : array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.
    """
    threshold = mne.stats.f_threshold_mway_rm(
        n_subjects = f_test_array.shape[0],
        factor_levels = [3],
        effects = 'A',
        pvalue = alpha
    )

    print("Running global CBPT for F-test across all conditions...")

    F_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        X = f_test_array,
        threshold = threshold, 
        n_permutations= n_permutations, 
        tail=1, # 0 = two-tailed (undirected) test, 1 = right-tailed test, -1 = left-tailed test
        stat_fun= None,
        adjacency=chn_adjacency,
        n_jobs=-1, # might be unnecessary 
        seed=seed, 
        out_type="mask"
    )

    if cluster_p_values is not None:
        print(f"  --- Global CBPT cluster p values: {cluster_p_values}")

    results = {}

    results = {
        'F_obs':      F_obs,
        'clusters':   clusters,
        'cluster_pv': cluster_p_values,
        'H0':         H0
    }
    
    return results

def cbpt_local(t_test_arrays, n_permutations, seed, chn_adjacency):
    """Performs cluster-based permutation tests for local t-tests.
    
    Parameters
    ----------
    t_test_arrays : dict
        Dictionary containing arrays for each t-test comparison.
    n_permutations : int
        Number of permutations to perform.
    alpha : float
        Significance level for cluster formation.
    seed : int
        Random seed for reproducibility.
    chn_adjacency : scipy.sparse.csr_matrix or csr_array
        Adjacency matrix for the channels.

    Returns
    -------
    same as cbpt_global, but for each t-test comparison in a dictionary.
    """
    results = {}

    for comp, arr in t_test_arrays.items():
        print(f"Running CBPT for {comp}")
    
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
            X = arr,
            threshold= None, 
            n_permutations = n_permutations, 
            tail=0, # 0 = two-tailed (undirected) test, 1 = right-tailed test, -1 = left-tailed test
            stat_fun=None, 
            adjacency=chn_adjacency,
            n_jobs=-1, # might be unnecessary, alt None
            seed=seed,
            out_type="mask", 
            )

        print(f"  --- Cluster p_vals for {comp}:", cluster_p_values)

        results[comp] = {
            'T_obs':      T_obs,
            'clusters':   clusters,
            'cluster_pv': cluster_p_values,
            'H0':         H0
        }

    return results


#####CBPTS ANALYSIS + GRAPHS 

# csv_path = os.path.join(result_dir, 'EEG_bands_hierprior.csv')
# fooof_cbpt_save_dir = os.path.join(result_dir, 'FOOOF_CBPT')
# os.makedirs(fooof_cbpt_save_dir, exist_ok=True)



def run_fooof_cbpt(condition_dict, cbpt_n_permutations, cbpt_alpha, cbpt_seed, epochs_info, csv_path, save_dir):
    
    chn_adjacency, _ = mne.channels.find_ch_adjacency(epochs_info, ch_type='eeg')
    cond_col, conditions = list(condition_dict.items())[0]
    
    data = pd.read_csv(csv_path, delimiter=',')
    
    components_cbpt = [
        "Aperiodic_Exponent",
        "Aperiodic_Offset",
        "total_delta_dB",
        "total_theta_dB",
        "total_alpha_dB",
        "total_beta_dB",
        "total_gamma_dB",
        "alpha_CF_Hz_peak"
    ]

    global_cbpt_results = {}
    local_cbpt_results = {}

    for component in components_cbpt:
        print(f'   ==== Testing {component}... ====')
        f_test_array, t_test_arrays = test_arrays(data, component, conditions)
        global_cbpt_results[component] = cbpt_global(f_test_array, cbpt_n_permutations, cbpt_alpha, cbpt_seed, chn_adjacency)
        local_cbpt_results[component] = cbpt_local(t_test_arrays, cbpt_n_permutations, cbpt_seed, chn_adjacency)

    # plot local results
    for component in components_cbpt:
        for comparison_key in local_cbpt_results[component]:
            cluster_p_values = local_cbpt_results[component][comparison_key]['cluster_pv']
            clusters = local_cbpt_results[component][comparison_key]['clusters']
            T_obs = local_cbpt_results[component][comparison_key]['T_obs']

            plot_cbpt_results(cluster_p_values, clusters, T_obs, cbpt_alpha, component, comparison_key, epochs_info, save_dir)

    # plot global results
    for component in components_cbpt:
        cluster_p_values = global_cbpt_results[component]['cluster_pv']
        clusters = global_cbpt_results[component]['clusters']
        F_obs = global_cbpt_results[component]['F_obs']
        sig_clusters = np.where(cluster_p_values < cbpt_alpha)[0]
        print(f"Found {len(sig_clusters)} significant clusters for {component}")
        for i_clu, clu_idx in enumerate(sig_clusters):
            mask = clusters[clu_idx]
            fig, ax = plt.subplots(figsize=(6, 6))
            im, _ = mne.viz.plot_topomap(F_obs, epochs_info, mask=mask, axes=ax, show=False, extrapolate='head', cmap='PiYG')
            ax.set_title(f"Cluster {clu_idx}\np = {cluster_p_values[clu_idx]:.4f}")
            colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label(f'{component} F Values')
            fig.savefig(os.path.join(save_dir, f"{component}_global_cluster{clu_idx}_cbpt_topoplot.png"), dpi=300, bbox_inches='tight')
            plt.close()

    return global_cbpt_results, local_cbpt_results



#_____________FOOOF______________



def run_fooof_analysis(epochs_clean, condition_dict, subject, bidspath_out_subject, tmax, narrowband_freqs, baseline, f_range, peak_threshold, log_df): #removed roi and added log_df
    """
    Fits a FOOOF model to the power spectrum, saves results (JSON), plots (basic + full), and summary CSVs.

    Parameters
    ----------
    freqs : array-like
        Frequencies from the power spectrum.
    spectrum : array-like
        Power spectrum values.
    subject : str
        Subject ID.
    condition : str
        Condition label.
    roi : str
        Channel name (e.g., 'Oz').
    bidspath_out_subject : mne_bids.BIDSPath
        BIDSPath object with .directory set to subject output.
    global_master_csv : str or None
        Path to master summary CSV (appended if provided).
    f_range : tuple
        Frequency range for FOOOF fitting.
    peak_threshold : float
        Threshold for peak detection.

    Returns
    -------
    fm : FOOOF
        The fitted FOOOF model.
    """

        # ----- Define paths -----
    save_dir = os.path.join(bidspath_out_subject.directory, 'FOOOF')
    os.makedirs(save_dir, exist_ok=True)
    summary_dir = os.path.join(save_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)

    # extract conditions
    cond_col, conditions = list(condition_dict.items())[0]

    # epochs_clean.apply_baseline(baseline=[-0.3, -0.1], verbose=False)
    epochs_clean.apply_baseline(baseline=baseline, verbose=False)

    df_fooofsum_list = []
    # df_alphapeak_list = []
    df_bandpeaks_list = []

    utils.log_msg(f"        fitting SpecParam with the following parameters: peak_threshold={peak_threshold}, max_n_peaks=6, peak_width_limits=(1.5, 4), f_range={f_range}")
    
    for condition in conditions:

        epochs_cond = epochs_clean[f"{cond_col} == '{condition}'"]

        # Pick the specified roi and crop to prediction window
        epochs_cond = epochs_cond.crop(tmin=0, tmax=tmax) #prediction window: tmin=0, tmax=tmax

        # Compute PSD
        psd = epochs_cond.compute_psd(method='welch', verbose=False)# whole epoch: n_fft=326, prestim interval: n_fft=251  fmin=f_range[0], fmax=f_range[1],
       

        # Get PSD data and freqs
        freqs = psd.freqs
        all_channels = psd.ch_names 

        psd_data = psd.get_data()  # extract once, pass as plain numpy array
        
        #for parallel processing 
        results = Parallel(n_jobs=-1, backend='loky')( # n_jobs = -1 uses every CPU on computer, loky = default 
            delayed(_fit_channel)(  
                channels, psd_data, psd.ch_names, freqs, condition, subject,
                save_dir, f_range, peak_threshold, narrowband_freqs
            )
            for channels in all_channels
        )

        for fm, df_fooofsum_cond, df_bandpeaks_cond in results:
            df_fooofsum_list.append(df_fooofsum_cond)
            df_bandpeaks_list.append(df_bandpeaks_cond)
        
    df_fooofsum = pd.concat(df_fooofsum_list, ignore_index=True)
    df_bandpeaks = pd.concat(df_bandpeaks_list, ignore_index=True)
    
    df_bandpeaks.loc[df_bandpeaks['Aperiodic_Exponent'] < 0, 'Aperiodic_Exponent'] = float('nan')
    df_bandpeaks.loc[df_bandpeaks['R_squared'] < 0.9, 'Aperiodic_Exponent'] = float('nan')

    
    #topoplot of aperiodic exponents 
    ch_orders = psd.ch_names
    avg_exponents = [df_bandpeaks[df_bandpeaks['channels'] == ch]['Aperiodic_Exponent'].mean() for ch in ch_orders ] #prevent NaN when no peaks found, so always one row per channel
    avg_exponents_plot = [0 if np.isnan(v) else v for v in avg_exponents] #replacing Nan with 0 for plots
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.array([not np.isnan(v) for v in avg_exponents]) #grays out the bad channels 
    im, _ = mne.viz.plot_topomap(avg_exponents_plot, psd.info, cmap='viridis', mask=mask, mask_params=dict(marker='o', markerfacecolor='grey', markersize=2), extrapolate='head', show=False, axes=ax)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('Aperiodic Exponent')  

    fig.savefig(os.path.join(save_dir, f"sub-{subject}_aperiodic_topomap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    utils.log_update(log_df, 'max_n_peaks', 6) # logs the parameters used

    
    return fm, df_fooofsum, df_bandpeaks

#def plot_topomap_generic(data, info, title="Topomap", cmap="viridis", mask=None, mask_params=None, vmin=None, vmax=None, colorbar_label="Value", save_path=None, figsize=(6, 6), extrapolate="head"):
   
#     fig, ax = plt.subplots(figsize=(6, ))

#     im, _ = mne.viz.plot_topomap(
#         data,
#         info,
#         axes=ax,
#         cmap=cmap,
#         mask=mask,
#         mask_params=mask_params,
#         show=False,
#         vmin=vmin,
#         vmax=vmax,
#         extrapolate=extrapolate
#     )

#     mask = np.array([not np.isnan(v) for v in avg_exponents]) 
#     colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     colorbar.set_label('Aperiodic Exponent')  

#     fig.savefig(os.path.join(save_dir, f"sub-{subject}_aperiodic_topomap.png"), dpi=300, bbox_inches='tight')
#     ax.set_title(title) 
#     plt.close(fig)


def _fit_channel(channels, psd_data, ch_names, freqs, condition, subject, save_dir, f_range, peak_threshold, narrowband_freqs):
    ch_idx = ch_names.index(channels) #to reduce redundancy with copying and so forth 
    spectrum = psd_data[:, ch_idx, :].mean(axis=0)
    
    #psd_channels = psd.copy().pick(channels)
    #spectrum = psd_channels.get_data().squeeze().mean(axis=0)  # avg over epochs only 

    model_path = os.path.join(save_dir, f"sub-{subject}_{condition}_{channels}_fooof_model.json")
    fig_path_basic = os.path.join(save_dir, f"sub-{subject}_{condition}_{channels}_fooof_basic.png")
    fig_path_full = os.path.join(save_dir, f"sub-{subject}_{condition}_{channels}_fooof_full.png") 
                
    # ----- Fit model -----
    fm = FOOOF(aperiodic_mode='fixed', peak_threshold=peak_threshold,
            max_n_peaks=6, peak_width_limits=(1.4, 8)) 
    fm.fit(freqs, spectrum, f_range)

    # ----- Save model JSON -----
    fm.save(model_path, save_results=True)

    # Detailed full iterative model
    plot_full_fooof_model_detailed(
        fm,
        subject=subject,
        condition=condition,
        save_path=fig_path_full,
        log_log=True
    )

    fm.plot()
    plt.savefig(fig_path_basic, dpi=300)
    plt.close()

    
    # ----- Extract and save summary -----
    rows_fooofsum = []
    for idx, (cf, pw, bw) in enumerate(fm.peak_params_, start=1):
        rows_fooofsum.append({
            "participant": f'sub-{subject}',
            "exp": condition,
            "Peak #": idx,
            "CF_Hz": cf,
            "PW_log10": pw,
            "PW_dB": 10 * pw,
            "BW_Hz": bw,
            "Aperiodic_Offset": fm.aperiodic_params_[0],
            "Aperiodic_Exponent": fm.aperiodic_params_[1],
            "R_squared": fm.r_squared_,
            "Error": fm.error_,
            "channels": channels
        })
    
    #discard negative aperiodic_exponents and r^2 filtering

    df_fooofsum_cond = pd.DataFrame(rows_fooofsum).round(6)
    if not df_fooofsum_cond.empty: #to prevent crash when checing for validity if it is empty/no peaks
        df_fooofsum_cond.loc[df_fooofsum_cond['Aperiodic_Exponent'] < 0, 'Aperiodic_Exponent'] = np.nan
        df_fooofsum_cond.loc[df_fooofsum_cond['R_squared'] < 0.9, 'R_squared'] = np.nan
        df_fooofsum_cond['Channels_Validity'] = (
            df_fooofsum_cond['Aperiodic_Exponent'].notna() &
            df_fooofsum_cond['R_squared'].notna()
        )
    #df_fooofsum_list.append(df_fooofsum_cond)
        

    



    # ----- Extract Delta, Theta, Alpha, Beta and Gamma Peak and save summary -----
    rows_bandpeaks = []
    # initializing single row (dict instead of list) with model params                
    rows_bandpeaks_dict = {
        "participant": f'sub-{subject}',
        "exp": condition,
        "channels": channels,
        "Aperiodic_Offset": fm.aperiodic_params_[0],
        "Aperiodic_Exponent": fm.aperiodic_params_[1],
        "R_squared": fm.r_squared_,
        "Error": fm.error_
        }
    
    
    #Loop through all bands 
    for band_name, freq_range in narrowband_freqs.items():
        band_peaks = fm.peak_params_[(fm.peak_params_[:, 0] > freq_range[0]) & (fm.peak_params_[:, 0] < freq_range[1])]
        if len(band_peaks) > 0:
            # extract peak with highest power
            max_band_peak = np.argmax(band_peaks[:, 1]) # if you leave this out, you get all peaks within the theta range. With averaging alpha_peak[1] you get the average power of all true oscillations within the alpha range.
            band_peak = band_peaks[max_band_peak]
            # extract sum of all peaks within narrowband range
            mean_band_cf = band_peaks[:, 0].mean() 
            sum_band_pw = band_peaks[:, 1].sum() #
            sum_band_bw = band_peaks[:, 2].sum()
        else:
            band_peak = [0, 0, 0]
            mean_band_cf = 0
            sum_band_pw = 0
            sum_band_bw = 0
        rows_bandpeaks_dict.update({
            f"{band_name}_CF_Hz_peak": band_peak[0],
            f"mean_{band_name}_cf": mean_band_cf,
            f"{band_name}_PW_dB_peak": 10 * band_peak[1],
            f"{band_name}_PW_dB_sum": 10 * sum_band_pw,
            f"{band_name}_BW_Hz_peak": band_peak[2],
            f"{band_name}_BW_Hz_sum": sum_band_bw
            })
        utils.log_msg(f"            - {condition}{channels}: CF: {band_peak[0]}, PW_dB {10 * band_peak[1]}")


    return fm, df_fooofsum_cond, pd.DataFrame([rows_bandpeaks_dict]).round(6)
        



# _____________________________Loading___________________________________________
## load inputs
inputs = utils.read_inputs(sys.argv[1])

# assign path variables
bidspath = utils.get_bidspath(inputs, 'bids_proc')

# make results directories
result_dir = os.path.join(bidspath.root, 'results')
os.makedirs(result_dir, exist_ok=True)
eeg_dir = os.path.join(result_dir, 'groupEEG')
os.makedirs(eeg_dir, exist_ok=True)

#  Time Frequency Representation (TFR)
perform_tfr = inputs['perform']['perform_tfr']
perform_cbpt = inputs['perform']['perform_cbpt']
perform_cbpt_group = inputs['perform']['perform_cbpt_group']
tmin = inputs['preprocessing']['epoch_min']
tmax = inputs['preprocessing']['epoch_max']
baseline = inputs['preprocessing']['baseline_correction']

# TFR parameters
fmin = inputs['Analysis']['fmin']
fmax = inputs['Analysis']['fmax']
time_res = inputs['Analysis']['time_res'] 
freq_res = inputs['Analysis']['freq_res'] 
alpha_freq_range = inputs['Analysis']['alpha_freq_range']
roi = inputs['Analysis']['ROI']
condition_dict = inputs['Analysis']['conditions']
eeg_contrasts = inputs['Analysis']['eeg_contrasts']
eeg_parameters = inputs['Analysis']['brain_parameters_to_plot']

# CBPT parameters
cbpt_threshold = inputs['Analysis']['cbpt']['threshold']
cbpt_n_permutations = inputs['Analysis']['cbpt']['n_permutations']
cbpt_alpha = inputs['Analysis']['cbpt']['alpha']
cbpt_seed = inputs['Analysis']['cbpt']['seed']


# FOOOF
fooof_f_range = inputs['Analysis']['fooof']['fooof_f_range']
fooof_peak_threshold = inputs['Analysis']['fooof']['fooof_peak_threshold']
compute_fooof = inputs ['perform']['compute_fooof']
narrowband_freq_ranges = inputs['Analysis']['narrowband_freqs']

# ERP
compute_erp = inputs['perform']['compute_erp']
plot_roi = inputs['Analysis']['plot_roi']    
times = inputs['Analysis']['times']


## extract subject list
subjects = utils.find_subjects(bidspath.root)

# process from subjex x onwards
# subjects = [sub for sub in subjects if int(sub) > 28]

# subjects to exclude
# subjects_to_exclude = ['028']
# subjects = [item for item in subjects if item not in subjects_to_exclude]

# statistics
stat_model = inputs['Analysis']['stat_model']
metric = inputs['Analysis']['metric']
comparisons = inputs['Analysis']['comparisons']

# fooof summaries dfs and save_dir
global_fooof_dfs = []  
tfr_alpha_dfs = []
fooof_alpha_dfs = []
global_master_csv = os.path.join(bidspath.root, "results", "master_fooof_summary.csv")

# _______________________________________________________________________________
  
  


# _____________________________Module_exe________________________________________
## Loop through participants
if __name__ == '__main__':
    print(f'\n\n\n\n')
    timepoint_start = utils.log_msg(f'START:  EEG Analysis Module - Subject Level')

    ## load log
    log_df = utils.log_load()

    # _____________________________Loading___________________________________________
    ## load inputs
    inputs = utils.read_inputs(sys.argv[1])

    # assign path variables
    bidspath = utils.get_bidspath(inputs, 'bids_proc')

    # make results directories
    result_dir = os.path.join(bidspath.root, 'results')
    os.makedirs(result_dir, exist_ok=True)
    eeg_dir = os.path.join(result_dir, 'groupEEG')
    os.makedirs(eeg_dir, exist_ok=True)

    #  Time Frequency Representation (TFR)
    perform_tfr = inputs['perform']['perform_tfr']
    perform_cbpt = inputs['perform']['perform_cbpt']
    perform_cbpt_group = inputs['perform']['perform_cbpt_group']
    tmin = inputs['preprocessing']['epoch_min']
    tmax = inputs['preprocessing']['epoch_max']
    baseline = inputs['preprocessing']['baseline_correction']

    # TFR parameters
    fmin = inputs['Analysis']['fmin']
    fmax = inputs['Analysis']['fmax']
    time_res = inputs['Analysis']['time_res'] 
    freq_res = inputs['Analysis']['freq_res'] 
    roi = inputs['Analysis']['ROI']
    condition_dict = inputs['Analysis']['conditions']
    eeg_contrasts = inputs['Analysis']['eeg_contrasts']
    eeg_parameters = inputs['Analysis']['brain_parameters_to_plot']

    # CBPT parameters
    cbpt_threshold = inputs['Analysis']['cbpt']['threshold']
    cbpt_n_permutations = inputs['Analysis']['cbpt']['n_permutations']
    cbpt_alpha = inputs['Analysis']['cbpt']['alpha']
    cbpt_seed = inputs['Analysis']['cbpt']['seed']


    # FOOOF
    fooof_f_range = inputs['Analysis']['fooof']['fooof_f_range']
    fooof_peak_threshold = inputs['Analysis']['fooof']['fooof_peak_threshold']
    compute_fooof = inputs ['perform']['compute_fooof']
    narrowband_freq_ranges = inputs['Analysis']['narrowband_freqs']

    #print(f"Loaded narrowband frequency ranges for FOOOF analysis: {narrowband_freq_range}")
    

    # ERP
    compute_erp = inputs['perform']['compute_erp']
    plot_roi = inputs['Analysis']['plot_roi']    
    times = inputs['Analysis']['times']


    ## extract subject list
    subjects = utils.find_subjects(bidspath.root)

    # process from subjex x onwards
    # subjects = [sub for sub in subjects if int(sub) > 28]

    # subjects to exclude
    # subjects_to_exclude = ['028']
    # subjects = [item for item in subjects if item not in subjects_to_exclude]

    # statistics
    stat_model = inputs['Analysis']['stat_model']
    metric = inputs['Analysis']['metric']
    comparisons = inputs['Analysis']['comparisons']

    # fooof summaries dfs and save_dir
    global_fooof_dfs = []  
    tfr_bands_dfs = []
    fooof_bands_dfs = []
    global_master_csv = os.path.join(bidspath.root, "results", "master_fooof_summary.csv")



    # __________________________________Subject Analysis - Start _________________________________
    # Loop through participants
    tfr_dict_sub = {}
    for subject in subjects:
        print(f'\n')
        utils.log_msg(f"_______ Processing Subject-{subject}_______")
        ## update subject and processing steps in inputs
        utils.update_inputs(sys.argv[1], 'basic','subject_ID', subject)
        utils.update_inputs(sys.argv[1], 'basic','current_step', '04epochsCorr')

        ## load  subject data
        epochs = utils.load_preprocessing_step(bidspath, subject, 'from_fif')
        
        # update subject  BIDSpath
        bidspath_processing_subject = bidspath.copy().update(subject=subject)

        
        #_________________Time-Frequency Analysis_________________
        # Compute time-frequency representation
        if perform_tfr:
            tfr_dict_sub[subject] = {}
            timepoint_start = utils.log_msg(f'        *** Time-Frequency Analysis ***')
            #tfr_dict_avg, power_df, log_df = subject_tfr(epochs, baseline, tmin, tmax, fmin, fmax, time_res, freq_res, alpha_freq_range, condition_dict, roi, bidspath_processing_subject, subject, log_df) #tfr_dict_epochs, 
            tfr_dict_avg, tfr_dict_epochs, power_df, log_df = subject_tfr(epochs, baseline, tmin, tmax, fmin, fmax, time_res, freq_res, alpha_freq_range, condition_dict, roi, bidspath_processing_subject, subject, log_df)
            # store TFR dict in dictionary as: dict[subject][condition] = tfr
            tfr_dict_sub[subject] = tfr_dict_avg
            power_df.insert(0, 'participant', f'sub-{subject}')
            tfr_alpha_dfs.append(power_df)
            utils.save_preprocessing_step(tfr_dict_avg, '05tfr', bidspath, subject) # save TFRs

            #_________Cluster-Based Permutation Tests - Logging_________
            if perform_cbpt:
                utils.log_update(log_df, 'cbpt_threshold', cbpt_threshold)
                utils.log_update(log_df, 'cbpt_n_permutations', cbpt_n_permutations)
                utils.log_update(log_df, 'cbpt_alpha', cbpt_alpha)
                utils.log_update(log_df, 'cbpt_seed', cbpt_seed)
                utils.log_update(log_df, 'cbpt_comparisons', comparisons)
                utils.log_update(log_df, 'cbpt_roi', roi)

                log_df = CBPT(tfr_dict_epochs, subject, comparisons, roi, bidspath_processing_subject, log_df, tmin,
                  threshold=cbpt_threshold, n_permutations=cbpt_n_permutations,
                  alpha=cbpt_alpha, seed=cbpt_seed)

        else:
            utils.log_msg(f'     -- Time-Frequency Analysis not performed')

    # ____________________ PSD & FOOOF Analysis ______________________
        if compute_fooof:
            utils.log_msg(f'        Computing FOOOF')
        # psd_dict = subject_psd(epochs, fmin, fmax, condition_dict, 'Oz', bidspath_processing_subject, subject)
            fm, df_fooofsum, df_fooofalpha  = run_fooof_analysis(epochs, condition_dict, subject=subject, bidspath_out_subject=bidspath_processing_subject, tmax=tmax, narrowband_freqs=narrowband_freq_ranges, baseline=baseline, f_range=fooof_f_range, peak_threshold=fooof_peak_threshold,log_df=log_df)
            global_fooof_dfs.append(df_fooofsum)
            fooof_alpha_dfs.append(df_fooofalpha)

        timepoint_end = utils.log_msg(f'DONE:   EEG Analysis Module - Subject Level')
        utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')
    # __________________________________Run CBPT FOOOF _________________________________

        epochs_info = epochs.info
        # epochs_info = mne.read_epochs(
        #         utils.load_preprocessing_step(bidspath, subjects[-1], 'from_fif'),
        #         preload=False,
        #         verbose=False
        # ).info
        csv_path = os.path.join(result_dir, 'EEG_bands_hierprior.csv')
        fooof_cbpt_save_dir = os.path.join(result_dir, 'FOOOF_CBPT')
        os.makedirs(fooof_cbpt_save_dir, exist_ok=True)

        global_cbpt_results, local_cbpt_results = run_fooof_cbpt(condition_dict, cbpt_n_permutations, cbpt_alpha, cbpt_seed, epochs_info, csv_path, fooof_cbpt_save_dir)      
    

    # ________________________Saving & Plotting - Subject Level ________________________
    # concatenating Wavelet (TFR) and FOOOF Alpha Frequency measures
    tfr_alpha = pd.concat(tfr_alpha_dfs, ignore_index=True)
    fooof_alpha = pd.concat(fooof_alpha_dfs, ignore_index=True)
    power_df = pd.merge(tfr_alpha, fooof_alpha, on=['participant', 'exp'], how='left')
    
    
    # added for different band names 
    band_rename = {}
    for band in narrowband_freq_ranges:
        band_rename[f"{band}_PW_dB_sum"] = f"total_{band}_dB"
        band_rename[f"{band}_PW_dB_peak"] = f"relative_{band}_dB"
    power_df = power_df.rename(columns=band_rename)
    power_df = power_df.loc[:, ~power_df.columns.duplicated(keep='first')]  #to deduplicate 
    power_file = f'{result_dir}/EEG_bands_hierprior.csv'
    power_df.to_csv(power_file, index=False)

    # Plot results
    #changed for variable naming purposes 
    eeg_parameters_valid = [p for p in eeg_parameters if p in power_df.columns and bool(power_df[p].notna().any())] # to account for nan
    #raincloud_plot(power_df, condition_dict, eeg_parameters_valid, eeg_dir)
    paired_plot(power_df, condition_dict, eeg_parameters_valid, eeg_dir)
    #raincloud_plot(power_df,condition_dict, eeg_parameters, eeg_dir)
    #paired_plot(power_df, condition_dict, eeg_parameters, eeg_dir)
    tfr_plots_subjects(tfr_dict_sub, condition_dict, fmin, fmax, tmin, tmax, eeg_dir)


    # Saving FOOF summary
    global_df = pd.concat(global_fooof_dfs, ignore_index=True)
    global_df.to_csv(global_master_csv, index=False)
    
    utils.log_msg(f"        Global FOOOF summary written to: {global_master_csv}")

    # Logging
    timepoint_end = utils.log_msg(f'DONE:   EEG Analysis Module - Subject Level')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')

# __________________________________Subject Analysis - END _________________________________

# __________________________________Group Analysis - START _________________________________
    print(f'\n\n\n\n')
    # bidspath_results = utils.get_bidspath(inputs, 'results')
    timepoint_start = utils.log_msg(f'START:  EEG Analysis Module - Group Level')
    utils.log_msg(f"_______ Analysis of subjects: {subjects}_______")

    ## Loading data and output paths
    bidspaths_epochs = utils.get_bidspath(inputs, 'epochs_list', subjects) # list of path-strings to epochs of each subject
    # bidspaths_tfrs = utils.get_bidspath(inputs, 'tfr_list', subjects) # list of path-strings to TFRs of each subject
    # Load epochs as dict: epochs_dict{'condition_1' : [mne.Evoked_sub001, mne.Evoked_sub002, ...],'condition_2' : [mne.Evoked_sub001, mne.Evoked_sub002, ...], ...
    # epochs_dict = utils.load_epochs_dict(bidspaths_epochs, condition_dict, roi)


    # Compute Grand Average TFR
    if perform_tfr:
        utils.log_msg(f'        *** TFRs - computing TFRs across all subjects ***')
        grand_avg_tfr = group_tfr(tfr_dict_sub, condition_dict, eeg_dir)# returns dictionary
        grand_avg_tfr_plots(grand_avg_tfr, condition_dict, comparisons, fmin, fmax, tmin, tmax, alpha_freq_range, eeg_dir)

        #_________Cluster-Based Permutation Tests - group level_________
        if perform_cbpt_group:
            utils.log_msg(f'        Computing group-CBPT over spectral ({fmin}-{fmax}Hz) and temporal ({tmin}-{tmax}s) dimensions')
            datatype = "group"
            log_df = CBPT(tfr_dict_sub, datatype, comparisons, roi, result_dir, log_df, tmin, threshold=cbpt_threshold, n_permutations=cbpt_n_permutations, alpha=cbpt_alpha, seed=cbpt_seed)
        else:
            utils.log_msg(f'     -- group-CBPT not performed')
    else:
        utils.log_msg(f'     -- Time-Frequency Analysis not performed')
    
        



    # ________________________________________________________________________________________
        # Compute ERP Features and Perform Statistical Analysis
    print(f'\n\n\n\n')
    if compute_erp:
            # extract ERP features and save to csv
            utils.log_msg(f'        *** ERP - extracting features ***')
            erp_results = erp_analysis(bidspaths_epochs, subjects, condition_dict, roi, tmin, tmax, result_dir)
            # perform statistical analysis and save to csv
            utils.log_msg(f'        *** ERP - performing statistical analysis ***')
            statistical_analysis(erp_results, result_dir)
    else:
            utils.log_msg(f'     -- No ERP features extracted or statistical analysis performed')

        ## compute average ERPs
        # utils.log_msg(f'        *** ERPs - computing ERPs for all subjects ***')
        # evoked_dict, grand_avg_evoked = grandAverage_ERP(baseline, epochs_dict, epoch_dict, roi, bidspath_results)
        # utils.save_preprocessing_step(grand_avg_evoked, '05GrandAvEvoked', bidspath_results)

    timepoint_end = utils.log_msg(f'DONE:   EEG Analysis Module - Group Level')
    utils.log_save(log_df,f'{bidspath.root}' ,'log_dataframe.csv')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')
    # __________________________________Grand Average Analysis - END _________________________________




# _____________________________LEGACY FUNCTIONS___________________________________________
def subject_psd(epochs_clean, fmin, fmax, condition_dict, channel, bidspath_out, subject):
    """
    Computes and saves PSDs per condition for a given subject and channel.

    Parameters
    ----------
    epochs_clean : mne.Epochs
        Cleaned EEG epochs.
    fmin, fmax : float
        Frequency range for computing PSD.
    condition_dict : dict
        Dict with one key (e.g., 'exp') and a list of condition labels.
    channel : str
        Channel to extract (e.g., 'Oz').
    bidspath_out : mne_bids.BIDSPath
        BIDSPath object with subject-specific output paths.
    subject : str
        Subject ID.

    Returns
    -------
    psd_dict : dict
        Dictionary of PSD arrays per condition.
    """
    utils.log_msg(f"        Computing PSDs for subject {subject} at channel {channel}...")

    cond_col, conditions = list(condition_dict.items())[0]
    psd_dir = os.path.join(bidspath_out.directory, 'FOOOF')
    os.makedirs(psd_dir, exist_ok=True)

    psd_dict = {}
    for cond in conditions:
        # Select epochs for the condition
        epochs_cond = epochs_clean[f"{cond_col} == '{cond}'"]

        # Pick the specified channel
        epochs_cond = epochs_cond.pick(channel)

        # Compute PSD
        psd = epochs_cond.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=326, verbose=False)

        # Get PSD data and freqs
        data = psd.get_data().squeeze()  # shape: (n_epochs, n_freqs)
        freqs = psd.freqs

        # Store in dict
        psd_dict[cond] = {
            'freqs': freqs,
            'psd': data
        }

        utils.log_msg(f"            Saved PSD for {cond} at {psd_dir}")

    return psd_dict




 




