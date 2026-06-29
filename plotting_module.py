# _____________________________plotting_module.py________________________________________
# Helper module imported by c_EEGAnalysis_module.py and d_BehavAnalysis_module.py
# (not run directly)
#
# * Allgemeine und Biologische Psychologie - AG Hesselman
# * Psychologische Hochschule
#
# ## Author(s)
# * Buchholz, Jonathan; Psychologische Hochschule Berlin, AG Hesselmann
# * Rowan, Dowd; Psychologische Hochschule Berlin, AG Hesselmann
#
# * last update: 2025.07.11
#
# * This script provides a set of utility functions for visualizing EEG analysis results,
# * including plotting time-frequency representations (TFRs), significant clusters from
# * cluster-based permutation tests (CBPT), and FOOOF spectral results.
# * Designed for modular use in EEG pipelines, particularly after statistical results are computed.
# =====================================================

# ================= Imports =====================
import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fooof import FOOOF
import mne
import utils_module as utils

# FOOF
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_peak_search


# ================= CBPT Plotting =====================

import seaborn as sns
# _______________________________________________________________________________



'''
TODO: 
- 
'''

# _____________________________Functions_________________________________________
def rt_descriptive_plots(df, rt_column, choice_column, choices, datatype, bidspath_out):
    """
    Plots a histogram + KDE, boxplot, and violinplot of reaction times.

    Parameters:
        df (pd.DataFrame): DataFrame with reaction times.
        rt_column (str): Name of the column holding reaction times.
        choice_column (str): Name of the column holding dichotonomous prior congruent (1) or incongruent (0) responses
    """
    # check for any remaining NaN
    rt = df.dropna(subset=[rt_column, choice_column])

    # remap condition labels
    rt[choice_column] = rt[choice_column].map({1: choices[0], 0: choices[1]})

    #color palette
    palette = ['dodgerblue', 'magenta']

    # Accuracy
    acc_df = rt.groupby(choice_column)['corr'].mean().reset_index()
    acc_df.columns = ['Condition', 'Accuracy']

    # Plotting
    plt.figure(figsize=(14, 12))

    # Histogram + KDE
    plt.subplot(3, 2, 1)
    sns.histplot(data=rt, x=rt_column, hue=choice_column, bins=25, kde=True, alpha=0.5, palette=palette)
    plt.title("Histogram")
    plt.xlabel("")
    plt.ylabel("Count")

    # Boxplot
    plt.subplot(3, 2, 2)
    sns.boxplot(x=choice_column, y=rt_column, data=rt, hue=choice_column, palette=palette, legend=False)
    plt.xlabel("")
    plt.title("Boxplot")

    # ECDF
    plt.subplot(3, 2, 3)
    sns.ecdfplot(data=rt, x=rt_column, hue=choice_column, palette=palette)
    plt.title("Cumulative Density Function (CDF)")
    plt.xlabel("")
    plt.ylabel("Cumulative Probability")

    # Violinplot
    plt.subplot(3, 2, 4)
    sns.violinplot(x=choice_column, y=rt_column, data=rt, hue=choice_column, palette=palette, legend=False)
    plt.xlabel("")
    plt.title("Violinplot")

    # PDF
    plt.subplot(3, 2, 5)
    sns.kdeplot(data=rt, x=rt_column, hue=choice_column, fill=True, common_norm=False, palette=palette)
    plt.title("Probability Density Function (PDF)")
    plt.xlabel("Reaction Time (s)")
    plt.ylabel("Density")

    # Accuracy
    plt.subplot(3, 2, 6)
    barplot = sns.barplot(data=acc_df, x='Condition', y='Accuracy',  hue='Condition', palette=palette, legend=False)
    for idx, row in acc_df.iterrows():
        barplot.text(idx, row.Accuracy, f"{row.Accuracy:.2f}", ha='center', va='bottom')

    plt.title("Accuracy")
    plt.ylabel("Mean Accuracy (% correct)")

    plt.suptitle("Overview of Reaction Time Distribution by Condition", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

    # save figure
    match datatype:
        case "group":
            fig_dir = bidspath_out
            # fig_dir = os.path.join(bidspath_out, 'groupBehavioral')
            # os.makedirs(fig_dir, exist_ok=True)
        case _ :
            # Ensure output directory exists
            fig_dir = os.path.join(bidspath_out.directory, 'behavioral')
            os.makedirs(fig_dir, exist_ok=True)
            # save all subject plots also the groupLevel results folder
            result_dir = os.path.join(bidspath_out.root, 'results/subjectBehavioral')
            os.makedirs(result_dir, exist_ok=True)
            datatype = "sub-" + datatype
            plt.savefig(os.path.join(result_dir, f"{datatype}_RT_distribution.png"))
    
    plt.savefig(os.path.join(fig_dir, f"{datatype}_RT_distribution.png"))

def model_validation(df, bidspath_out):
        
    metrics = ['mean_rt', 'median_rt', 'std_rt']

    ### Plot comparic RT measures of empirical and simulated RT data
    plt.figure(figsize=(14, 12))
    for idx, metric in enumerate(metrics):

        plt.subplot(3, 2, (idx * 2 + 1))
        # prior congruent trials
        sns.scatterplot(data=df, x=(f'{metric}_cong'), y=(f'{metric}_cong_rec'), hue='exp', size=None, style=None, palette='cool', hue_order=None, hue_norm=None, 
                        sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, legend='auto', ax=None)
        plt.title(f"{metric} - Prior Congruent")
        plt.xlabel("Empirical Data")
        plt.ylabel("Simulated Data")

        # prior incongruent trials
        plt.subplot(3, 2, (idx * 2 + 2))
        sns.scatterplot(data=df, x=(f'{metric}_incong'), y=(f'{metric}_incong_rec'), hue='exp', size=None, style=None, palette='cool', hue_order=None, hue_norm=None, 
                    sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, legend='auto', ax=None)
        
        plt.title(f"{metric} - Prior Incongruent")
        plt.xlabel("Empirical Data")
        plt.ylabel("Simulated Data")

    plt.suptitle(f"DDM Validation - Empirical vs. Simulated RT metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # save figure
    plt.savefig(os.path.join(bidspath_out, "ModelEval_RTmetrics.png"))
    plt.close()

    ### Plot comparing estimated and recovered DDM parameter
    parameter = ['drift', 'B', 'x0']

    plt.figure(figsize=(14, 12))
    for idx, para in enumerate(parameter):

        plt.subplot(3, 2, (idx + 1))
        sns.scatterplot(data=df, x=para, y=(f'{para}_rec'), hue='exp', style=None, palette='cool', hue_order=None, hue_norm=None, #size='participant',
                        sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, legend='auto', ax=None)
        plt.title(f"{para}")
        plt.xlabel("Estimated Parameters")
        plt.ylabel("Recovered Parameters")

    plt.suptitle(f"DDM Validation - Estimated vs. Recovered DDM-Parameters", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(bidspath_out, "ModelEval_recoveredParameters.png"))
    plt.close()

def raincloud_plot(df, condition_dict, parameters, bidspath_out):

    """Raincloud-style plot with grouped offset per condition."""

    # extract condition column and conditions
    cond_col, conditions = list(condition_dict.items())[0]
    # guarantee that 'base' is in the middle
    conditions = [cond for cond in conditions if cond != 'base']
    conditions = [conditions[0], 'base', conditions[1]]

    cmap = ['mediumblue', 'darkorchid', 'magenta']

    # saving directory
    ddm_dir = os.path.join(bidspath_out, 'groupDDM_SDT/raincloud')
    os.makedirs(ddm_dir, exist_ok=True)

    # Define x positions
    x_pos = np.arange(len(conditions))
    offset_violin = 0.2
    offset_box = 0.0
    offset_scatter = -0.2
    width = 0.25

    for param in (parameters):

        # initialize plot
        plt.figure(figsize=(10, 5))
        ax = plt.gca()

        for i, cond in enumerate(conditions):
            y = df[df[cond_col] == cond][param]

            # Violin plot (half - left side)
            parts = ax.violinplot(y, positions=[x_pos[i] + offset_violin], widths=width, showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                # Get the center
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                # Make the violin plot half (left side only)
                pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf) # have distribution point to the other direction: -np.inf, m
                pc.set_facecolor(cmap[i])
                pc.set_alpha(0.8)

            # Box plot
            bp = ax.boxplot(y, positions=[x_pos[i] + offset_box], widths=width * 0.6, patch_artist=True)
            for element in ['boxes', 'whiskers', 'caps', 'medians']:
                for line in bp[element]:
                    line.set_color('black')
            for patch in bp['boxes']:
                patch.set_facecolor(cmap[i])

            # Scatter plot
            jitter = np.random.normal(loc=0, scale=0.03, size=len(y))
            ax.scatter(np.full_like(y, x_pos[i] + offset_scatter) + jitter, y, color=cmap[i], alpha=0.7, s=20)

        # Final touches
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(param)
        ax.set_title(f'{param}-parameter by Prior Condition')
        plt.tight_layout()

        # save and close
        plt.savefig(os.path.join(ddm_dir, f"{param}_raincloud.png"))
        plt.close()

    utils.log_msg(f'        Raincloud-plots saved to {ddm_dir}')

def paired_plot(df, condition_dict, parameters, bidspath_out):
    '''
    paired plot for each DDM parameter
    '''
    
    # Conditions and ordering
    cond_col, conditions = list(condition_dict.items())[0]
    # guarantee that 'base' is in the middle
    conditions = [cond for cond in conditions if cond != 'base']
    conditions = [conditions[0], 'base', conditions[1]]
    x_map = {cond: i for i, cond in enumerate(conditions)}
    df['x'] = df[cond_col].map(x_map)

    # saving directory
    ddm_dir = os.path.join(bidspath_out, 'groupDDM_SDT/paired')
    os.makedirs(ddm_dir, exist_ok=True)

    # Get unique participants and color map
    cmap = cm.get_cmap('viridis', len(df['participant'].unique()))  # Discrete colors

    # iterate through DDM-parameters
    for param in parameters:
        # Set up plot
        plt.figure(figsize=(8, 6))

        # iterate through subjects
        for i, (subject, df_sub) in enumerate(df.groupby('participant')):
            df_sub = df_sub.sort_values('x')

            # Add jitter to x values
            jitter = np.random.normal(loc=0, scale=0.03, size=len(df_sub))
            x_jittered = df_sub['x'].astype(float).values + jitter

            # Assign a color
            color = cmap(i)

            # Plot line and scatter
            plt.plot(x_jittered, df_sub[param], marker='o', linestyle='-', color=color, alpha=0.8, label=subject)

        # Axis formatting
        plt.xticks(ticks=range(len(conditions)), labels=conditions)
        plt.xlabel("Prior Condition")
        plt.ylabel(f"{param}")
        plt.title(f"DDM-parameter - {param}")
        plt.grid(True, linestyle='--', alpha=0.3)

        # Optional: show legend if not too crowded
        # plt.legend(title="Participants")

        plt.tight_layout()

        # save
        plt.savefig(os.path.join(ddm_dir, f"{param}_pairedScatter.png"))
        plt.close()
    
    utils.log_msg(f'        Pair-plots saved to {ddm_dir}')

def tfr_plots_subjects(tfr_dict, condition_dict, fmin, fmax, tmin, tmax, eeg_dir, vmin=None, vmax=None):
    '''
    diagnostic plots for TFR data
    '''
    # extract condition column and conditions
    cond_col, conditions = list(condition_dict.items())[0]
    subjects = list(tfr_dict.keys())

    # define predict window and baseline
    baseline = [tmin, -0.1]

    # saving directory    # saving directory
    tfr_diag_dir = os.path.join(eeg_dir, 'diagnostic_plots') # Check again if this is the correct directory
    os.makedirs(tfr_diag_dir, exist_ok=True)

    #_________________Plot TFR by subject and condition_________________
    # Create subplot grid: subjects x conditions
    fig, axes = plt.subplots(len(subjects), len(conditions), 
                            figsize=(4*len(conditions), 3*len(subjects)))
    
    for i, subject in enumerate(subjects):
        for j, condition in enumerate(conditions):
            ax = axes[i, j] if len(subjects) > 1 else axes[j]
            
            if condition in tfr_dict[subject]:
                tfr = tfr_dict[subject][condition]
                tfr.plot(baseline=baseline,fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax-0.1,
                        cmap='RdBu_r',
                        #vmin=vmin, vmax=vmax,
                        axes=ax, show=False, colorbar=False, verbose=False)
                ax.set_title(f'{subject} - {condition}')
                ax.axhline(y=7, color='dodgerblue', linestyle='--', linewidth=1, alpha=0.8, label=f'Alpha Band (7-15Hz)')
                ax.axhline(y=15, color='dodgerblue', linestyle='--', linewidth=1, alpha=0.8)
                ax.axvline(x=0, color='magenta', linestyle='--', linewidth=1, alpha=0.8, label='Cue Onset')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{subject} - {condition} (No Data)')
    plt.tight_layout()

    # save
    plt.savefig(os.path.join(tfr_diag_dir, f"TFR_allSubjects_conditions.png"))
    plt.close()

    return fig

def grand_avg_tfr_plots(grand_avg_tfr, condition_dict, comparisons, fmin, fmax, tmin, tmax, alpha_freq_range, eeg_dir=None):
    """
    Compute grand average TFR across subjects for each condition.
    
    Returns
    -------
    dict
        Dictionary mapping each condition to its grand-averaged TFR object.
    """
    
    # define predict window and baseline
    # lower_limit = abs(epoch_min)
    # upper_limit = lower_limit + abs(epoch_max)
    # predict_window = [lower_limit, upper_limit]
    baseline = [tmin, -0.1]

    tfr_diag_dir = os.path.join(eeg_dir, 'diagnostic_plots') # Check again if this is the correct directory
    os.makedirs(tfr_diag_dir, exist_ok=True)

    # extract condition column and conditions
    cond_col, conditions = list(condition_dict.items())[0]

    #_________________Plot grand average TFR by condition_________________
    for condition in conditions:
        tfr = grand_avg_tfr[condition]
        print(tfr.times)
        fig = tfr.plot(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                baseline=baseline,
                cmap='RdBu_r',
                title=f'Grand Average TFR - {condition}',
                show=False, colorbar=True, verbose=False)[0]
                # add horizontal lines - alpha band
        ax = fig.axes[0]  # Get the main plot axis
        ax.axhline(y=alpha_freq_range[0], color='dodgerblue', linestyle='--', linewidth=1, alpha=0.8, label=f'Alpha Band (7-15Hz)')
        ax.axhline(y=alpha_freq_range[1], color='dodgerblue', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(x=0, color='magenta', linestyle='--', linewidth=1, alpha=0.8, label='Cue Onset')
        ax.legend(loc='upper right')
        outpath = os.path.join(tfr_diag_dir, f"GrandAvgTFR_{condition}.png")
        utils.log_msg(f"        Saving Grand Average TFR plot to {outpath}")
        plt.savefig(outpath)
        plt.close()


    #_________________Plot differenceTFR between conditions_________________
    for cond1, cond2 in comparisons:
    
        tfr1 = grand_avg_tfr[cond1]
        tfr2 = grand_avg_tfr[cond2]
        # Create a copy of the first TFR for the difference
        diff_tfr = grand_avg_tfr[cond1].copy()

        # Compute difference: cond1 - cond2
        diff_tfr.data = tfr1.data - tfr2.data

        # Update the title/comment to reflect the comparison
        diff_tfr.comment = f"Difference: {cond1} - {cond2}"

        fig = diff_tfr.plot(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                baseline=baseline,
                cmap='RdBu_r',
                title=f'{diff_tfr.comment}',
                show=False, colorbar=True, verbose=False)[0]
        # add horizontal lines - alpha band
        ax = fig.axes[0]  # Get the main plot axis
        ax.axhline(y=alpha_freq_range[0], color='dodgerblue', linestyle='--', linewidth=1, alpha=0.8, label=f'Alpha Band (7-15Hz)')
        ax.axhline(y=alpha_freq_range[1], color='dodgerblue', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(x=0, color='magenta', linestyle='--', linewidth=1, alpha=0.8, label='Cue Onset')
        ax.legend(loc='upper right')
        # save
        outpath = os.path.join(tfr_diag_dir, f"GrandAvgTFR_diff_{cond1}vs{cond2}.png")
        utils.log_msg(f"        Saving Grand Average TFR difference plot to {outpath}")
        plt.savefig(outpath)
        plt.close()

    return grand_avg_tfr

# ================= CBPT Utility Functions =====================
def plot_and_save_cbpt_results(
    data_1, data_2, freqs, times_ms, T_obs, clusters, cluster_p_values,
    cond1, cond2, channel, datatype, alpha, tfr_dir
):
    # Function to visualize and save CBPT results for a given condition pair
    # Generates: (1) T-stat map, (2) plots for significant clusters, (3) cluster info CSVs
    rows = []
    evoked_diff = data_1.mean(axis=0) - data_2.mean(axis=0)
    signs = np.sign(evoked_diff)

    T_obs_plot = np.full_like(T_obs, np.nan)
    for c, p in zip(clusters, cluster_p_values):
        if p < alpha:
            T_obs_plot[c] = T_obs[c] * signs[c]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(T_obs, extent=[times_ms[0], times_ms[-1], freqs[0], freqs[-1]],
              aspect='auto', origin='lower', cmap='gray')
    max_T = np.nanmax(np.abs(T_obs_plot))
    ax.imshow(T_obs_plot, extent=[times_ms[0], times_ms[-1], freqs[0], freqs[-1]],
              aspect='auto', origin='lower', cmap='RdBu_r', vmin=-max_T, vmax=max_T)
    ax.set(title=f"{cond1} vs. {cond2} - {channel} - {datatype}",
           xlabel="Time (ms)", ylabel="Frequency (Hz)")
    outpath = os.path.join(tfr_dir, f"{datatype}_{cond1}vs{cond2}_TstatMap.png")
    utils.log_msg(f"        Saving T-stat map to {outpath}")
    plt.savefig(outpath)
    plt.close()

    # For each cluster, extract summary stats and generate plot (irrespective of p-value)
    for idx, p_val in enumerate(cluster_p_values):
        cluster_mask = clusters[idx]
        cluster_data = np.where(cluster_mask, evoked_diff, np.nan)
        cluster_mean = np.nanmean(cluster_data)
        freq_idxs, time_idxs = np.where(cluster_mask)
        freq_range = freqs[freq_idxs.min()], freqs[freq_idxs.max()]
        time_range = times_ms[time_idxs.min()], times_ms[time_idxs.max()]
        cluster_T = np.where(cluster_mask, T_obs, 0)
        peak_idx = np.unravel_index(np.abs(cluster_T).argmax(), cluster_T.shape)
        peak_freq = freqs[min(peak_idx[0], len(freqs) - 1)]
        peak_time = times_ms[peak_idx[1]]
        
        # Calculate cluster size (number of points in cluster)
        cluster_size = np.sum(cluster_mask)
        # Calculate cluster statistic (sum of absolute T-values)
        cluster_statistic = np.sum(np.abs(cluster_T[cluster_mask]))

        rows.append({
            "subject": datatype,
            "comparison": f'{cond1}vs{cond2}',
            "cluster": idx,
            "p_value": round(p_val, 4),
            "significant": p_val < alpha,  # Boolean flag for significance
            "mean_value": round(cluster_mean, 4),
            "cluster_statistic": round(cluster_statistic, 4),  # Sum of |T| values
            "cluster_size": cluster_size,  # Number of points in cluster
            "freq_range": f"{freq_range[0]:.1f}-{freq_range[1]:.1f}",
            "time_range_ms": f"{time_range[0]:.1f}-{time_range[1]:.1f}",
            "peak_freq": round(peak_freq, 1),
            "peak_time_ms": round(peak_time, 1),
        })
        
        # Save cluster heatmap only for significant clusters
        if p_val < alpha:
            fig, ax = plt.subplots()
            ax.imshow(cluster_data, aspect='auto', origin='lower',
                      extent=[times_ms[0], times_ms[-1], freqs[0], freqs[-1]],
                      cmap='RdBu_r')
            ax.set(title=f"{cond1} vs {cond2} - Cluster #{idx} - {datatype}",
                   xlabel="Time (ms)", ylabel="Frequency (Hz)")
            plt.colorbar(ax.images[0], ax=ax, label='Power Difference')
            outpath = os.path.join(tfr_dir, f"{datatype}_{cond1}vs{cond2}_Cluster{idx}.png")
            utils.log_msg(f"        Saving Cluster #{idx} plot to {outpath}")
            plt.savefig(outpath)
            plt.close()

    # Convert rows to DataFrame and save using pandas
    if rows:
        df_clusters = pd.DataFrame(rows)
        
        # Save per-subject/per-group cluster summary
        subject_csv = os.path.join(tfr_dir, f"{datatype}_{cond1}vs{cond2}_summary.csv")
        df_clusters.to_csv(subject_csv, index=False)
        utils.log_msg(f"        CBPT-results saved to {subject_csv}")
        
        # Append to master CSV (across all subjects) using pandas
        master_csv = os.path.join(tfr_dir, f"all_subjects_cluster_summary.csv")
        if os.path.exists(master_csv):
            # Append mode: read existing, concatenate, save
            df_existing = pd.read_csv(master_csv)
            df_combined = pd.concat([df_existing, df_clusters], ignore_index=True)
            df_combined.to_csv(master_csv, index=False)
        else:
            # Create new file
            df_clusters.to_csv(master_csv, index=False)
    else:
        utils.log_msg(f"        No clusters found for {cond1} vs {cond2}")

    return rows

# ================= FOOOF Utility Functions =====================
def plot_full_fooof_model_detailed(fm, subject, condition, save_path, log_log=True):

    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    init_flat_spec = fm.power_spectrum - init_ap_fit

    fig, axs = plt.subplots(3, 2, figsize=(20, 12))  
    axs = axs.flatten()

    summary_text = (
        f"R²: {fm.r_squared_:.3f}\n"
        f"Error: {fm.error_:.3f}\n"
        f"# Peaks: {len(fm.peak_params_)}\n"
        f"Aperiodic Offset: {fm.aperiodic_params_[0]:.2f}\n"
        f"Aperiodic Exponent: {fm.aperiodic_params_[1]:.2f}"
    )
    axs[2].axis('off')
    axs[2].text(0.05, 0.9, "FOOOF Model Summary:", fontsize=12, fontweight='bold')
    axs[2].text(0.05, 0.75, summary_text, fontsize=11, verticalalignment='top')
    axs[2].set_title("Model Summary")


    plot_spectra(fm.freqs, fm.power_spectrum, log_log, label='Raw Power Spectrum', color='black', ax=axs[0])
    plot_spectra(fm.freqs, init_ap_fit, log_log, label='Initial Aperiodic Fit', linestyle='--', color='blue', alpha=0.5, ax=axs[0])
    axs[0].set_title('Raw Spectrum + Initial Aperiodic Fit')
    axs[0].legend()

    plot_spectra(fm.freqs, init_flat_spec, log_log, label='Flattened Spectrum', color='gray', ax=axs[1])
    axs[1].set_title('Flattened Spectrum')
    axs[1].legend()

    # Save standalone: iterative peak search
    fig_iter = plt.figure(figsize=(10, 6))
    plot_annotated_peak_search(fm)
    plt.suptitle(f"Peak Search - {subject}, {condition}")
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_peaksearch.png'), dpi=300)
    plt.close(fig_iter)

    plot_spectra(fm.freqs, fm._peak_fit, log_log, label='Periodic (Peaks) Fit', color='green', ax=axs[3])
    axs[3].set_title('Final Periodic (Peak) Fit')
    axs[3].legend()

    plot_spectra(fm.freqs, fm._spectrum_peak_rm, log_log, label='Peak-Removed Spectrum', color='black', ax=axs[4])
    plot_spectra(fm.freqs, fm._ap_fit, log_log, label='Final Aperiodic Fit', linestyle='--', color='blue', alpha=0.5, ax=axs[4])
    axs[4].set_title('Peak-Removed Spectrum + Final Aperiodic')
    axs[4].legend()

    plot_spectra(fm.freqs, fm.power_spectrum, log_log, label='Raw Spectrum', color='black', ax=axs[5])
    plot_spectra(fm.freqs, fm.fooofed_spectrum_, log_log, label='Full Model Fit', linestyle='-.', color='red', ax=axs[5])
    axs[5].set_title('Raw vs Full Model')
    axs[5].legend()

    # Save standalone: shaded annotated model
    fig_peaks = plt.figure(figsize=(10, 6))
    fm.plot(plot_peaks='shade')
    plt.suptitle(f"FOOOF Annotated Model - {subject}, {condition}")
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_annotated.png'), dpi=300)
    plt.close(fig_peaks)

    fig.suptitle(f"FOOOF Full Model - Subject: {subject} | Condition: {condition}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def run_fooof_analysis(psd_dict, subject, channel, bidspath_out_subject,
                       global_master_csv=None, f_range=(1, 45), peak_threshold=0.1):
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
    channel : str
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

    df_sub_list = []

    for condition, data in psd_dict.items():
        freqs = data['freqs']
        spectrum = data['psd'].mean(axis=0)
        
        model_path = os.path.join(save_dir, f"sub-{subject}_{condition}_{channel}_fooof_model.json")
        fig_path_basic = os.path.join(save_dir, f"sub-{subject}_{condition}_{channel}_fooof_basic.png")
        fig_path_full = os.path.join(save_dir, f"sub-{subject}_{condition}_{channel}_fooof_full.png")
                    
        # ----- Fit model -----
        fm = FOOOF(aperiodic_mode='fixed', peak_threshold=peak_threshold,
                max_n_peaks=6, peak_width_limits=(1.5, 4))
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
        rows = []
        for idx, (cf, pw, bw) in enumerate(fm.peak_params_, start=1):
            rows.append({
                "Subject": subject,
                "Condition": condition,
                "Peak #": idx,
                "CF_Hz": cf,
                "PW_log10": pw,
                "PW_dB": 10 * pw,
                "BW_Hz": bw,
                "Aperiodic_Offset": fm.aperiodic_params_[0],
                "Aperiodic_Exponent": fm.aperiodic_params_[1],
                "R_squared": fm.r_squared_,
                "Error": fm.error_
            })

        df_cond = pd.DataFrame(rows).round(6)
        df_sub_list.append(df_cond)
    df_sub = pd.concat(df_sub_list, ignore_index=True)
    return fm, df_sub


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


# ================= HSSM posterior plotting =====================
# Visualizations of HSSM/DDM Bayesian posteriors, kept here so e_HSSM_module.py stays
# clean. Templates: Romei & Tarasi (2026) Fig 4C (posterior-coefficient histograms) and
# Franzen et al. (2025) Fig 5 (condition ridgelines), plus a drift-diffusion schematic
# adapted (and corrected) from a conference colleague's DDM_Plots_functions.py.
# All functions take the ArviZ InferenceData (model._inference_obj) and save into result_dir.

def _hssm_draws(idata, var, coord=None):
    """Flatten posterior draws for one variable into a 1-D array. Selecting a level by
    its coord *value* (not the auto-generated dim name like 'v_exp_dim') keeps these
    plots robust if the formulas change."""
    da = idata.posterior[var]
    if coord is not None:
        extra = [d for d in da.dims if d not in ('chain', 'draw')][0]
        da = da.sel({extra: coord})
    return np.asarray(da.values).reshape(-1)


def _hssm_expit(x):
    """Inverse of HSSM's generalized-logit link for bounds (0, 1) -- the logistic."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _hssm_mean_hdi(draws, hdi_prob=0.94):
    import arviz as az
    lo, hi = az.hdi(np.asarray(draws), hdi_prob=hdi_prob)
    return float(np.mean(draws)), float(lo), float(hi)


def _hssm_factor_levels(idata, offset_var):
    """Coord values present on a regression offset variable (the non-reference levels)."""
    if offset_var not in idata.posterior.data_vars:
        return []
    dim = [d for d in idata.posterior[offset_var].dims if d not in ('chain', 'draw')][0]
    return [str(c) for c in idata.posterior[offset_var][dim].values]


def _hssm_coefficient_vars(idata):
    """Discover 'slope-like' posterior coefficients worth a zero-crossing test -- skips
    intercepts, the bare scalar parameters (a/t/z), random-effect terms, and the
    non-centered offsets/sigmas, for which a line at 0 is not the question."""
    out = []
    for var in idata.posterior.data_vars:
        name = str(var)
        if ('Intercept' in name or name.endswith('_sigma') or name.endswith('_offset')
                or '1|' in name or name in ('a', 't', 'z')):
            continue
        extra = [d for d in idata.posterior[name].dims if d not in ('chain', 'draw')]
        if extra:
            for lvl in [str(c) for c in idata.posterior[name][extra[0]].values]:
                out.append((name, lvl, f'{name}[{lvl}]'))
        else:
            out.append((name, None, name))
    return out


def hssm_trace(idata, result_dir, fname='hssm_trace.png'):
    """Readable convergence trace. The default az.plot_trace crams every variable with no
    spacing, so titles collide with the axis above. Here we scale the figure to the row
    count, drop the redundant non-centered '_offset' duplicates, add row spacing, and
    shrink the long variable titles."""
    import arviz as az
    var_names = [str(v) for v in idata.posterior.data_vars if not str(v).endswith('_offset')]
    n = max(len(var_names), 1)
    axes = az.plot_trace(idata, var_names=var_names, compact=True,
                         figsize=(12, max(2.4 * n, 4)))
    fig = axes.ravel()[0].figure
    for ax in axes.ravel():
        ax.set_title(ax.get_title(), fontsize=9)
    fig.subplots_adjust(hspace=0.9)
    path = os.path.join(result_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    utils.log_msg(f'        Trace plot: {path}')
    return path


def hssm_posterior_coefficients(idata, result_dir, coefficients=None,
                                fname='hssm_posterior_coefficients.png', hdi_prob=0.94):
    """Romei & Tarasi (2026) Fig 4C style: one histogram per regression coefficient's
    posterior samples, with a dashed reference line at 0. Each panel is annotated with
    the posterior mass on the opposite side of 0 (the Bayesian 'probability of direction'
    q-value) and the HDI -- the narrower that tail, the more credible the effect."""
    import arviz as az
    if coefficients is None:
        coefficients = _hssm_coefficient_vars(idata)
    if not coefficients:
        utils.log_msg('        (no regression coefficients to plot)')
        return None

    n = len(coefficients)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')

    for ax, (var, coord, label) in zip(axes.ravel(), coefficients):
        d = _hssm_draws(idata, var, coord)
        q = float(min((d > 0).mean(), (d < 0).mean()))
        lo, hi = az.hdi(d, hdi_prob=hdi_prob)
        ax.hist(d, bins=40, color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.3)
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2)
        ax.axvspan(lo, hi, color='steelblue', alpha=0.12)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('coefficient')
        ax.set_ylabel('count')
        ax.text(0.04, 0.96, f'q = {q:.3f}\n{int(hdi_prob * 100)}% HDI\n[{lo:.3f}, {hi:.3f}]',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    fig.suptitle('HSSM posterior regression coefficients', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(result_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    utils.log_msg(f'        Posterior coefficient plot: {path}')
    return path


def hssm_posterior_ridgeline(idata, param, factor, levels, result_dir,
                             link='identity', fname=None, hdi_prob=0.94):
    """Franzen et al. (2025) Fig 5 style: stacked posterior densities (a ridgeline) of one
    DDM parameter across the levels of one factor. Per-condition draws are reconstructed
    from the regression: reference level = the parameter's intercept; each other level =
    intercept + that level's offset (other covariates held at their centered mean of 0).
    For a bounded parameter pass link='logit' to map back to the natural [0, 1] scale."""
    from scipy.stats import gaussian_kde

    intercept = _hssm_draws(idata, f'{param}_Intercept')
    offset_var = f'{param}_{factor}'
    coords = _hssm_factor_levels(idata, offset_var)
    has_offset = bool(coords)
    ref = next((l for l in levels if l not in coords), levels[0])

    inv = _hssm_expit if link == 'logit' else (lambda v: v)
    per_cond = {}
    for lvl in levels:
        lin = intercept if (lvl == ref or not has_offset) else intercept + _hssm_draws(idata, offset_var, lvl)
        per_cond[lvl] = inv(lin)

    cmap = ['mediumblue', 'darkorchid', 'magenta', 'teal', 'darkorange']
    fig, ax = plt.subplots(figsize=(8, 1.4 * len(levels) + 1.5))
    for i, lvl in enumerate(levels):
        d = per_cond[lvl]
        kde = gaussian_kde(d)
        xs = np.linspace(d.min(), d.max(), 200)
        dens = kde(xs)
        ys = dens / dens.max() * 0.9
        base = float(i)
        ax.fill_between(xs, base, base + ys, color=cmap[i % len(cmap)], alpha=0.7)
        ax.plot(xs, base + ys, color='black', linewidth=0.6)
        m = float(np.mean(d))
        hm = float(kde(m)[0] / dens.max() * 0.9)
        ax.plot([m, m], [base, base + hm], color='black', linewidth=1.0)
        ax.text(xs.min(), base + 0.05, f'  {lvl}', va='bottom', ha='left', fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel(f'{param}' + (' (natural [0,1] scale)' if link == 'logit' else ''))
    ax.set_title(f'{param} posterior by {factor}')
    fig.tight_layout()
    fname = fname or f'hssm_ridgeline_{param}_by_{factor}.png'
    path = os.path.join(result_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    utils.log_msg(f'        Ridgeline plot: {path}')
    return path


def hssm_ddm_schematic(idata, condition_dict, result_dir, t_value=None,
                       fname='hssm_ddm_schematic.png', hdi_prob=0.94):
    """Drift-diffusion 'anatomy' cartoon per condition (adapted & corrected from a
    conference colleague's DDM_Plots_functions.py): boundary separation a (top) and 0
    (bottom), the start point, and the mean drift slope v, each with its posterior HDI.
    Drift v is shown per condition (from condition_dict); a and the start point are shared
    across conditions in the current model (z is modulated by prior, not the drift factor),
    so they repeat. Fixes vs the original: HSSM z is RELATIVE in [0,1], so the start point
    is drawn at z*a (not z); HDIs come from real posterior draws, not a shifted intercept."""
    cond_col, conditions = list(condition_dict.items())[0]

    v_ic = _hssm_draws(idata, 'v_Intercept')
    offset_var = f'v_{cond_col}'
    coords = _hssm_factor_levels(idata, offset_var)
    has_off = bool(coords)
    ref = next((c for c in conditions if c not in coords), conditions[0])
    v_by = {c: (v_ic if (c == ref or not has_off) else v_ic + _hssm_draws(idata, offset_var, c))
            for c in conditions}

    if 'a' in idata.posterior.data_vars:
        a_m, a_lo, a_hi = _hssm_mean_hdi(_hssm_draws(idata, 'a'), hdi_prob)
    elif 'a_Intercept' in idata.posterior.data_vars:
        a_m, a_lo, a_hi = _hssm_mean_hdi(np.exp(_hssm_draws(idata, 'a_Intercept')), hdi_prob)
    else:
        a_m, a_lo, a_hi = 1.0, 1.0, 1.0

    if 'z_Intercept' in idata.posterior.data_vars:
        z_rel = float(np.mean(_hssm_expit(_hssm_draws(idata, 'z_Intercept'))))
    elif 'z' in idata.posterior.data_vars:
        z_rel = float(np.mean(_hssm_draws(idata, 'z')))
    else:
        z_rel = 0.5
    z_abs = z_rel * a_m

    if t_value is not None:
        t_m = float(t_value)
    elif 't' in idata.posterior.data_vars:
        t_m = float(np.mean(_hssm_draws(idata, 't')))
    else:
        t_m = 0.0

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    cmap = ['mediumblue', 'darkorchid', 'magenta', 'teal', 'darkorange']
    xmax = 1.0
    x0 = min(t_m, 0.5 * xmax)  # keep the (a.u.) non-decision band inside the window
    for ax, cond, color in zip(axes.ravel(), conditions, cmap):
        v_m, v_lo, v_hi = _hssm_mean_hdi(v_by[cond], hdi_prob)
        ax.axhline(a_m, color='firebrick', linewidth=2)
        ax.axhspan(a_lo, a_hi, color='firebrick', alpha=0.12)
        ax.axhline(0.0, color='royalblue', linewidth=2)
        ax.plot([x0], [z_abs], 'ko', ms=7)
        span = xmax - x0
        ax.plot([x0, xmax], [z_abs, z_abs + v_m * span], color=color, linewidth=2)
        ax.plot([x0, xmax], [z_abs, z_abs + v_lo * span], color=color, linewidth=0.7, linestyle='--')
        ax.plot([x0, xmax], [z_abs, z_abs + v_hi * span], color=color, linewidth=0.7, linestyle='--')
        if t_m:
            ax.axvspan(0, x0, color='gray', alpha=0.12)
        ax.set_xlim(0, xmax)
        ax.set_ylim(min(0, a_lo) - 0.1 * a_m, max(a_m, a_hi) + 0.1 * a_m)
        ax.set_title(f'{cond_col} = {cond}', fontsize=10)
        ax.set_xlabel('time (a.u.)')
        ax.set_ylabel('evidence')
        ax.text(0.02, a_m, f'a={a_m:.2f}', va='bottom', fontsize=8, color='firebrick')
        ax.text(x0, z_abs, f' z={z_rel:.2f}', va='bottom', fontsize=8)
        ax.text(x0 + span * 0.5, z_abs + v_m * span * 0.5, f'v={v_m:.2f}', fontsize=8, color=color)
        if t_m:
            ax.text(x0 / 2 if x0 else 0.0, a_m * 0.5, f't={t_m:.2f}', fontsize=7, ha='center', color='gray')
    fig.suptitle('HSSM drift-diffusion structure by condition', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(result_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    utils.log_msg(f'        DDM schematic: {path}')
    return path