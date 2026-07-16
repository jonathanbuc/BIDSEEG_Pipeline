# _____________________________ICA_preprocessing_module______________________________
# run with
# python a_ica_preprocessing_module.py inputs.json
#
# Step 1 of the Hülsemann et al. (2025) two-stage ICA approach:
#   - Preprocess data specifically for ICA decomposition
#   - Fit extended infomax ICA and classify components with ICLabel
#   - Save ICA weight matrix and exclusion metadata for step 2
#
# Step 2 is handled by b_ArtifactCorrection_ica_module.py, which runs the
# standard analysis preprocessing (module a) and applies the saved weights.
# _______________________________________________________________________________


# _____________________________Imports___________________________________________
import sys
import os
import json
import shutil
import warnings

import mne
import numpy as np
import utils_module as utils
from mne_icalabel import label_components
from autoreject import Ransac

from a_preprocessing_module import diagnostic_plots, down_sample
# _______________________________________________________________________________


# _____________________________Functions_________________________________________
def ica_filter(raw, low_cutoff, high_cutoff, filter_method, log_df):
    '''Band-pass filter continuous data for ICA decomposition.'''
    raw_filt = raw.copy()
    raw_filt.filter(
        l_freq=low_cutoff,
        h_freq=high_cutoff,
        method=filter_method,
        phase='zero',
        n_jobs=4,
        verbose=False,
    )
    utils.log_msg(
        f"        ICA prep: FIR band-pass filtered at {low_cutoff} - {high_cutoff} Hz ({filter_method})"
    )
    raw_filt.info['description'] = f'#ICAprep_filter({low_cutoff}Hz - {high_cutoff}Hz)'
    utils.log_update(log_df, 'ica_prep_low_cutoff', low_cutoff)
    utils.log_update(log_df, 'ica_prep_high_cutoff', high_cutoff)
    return raw_filt, log_df


def detect_bad_channels_neighbor_corr(epochs, threshold, log_df):
    '''
    Reject channels correlating below threshold with neighboring channels.
    Implements the clean_artefacts channel criterion from Hülsemann et al. (2025).
    '''
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    ch_names = [epochs.ch_names[p] for p in picks]
    data = epochs.get_data(picks=picks).mean(axis=0)

    try:
        adjacency, adj_ch_names = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')
    except RuntimeError:
        utils.log_msg("        ICA prep: adjacency not found, skipping neighbor-correlation bad-channel detection")
        return [], log_df

    ch_to_idx = {ch: i for i, ch in enumerate(adj_ch_names)}
    adjacency = adjacency.tocoo()
    bads = []

    for ch in ch_names:
        if ch not in ch_to_idx:
            continue
        i = ch_to_idx[ch]
        neighbors = adjacency.row[(adjacency.col == i) & (adjacency.row != i)]
        if len(neighbors) == 0:
            continue
        neighbor_signal = data[neighbors].mean(axis=0)
        r = np.corrcoef(data[i], neighbor_signal)[0, 1]
        if np.isnan(r) or r < threshold:
            bads.append(ch)

    utils.log_msg(f"        ICA prep: {len(bads)} bad channels (neighbor r < {threshold}): {bads}")
    utils.log_update(log_df, 'ica_prep_bad_channels', bads)
    utils.log_update(log_df, 'ica_prep_n_bad_channels', len(bads))
    return bads, log_df


def badChannels(epochs, rd_state, log_df):
    '''
    Estimates bad channels on epoched data for masking before ICA.
    Implements RAndom SAmple Consensus (RANSAC) to detect bad sensors.
    '''
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False)

    utils.log_msg("        ICA prep: fitting RANSAC to epochs...")
    ransac = Ransac(picks=picks, random_state=rd_state, n_jobs=4, verbose=False)
    ran_fit = ransac.fit(epochs)

    bad_chs = list(ran_fit.bad_chs_)
    utils.log_msg(f"        ICA prep: {len(bad_chs)} channels marked as bad by RANSAC: {bad_chs}")
    utils.log_update(log_df, 'ica_prep_ransac_bad_channels', bad_chs)
    utils.log_update(log_df, 'ica_prep_ransac_n_bad_channels', len(bad_chs))
    return bad_chs, ran_fit, log_df


def ica_rereference(raw, rereference, drop_ref_channel, log_df):
    '''Average-reference continuous data for ICA; optionally drop online reference electrode.'''
    raw_ref = raw.copy()
    if rereference:
        raw_ref.set_eeg_reference(ref_channels=rereference, projection=False, verbose=False)
        utils.log_msg(f"        ICA prep: re-referenced to {rereference}")

    raw_ref.info['description'] = '#ICAprep_rereferenced'
    utils.log_update(log_df, 'ica_prep_rereference', rereference)
    return raw_ref, log_df


def ica_create_epochs(raw, epoch_dict, tmin, tmax, log_df):
    '''Create epochs for ICA from annotated events.'''
    if set(epoch_dict.keys()).intersection(raw.annotations.description):
        events, _ = mne.events_from_annotations(raw, event_id=epoch_dict, verbose=False)
        epochs = mne.Epochs(
            raw, events, event_id=epoch_dict, tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False,
        )
    utils.log_msg(f"        ICA prep: created {len(epochs)} consecutive {tmax - tmin}s epochs")
    utils.log_update(log_df, 'ica_prep_n_epochs', len(epochs))
    epochs.info['description'] = f'#ICAprep_epochs({tmax - tmin}s)'
    return epochs, log_df


def ica_reject_epochs(epochs, reject_uv, log_df):
    '''Reject artefactual epochs using amplitude threshold (±1000 µV).'''
    epochs_clean = epochs.copy()
    n_before = len(epochs_clean)
    reject = dict(eeg=reject_uv * 1e-6)
    flat = dict(eeg=1e-6)
    epochs_clean.drop_bad(reject=reject, flat=flat, verbose=False)
    n_rejected = n_before - len(epochs_clean)
    pct_rejected = round(n_rejected / n_before * 100, 2) if n_before else 0.0

    utils.log_msg(
        f"        ICA prep: rejected {n_rejected}/{n_before} epochs ({pct_rejected}%) "
        f"with amplitude > {reject_uv} µV"
    )
    utils.log_update(log_df, 'ica_prep_epochs_rejected', n_rejected)
    utils.log_update(log_df, 'ica_prep_epochs_rejected_pct', pct_rejected)
    epochs_clean.info['description'] = '#ICAprep_epochs_rejected'
    return epochs_clean, log_df


def ica_fit(epochs, n_components, max_iter, rd_state, method, log_df):
    '''Fit extended infomax ICA on ICA-preprocessed epochs.'''
    epochs_ica = epochs.copy()
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=rd_state,
        method=method,
        verbose=False,
    )
    if epochs_ica.info['bads']:
        utils.log_msg(f"        ICA prep: excluding bad channels from ICA fit: {epochs_ica.info['bads']}")
    else:
        utils.log_msg("        ICA prep: no bad channels excluded for ICA fit")

    utils.log_msg(
        f"        ICA prep: fitting ICA ({n_components} components, {method}) "
        f"on {len(epochs_ica)} epochs"
    )
    ica.fit(epochs_ica, verbose=False)

    utils.log_update(log_df, 'ica_prep_n_components', n_components)
    utils.log_update(log_df, 'ica_prep_max_iter', max_iter)
    utils.log_update(log_df, 'ica_prep_method', method)
    return ica, log_df


def ica_label_components(epochs, ica, reject_labels, rej_thresholds, bidspath_processing, log_df):
    '''
    Classify ICs with ICLabel, save diagnostic IC plots, and determine exclusion list.
    Hülsemann et al. (2025): eye > 25%, muscle > 90%, heart > 60%.
    '''
    ic_labels = label_components(epochs, ica, method="iclabel")
    labels = list(ic_labels["labels"])
    prob = list(ic_labels["y_pred_proba"])

    # identify artifact and brain components (same logic as module b)
    exclude_dict = {idx: label for idx, label in enumerate(labels) if label in reject_labels}
    exclude_dict = {idx: label.replace(' ', '-') for idx, label in exclude_dict.items()}
    include_dict = {idx: label for idx, label in enumerate(labels) if label not in reject_labels}
    include_dict = {idx: label.replace(' ', '-') for idx, label in include_dict.items()}

    # save diagnostic plots to BIDSprocessed/ICA/ica_prep
    ica_dir = os.path.join(bidspath_processing.directory, 'ICA')
    if os.path.exists(ica_dir):
        shutil.rmtree(ica_dir)
    os.makedirs(ica_dir)

    artIC_dir = os.path.join(ica_dir, 'artefactICs')
    os.makedirs(artIC_dir)
    brainIC_dir = os.path.join(ica_dir, 'brainICs')
    os.makedirs(brainIC_dir)

    exclude_keys = list(exclude_dict.keys())
    include_keys = list(include_dict.keys())
    artIC_plot = []
    brainIC_plot = []

    if exclude_keys:
        artIC_plot = ica.plot_properties(epochs, picks=exclude_keys, show=False, verbose=False)
    if include_keys:
        brainIC_plot = ica.plot_properties(epochs, picks=include_keys, show=False, verbose=False)

    for i, fig in enumerate(artIC_plot):
        idx = exclude_keys[i]
        fig.savefig(
            f"{artIC_dir}/IC{idx}_{exclude_dict[idx]}_Prob{prob[idx]:.4f}.png",
            format="png", dpi=300,
        )
        fig.clf()
    for i, fig in enumerate(brainIC_plot):
        idx = include_keys[i]
        fig.savefig(
            f"{brainIC_dir}/IC{idx}_{include_dict[idx]}_Prob{prob[idx]:.4f}.png",
            format="png", dpi=300,
        )
        fig.clf()

    # only keep artefact labels above their label-specific rejection threshold
    # and among the first 10 ICA components (idx 0-9) — same logic as module b
    max_reject_idx = 10
    default_threshold = rej_thresholds.get('default', 0.9)
    exclude_tmp = exclude_dict.copy()
    for idx in exclude_tmp.keys():
        label_threshold = rej_thresholds.get(labels[idx], default_threshold)
        if prob[idx] < label_threshold or idx >= max_reject_idx:
            exclude_dict.pop(idx, None)

    utils.log_msg(
        f"        ICA prep: {len(exclude_dict)} ICs exceed their label-specific artifact "
        f"thresholds ({rej_thresholds}) and are marked for removal:"
    )
    utils.log_msg(f"            {exclude_dict}")
    utils.log_update(log_df, 'ica_prep_ics_removed', exclude_dict)
    utils.log_update(log_df, 'ica_prep_n_ics_removed', len(exclude_dict))
    utils.log_update(log_df, 'ica_prep_ic_removal_thresholds', rej_thresholds)
    utils.log_update(log_df, 'ica_prep_label_ics_removed', list(exclude_dict.values()))
    utils.log_update(log_df, 'ica_prep_idx_ics_removed', list(exclude_dict.keys()))
    prob_arr = np.array(prob)
    if exclude_dict:
        utils.log_update(log_df, 'ica_prep_prob_ics_removed', prob_arr[list(exclude_dict.keys())].tolist())
    else:
        utils.log_update(log_df, 'ica_prep_prob_ics_removed', [])

    metadata = {
        'exclude_indices': list(exclude_dict.keys()),
        'exclude_labels': list(exclude_dict.values()),
        'ic_labels': labels,
        'ic_probs': [float(p) for p in prob],
        'rej_thresholds': rej_thresholds,
    }
    return exclude_dict, metadata, log_df


def save_ica_metadata(metadata, bad_channels, bidspath_processing, subject):
    '''Save ICA exclusion metadata as JSON alongside the ICA .fif weights.'''
    metadata = metadata.copy()
    metadata['bad_channels'] = bad_channels

    bids_path_step = bidspath_processing.copy().update(processing='00ICAweights', subject=subject)
    filepath = f"{bids_path_step.directory}/{bids_path_step.basename}_metadata.json"
    os.makedirs(bids_path_step.directory, exist_ok=True)
    with open(filepath, 'w') as jsonfile:
        json.dump(metadata, jsonfile, indent=4)
    utils.log_msg(f"        ICA prep: saved metadata to {filepath}")
    return filepath


def load_ica_metadata(bidspath_processing, subject):
    '''Load ICA exclusion metadata saved by a_ica_preprocessing_module.'''
    bids_path_step = bidspath_processing.copy().update(processing='00ICAweights', subject=subject)
    filepath = f"{bids_path_step.directory}/{bids_path_step.basename}_metadata.json"
    with open(filepath) as jsonfile:
        return json.load(jsonfile)


def load_pretrained_ica(bidspath_processing, subject):
    '''Load ICA weight matrix saved by a_ica_preprocessing_module.'''
    bids_path_step = bidspath_processing.copy().update(processing='00ICAweights', subject=subject)
    filepath = f"{bids_path_step.directory}/{bids_path_step.basename}.fif"
    utils.log_msg(f"        Loading pretrained ICA weights from {filepath}")
    return mne.preprocessing.read_ica(filepath, verbose=False)


def apply_pretrained_ica(epochs, ica, exclude_indices, bad_channels, log_df):
    '''Apply saved ICA weights to analysis-preprocessed epochs and remove artefact ICs.'''
    epochs_out = epochs.copy()
    if bad_channels:
        epochs_out.info['bads'] = list(bad_channels)
        utils.log_msg(f"        Masking ICA-prep bad channels before apply: {bad_channels}")

    ica.apply(epochs_out, exclude=list(exclude_indices), verbose=False)
    epochs_out.info['bads'] = []

    utils.log_msg(f"        Applied pretrained ICA, removed {len(exclude_indices)} components: {exclude_indices}")
    utils.log_update(log_df, 'ica_pretrained_exclude_indices', exclude_indices)
    utils.log_update(log_df, 'ica_pretrained_n_ics_removed', len(exclude_indices))
    epochs_out.info['description'] = '#6ICAcorrected_pretrained'
    return epochs_out, ica, log_df


def ica_preprocessing_subject(subject, log_df):
    '''Run ICA preprocessing for a single subject.'''
    warnings.filterwarnings("ignore")

    bidspath_processing_subject = bidspath_out.copy().update(subject=subject)

    if not perform_ica_prep:
        utils.log_msg("     -- ICA preprocessing disabled in inputs.json")
        return log_df

    raw = utils.load_preprocessing_step(bidspath_in, subject, 'from_bids')
    raw.info['description'] = '#0.1_raw'
    diagnostic_plots(raw, bidspath_processing_subject)

    # 1) Band-pass filter
    raw_step, log_df = ica_filter(raw, low_cutoff, high_cutoff, filter_method, log_df)
    diagnostic_plots(raw_step, bidspath_processing_subject)

    # 2) Downsample data
    raw_step, log_df = down_sample(raw_step, downsample_to, log_df)
    diagnostic_plots(raw_step, bidspath_processing_subject)

    # 3) Create epochs
    epochs, log_df = ica_create_epochs(raw_step, epoch_dict, tmin, tmax, log_df)

    # 4) Detect bad channels; ransac or neighbor correlation
    match ch_interpolation_method:
        case 'neighbor_corr':
            bad_channels, log_df = detect_bad_channels_neighbor_corr(
                epochs, neighbor_corr_threshold, log_df
            )
            epochs.info['bads'] = bad_channels
        case 'ransac':
            bad_channels, _, log_df = badChannels(epochs, rd_state, log_df)
            raw_step.info['bads'] = bad_channels
        case 'none':
            utils.log_msg('     -- ICA prep: no bad-channel detection')
            bad_channels = []
    
    # 5) Re-reference (on continuous data)
    raw_step, log_df = ica_rereference(raw_step, rereference, drop_ref_channel, log_df)
    diagnostic_plots(raw_step, bidspath_processing_subject)

    # 6) Re-epoch
    epochs, log_df = ica_create_epochs(raw_step, epoch_dict, tmin, tmax, log_df)
    epochs.info['bads'] = bad_channels

    # 7) Reject epochs
    epochs, log_df = ica_reject_epochs(epochs, reject_uv, log_df)
    diagnostic_plots(epochs, bidspath_processing_subject)

    if len(epochs) == 0:
        utils.log_msg(f"      X ERROR: No epochs remaining after rejection for Subject-{subject}")
        return log_df

    # 8) Fit ICA
    ica, log_df = ica_fit(epochs, n_components, max_iter, rd_state, method, log_df)

    # 9) Label components
    _, metadata, log_df = ica_label_components(
        epochs, ica, reject_labels, rej_thresholds, bidspath_processing_subject, log_df
    )

    # 10) Save ICA weights and metadata
    utils.save_preprocessing_step(ica, '00ICAweights', bidspath_out, subject)
    save_ica_metadata(metadata, bad_channels, bidspath_out, subject)

    return log_df


# _______________________________________________________________________________


# _____________________________Loading___________________________________________
## load inputs
inputs = utils.read_inputs(sys.argv[1])

# BIDS path for in- and output data
bidspath_in = utils.get_bidspath(inputs)
bidspath_out = utils.get_bidspath(inputs, 'bids_proc')

# assign module variables
ica_cfg = inputs['ICA_preprocessing']
perform_ica_prep = ica_cfg['perform']
low_cutoff = ica_cfg['low_cutoff']
high_cutoff = ica_cfg['high_cutoff']
filter_method = ica_cfg['filter_method']
downsample_to = ica_cfg.get('downsample_to')
neighbor_corr_threshold = ica_cfg['neighbor_corr_threshold']
rereference = ica_cfg['rereference']
drop_ref_channel = ica_cfg.get('drop_ref_channel')
reject_uv = ica_cfg['reject_amplitude_uv']
n_components = ica_cfg['n_components']
max_iter = ica_cfg['max_iter']
method = ica_cfg['ICA_method']
rd_state = ica_cfg['random_state_seed']
rej_thresholds = ica_cfg['rej_thresholds']
reject_labels = list(rej_thresholds.keys())
ch_interpolation_method = ica_cfg['ch_interpolation_method']
epoch_dict = ica_cfg['epoch_dict']
tmin = ica_cfg['tmin']
tmax = ica_cfg['tmax']

# extract subject list
subjects = utils.find_subjects(bidspath_in.root)
# start at a given subject
# subjects = [sub for sub in subjects if int(sub) > 1]


if __name__ == '__main__':
    print(f'\n\n\n\n')
    timepoint_start = utils.log_msg('START:  ICA preprocessing')

    ## load log
    log_df = utils.log_load()

    # Loop through participants
    for subject in subjects:
        utils.log_msg(f"\n_______ ICA Preprocessing Subject-{subject}_______")
        utils.update_inputs(sys.argv[1], 'basic', 'subject_ID', subject)
        utils.update_inputs(sys.argv[1], 'basic', 'current_step', None)

        log_df = ica_preprocessing_subject(subject, log_df)

    utils.log_save(log_df, bidspath_out.root, 'log_dataframe.csv')

    timepoint_end = utils.log_msg('DONE:   ICA preprocessing')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end - timepoint_start)}\n\n')
