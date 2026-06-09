# _____________________________ArtifactCorrection_______________________________________
# run with
# python c_ArtifactCorrection_module.py inputs.json
#
# * Allgemeine und Biologische Psychologie - AG Hesselman
# * Psychologische Hochschule
#
# ## Author(s)
# * Buchholz, jonathan; Psychologische Hochschule Berlin, AG Hesselmann
# 
#
# * last update: 2025.02.05
#
#
# This script is provides MNE-based functions for EEG artifact rejection and correction, including interpolation methods like autoreject and Ransac as well as ICA 
#
#
# _______________________________________________________________________________



# _____________________________Imports___________________________________________
# basics
import sys
import utils_module as utils
import mne
import numpy as np
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil

# module specifics
from a_preprocessing_module import diagnostic_plots, rereferencing
from d_BehavAnalysis_module import behavdata_prep
from autoreject import AutoReject, Ransac, RejectLog
from mne_icalabel import label_components
from mne_icalabel.gui import label_ica_components


# _______________________________________________________________________________




'''
TODO: 
- 
'''


# _____________________________Functions_________________________________________
def epoching(raw, df, epoch_dict, tmin, tmax, log_df, baseline=None):
    """
    Perform epoching on continuous preprocessed EEG data and include behavioral metadata.

    Parameters
    ----------
    raw : mne.io.Raw
        The continuous, preprocessed EEG recording.
    df : pandas.DataFrame
        Behavioral data including trial-level metadata.
    epoch_dict : dict
        Dictionary mapping event labels to annotation strings.
    tmin : float
        Start time before event (in seconds).
    tmax : float
        End time after event (in seconds).
    log_df : pandas.DataFrame
        Log dataframe.
    baseline : tuple or None, optional
        The time interval to apply baseline correction (in seconds), or None.

    Returns
    -------
    mne.Epochs or None
        The epoched data with metadata, or None if specified events not found.
    """

    eeg_data = raw.copy().pick('eeg')

    if set(epoch_dict.keys()).intersection(eeg_data.annotations.description):
        events, _ = mne.events_from_annotations(eeg_data, event_id=epoch_dict, verbose=False)
        epochs = mne.Epochs(
            eeg_data, events, tmin=tmin, tmax=tmax, baseline=baseline,
            event_id=epoch_dict, preload=True, verbose=False, event_repeated='merge'
        )

        utils.log_msg(f"        epochs computed for the following events: {list(epoch_dict.keys())} with window = [{tmin}, {tmax}] and baseline correction: {baseline}")

        # all remaining stimuli are summarizes as cue (1)
        epochs.events[:, 2] = 1
        epochs.event_id = {'cue': 1}

        ### Include metadata from behavioral df
        analyzed_blocks = inputs['basic']['analyzed_blocks']
        metadata = df[df['block_cond'].isin(analyzed_blocks)].reset_index(drop=True)
        metadata.insert(0, 'trigger', epochs.events[:, 2])

        utils.log_msg(f"        Behavioral data is included as metadata for each epoch. Including: {list(metadata.columns)[:8]}")

        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                epochs.metadata = metadata
        

        epochs.info['description'] = '#5epoched'

        # logging
        utils.log_update(log_df, 'epoch_min', tmin)
        utils.log_update(log_df, 'epoch_max', tmax)
        return epochs, log_df

    return None
    
## Compute interpolation_mask for bad channels
def badChannels(epochs, rd_state, log_df):
    """
    Estimates bad channels on epoched data for post-ICA interpolation. Implements RAndom SAmple Consensus (RANSAC) method to detect bad sensors.

    Parameters
    ----------
    epochs : mne.Epochs 
        <br>The data split into arbitrary epochs.

    Returns
    -------
    ransac : instance of autoreject.Ransac 
        <br>The RANSAC object used to identify bad channels.
    epochs : instance mne.Epochs 
        <br>The epoched data.

    See Also
    -------
    interpolate_RANSAC()

    Examples
    --------
    >>> import sys
    >>> from utils_module import read_inputs, load_preprocessing_step, get_bidspath
    >>> from helper_module import arbitrary_epochs
    >>> inputs = utils.read_inputs(sys.argv[1])
    >>> # generate the BIDS path and load raw data from BIDS
    >>> bids_path_preprocessing = utils.get_bidspath(inputs, 'preprocessing')
    >>> raw_filt = utils.load_preprocessing_step(bids_path_preprocessing, 'from_bids', '06filter')
    >>> epochs, epoch_duration = arbitrary_epochs(raw_filt, epoch_duration)
    >>> # compute bad channels, ransac object and epoch the data
    >>> bad_chs, ransac = interpolation_mask(raw, bids_path) 

    Notes
    -----
    This function performs the following steps:
    3. Fits an AutoReject object to the epochs.
    4. Extracts the rejection log object holding overall bad epochs, bad channels, and channel names.
    5. Computes overall bad channels using RANSAC.
    6. Adds channel(s) identified as bad by RANSAC to info['bads'].
    7. Saves the rejection log array to a TSV file.
    """

    # pick channels types; not REALLY necessary
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False)

    ### compute overall bad channels using RANSAC
    utils.log_msg(f"        Fitting RANSAC to epochs...")
    ransac = Ransac(picks=picks, random_state = rd_state, n_jobs=4, verbose=False)
    ran_fit = ransac.fit(epochs)

    # logging
    bad_chs = list(ran_fit.bad_chs_)
    utils.log_msg(f"        {len(bad_chs)} channels marked as bad: {bad_chs}")
    utils.log_update(log_df, 'bad_chs', bad_chs)
    utils.log_update(log_df, 'n_bad_chs', len(bad_chs))

    # add channel(s) identified as bad by RANSAC to info['bads']
    epochs.info['bads'] = bad_chs

    return epochs, ran_fit, log_df

# Independent Component Analysis
def ICA_mne(epochs, n_components, max_iter, rd_state, method, log_df):
    
    epochs_ICA = epochs.copy()

    # create  ica object
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter=max_iter, random_state=rd_state, method=method, verbose = False)

    if not epochs.info['bads']:
        utils.log_msg(f'        No channels have been marked as bad therefore excluded for ICA.')
    else:
        utils.log_msg(f'        The following channels were excluded before computing ICA: {epochs_ICA.info["bads"]}')

    # fit ICA    
    utils.log_msg(f"        Fitting ICA ({n_components} components) using {method} method and {len(epochs_ICA.info['ch_names'])} channels")
    ica = ica.fit(epochs_ICA, verbose=False)
    
    #epochs_ICA.info['description'] = '_fitted_ICA'

    # remove bad channels from info[bads]
    epochs_ICA.info['bads'] = []

    # logging
    utils.log_update(log_df, 'n_components', n_components)
    utils.log_update(log_df, 'max_iter', max_iter)
    utils.log_update(log_df, 'method', method)

    return epochs_ICA, ica, log_df

def autoICLabel(epochs, ica, reject_labels, rej_threshold, bidspath_processing, log_df):
    '''
    Automatically identifies delete artifactual ICs before back-projection into electrode space.  
    '''
    # labelling all ICs
    ic_labels = label_components(epochs, ica, method="iclabel")
    labels = list(ic_labels["labels"])#artefact labels
    prob = list(ic_labels["y_pred_proba"])#certainty (probability) of respective label

    # identify artifact and brain components
    #idx(key) = Component indes, label (value) = Component label (e.g., eye blink)
    exclude_dict = {idx : label for idx, label in enumerate(labels) if label in reject_labels}
    exclude_dict = {idx: label.replace(' ','-') for idx, label in exclude_dict.items()}# replace spaces in labels with _

    include_dict = {idx : label for idx, label in enumerate(labels) if label not in reject_labels}
    include_dict = {idx: label.replace(' ','-') for idx, label in include_dict.items()}# replace spaces in labels with _



    ### save diagnostic plots to BIDSprocessed
    # create folder
    ica_dir = os.path.join(bidspath_processing.directory, 'ICA')
    if os.path.exists(ica_dir):
        shutil.rmtree(ica_dir)# delete folder and contents if it exists
    os.makedirs(ica_dir)

    # make subfolders
    artIC_dir = os.path.join(ica_dir, 'artefactICs')
    os.makedirs(artIC_dir)
    brainIC_dir = os.path.join(ica_dir, 'brainICs')
    os.makedirs(brainIC_dir)

    # compute diagnostic plots only if there are components to plot
    exclude_keys = list(exclude_dict.keys())
    include_keys = list(include_dict.keys())
    artIC_plot = []
    brainIC_plot = []

    if exclude_keys:
        artIC_plot = ica.plot_properties(epochs, picks=exclude_keys, show=False, verbose=False)
    if include_keys:
        brainIC_plot = ica.plot_properties(epochs, picks=include_keys, show=False, verbose=False)

    # iterate through artComp_plot (storing all plots) to individually save diagnostic plots
    for i, fig in enumerate(artIC_plot):
        idx = exclude_keys[i]
        fig.savefig(f"{artIC_dir}/IC{idx}_{exclude_dict[idx]}_Prob{prob[idx]:.4f}.png", format = "png", dpi=300)
        fig.clf()  # Clear each figure to free memory]
    # BRAIN components
    for i, fig in enumerate(brainIC_plot):
        idx = include_keys[i]
        fig.savefig(f"{brainIC_dir}/IC{idx}_{include_dict[idx]}_Prob{prob[idx]:.4f}.png", format = "png", dpi=300)
        fig.clf()  # Clear each figure to free memory

    # only keep artefact labels above rejection threshold (90%)
    exclude_tmp = exclude_dict.copy()
    for idx in exclude_tmp.keys():
        if prob[idx] < rej_threshold:
            exclude_dict.pop(idx, None)

    # remove artifactual components and project data back to electrode space
    reconst_epochs = epochs.copy()
    ica.apply(reconst_epochs, exclude=list(exclude_dict.keys()), verbose=False)

    # logging 
    utils.log_msg(f'        {len(exclude_dict)} ICs exceed artifact threshold ({rej_threshold}) and are removed before back-projection:')
    utils.log_msg(f'            {exclude_dict}')
    utils.log_update(log_df, 'n_ICs_removed', len(exclude_dict))
    utils.log_update(log_df, 'ICs_removed', exclude_dict)

    reconst_epochs.info['description'] = f'#6ICAcorrected' 


    # logging
    utils.log_update(log_df, 'ic_removal_method', str('automatic'))
    utils.log_update(log_df, 'ic_removal_threshold', rej_threshold)
    utils.log_update(log_df, 'n_ics_removed', len(exclude_dict))
    utils.log_update(log_df, 'label_ics_removed', list(exclude_dict.values()))
    utils.log_update(log_df, 'idx_ics_removed', list(exclude_dict.keys()))
    # Convert to numpy array to support advanced indexing with a list of indices
    # Check this again, AI generated code!!
    prob_arr = np.array(prob)
    utils.log_update(log_df, 'prob_ics_removed', prob_arr[list(exclude_dict.keys())].tolist())

    return reconst_epochs, ica, log_df


def manualICLabel(epochs, ica, bidspath_processing, log_df):
    '''
    Requires experimentor to manually identify artifactual ICs to delete them before back-projection into electrode space.  
    '''
    epochs_ICA = epochs.copy()

    # opens GUI to flag artifactual ICs
    # redirects standard output to temporary file
    # redirecting both standard output (stdout) and standard error messages (stderr) to temporary file (devnull)
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # open GUI to flag artifactual ICs 
                gui = label_ica_components(epochs_ICA, ica, show=True, block=True)

    # reject labels aligned with ica.labels_ object (deviant from labels returned by label_components())
    reject_labels = [
            "muscle",
            "eog",
            "ch_noise",
            "ecg",
            "line_noise"]
    
    #individually extracts idx and labels of artifactual components (defined by reject_labels) and save it as dict 
    exclude_dict = {
    idx: label for label in ica.labels_ if label in reject_labels for idx in ica.labels_[label]
    }

    # Sort the dictionary by keys
    exclude_dict = {key: exclude_dict[key] for key in sorted(exclude_dict)}

    ### save diagnostic plots to BIDSprocessed
    # create folder
    ic_prop_dir = os.path.join(bidspath_processing.directory, 'artefactICs')
    if os.path.exists(ic_prop_dir):
        shutil.rmtree(ic_prop_dir)# delete folder and contents if it exists
    os.makedirs(ic_prop_dir)

    # compute diagnostic plot for each artifactual IC and save as single .png
    artComp_plot = ica.plot_properties(epochs, picks=list(exclude_dict.keys()), show=False, verbose=False)

    # iterate through artComp_plot (storing all plots) to individualy save diagnostic plots for all artefactual ICs
    for i, fig in enumerate(artComp_plot):
        fig.savefig(f"{ic_prop_dir}/IC{list(exclude_dict.keys())[i]}_{list(exclude_dict.values())[i]}.png", format = "png", dpi=300)
        fig.clf()  # Clear each figure to free memory

    # remove artifactual components and project data back to electrode space
    reconst_epochs = epochs.copy()
    ica.apply(reconst_epochs, exclude=list(exclude_dict.keys()), verbose=False)

    # logging
    utils.log_msg(f'        {len(exclude_dict)} ICs were selected to be removed before back-projection:')
    utils.log_msg(f'            {exclude_dict}')
    utils.log_update(log_df, 'n_artICs', len(exclude_dict))
    utils.log_update(log_df, 'idx_artICs', list(exclude_dict.keys()))
    utils.log_update(log_df, 'label_artICs', list(exclude_dict.values()))

    reconst_epochs.info['description'] = f'#6ICAcorrected'

    # logging
    utils.log_update(log_df, 'ic_removal_method', str('manual'))
    utils.log_update(log_df, 'n_ics_removed', len(exclude_dict))
    utils.log_update(log_df, 'label_ics_removed', list(exclude_dict.values()))
    utils.log_update(log_df, 'idx_ics_removed', list(exclude_dict.keys()))

    return reconst_epochs, ica, log_df

# detect and remove bad epochs
def badEpochs(epochs, rd_state, bidspath_processing, log_df, epoch_rejection_type = 'auto'):
    
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False)

    # create autoreject object and fit it to data
    ar = AutoReject(picks=picks, random_state=rd_state, n_jobs=4, verbose=False)
    ar.fit(epochs[:40])#[:40] --> fitted/trained on how many epochs

    #____________________________ make actual interpolation____________________________
    epochs_inter = epochs.copy() 

    # perform interpolation 
    # redirects standard output to temporary file
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Interpolation
            epochs_clean_auto, transform_rej_log = ar.transform(epochs_inter, return_log=True) #, reject_log=rejection_log

    ### compute epoch measures
    # global epochs
    n_rejected_auto = np.count_nonzero(transform_rej_log.bad_epochs) #global bad epochs, which will be entirely removed 
    n_global_epochs = transform_rej_log.labels.shape[0]
    nrejected_ratio_auto = round((n_rejected_auto/n_global_epochs*100),2)

    match epoch_rejection_type:
        # automated epoch rejection with autoreject
        case 'auto':
            utils.log_msg(f"        Automated epoch rejection with Autoreject performed.")
            epochs_clean = epochs_clean_auto.copy()
            n_rejected = n_rejected_auto
            n_rejected_ratio = nrejected_ratio_auto
        # manual epoch rejection
        case 'manual':
            # if autoreject would have rejected more than 15% of trials, perform manual epoch rejection
            if nrejected_ratio_auto > 15: # do 

                utils.log_msg(f"        Autoreject would have rejected {nrejected_ratio_auto}% of trials. Manual epoch rejection performed instead.")
                
                # Perform manual epoch rejection in data browser
                epochs_inter.plot(block=True)

                # extract number of manually rejected epochs
                n_rejected_manual = sum('USER' in reason for reason in epochs_inter.drop_log)

                # drop bad epochs
                epochs_inter.drop_bad()
                nrejected_ratio_manual = round((n_rejected_manual/n_global_epochs*100),2)
                
                # assign manual 
                epochs_clean = epochs_inter.copy()
                n_rejected = n_rejected_manual
                n_rejected_ratio = nrejected_ratio_manual

            else: # go with automated epoch rejection
                epochs_clean = epochs_clean_auto.copy()
                n_rejected = n_rejected_auto
                n_rejected_ratio = nrejected_ratio_auto
    
    # local epochs
    epochs_interp = np.count_nonzero(transform_rej_log.labels == 2) # bad 
    n_local_epochs = np.prod(np.shape(transform_rej_log.labels))
    local_epochs_ratio = round((n_rejected_auto/n_local_epochs*100),2)

    #____________logging____________
    ### save autoreject rejection log matrix as png. Good = 0, bad = 1, interpolated = 2
    # create directory
    autoreject_dir = os.path.join(bidspath_processing.directory, 'autoreject')
    if os.path.exists(autoreject_dir):
        shutil.rmtree(autoreject_dir)# delete folder and contents if it exists
    os.makedirs(autoreject_dir)

    # save plot
    rej_log_plot = transform_rej_log.plot('horizontal', show=False)
    rej_log_plot.savefig(f"{autoreject_dir}/reject_log.png", format = "png", dpi=300)
    rej_log_plot.clf()

    ### logging
    # global 
    utils.log_msg(f"        {n_rejected}/{n_global_epochs} ({n_rejected_ratio}%) global epochs identified removed from dataset.") 
    utils.log_update(log_df, 'n_epochs_removed', n_rejected)
    utils.log_update(log_df, 'global_epochs_removed_ratio', n_rejected_ratio)
    utils.log_update(log_df, 'n_epochs_total', n_global_epochs)

    # local
    utils.log_msg(f"        {epochs_interp}/{n_local_epochs} ({local_epochs_ratio}%) local epochs were interpolated.")
    utils.log_update(log_df, 'n_local_epochs_interpolated', epochs_interp)
    utils.log_update(log_df, 'local_epochs_interpolated_ratio', local_epochs_ratio)
    utils.log_update(log_df, 'n_local_epochs_total', n_local_epochs)

    epochs_clean.info['description'] = f'#8EpochsRejected'

    return epochs_clean, log_df

def badChannels_interpolate(epochs, ransac, log_df):
    """
    Interpolate bad channels in raw data using RANSAC algorithm. The function incorporates the 'autoreject.Ransac' 
    object returned by interpolation_mask(). 

    Parameters
    ----------
    raw : instance of mne.io.Raw
        <br>The raw data used to pass the meta information (e.g., 'mne.info') to the cleaned, continuous data.
        <br>**Must** include every meta information that the fully preprocessed data should incorporate.
    epochs : instance of mne.Epochs
        <br>The epoched data containing the bad channels to be interpolated.
    ransac : instance of autoreject.Ransac
        <br>RANSAC estimator used to interpolate the bad channels.

    Returns
    -------
    raw_clean : mne.io.Raw
        <br>The clean continous raw data with the bad channels interpolated.


    Notes
    -----
    The computation of bad channels and the subsequent interpolation is subdivided into two functions, since the bad channels
    are masked during ICA training and fitting to not introduce channel noise into ICA.

    This function performs the following steps:
    1. performs RANSAC channel interpolation on the bad channels (identified by interpolation_mask) in the epoched data. 
    2. converts the epoched data into continous data again and adds important meta information (e.g. 'mne.info, annotations)
    3. Finally, it resets 'raw_clean.info['bads']'.
    """

    # perform actual interpolation
    # redirects standard output to temporary file
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            epochs_clean = ransac.transform(epochs)
    
    # logging
    utils.log_msg(f"        Following channels will be interpolated: {epochs.info['bads']}")
    utils.log_update(log_df, 'chs_interpolated', epochs.info['bads'])
    utils.log_update(log_df, 'n_chs_interpolated', len(epochs.info['bads']))
    
    epochs_clean.info['description'] = f'#7ChannelsInterp'

    return epochs_clean, log_df

# _______________________________________________________________________________




# _____________________________Loading___________________________________________
## load inputs
inputs = utils.read_inputs(sys.argv[1])

# assign path variables
bidspath_processing = utils.get_bidspath(inputs, 'bids_proc')
sourcedata_dir = inputs['basic']['sourcedata']

## assign module variables

# rereference
perform_rereferencing = inputs['perform']['rereferencing']
rereference = inputs['preprocessing']['rereference']

# epoching
perform_baseline = inputs['perform']['baseline']
epoch_dict = inputs['preprocessing']['epoch_dict']
tmin = inputs['preprocessing']['epoch_min']
tmax = inputs['preprocessing']['epoch_max']
baseline = inputs['preprocessing']['baseline_correction']

# Independent Component Analysis
mask_ICA = inputs['perform']['mask_ICA']
perform_ica = inputs['perform']['ICA']
n_components = inputs['ArtifactCorrection']['n_components']
method = inputs['ArtifactCorrection']['ICA_method']
max_iter = inputs['ArtifactCorrection']['max_iter']
keep_labels = inputs['ArtifactCorrection']['keep_labels']
reject_labels = inputs['ArtifactCorrection']['reject_labels']
rej_threshold = inputs['ArtifactCorrection']['rej_threshold']
ic_labeling = inputs['ArtifactCorrection']['ic_labeling']

# Interpolation
rd_state = inputs['ArtifactCorrection']['random_state_seed']
perform_channel_interpolation = inputs['perform']['channel_interpolation']
perform_epoch_removal = inputs['perform']['epoch_removal']
epoch_rejection_type = inputs['ArtifactCorrection']['epoch_rejection_type']

# Time-Frequency Analysis
fmin = inputs['Analysis']['fmin']
fmax = inputs['Analysis']['fmax']
roi = inputs['Analysis']['ROI']
conditions = inputs['Analysis']['conditions']
eeg_contrasts = inputs['Analysis']['eeg_contrasts']

#ERP
compute_erp = inputs['perform']['compute_erp']
average_stim_trigger = inputs['perform']['average_stim_trigger']



## extract subject list
subjects = utils.find_subjects(bidspath_processing.root)
# subjects = [item for item in subjects if item not in ['028']]

# start at a given subject
# subjects = [sub for sub in subjects if int(sub) > 38]
# _______________________________________________________________________________
  
  
    

# _____________________________Module_exe________________________________________
## step A

if __name__ == '__main__':

    print(f'\n\n\n\n')
    timepoint_start = utils.log_msg(f'START:  Artifact Correction Module')

    ## load log
    log_df = utils.log_load()

    
     # Loop through participants
    for subject in subjects:
        print(f'\n\n')
        utils.log_msg(f"_______ Processing Subject-{subject}_______")
        ## update subject variable in inputs
        utils.update_inputs(sys.argv[1], 'basic','subject_ID', subject)
        utils.update_inputs(sys.argv[1], 'basic','current_step', '01rawfilter')
        # update subject path
        bidspath_processing_subject = bidspath_processing.copy().update(subject=subject)
        ## load data
        raw_filt = utils.load_preprocessing_step(bidspath_processing, subject, 'from_bids')
        # _______________________________________________________________________________


        # _____________________________Module_exe________________________________________
        # prepare behavioral data
        df, log_df = behavdata_prep(sourcedata_dir, subject, log_df)

        # # perform epoching
        epochs, log_df = epoching(raw_filt, df, epoch_dict, tmin, tmax, log_df)# >>> IMPORTANT: currently no baseline correction performed here <<<
        diagnostic_plots(epochs, bidspath_processing_subject)
        
        if epochs is None or not bool(epochs):  # If none, events for epoching do not exist in raw.annotations
            utils.log_msg(f'      X ERROR: None of the following events were found in raw.annotations: {list(epoch_dict.keys())}. Continuing with Subject {int(subject) + 1}')
            continue  # Skip to the next subject

        # identify bad channels to mask before ICA
        if mask_ICA:
            utils.log_msg(f'        Masking bad channels before ICA')
            epochs, ransac, log_df = badChannels(epochs, rd_state, log_df)
        else:
            utils.log_msg(f'     -- No channels masked before ICA')

        # perform ICA
        if perform_ica:
            utils.log_msg(f'        *** Independent Component Analysis ***')
            epochs, ica, log_df = ICA_mne(epochs, n_components, max_iter, rd_state, method, log_df)

            match ic_labeling:
                case 'automatic':
                    # if subject == '003':
                    #    epochs, ica = manualICLabel(epochs, ica, bidspath_processing_subject)
                    #else:
                    epochs, ica, log_df = autoICLabel(epochs, ica, reject_labels, rej_threshold, bidspath_processing_subject, log_df)
                    diagnostic_plots(epochs, bidspath_processing_subject)
                    
                case 'manual':
                    epochs, ica, log_df = manualICLabel(epochs, ica, bidspath_processing_subject, log_df)
                    diagnostic_plots(epochs, bidspath_processing_subject)
            
            # save ICA
            utils.save_preprocessing_step(ica, '02ICA', bidspath_processing, subject)
            utils.update_inputs(sys.argv[1], 'basic','current_step', '02ICA')
        else:
            utils.log_msg(f'     -- ICA not performed')

        ## Apply Baseline
        if perform_baseline:
            epochs.apply_baseline(baseline)
            utils.log_msg(f'        Epochs were baseline corrected to {baseline}')
        else:
            utils.log_msg(f'     -- No baseline correction')

        ## rereference data
        if perform_rereferencing:
            epochs, log_df = rereferencing(epochs, rereference, log_df)
            diagnostic_plots(raw_step, bidspath_processing_subject)
            # utils.save_preprocessing_step(raw_step, '04rawreref')
        else:
            utils.log_msg(f'     -- Rereferencing not performed')

        ### Channel Interpolation
        if perform_channel_interpolation:
            utils.log_msg(f'        *** RANSAC - channel interpolation ***')
            epochs, ransac, log_df = badChannels(epochs, rd_state, log_df)

            epochs, log_df = badChannels_interpolate(epochs, ransac, log_df)
            diagnostic_plots(epochs, bidspath_processing_subject)

            utils.save_preprocessing_step(epochs, '03chInterp', bidspath_processing, subject)
            utils.update_inputs(sys.argv[1], 'basic','current_step', '03chInterp')  
        else:
            utils.log_msg(f'     -- RANSAC not performed')

        ### Removal of Bad Epochs
        if perform_epoch_removal:
            utils.log_msg(f'        *** Autoreject - removal of bad epochs ***')
            epochs, log_df = badEpochs(epochs, rd_state, bidspath_processing_subject, log_df, epoch_rejection_type)
            diagnostic_plots(epochs, bidspath_processing_subject)
        else:
            utils.log_msg(f'     -- Autoreject not performed')
              
        ## save continuous data
        utils.save_preprocessing_step(epochs, '04epochsCorr', bidspath_processing, subject)
        
        
        # _____________________________logging________________________________________________
        ## update subject variable in inputs
        utils.update_inputs(sys.argv[1], 'basic','current_step', '04epochsCorr')
        utils.log_save(log_df,f'{bidspath_processing.root}' ,'log_dataframe.csv')

    utils.update_inputs(sys.argv[1], 'basic','current_step', '04epochsCorr')
    timepoint_end = utils.log_msg(f'DONE:   Artifact Correction')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')
        # ______________________________END - Artifact Correction_____________________________


# _____________________________logging___________________________________________
# save
# _______________________________________________________________________________



def badEpochs_legacy(epochs, rd_state, bidspath_processing):
    
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False)

    # create autoreject object and fit it to data
    ar = AutoReject(picks=picks, random_state=rd_state, n_jobs=4, verbose=False)
    ar.fit(epochs)#[:40] --> fitted/trained on how many epochs

    #____________________________ make actual interpolation____________________________
    epochs_inter = epochs.copy() 


    # perform interpolation 
    # redirects standard output to temporary file
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Interpolation
            epochs_clean, transform_rej_log = ar.transform(epochs_inter, return_log=True) #, reject_log=rejection_log

    ### compute epoch measures
    # global epochs
    epochs_del = np.count_nonzero(transform_rej_log.bad_epochs) #global bad epochs, which will be entirely removed 
    n_global_epochs = transform_rej_log.labels.shape[0]
    global_epochs_ratio = round((epochs_del/n_global_epochs*100),2)

    # Check if more than 20 percent of epochs (trials) are removed. If so, abort. --> Luck (2014) recommendation
    if global_epochs_ratio > 20:
        utils.log_msg(f"        ERROR: autoreject removed {global_epochs_ratio}% of all trials (>20%). Reevaluate subject!")
        #raise ValueError(f"ERROR: autoreject removed {global_epochs_ratio}% of all trials (>20%). Reevaluate subject!")
    else:
        None

    # local epochs
    epochs_interp = np.count_nonzero(transform_rej_log.labels == 2) # bad 
    n_local_epochs = np.prod(np.shape(transform_rej_log.labels))
    local_epochs_ratio = round((epochs_del/n_local_epochs*100),2)

    #____________logging____________
    ### save autoreject rejection log matrix as png. Good = 0, bad = 1, interpolated = 2
    # create directory
    autoreject_dir = os.path.join(bidspath_processing.directory, 'autoreject')
    if os.path.exists(autoreject_dir):
        shutil.rmtree(autoreject_dir)# delete folder and contents if it exists
    os.makedirs(autoreject_dir)

    # save plot
    rej_log_plot = transform_rej_log.plot('horizontal', show=False)
    rej_log_plot.savefig(f"{autoreject_dir}/reject_log.png", format = "png", dpi=300)
    rej_log_plot.clf()

    # global 
    utils.log_msg(f"        {epochs_del}/{n_global_epochs} ({global_epochs_ratio}%) global epochs identified removed from dataset.") 
    utils.log_update(log_df, 'num_epochs_removed', epochs_del)
    utils.log_update(log_df, '%_epochs_removed', global_epochs_ratio)
    utils.log_update(log_df, 'num_epochs_total', n_global_epochs)

    # local
    utils.log_msg(f"        {epochs_interp}/{n_local_epochs} ({local_epochs_ratio}%) local epochs were interpolated.")
    utils.log_update(log_df, 'num_epochs_interpolated', epochs_interp)
    utils.log_update(log_df, '%_epochs_removed', local_epochs_ratio)

    epochs_clean.info['description'] = f'#8EpochsRejected'
    

    return epochs_clean #, transform_rej_log