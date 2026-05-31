# _____________________________preprocessing module______________________________
# run with
# python a_preprocessing_module.py inputs.json
# _______________________________________________________________________________



# _____________________________Imports___________________________________________
import sys
import os
import shutil
import warnings
from contextlib import redirect_stdout, redirect_stderr
import utils_module as utils

#from meegkit import dss
import mne
import numpy as np
import matplotlib.pyplot as plt
# _______________________________________________________________________________




# _____________________________Functions_________________________________________
## downsampling
def down_sample(raw, freq_ds, log_df):
    '''
    Downsample the raw data to specified sampling rate from its original sampling frequency.

    Parameters
    ----------
    raw : mne.io.Raw
        <br>The raw data to be downsampled.
    sfreq_ds : int
        <vr>The sampling rate after downsampling

    Returns
    -------
    raw : mne.io.Raw
        <br>The downsampled raw data.

    Notes
    -----
    This function resamples the raw data to a chosen target sampling rate.<br>
    This is performed using the `resample` method from the MNE-Python library.

    Examples
    --------
    >>> from preproc.def import prerequisites, preprocessing
    >>> raw = prerequisites.read_bids(bids_path, notch)
    >>> downsampled_raw = preprocessing.downsample(raw,sfreq_ds)
    '''
    raw_downs = raw.copy()

    utils.log_msg(f"        Downsampling from { raw_downs.info['sfreq'] } to {freq_ds}")
    
    raw_downs = raw_downs.resample(sfreq=freq_ds)
    raw_downs.info['description'] += f"\n#d_downsampled_to_{raw_downs.info['sfreq']}_Hz"
    
    # logging
    utils.log_update(log_df, 'samplingrate_down', freq_ds)

    raw_downs.info['description'] = f'#1downsampled({freq_ds}Hz)'

    return raw_downs, log_df

## rereferencing
def rereferencing(raw, rereference, log_df):
    """
    Rereferences data to the given reference channels.

    Parameters
    ----------
    raw : mne.io.Raw
        <br>raw data.
    rereference : list of str
        <br>The names of the reference channels to use for rereferencing the EEG data. List can be cultivated with any existing
        channel names. Other arguments are 
            <br>-``REST``: uses the Reference Electrode Standardization Technique infinity reference
            <br>-``[]``: empty list. In which case the data is not rereferenced.

    Returns
    -------
    raw : mne.io.Raw
        The rereferenced mne.io.Raw object.

    """

    

    match rereference:
        
        case list():
            raw_reref = raw.copy().set_eeg_reference(ref_channels=rereference, projection=False, verbose = False)
        
        case 'average':
            raw_reref = raw.copy().set_eeg_reference(ref_channels=rereference, projection=False, verbose = False)
        
        case 'infinite':
            deriv_path = utils.get_bidspath(inputs, 'deriv')
            # coregistration
            utils.log_msg(f"        Preparing rereferencing to infinite reference")
            #trans_mat = auto_coreg(deriv_path,inputs['source_localisation']['icp_niters'])
            
            # forward solution
            #fwd = forward_solution(deriv_path, raw.info, trans_mat, inputs['source_localisation']['conductivity'])

            # rereference
            raw_reref = raw.copy().set_eeg_reference("REST", ch_type='eeg')#forward=fwd,
    # logging
    utils.log_update(log_df, 'rereference', rereference)
    utils.log_msg(f"        Rereferencing data to {rereference}")

    raw_reref.info['description'] = f'#2rereferenced_({rereference})'


    return raw_reref, log_df

## notch filter
def remove_linenoise(raw, notch_freq, line_method, high_cutoff, notch_width, log_df):
    '''
    Removes line noise components from the continuous signal
    Includes harmonics below set high_cutoff of bandpassfilter

    Parameters
    ----------
    raw : mne.io.Raw
        <br> The raw data to be notch_filtered. 
    notch_freq : float
        <br> The frequency in Hz to notchfilter. Usually 50/60 Hz
    line_method : string
        <br> if 'stop_band' applies a mne FIR stop band filter
        <br> if 'zapline' applies the iterative Denoising source separation algorithm (de Cheveigné, A. (2019))
    high_cutoff : float
        <br> Used to remove harmonics below the high_cuttoff freqency

    Returns
    -------
    raw : mne.io.Raw
        <br>The raw data without line noise.

    Notes
    -----
    This function resamples the raw data to a chosen target sampling rate.<br>
    This is performed using the `resample` method from the MNE-Python library.

    Examples
    --------
    >>> from preproc.def import prerequisites, preprocessing
    >>> raw = prerequisites.read_bids(bids_path, notch)
    >>> downsampled_raw = preprocessing.downsample(raw,sfreq_ds,'zapline',high_cutoff)
    
    References
    ----------
    [1] : de Cheveigné, A. (2019). ZapLine: A simple and effective method to remove power line artifacts [Preprint]. https://doi.org/10.1101/782029
    '''
    
    # check if harmonics need to be removed as well
    if  2 * notch_freq < high_cutoff:
        notch_freqs = np.arange(notch_freq,high_cutoff,notch_freq)
    else:
        notch_freqs = [notch_freq]
        
    raw_notch = raw.copy()

    # remove notch
    match line_method:
        case 'band_stop':
            #mne.filter.notch_filter(raw_notch, Fs=raw.info['sfreq'], freqs=notch_freqs, filter_length='auto', notch_widths=2, trans_bandwidth=1, method='fir', verbose=False)
            raw_notch.notch_filter(freqs = notch_freqs, notch_widths=notch_width, verbose = False)
            utils.log_msg(f"        Line noise removed at {notch_freqs} Hz, using {line_method}")
                
            
        case 'zapline':
            # meegkit/zapline needs data as numpy array
            zap_data = np.array(raw_notch.get_data()).transpose()

            for notch in notch_freqs:
                zap_data, iterations = dss.dss_line_iter(zap_data, notch, raw_notch.info['sfreq'])
                utils.log_msg(f"        Line noise removed at {notch_freqs} Hz, using {line_method}")
                # create mne.raw from numpy array - copy .info from prior step
                raw_notch = mne.io.RawArray(zap_data.transpose(), raw_notch.info, first_samp=0, copy='auto', verbose=True)
            raw_notch.set_annotations(raw.annotations)
                
            
        case 'multi_taper':
            print('multitapering note yet implemented')
            # raw_notch.set_annotations(raw.annotations)
            
            
     
    raw_notch.info['description'] = f'#3line_noise_({line_method})'   
    
    
    
    # logging
    utils.log_update(log_df, 'notch_freq', notch_freq)
    utils.log_update(log_df, 'line_method', line_method)
    
    return raw_notch, log_df

## band pass filter
def filtering(raw, filter_method, low_cutoff, high_cutoff, log_df):
    '''
    Apply band-pass filter to the raw data.
    

    Parameters
    ----------
    raw : mne.io.Raw
        <br>The raw data to be filtered.
    filter_method : string
        <br>If can be 'fir' or 'iir_butterworth' and uses the respective mne filter function
    low_cutoff : float | int
        <br>The low cutoff frequency of the band-pass filter.
    high_cutoff : float | int
        <br>The high cutoff frequency of the band-pass filter.

    Returns
    -------
    raw : mne.io.Raw
        <br>The filtered raw data.

    Notes
    -----

    Examples
    --------
    >>> import mne_bids
    >>> from preproc_def import prerequisites, preprocessing
    >>> # generate the BIDS path and load raw data from BIDS
    >>> bids_path = mne_bids.BIDSPath(subject=subject_id, task=task, root=root, datatype= 'eeg')
    >>> raw = prerequisites.read_bids(bids_path, notch=50)
    >>> filtered_raw = preprocessing.filter(raw, notch=60, low_cutoff=1, high_cutoff=40)
    '''
    
    raw_filt = raw.copy()
    match filter_method:
        case 'fir':
            raw_filt.filter(
                l_freq = low_cutoff, 
                h_freq = high_cutoff, 
                l_trans_bandwidth = 'auto', # 'auto' = min(max(l_freq * .25, 2), l_freq)
                h_trans_bandwidth = 'auto', # 'auto' = min(max(h_freq * .25, 2), info['sfreq']/2 - h_freq)
                n_jobs=n_jobs,
                fir_design = 'firwin', 
                phase = 'zero', # compensate for phase shift
                verbose = False
            )
            utils.log_msg(f"        Signal band pass filtered, using {filter_method} - (low_cutoff = {low_cutoff}, high_cutoff = {high_cutoff})")
            
        case 'iir_butterworth':
            raw_filt.filter(
                l_freq = low_cutoff, 
                h_freq = high_cutoff, 
                l_trans_bandwidth = 'auto', # 'auto' = min(max(l_freq * .25, 2), l_freq)
                h_trans_bandwidth = 'auto', # 'auto' = min(max(h_freq * .25, 2), info['sfreq']/2 - h_freq)
                n_jobs=n_jobs,
                method = 'iir',
                iir_params = None,
                phase = 'zero', # compensate for phase shift
                verbose = False
            )
            utils.log_msg(f"        Signal band pass filtered, using {filter_method} - (low_cutoff = {low_cutoff}, high_cutoff = {high_cutoff})")
            
    raw_filt.info['description'] = f'#4filtered({low_cutoff}-{high_cutoff}Hz)'
    
    # logging
    utils.log_update(log_df, 'filter_method', filter_method)
    utils.log_update(log_df, 'low_cutoff_freq', low_cutoff)
    utils.log_update(log_df, 'low_transition_band', min(max(low_cutoff * .25, 2), low_cutoff))

    utils.log_update(log_df, 'high_cutoff_freq', high_cutoff)
    utils.log_update(log_df, 'high_transition_band', min(max(high_cutoff * .25, 2), raw_filt.info['sfreq']/2 - high_cutoff))
        
    
    return raw_filt, log_df

def get_evoked(raw, erp_stim, event_dict, tmin=-0.5, tmax=1):
    '''
    Computing epochs and evoked potentials around a given stimulus
    '''
    eeg_data = raw.copy().pick('eeg')
    
    events, event_dict = mne.events_from_annotations(eeg_data, event_id = event_dict, verbose=False)

    if erp_stim not in event_dict:
        return('WARNING: Stimulus to compute evoked potential does not exist')

    else:
        # epoching and computing evokes around Stimulus  2 (grip stim onset)

        epochs = mne.Epochs(eeg_data, events, tmin=tmin, tmax=tmax, event_id=event_dict, preload=True, verbose=False, event_repeated='merge')
        evoked = epochs[erp_stim].average()
        utils.log_msg(f"        Evoked potential calculated for event: \'{erp_stim}\' with window = [{tmin}, {tmax}].")

    return epochs
def diagnostic_plots(data, bidspath_processing_subject):
    '''
    makes all sorts of diagnostic plots after selected preprocessing steps
    '''

    # create folder
    diag_dir = os.path.join(bidspath_processing_subject.directory, 'diagnostics')
    if not os.path.exists(diag_dir):
        os.makedirs(diag_dir)
        #shutil.rmtree(diag_dir)# delete folder and contents if it exists
    

    # Power Spectrum Density
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):#surpress output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psd = data.compute_psd(verbose=False).plot(show=False)
            psd.savefig(f"{diag_dir}/PSD_{data.info['description']}.png", format = "png", dpi=diag_dpi)
            plt.close(psd)




# _______________________________________________________________________________




# _____________________________Loading___________________________________________
## load inputs
inputs = utils.read_inputs(sys.argv[1])

# BIDS path for in- and output data
bidspath_in = utils.get_bidspath(inputs)
bidspath_out = utils.get_bidspath(inputs, 'bids_proc')


# assign module variables
perform_downsampling = inputs['perform']['downsampling']
perform_rereferencing = inputs['perform']['rereferencing']
perform_linenoise_filtering = inputs['perform']['linenoise_filtering']
perform_filtering = inputs['perform']['filtering']
event_dict = inputs['basic']['event_dict']
n_jobs = inputs['basic']['n_jobs']
diag_dpi = inputs['basic']['diagnostic_dpi']

samplingrate_down = inputs['preprocessing']['samplingrate_down']
rereference = inputs['preprocessing']['rereference']
notch = inputs['preprocessing']['notch']
line_method = inputs['preprocessing']['line_method']
notch_width = inputs['preprocessing']['notch_width']
low_cutoff_freq = inputs['preprocessing']['low_cutoff_freq']
high_cutoff_freq = inputs['preprocessing']['high_cutoff_freq']
filter_method = inputs['preprocessing']['filter_method']

# extract subject list
subjects = utils.find_subjects(bidspath_in.root)
# start at a given subject
# subjects = [sub for sub in subjects if int(sub) < 5]


if __name__ == '__main__':
    print(f'\n\n\n\n')
    timepoint_start = utils.log_msg(f'START:  Data preprocessing')

    ## load log
    log_df = utils.log_load()

    # Loop through participants
    for subject in subjects:

        utils.log_msg(f"\n_______ Processing Subject-{subject}_______")
        ## update subject variable in inputs
        utils.update_inputs(sys.argv[1], 'basic','subject_ID', subject)
        utils.update_inputs(sys.argv[1], 'basic','current_step', None)
        
        # define subject BIDS-path
        bidspath_processing_subject = bidspath_out.copy().update(subject=subject)

        ## load data
        # raw, bidspath_sub = utils.read_rawBIDS(subject, bidspath_in)
        raw = utils.load_preprocessing_step(bidspath_in, subject, 'from_bids')
        raw.info['description'] = f'#0.1_raw'
        diagnostic_plots(raw, bidspath_processing_subject)

        #raw_step = utils.load_preprocessing_step(bidspath, subject, 'from_bids')
        # ______________________________________________________________________________
            


        # _____________________________preprocessing_____________________________________
        ## downsample data
        if perform_downsampling:
            raw_step, log_df = down_sample(raw, samplingrate_down, log_df) 
            diagnostic_plots(raw_step, bidspath_processing_subject)
            #utils.save_preprocessing_step(raw_step, '03rawdownsample')
        else:
            utils.log_msg(f"     -- Downsampling not performed, sampling rate remains at {raw_step.info['sfreq']}")


        ## rereference data
        # if perform_rereferencing:
        #     raw_step = rereferencing(raw_step, rereference)
        #     diagnostic_plots(raw_step, bidspath_processing_subject)
        #     #utils.save_preprocessing_step(raw_step, '04rawreref')
        # else:
        #     utils.log_msg(f'     -- Rereferencing not performed')


        ## filter line noise (including harmonics)
        if perform_linenoise_filtering:
            raw_step, log_df = remove_linenoise(raw_step, notch, line_method, high_cutoff_freq, notch_width, log_df)
            diagnostic_plots(raw_step, bidspath_processing_subject)
            #utils.save_preprocessing_step(raw_step, '05rawnotch')
        else:
            utils.log_msg(f'     -- Line noise filtering not performed')

        ## filter data
        if perform_filtering:
            raw_step, log_df = filtering(raw_step, filter_method, low_cutoff_freq, high_cutoff_freq, log_df)
            diagnostic_plots(raw_step, bidspath_processing_subject)
        else:
            utils.log_msg(f'     -- Data filtering not performed')

    # _____________________________SAVING___________________________________________
        # BIDS path to write preprocessed data to
        utils.save_preprocessing_step(raw_step, '01rawfilter', bidspath_out, subject)
    # _____________________________logging___________________________________________
    utils.log_save(log_df, bidspath_out.root, 'log_dataframe.csv')
    #
    timepoint_end = utils.log_msg(f'Done:   Data preprocessing')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')
    # _______________________________________________________________________________
