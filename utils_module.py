# _____________________________Utility_module_____________________________

# _____________________________Imports_____________________________
import sys
import os
import json
import pickle
import warnings
import datetime
import h5py

#import statistics_module as stat


import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

import mne
# from mne import read_epochs
from mne_bids import read_raw_bids, write_raw_bids, BIDSPath
# from mne.minimum_norm import write_inverse_operator
from autoreject import read_auto_reject, read_reject_log


#import autoreject

'''
TODO: 

- offload bids_path generation
- save source_localisation volume
'''



# _____________________________saving/loading__________________________
## Load
# load inputs
def read_inputs (input_file):
    with open(input_file) as file:
        input = json.load(file)       
    return input

def find_subjects(bidsroot):
    '''
    finds all subjects in participants.tsv and returns list of subjects
    '''
    # read subject file and convert it into list
    filepath_sub = os.path.join(bidsroot, 'participants.tsv')
    subfile = pd.read_csv(filepath_sub, sep='\t')
    sublist = subfile['participant_id'].tolist()
    # remove sub- from subject list
    sublist = [entry.replace('sub-', '') for entry in sublist]

    return sublist

# load filepaths
def get_bidspath(inputs, option=None, subjects=None, processingstep=None):
    
    # raw, unprocessed BIDS
    bidspath = BIDSPath(subject = inputs['basic']['subject_ID'], task=inputs['basic']['task'], processing = inputs['basic']['current_step'], session=inputs['basic']['session'], root=inputs['basic']['bids_root_in'], datatype= 'eeg')
    
    match option:
        case 'None':
            pass
        case 'bids_proc': #
            bidspath = bidspath.update(root = inputs['basic']['bids_root_out']+'/BIDSprocessed')
        case 'epochs_list':# store BIDSpaths of Epochs of all subjects in a list
            bidspath = bidspath.update(root = inputs['basic']['bids_root_out']+'/BIDSprocessed', processing = '04epochsCorr')
            bidspath_list = []
            for subject in subjects:
                bidspath_sub = bidspath.update(subject = subject)
                bidspath_list.append(bidspath_sub.fpath)
            bidspath = bidspath_list
        case 'tfr_list':# reads BIDSpaths of subject TFRs into a list
            bidspath = bidspath.update(root = inputs['basic']['bids_root_out']+'/BIDSprocessed', processing = '05tfr')
            bidspath_list = []
            for subject in subjects:
                bidspath_sub = bidspath.update(subject = subject)
                bidspath_list.append(bidspath_sub.fpath)
            
            bidspath = bidspath_list



            
    if processingstep is not None:
        bidspath.update(processing=processingstep)


    return bidspath



# update inputs
def update_inputs(input_file, inputA, inputB, update):
    
    with open(input_file, 'r') as file:
        json_object = json.load(file)
        json_object[inputA][inputB] = update
        file.close()
        
    with open(input_file, 'w') as file:
        json.dump(json_object,file,indent=4)
        file.close()
        

## load raw

# load step
def load_preprocessing_step(bids_path_preprocessing, subject, file_format):
    
    inputs = read_inputs(sys.argv[1])
    
    # update BIDSpath on subject and processing step
    step = inputs['basic']['current_step']
    bids_path_step = bids_path_preprocessing.copy().update(processing=step, subject = subject)

    
    #suppress warnings
    warnings.filterwarnings("ignore")

    # load data
    match file_format:
        case 'clean_epochs':
            log_msg(f'        Loading epochs ({step[2:]}) from {bids_path_step.basename}')
            raw = mne.read_epochs(bids_path_step, verbose=False)
            raw.load_data(verbose=False)
        case 'from_bids':
            if not step:
                step = "  BIDSified"
            log_msg(f'        Loading raw data ({step[2:]}) from {bids_path_step.basename}')
            raw = read_raw_bids(bids_path_step, verbose=False)
            raw.load_data(verbose=False)
        case 'from_fif':
            log_msg(f'        Loading epoched data from {bids_path_step.basename}')
            # bids_path_step = bids_path_step.copy().update(processing='04epochsCorr')
            filepath = f'{bids_path_step.directory}/{bids_path_step.basename}.fif'
            raw = mne.read_epochs(filepath, verbose=False)
            raw.load_data()
        case 'from_hdf5':
            log_msg(f'        Loading data from {bids_path_step.basename}')
            filepath = f'{bids_path_step.directory}/{bids_path_step.basename}.hdf5'
            raw = read_auto_reject(filepath)
        case 'from_npz':
            log_msg(f'        Loading rejection log from {bids_path_step.basename}')
            filepath = f'{bids_path_step.directory}/{bids_path_step.basename}.npz'
            raw = read_reject_log(filepath)
        case 'from_dict':
            log_msg(f'        Loading iclabel component dict from {bids_path_step.basename}')
            filepath = f'{bids_path_step.directory}/{bids_path_step.basename}.json'
            with open(filepath) as jsonfile:
                raw = json.load(jsonfile)
            
    return raw

def load_epochs_dict(bidspathslist_epochs, condition_dict, roi):
    """
    Loads all preprocessed epochs separately for each experimental condition and stores them in dictionary
    
    return:
    epochs_dict{'condition_1' : [mne.Evoked_sub001, mne.Evoked_sub002, ...],
                'condition_2' : [mne.Evoked_sub001, mne.Evoked_sub002, ...],
                ...}
    """

    # extract experimental conditions
    cond_col, conditions = list(condition_dict.items())[0]
    # load all epochs and store them in list
    epochs_list = [mne.read_epochs(file, verbose=False).pick(roi)
                            for file in bidspathslist_epochs]

    # define epoch_dict
    epochs_dict = {}
    ### loop through experimental conditions  
    for condition in conditions:
        epochs_dict[condition] = {}

        epochs_dict[condition] = [epochs[f"{cond_col} == '{condition}'"]# epochs_dict{'condition_1' : [mne.Evoked_sub001, mne.Evoked_sub002, ...], ..}
                                        for epochs in epochs_list]

    return epochs_dict

def save_preprocessing_step(file, step, bidspath_preprocessing, subject=None):
    '''
    Collection of saving methods integrated to the BIDS Path procedure
    '''
    
    #inputs = read_inputs(sys.argv[1])
    
    #defines filepath/name based on parameters step and bidspath_preprocessing
    bids_path_step = bidspath_preprocessing.copy().update(processing=step, subject = subject)
    filepath = f'{bids_path_step.directory}/{bids_path_step.basename}'
    filename = step[2:]
   
    # check file format
    match str(type(file)):
        # continuous Data
        case "<class 'mne.io.edf.edf.RawEDF'>"|"<class 'mne.io.array.array.RawArray'>":
            inputs = read_inputs(sys.argv[1])
            update_inputs(sys.argv[1], 'basic','current_step', step)
            event_dict = inputs['basic']['event_dict']
            log_msg(f'        Saving continuous data ({filename}) in BIDS format to {bids_path_step.directory}')
            write_raw_bids(file, bids_path_step, event_id=event_dict, format = 'EDF', overwrite=True, allow_preload = True, verbose = False)
            
            # get channel statistics and log to csv
            #stat.log_stats_to_csv(file)
    
        # Epoched Data
        case "<class 'mne.epochs.Epochs'>"|"<class 'mne.epochs.EpochsFIF'>":
            log_msg(f'        Saving epochs as fif file to {bids_path_step.directory}')
            file.save(f'{filepath}.fif', overwrite=True, verbose=False)
        
        # Average and EpochsTFRs
        case "<class 'mne.time_frequency.tfr.AverageTFR'>"|"<class 'mne.time_frequency.tfr.EpochsTFR'>":
            log_msg(f'        Saving TFRs as fif file to {bids_path_step.directory}')
            file.save(f'{filepath}.fif', overwrite=True, verbose=False)

        # mne.Evoked array (list)
        case "<class 'list'>":
            log_msg(f'        Saving Evoked (ERP) as fif file to {bids_path_step.directory}')
            mne.write_evokeds(f'{filepath}.fif', file, overwrite=True, verbose=False)

        # Epochs / Evokeds / averageTFRs for each condition stored in dictionary: dict{cond: mne.Object}; 
        case "<class 'dict'>": # epochs_dict{'base': <mne.Epochs>, lowlevel: <mne.Epochs>, highlevel : <mne.Epochs>}, tfr_dict{'base': <mne.AverageTFR>, lowlevel: <mne.AverageTFR>, highlevel : <mne.AverageTFR>}
            for cond in list(file.keys()):# iterate through dictionary
                match str(type(file[cond])):
                    case "<class 'mne.epochs.Epochs'>"|"<class 'mne.time_frequency.tfr.AverageTFR'>"|"<class 'mne.time_frequency.tfr.EpochsTFR'>":
                        log_msg(f'        Saving {(type(file[cond]))} as fif file ({cond}) to {bids_path_step.directory}')
                        file[cond].save(f'{filepath}_{cond}.fif', overwrite=True, verbose=False)


            # Evokeds for each trigger stored in dictionary: dict{trigger: mne.Evoked}
            # triggers = list(file.keys())
            # log_msg(f'        Saving Evoked (ERP) of {triggers} as fif file to {bids_path_step.directory}')
            # for trigger in triggers:
            #     file[trigger].save(f'{filepath}_{trigger}.fif', overwrite=True, verbose=False)

        # Evoked collapsed over triggers
        case "<class 'mne.evoked.Evoked'>":
            if subject:
                log_msg(f'        Saving Evoked (ERP) as fif file to {bids_path_step.directory}')
                file.save(f'{filepath}.fif', overwrite=True, verbose=False)
            else:
                filepath = f'{bids_path_step.root}/{step}'# ne filepath for GrandAverage Evoked
                log_msg(f'        Saving Grand Average Evoked (ERP) as fif file to {bids_path_step.root}')
                file.save(f'{filepath}.fif', overwrite=True, verbose=False)
        # ransac
        case "<class 'autoreject.ransac.Ransac'>":
            log_msg(f'        Saving {filename} (pickle) to {bids_path_step.directory}')
            with open (filepath,'wb') as f:
                pickle.dump(file,f)
        
        # autoreject obj    
        case "<class 'autoreject.autoreject.AutoReject'>":
            filepath = f'{filepath}.hdf5'
            log_msg(f'        Saving {filename} as .hdf5 to {bids_path_step.directory}')
            file.save(filepath, overwrite=True)
        
        # autoreject rejectionlog
        case "<class 'autoreject.autoreject.RejectLog'>":
            filepath = f'{filepath}.npz'
            log_msg(f'        Saving {filename} as .npz to {bids_path_step.directory}')
            file.save(filepath, overwrite=True)
        
        # ICA object
        case "<class 'mne.preprocessing.ica.ICA'>":
            filepath = f'{filepath}.fif'
            log_msg(f'        Saving {filename} as .fif to {bids_path_step.directory}')
            file.save(filepath, overwrite=True, verbose=False)
            
        # trans mat
        case "<class 'mne.transforms.Transform'>":
            deriv_path = get_bidspath(inputs, 'deriv').update(datatype = 'eeg').mkdir()
            filepath = os.path.join(f'{deriv_path.directory}/trans.fif')
            log_msg(f'        Saving {filename} as .fif to {deriv_path.directory}')
            mne.write_trans(filepath, file, overwrite=True, verbose = False)
        
        # forward solution
        case "<class 'mne.forward.forward.Forward'>":
            deriv_path = get_bidspath(inputs, 'deriv').update(datatype = 'eeg')
            filepath = os.path.join(f'{deriv_path.directory}/fwd.fif')
            log_msg(f'        Saving {filename} as .fif to {deriv_path.directory}')
            mne.write_forward_solution(filepath, file, overwrite=True, verbose = False)
            
        # inverse operator
        case "<class 'mne.minimum_norm.inverse.InverseOperator'>":
            filepath = f'{filepath}-inv.fif'
            log_msg(f'        Saving {filename} as .fif to {bids_path_step.directory}')
            mne.minimum_norm.write_inverse_operator(filepath, file, overwrite=True, verbose=False)
        
        # source estimate
        case "<class 'mne.source_estimate.VolSourceEstimate'>":
            filepath = f'{filepath}.hdf5'
            log_msg(f'        Saving {filename} as .hdf5 to {bids_path_step.directory}')
            file.save(filepath, overwrite=True, verbose = False)
        case "<class 'mne.io.brainvision.brainvision.RawBrainVision'>":
            inputs = read_inputs(sys.argv[1])
            update_inputs(sys.argv[1], 'basic','current_step', step)
            event_dict = inputs['basic']['event_dict']
            log_msg(f'        Saving continuous data ({filename}) in BIDS format to {bids_path_step.directory}')
            write_raw_bids(file, bids_path_step, event_id=event_dict, format = 'BrainVision', overwrite=True, allow_preload = True, verbose = False)
           
        # dict - eg IClabel    
        # case "<class 'dict'>":
        #     clean_dict = {}
        #     for k,v in file.items():
        #         if type(v) == np.ndarray:
        #             clean_dict[k] = v.tolist()
        #         else:
        #             clean_dict[k] = v  
        #     filepath = f'{filepath}.json'
        #     log_msg(f'        Saving {filename} as .json to {bids_path_step.directory}')
        #     with open(filepath, 'w') as jsonfile:
        #         json.dump(clean_dict, jsonfile)
        
        case _ :
            print(f"{filename} NOT SAVED __________________________ is type {type(file)}")
# _____________________________TFR helper functions_____________________________
def make_roi_channel(epochs, roi):
    """
    Makes a virtual channel by averaging the specified electrodes.
    """
    data = epochs.copy().pick(roi).get_data()
    data_roi = data.mean(axis=1)
    info = mne.create_info(ch_names=['roi'], sfreq=epochs.info['sfreq'], ch_types='eeg', verbose=False)
    epochs_roi = mne.EpochsArray(data_roi[:, np.newaxis, :], info, events=epochs.events, event_id=epochs.event_id, metadata=epochs.metadata, verbose=False)
    return epochs_roi
# _____________________________Data Cleansing_____________________________
def rt_cleansing(df, log_df):
    """
    Drops trials with "extreme" reaction times based on absolute cut-offs.
    Prints warning when n of dropped trials exceeds 2%.
    possible cut-offs: < 100 msec and RT > 1 or 1.5 s.
    """

    ### make rt cleansing flag column
    # compute upper and lower limit 0.1 < RT < 2 * sd_rt
    rt_mean = df['rt'].mean()
    rt_std  = df['rt'].std()
    upper_limit = round(rt_mean + 2 * rt_std, 3)
    lower_limit = round(rt_mean - 2 * rt_std, 3)

    # Create rt_flag column with default = False
    df['rt_flag'] = False

    ### Flagging trials excluded from behavioral analysis
    # True when RTs are outside of range |2 * sd|
    df.loc[(df['rt'] <= lower_limit) | (df['rt'] >= upper_limit), 'rt_flag'] = True
    # True when RTs are below 100 ms
    df.loc[(df['rt'] <= 0.1 ), 'rt_flag'] = True
    # True for all missing responses
    df.loc[df['rt'].isna(), 'rt_flag'] = True

    ### Logging
    log_msg(f"        Reaction time outlier correction:")

    ## all flagged trials
    # |2 * sd|
    rt_outliers_sd = len(df.loc[(df['rt'] <= lower_limit) | (df['rt'] >= upper_limit)])
    log_update(log_df, 'rt_outliers_sd', rt_outliers_sd)
    log_msg(f"          - {rt_outliers_sd} are outside of range |2 * sd|")
    # < 100 ms
    rt_outliers_100ms = len(df.loc[(df['rt'] <= 0.1 )])
    log_update(log_df, 'rt_outliers_100ms', rt_outliers_100ms)
    log_msg(f"          - {rt_outliers_100ms} RTs are < 100 ms")
    # missing responses
    no_responses = len(df.loc[df['rt'].isna(), 'rt_flag'])
    log_update(log_df, 'no_responses', no_responses)
    log_msg(f"          - {no_responses} missing responses")

    # print warning when excluded trials exceed 10%
    perc_removed = round(df['rt_flag'].mean() * 100, 2)
    trials_removed = len(df.loc[df['rt_flag'] == True])
    log_update(log_df, 'perc_removed_rt', perc_removed)
    log_update(log_df, 'trials_removed', trials_removed)
    if perc_removed > 10:
        log_msg(f"        WARNING: {perc_removed}% of trials flagged as outliers/missings. Re-evaluate {df['participant'].iloc[0]}")
    else:
        log_msg(f"          - {perc_removed}% of trials flagged as outliers/missings")
    
    return df, log_df

# _____________________________Statistical Helper Functions_____________________________
        # Define wrapper function for paired t-test that returns t-statistic array
def ttest_rel_wrapper(x, y):
    """
    Wrapper for scipy.stats.ttest_rel that returns only the t-statistic array.
    Required because permutation_cluster_test expects stat_fun to return an ndarray,
    not a TtestResult object.
    """
    t_stat, _ = scipy.stats.ttest_rel(x, y, axis=0)
    return t_stat

# _____________________________logging_____________________________
# loads log_df from csv / creates log_df if no csv   
def log_load():
    from utils_module import bids_path_preprocessing

    filepath = os.path.join(bids_path_preprocessing.root, 'log_dataframe.csv')

    if os.path.isfile(filepath):
        log_df = pd.read_csv(filepath)
    else:
        log_df = pd.DataFrame(columns=['subject_ID', 'session', 'task'])

    return log_df

# check if log entry (sub/sess/task already exists) returns bool
def log_entry_exists(df, subject_ID, session, task):
    return (df['subject_ID'].isin([subject_ID]) & df['session'].isin([session]) & df['task'].isin([task])).any()


# adds row in df for specified sub and ses
def log_add_entry(df, subject_ID, session, task):
    warnings.filterwarnings("ignore")
    if not log_entry_exists(df, subject_ID, session, task):
        df.loc[len(df), ['subject_ID','session','task']] = [subject_ID,session,task]
    else:
        pass


# sets value to column for specified df entry
def log_update(df, column, value):
    inputs = read_inputs(sys.argv[1])
    subject_ID = int(inputs['basic']['subject_ID']) # ensure subject_ID is an integer
    session = int(inputs['basic']['session']) # ensure session is an integer
    task = inputs['basic']['task']


    # ensure entry for current subject/session/task exists
    if not log_entry_exists(df, subject_ID, session, task):
        log_add_entry(df, subject_ID, session, task)

    # ensure column exists
    if column not in df.columns:
        dtype = "object" if isinstance(value, (str, list, dict, bool)) else "float64"
        df[column] = pd.Series([np.nan] * len(df), dtype=dtype)
    
    # ensure column can store strings/JSON
    if isinstance(value, (str, list, dict, bool)):
        df[column] = df[column].astype("object")
        # if entry is a list or dict, convert to json string
        value = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    
    entry = (df['subject_ID'] == subject_ID) & (df['session'] == session) & (df['task'] == task)
    
    df.loc[entry, column] = value

    '''
    # ensure column for the preprocessing step exists
    if column not in df.columns:
        df[column] = np.nan

    # add entry either as list or scalar value
    entry = (df['subject_ID'] == subject_ID) & (df['session'] == session) & (df['task'] == task)
    if isinstance(value, list):
        df.loc[entry, column] = json.dumps(value)
    else:
        df.loc[entry, column] = value
        
    '''    
# saves to csv (overwrites)    
def log_save(df, filedir, filename):
    filepath = os.path.join(filedir,filename)
    df.to_csv(filepath, index=False)
    
    
# sorts log by subject first and session,task after
def log_sort(df):
    df.sort_values(['subject_ID','session','task']).reset_index(drop=True)
    
    

# saves logs with time stamps to txt
def log_msg(_string, path_option=None):
    
    '''
    logging function printing date, scriptname & input string 
    once to console, once to module_specific textfile 
    '''
    if path_option is None:
        from utils_module import bids_path_preprocessing
        logmsg_path = bids_path_preprocessing.copy().update(task=None, subject=None, session=None, datatype= None)
    else:
        logmsg_path = path_option.copy().update(task=None, subject=None, session=None, datatype= None)
        
    log_filepath = f'{logmsg_path.directory}/logfiles'
    if not os.path.exists(log_filepath):
        os.makedirs(log_filepath)
    
    # print first to console
    print(f'* {_string}')    
    
    # make sure you log to the current module
    caller_file = os.path.basename(sys.argv[0])
    caller_name = os.path.splitext(caller_file)[0]
    
    # direct output to textfile instead of console
    log_file = open(f'{log_filepath}/{caller_name}_log.txt', 'a')
    sys.stdout = log_file
    
    timepoint = datetime.datetime.today()
    
    # log message to textfile
    print(timepoint.strftime("%a %B %d %H:%M:%S %Z %Y") + " " + str(os.path.basename(sys.argv[0])) + ": " + str(_string))
   
    # close the file and reset output to console
    log_file.close()
    sys.stdout = sys.__stdout__
    return timepoint





# _____________________________SETUP________________________________
# if __name__ == "__main__":
inputs = read_inputs(sys.argv[1])
bids_path_preprocessing = get_bidspath(inputs, 'bids_proc')

#log_df = log_load(f'{bids_path_preprocessing.root}', 'log_dataframe.csv', subject_ID, session, task)
log_df = log_load()
    
    
    
    