# _____________________________Data_preparation_module___________________________
# run with
# python a_preparation_module.py raw_.fif inputs.json
#
# * Allgemeine und Biologische Psychologie - AG Hesselman
# * Psychologische Hochschule
#
# ## Author(s)
# * Buchholz, jonathan; Psychologische Hochschule Berlin, AG Hesselmann
# 
#
# * last update: 2025.23.06
#
# * this scripts provides a BIDSification function, transforming EEG-data in BrainVision format into BIDS 
# * while extending for precise reaction times from standard PsychoPy output
# _______________________________________________________________________________



# _____________________________Imports___________________________________________
import sys
import os
import json
import utils_module as utils
import warnings

import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids
# _______________________________________________________________________________




# _____________________________Functions_________________________________________

def bidsification(inputs):
    """
    BIDSification of EEG raw data (brainvision).
    """
    #INPUTS
    sourcedata_dir = inputs['basic']['sourcedata']
    response_rt = inputs['basic']['response_rt']
    analyzed_blocks = inputs['basic']['analyzed_blocks']
    # set event dictionairy
    event_dict = inputs['basic']['event_dict']
    vis_stim_dict = inputs['basic']['target_events']



    # Pre-requisites
    # find all responses in event_dictionary
    r = 'R'
    response_desc = [key for key in event_dict.keys() if r in key]
    # find all stimuli in event_dictionary
    s = 'S'
    stimulus_desc = [key for key in vis_stim_dict.keys() if s in key]

    subjects = []
    print(sourcedata_dir)
    # find and loop through .vhdr files 
    for dirpath_vdhr, _, filenames in os.walk(sourcedata_dir): #eegInput_dir
        for fname_vdhr in filenames:
            if fname_vdhr.endswith('.vhdr'):
                vhdr_file = os.path.join(dirpath_vdhr, fname_vdhr)
                
                

                #extract subject ID  
                subjectID = fname_vdhr.split('.')[0][-3:]
                # Append to the subject list
                subjects.append(subjectID)
                # BIDSpath
                bidspath = BIDSPath(subject=subjectID, task=inputs['basic']['task'], session=inputs['basic']['session'], root=inputs['basic']['bids_root_in'], datatype= 'eeg')

                # suppress warnings
                warnings.filterwarnings("ignore")

                # load raw data
                utils.log_msg(f"\n______________ Processing Subject-{subjectID}______________", bidspath)
                raw = mne.io.read_raw_brainvision(vhdr_file, ignore_marker_types=True, preload = False, verbose= False)

                # assign montage and line noise frequency
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage)
                raw.info['line_freq'] = 50

                # replace spaces and double spaces with _
                raw.annotations.description = np.array([entry.replace('  ', '_').replace(' ','_') for entry in raw.annotations.description])

                ### find corresponding .csv file holding  response times from PsychoPy
                for dirpath_csv, _, filenames in os.walk(sourcedata_dir): # behavInput_dir
                    for csv_fname in filenames:
                        if subjectID in csv_fname and csv_fname.endswith('.csv'):#finds csv of the currently processed subject
                            csv_dir = os.path.join(dirpath_csv, csv_fname)
                            response_file = pd.read_csv(csv_dir, header = 0, sep =',')

                            # Logg
                            utils.log_msg(f"\tExtending raw.annotation for precise response times from {csv_fname}...", bidspath)

                            #### PsychoPy reaction times
                            rt_series = response_file[response_file['block_cond'].isin(analyzed_blocks)][response_rt]
                            rt_series = rt_series.reset_index(drop=True)#reset index
                           
                            # stimuli onsets from BrainVision
                            stimulus_onsets = pd.Series([onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description) if desc in stimulus_desc])

                            #______________Write BIDS without responses, if a third of responses are Missing______________
                            onethird = len(rt_series) / 3

                            # Count NaN values
                            num_nans = rt_series.isna().sum()

                            # Skip subject if more than a third of reaction times are missing
                            if num_nans > onethird:
                                utils.log_msg(f"\t>>>WARNING<<< sub-{subjectID} misses more than a third of reaction times.", bidspath)
                                if set(vis_stim_dict.keys()).intersection(raw.annotations.description):         
                                    
                                    # delete old response triggers from annotations
                                    # --> loop in function: finds response idx for all responses passed in response_desc
                                    raw.annotations.delete([i for i, desc in enumerate(raw.annotations.description) if desc in response_desc])

                                    # remove all responses from event dictionary
                                    event_dict_noresp = {key: value for key, value in event_dict.items() if key not in response_desc}
                                    
                                    # write to BIDS
                                    utils.log_msg(f"\tdata of sub-{subjectID} is converted to BIDS with without Responses:\n*\t{event_dict_noresp}", bidspath)
                                    write_raw_bids(raw, bidspath, event_id=event_dict_noresp, format = 'EDF', overwrite=True, verbose = False, allow_preload= False)#EDF
                                else:
                                    # Stop proces, if the wrong labels are provided
                                    missing_event = np.setdiff1d(raw.annotations.description, list(event_dict.keys()))# what event is missing in annotations
                                    raise ValueError(f"\tERROR: data of sub-{subjectID} contains annotations which are not passed by event_dict. Missing event: {missing_event}")
                                
                                continue


                            ###_________Add Response Onsets to Raw.Annotations
                            if len(stimulus_onsets) != len(rt_series):
                                utils.log_msg(f"\tMismatch in stimuli and responses count. Skip subject")
                                utils.log_msg(f'\t\tEEG Stimulus Onsets: {len(stimulus_onsets)}\n\t\tReaction times: {len(rt_series)}')
                                continue
                           
                            #_________Add Response Onsets to Raw.Annotations - final BIDSification_________
                            # add stimuli onsets to PsychoPy RTs for BrainVision RTs (respective to beginnig of recording)
                            new_response_onsets = stimulus_onsets + rt_series
                            new_response_onsets.dropna(inplace=True)# remove nan of non-response trials

                            # extract response descriptions of correct (R_1) and wrong (R_2) responses
                            response_desc_new = [resp for resp in raw.annotations.description if resp in response_desc]
                            
                            # delete old response trigger
                            raw.annotations.delete([i for i, desc in enumerate(raw.annotations.description) if desc in response_desc])

                            # add new response trigger differentiating correct/wrong responses
                            raw.annotations.append(new_response_onsets, 0.001, response_desc_new, ch_names=None)

                            if set(event_dict.keys()).intersection(raw.annotations.description):
                                # define BIDSpath
                                utils.log_msg(f"\tdata of sub-{subjectID} is converted to BIDS with #{len(stimulus_onsets)} total trials", bidspath)
                                write_raw_bids(raw, bidspath, event_id=event_dict, format = 'EDF', overwrite=True, verbose = False, allow_preload= False)#EDF
                                # break this loop and start with next subject.
                                continue
                            else:
                                missing_event = np.setdiff1d(raw.annotations.description, list(event_dict.keys()))# what event is missing in annotations
                                raise ValueError(f"\tERROR: data of sub-{subjectID} contains annotations which are not passed by event_dict. Missing event: {missing_event}")
                                # break this loop and start with next subject.
                                continue
    return subjects

# _______________________________________________________________________________



# _____________________________Loading___________________________________________
## load inputs
inputs = utils.read_inputs(sys.argv[1])
bidspath_in = utils.get_bidspath(inputs)


## Start Processing
if __name__ == '__main__':
    print(f'\n\n\n\n')
    timepoint_start = utils.log_msg(f'START:  BIDSification', bidspath_in)
   

    # _____________________________Module_exe________________________________________
    # BIDSificaiton of data set
    subjects = bidsification(inputs)
    utils.update_inputs(sys.argv[1], 'basic','current_step', None)
    # _______________________________________________________________________________




    # _____________________________logging___________________________________________
    timepoint_end = utils.log_msg(f'DONE: BIDSIfication', bidspath_in)
    utils.log_msg(f'\tTime elapsed: {str(timepoint_end-timepoint_start)}\n\n', bidspath_in)
    # _______________________________________________________________________________



