# _____________________________Behavioral Analysis_______________________________________
# run with
# python b_ArtifactCorrection_module.py inputs.json
#
# * Allgemeine und Biologische Psychologie - AG Hesselman
# * Psychologische Hochschule
#
# ## Author(s)
# * Buchholz, jonathan; Psychologische Hochschule Berlin, AG Hesselmann
# 
#
# * last update: 2025.07.10
#
#
# This script is provides MNE-based functions for EEG artifact rejection and correction, including interpolation methods like autoreject and Ransac as well as ICA 
#
#
# _______________________________________________________________________________



# _____________________________Imports___________________________________________
# basics
import sys
import pandas as pd
import numpy as np
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil

# custom functions
import utils_module as utils
import plotting_module as plotting

# module specifics
import io
import pyddm
from scipy import stats


# _______________________________________________________________________________

def behavdata_prep(sourcedata_dir, subject, log_df):
    """
    load behavioral data and prepare data frame
    """

    # Initialize empty return values
    df, base, low_df, high_df = None, None, None, None

    # Find the corresponding CSV file
    csv_file_found = False
    ### find corresponding .csv file holding  response times from PsychoPy
    for dirpath_csv, _, filenames in os.walk(sourcedata_dir):
        for csv_fname in filenames:
            if subject in csv_fname and csv_fname.endswith('.csv'):#finds csv of the currently processed subject
                csv_dir = os.path.join(dirpath_csv, csv_fname)

                # Load data
                try:
                    df = pd.read_csv(csv_dir, header=0, sep=',')
                except Exception as e:
                    raise ValueError(f"Error reading CSV file {csv_dir}: {e}")
                
                csv_file_found = True
                break

        # stop loop after file was found 
        if csv_file_found:
            break
    
    # select and rename columns
    cols = ['exp', 'block_cond', 'block_order', 'thisN', 'cueAss', 'cueHz', 'motionDir', 'prior', 'keyMotionResp_2.keys', 
            'keyMotionResp_2.corr', 'keyMotionResp_2.rt', 'thresh75', 'BiasFirst', 'trialCoh', 'trialCoh_level', 'participant', 'filename']#'prior',
    df = df[cols]
    df = df.rename(columns={'keyMotionResp_2.keys': 'response', 
                            'keyMotionResp_2.corr': 'corr', 
                            "motionDir": "motion_direction", 
                            'keyMotionResp_2.rt': 'rt',
                            'trialCoh' : 'coh',
                            'trialCoh_level' : 'coh_level'})
    # repeat thresh75
    thresh75 = df['thresh75'].dropna().unique()[0]  # get 75% motion coherence threshold
    df['thresh75'] = thresh75  # repeat it across the column
    
    # remove all unnecessary lines
    df = df[df['exp'].isin(['base', 'lowlevel', 'highlevel'])]

    
    # Data Cleansing - Reaction time
    df, log_df = utils.rt_cleansing(df, log_df)
    

    # Detection rates across all three coherence level
    coh_means = df[df['rt_flag'] == False].groupby('coh_level')['corr'].mean().round(3)
    df.loc[:, ['lowCoh_performance', 'medCoh_performance', 'highCoh_performance']] = [
        coh_means.get('low', None),
        coh_means.get('medium', None),
        coh_means.get('high', None)
        ]
    # logg
    utils.log_msg(f"        Detection rate according to motion coherence level: low {coh_means['low']}, medium: {coh_means['medium']}, high: {coh_means['high']})")

    # Accuracy (detection rate) and mean reaction time according to prior condition
    # df['prior_accuracy'] = df.groupby('exp')['corr'].transform('mean')
    # df['prior_rt'] = df.groupby('exp')['rt'].transform('mean')
    
    # Remove all trials which are not in analyzed_blocks (= test, base)
    analyzed_blocks = inputs['basic']['analyzed_blocks']
    df = df[df['block_cond'].isin(analyzed_blocks)].reset_index(drop=True)

    # add dichotomous prior/noprior column
    df['prior_dic'] = df['prior'].replace({
        'left': 'prior',
        'right': 'prior',
        'noprior': 'noprior'
    })

    ### Make dummy variable coding prior congruent/incongruent decisions
    # for low- or high-level prior: prior congruent = 1, prior incongruent = 0
    df['response_prior'] = 0
    mask_prior = (df['prior'] == df['response']) & (df['exp'].isin(['lowlevel', 'highlevel']))
    df.loc[mask_prior, 'response_prior'] = 1
    # for base / no-prior: left = 1, right = 0
    df.loc[df["exp"] == "base", "response_prior"] = (
        df.loc[df["exp"] == "base", "response"].map({"left": 1, "right": 0})
    )

    # place participant column at columns[0]
    df = df[['participant'] + [col for col in df.columns if col != 'participant']]
   
    # df.to_csv(f'{subject}_df.csv', index=False)
    return df, log_df

#______________________________Signal Detection Theory Measures______________________________
def SDT(df, condition_dict):
    '''
    Computes Signal Detection Theory measures, namely d_prime (stimulus sensitivity) and c criterion (decision bias).
    "signal" is defined as leftwards moving dots. Therefore, c criterion is separately computed for trials with left prior (expecting negative c)
    and trials with right prior (expecting positive c). Computation is equivalent for low- and high-level prior.
    '''

    # extract condition column and conditions
    cond_col, conditions = list(condition_dict.items())[0]

    # open SDT df
    sdt_df_list = []
    # sdt_df_cond = pd.DataFrame()
    sdt_dict = {}

    for cond in conditions:
        # insert prior condition
        sdt_dict[cond_col] = cond
        # make subset condition df
        df_cond = df[df[cond_col] == cond].copy()
        match cond:
            case 'lowlevel' | 'highlevel':
                signal_present = df_cond[df_cond['prior'] == df_cond['motion_direction']]
                signal_absent = df_cond[df_cond['prior'] != df_cond['motion_direction']]
                # Hits
                hits = signal_present['prior'] == signal_present['response'] #.mean() #answering correctly and in congruency with prior.mean()
                # False Alarms           
                fas = signal_absent['prior'] == signal_absent['response'] #.mean() #answering wrongly and in congruency with prior.mean()
                
                # Hit rate
                # hits = ((df_cond['prior'] == df_cond['response']) & (df_cond['response'] == df_cond['motion_direction']))#.mean() #answering correctly and in congruency with prior.mean()
                # False Alarm rate           
                # fas = ((df_cond['prior'] == df_cond['response']) & (df_cond['response'] != df_cond['motion_direction']))#.mean() #answering wrongly and in congruency with prior.mean()
                
                # Miss: incorrect response, *not* in line with prior
                misses = ((df_cond['prior'] != df_cond['response']) & 
                        (df_cond['response'] != df_cond['motion_direction']))

                # Correct Rejection: correct response, *not* in line with prior
                crs = ((df_cond['prior'] != df_cond['response']) & 
                    (df_cond['response'] == df_cond['motion_direction']))
                                
            case 'base':
                # make signal_present (left motion) and signal_absent (right motion) dfs
                signal_present = df_cond[df_cond["motion_direction"] == "left"]
                signal_absent = df_cond[df_cond["motion_direction"] == "right"]

                # Hits
                hits = (signal_present["response"] == "left")#.mean()# hit: answering left in leftward trials.mean()
                # false alarms
                fas = (signal_absent["response"] == "left")#.mean()# false alarm: answering left in rightward trials.mean()

                # Misses
                misses = (signal_present["response"] == "right")
                # Correct Rejection
                crs = (signal_absent["response"] == "right")

        # compute hit and false alarm rate
        hit_rate = hits.mean()
        fa_rate = fas.mean()

        # Apply correction for extreme values (avoid 0 or 1 probabilities)
        eps = 1e-6  # Small correction factor
        hit_rate = np.clip(hit_rate, eps, 1 - eps)
        fa_rate = np.clip(fa_rate, eps, 1 - eps)

        # Compute d' (d-prime)
        d_prime = round(stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate), 4)

        # Compute criterion (c)
        criterion = round(-0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate)), 4)

        # compute accuracy and mean RT
        accuracy = df_cond['corr'].mean().round(3)
        mean_rt = df_cond['rt'].mean().round(3)

        # logg
        utils.log_msg(f"        {cond}: c-criterion: {criterion:.2f}, d prime: {d_prime:.2f}; accuracy: {accuracy:.3f}, mean RT: {mean_rt:.3f}")
        # print(f'length signal presence: {len(signal_present)}. length signal : {len(signal_absent)}')

        # add SDT and basic behavioral measures to condition df
        sdt_dict['accuracy'] = accuracy
        sdt_dict['mean_rt'] = mean_rt
        sdt_dict['c'] = criterion
        sdt_dict['dprime'] = d_prime
        sdt_dict['hits'] = sum(hits)
        sdt_dict['fas'] = sum(fas)
        sdt_dict['misses'] = sum(misses)
        sdt_dict['cr'] = sum(crs)
        # add hits/misses/false alarms/correct rejections to df
        #convert to df
        sdt_df = pd.DataFrame([sdt_dict])
        sdt_df_list.append(sdt_df)

    # concat and add subject
    sdt_df = pd.concat(sdt_df_list, ignore_index=True, sort=False)

    

    return sdt_df



#______________________________Drift Diffusion Modelling______________________________
def rt_descriptives(parent_df, df):

    # make extension to differentiate original from recovered RT measures
    if len(df.columns) > 4:
        string_extension = ""
    else:
        string_extension = "_rec"

    # compute stats on empirical RT data and write it to df
    for prior_cong in df['response_prior'].unique():
        df_subset = df[df['response_prior'] == prior_cong]
        if prior_cong == 1.0:
            prior_string = 'cong' + string_extension
        else:
            prior_string = 'incong' + string_extension
        parent_df[f'mean_rt_{prior_string}'] = df_subset['rt'].mean()
        parent_df[f'median_rt_{prior_string}'] = df_subset['rt'].median()
        parent_df[f'std_rt_{prior_string}'] = df_subset['rt'].std()


    return parent_df

def extract_parameters(ddm, datatype='original'):
    # extract parameters
    parameters = ddm.parameters()
    parameter_dict = {}

    for key in parameters:
        value = float(next(iter(parameters[key].values())))
        parameter = str(next(iter(parameters[key].keys())))
        match datatype:
            case 'original':
                parameter_dict[parameter] = round(value, 2)
            case 'recovered':
                parameter_dict[f'{parameter}_rec'] = round(value, 2)

    # convert to DataFrame
    parameter_df = pd.DataFrame([parameter_dict])
    # add Bayesian Information Criterion
    match datatype:
        case 'original':
            parameter_df['bic'] = round(ddm.get_fit_result().value(), 2)
        case 'recovered':
            parameter_df['bic_rec'] = round(ddm.get_fit_result().value(), 2)
    
    return parameter_df, parameter_dict



def parameter_recovery(ddm, df):

    # extract coherence values
    coh_values = list(sorted(df['coh'].unique()))

    # extract number of per coherence condition trials
    n_medium_coh = len(df[df['coh_level'] == 'medium'])
    n_low_coh = len(df[df['coh_level'] == 'low'])

    # simulates data based on passed ddm separately for each coherence level
    samp_coh_low = ddm.solve(conditions={"coh": coh_values[0]}).sample(n_low_coh)
    samp_coh_med = ddm.solve(conditions={"coh": coh_values[1]}).sample(n_medium_coh)
    sample_simulated = samp_coh_low + samp_coh_med # This preserves information about the conditions

    # fit the model to simulated data (with suppressed output)
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ddm.fit(sample_simulated, fitting_method='differential_evolution', lossfunction=pyddm.LossRobustBIC, verbose=False)


    # stdout_buffer = io.StringIO()
    # with redirect_stdout(stdout_buffer):
    #     ddm.fit(sample_simulated, lossfunction=pyddm.LossRobustBIC, verbose=False)
    # utils.log_msg(f"        Output should appear here:{stdout_buffer.getvalue()}")

    # extract parameters
    df_recovered, dict_recovered = extract_parameters(ddm, datatype='recovered')

    # compute stats on simulated RT data
    df_sim = sample_simulated.to_pandas_dataframe(rt_column_name='rt', choice_column_name='response_prior', drop_undecided=True)

    df_recovered = rt_descriptives(df_recovered, df_sim)

    return df_recovered, dict_recovered


def fit_ddm(df, condition_dict, rt_column, choice_column, choices=("correct", "error"), noise = 1, nondecision = 0.2, T_dur = 3):

    # convert to tuple
    choices = tuple(choices)

    # extract condition column and conditions
    cond_col, conditions = list(condition_dict.items())[0]

    # open df list to concatenate condition-wise df
    df_ddm_cond_list = []

    # Loop through conditions
    for cond in conditions:
        utils.log_msg(f"        Computing DDM on {cond} trials...")
       
        # extract condition df
        df_cond = df[df[cond_col] == cond]


        ddm_sample = pyddm.Sample.from_pandas_dataframe(df_cond, rt_column_name=rt_column, choice_column_name=choice_column, choice_names=choices)
        # all other columns remain accessible as conditions

        # define model
        ### TO DO: compute separately for different coherence level
        ddm = pyddm.gddm(
                    drift= lambda coh,drift : coh*drift,#'drift',#
                    bound= "B",
                    noise = noise,
                    nondecision= nondecision,
                    T_dur=T_dur,
                    starting_position='z',#lambda coh, z: (1 - coh) * z, # compute 
                    parameters={"drift": (-10,10), "B": (0.3, 2) , "z": (-1, 1)}, #, "B": (0.3, 1)
                    conditions=["coh"],
                    choice_names=choices,
                    name=f'DDM on {cond}')
        
        # Fit model to data
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ddm.fit(sample=ddm_sample, fitting_method='differential_evolution', lossfunction=pyddm.LossRobustBIC, verbose=False)

        # stdout_buffer = io.StringIO()
        # with redirect_stdout(stdout_buffer):
        #     ddm.fit(sample_simulated, lossfunction=pyddm.LossRobustBIC, verbose=False)
        # utils.log_msg(f"        Output should appear here:{stdout_buffer.getvalue()}")

        # extract parameters
        df_original_ddm, dict_original = extract_parameters(ddm)
        utils.log_msg(f"        Fitted    Parameters: {dict_original}")
        # add descriptive RT metrics (median, mean, sd)
        df_original_ddm = rt_descriptives(df_original_ddm, df_cond)

        ### SANITY CHECK
        ## Use fitted ddm to recover parameters
        df_recovered_ddm, dict_recovered = parameter_recovery(ddm, df_cond)
        utils.log_msg(f"        Recovered Parameters: {dict_recovered}")

        # concatenate original and recovered and append to condition df list
        df_ddm_cond = pd.concat([df_original_ddm, df_recovered_ddm], axis=1)
        df_ddm_cond.insert(0, cond_col, cond)
        df_ddm_cond_list.append(df_ddm_cond)
        

    # concat and add subject
    df_ddm = pd.concat(df_ddm_cond_list, ignore_index=True, sort=False)

    return df_ddm

#______________________________Psychosis Proneness______________________________
# Function to compute averages per prefix and ending digit
def compute_averages(df, measures, subscores):
    measure_cols = {}
    for measure in measures:
        for subscore in subscores:
            key = f'{measure}{subscore}'
            measure_cols[key] = [
                col for col in df.columns if col.startswith(measure) and col[-1] == str(subscore)# group columns of same subscore
                ]
            if measure_cols:
                new_col = f'{measure}{subscore}_sum'
                df[new_col] = df[measure_cols[key]].sum(axis=1)
    return df

def psych_prone(df):

    # Define prefixes and suffix types
    measures = ['pdi_', 'caps_']
    subscores = [0, 1, 2, 3]

    # Apply the function
    df = compute_averages(df, measures, subscores)

    score_cols = ['pdi_0_sum', 'caps_0_sum']

    # z-transformation & weighted sums 
    for col in score_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f'z_{col}'] = (df[col] - mean) / std  # z-transform
        # weigthed pdi & caps (according to relative contribution to total item number)
        match col:
            case 'pdi_0_sum':
                df[f'weight_{col}'] = (df[col] * (40/72))
            case 'caps_0_sum':
                df[f'weight_{col}'] = (df[col] * (32/72))    

    df['psych_prone'] = df['pdi_0_sum'] + df['caps_0_sum']
    df['z_psych_prone'] = df['z_pdi_0_sum'] + df['z_caps_0_sum']
    df['weighted_psych_prone'] = df['weight_pdi_0_sum'] + df['weight_caps_0_sum']

    # select columns
    cols = ['participant_id','gender', 'age', 'handedness', 'education','highprior_strength', 
            'pdi_0_sum', 'caps_0_sum', 'z_pdi_0_sum', 'z_caps_0_sum', 'weight_pdi_0_sum', 'weight_caps_0_sum', 
            'psych_prone','weighted_psych_prone', 'z_psych_prone',
            'score_mpfi_accept_1','score_mpfi_aware_2','score_mpfi_context_3', 'score_mpfi_defusion_4', 'score_mpfi_values_5','score_mpfi_action_6',
            'score_mpfi_avoidance_1', 'score_mpfi_moment_2', 'score_mpfi_content_3', 'score_mpfi_fusion_4', 'score_mpfi_lackvalues_5', 'score_mpfi_inaction_6',
            'score_mpfi_total']#'prior'
    df = df[cols]

    # rename participant column to align with SDT/DDM results
    df = df.rename(columns={'participant_id': 'participant'})

    return df

'''
TODO: 
- 
'''




# _____________________________Functions_________________________________________


## def function B
## ...
# _______________________________________________________________________________




# _____________________________Loading___________________________________________
## load inputs
inputs = utils.read_inputs(sys.argv[1])

# assign path variables
sourcedata_dir = inputs['basic']['sourcedata']
bidspath_processing = utils.get_bidspath(inputs, 'bids_proc')
# make results directory
result_dir = os.path.join(bidspath_processing.root, 'results/groupBehavioral')
os.makedirs(result_dir, exist_ok=True)

### assign module variables
# epoching
epoch_dict = inputs['preprocessing']['epoch_dict']
tmin = inputs['preprocessing']['epoch_min']
tmax = inputs['preprocessing']['epoch_max']
baseline = inputs['preprocessing']['baseline_correction']

## Signal Detection Theory
compute_sdt = inputs['perform']['compute_sdt']



## Drift Diffusion Modelling
compute_ddm = inputs['perform']['compute_ddm']
rt_column = inputs['Analysis']['ddm']['rt_column']
choice_column = inputs['Analysis']['ddm']['choice_column']
choices = inputs['Analysis']['ddm']['choices']
condition_dict = inputs['Analysis']['conditions']
parameters = inputs['Analysis']['parameters_to_plot']
# model parameters 
noise = inputs['Analysis']['ddm']['noise']
t_dur = inputs['Analysis']['ddm']['t_dur']
nondecision = inputs['Analysis']['ddm']['nondecision']

## Trait Variables - Psychosis Pronenes / Psychological Flexibility


## load log
# extract subject list
subjects = utils.find_subjects(bidspath_processing.root)

# subjects to exclude
# subjects_to_exclude = ['019', '039']
# subjects = [item for item in subjects if item not in subjects_to_exclude]

# process from subjex x onwards
# limit = 40  # process subjects > 40
# subjects = [s for s in subjects if int(s) >= limit]
## load log


## load data
# _______________________________________________________________________________
  
  
    

# _____________________________Module_exe________________________________________
## step A
if __name__ == '__main__':

    print(f'\n\n\n\n')
    timepoint_start = utils.log_msg(f'START:  Behavioral Analysis Module')

    log_df = utils.log_load()


     # Loop through participants
    df_data_list = []
    df_result_list = []
    for subject in subjects:
        utils.log_msg(f"_______ Processing Subject-{subject}_______")
        ## update subject variable in inputs
        utils.update_inputs(sys.argv[1], 'basic','subject_ID', subject)
        bidspath_processing_subject = bidspath_processing.copy().update(subject=subject)

        ### Prerequisites
        # Behavioral Preprocessing
        utils.log_msg(f'        *** Behavioral Data Preprocessing ***')
        df_sub, log_df = behavdata_prep(sourcedata_dir, subject, log_df)
        df_sub = df_sub[df_sub["rt_flag"] == False]
        # append subject data to grouplevel df list
        df_data_list.append(df_sub)
       
        # plotting descriptive plots of RT distribution by condition
        datatype = subject
        plotting.rt_descriptive_plots(df_sub, rt_column, choice_column, choices, datatype, bidspath_processing_subject)
        #raw_filt, bidspath_sub = utils.read_bids(subject, bids_path_preprocessing)
        # _______________________________________________________________________________




        # _____________________________Behavioral Analysis________________________________________
        ### Signal Detection Theory Measures
        if compute_sdt:
            utils.log_msg(f'        *** Signal Detection Theory (SDT) ***')
            sdt_results = SDT(df_sub, condition_dict)
        else:
            utils.log_msg(f'     -- SDT-analysis not performed')

        ### Drift Diffusion Modelling
        if compute_ddm:
            # df_sub = df_sub[df_sub["rt_flag"] == False]
            utils.log_msg(f'        *** Drift Diffusion Modelling (DDM) ***')
            ddm_results = fit_ddm(df_sub, condition_dict, rt_column, choice_column, choices)
        else:
            utils.log_msg(f'     -- DDM-analysis not performed')       

        if compute_sdt & compute_ddm:
            # append SDT and DDM results and delete duplicates
            df_result = pd.concat([sdt_results, ddm_results], axis=1)
            df_result.insert(0, 'participant', f'sub-{subject}')
            df_result = df_result.loc[:, ~df_result.T.duplicated()] # delete duplicate row
            df_result_list.append(df_result)
        else:
            utils.log_msg(f'     -- neither SDT- nor DDM-analysis was performed') 


    # _____________Saving_____________
    # concatenated and save behavioral data
    df_data = pd.concat(df_data_list, ignore_index=True)
    behav_data_file = f'{result_dir}/behavioraldata_hierprior.csv'
    df_data.to_csv(behav_data_file, index=False)

    # concatenated and save behavioral results
    df_results_total = pd.concat(df_result_list, ignore_index=True)
    result_file = f'{result_dir}/behav_results_hierprior.csv'
    df_results_total.to_csv(result_file, index=False)

    timepoint_end = utils.log_msg(f'DONE:   Behavioral Analysis Module - Subject Level')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')
    # '''

# __________________________________Subject Analysis - END _________________________________


    # __________________________________Group Analysis - START _________________________________
    print(f'\n\n\n\n')
    # bidspath_results = utils.get_bidspath(inputs, 'results')
    timepoint_start = utils.log_msg(f'START:  Behavioral Analysis Module - Group Level')
    utils.log_msg(f"_______ Analysis of subjects: {subjects}_______")

    # load behavioral data
    behav_data_file = f'{result_dir}/behavioraldata_hierprior.csv'
    df_data_group = pd.read_csv(behav_data_file, header = 0, sep =',')
    datatype = 'group'

    # make descriptive RT plots across all subjects
    plotting.rt_descriptive_plots(df_data_group, rt_column, choice_column, choices, datatype, result_dir)

    ### Drift Diffusion Modelling
    if compute_ddm:
        utils.log_msg(f'        *** SDT and DDM Analysis ***')
        # df_result_grandAverage = fit_ddm(df_data_group, condition_dict, rt_column, choice_column, choices)
        # Model evaluation by assessing DDM-parameters recovered from simulated data
        plotting.model_validation(df_results_total, result_dir)
    else:
        utils.log_msg(f'        -- DDM-analysis not performed')       

    # plot DDM parameter
    plotting.raincloud_plot(df_results_total, condition_dict, parameters, result_dir)
    plotting.paired_plot(df_results_total, condition_dict, parameters, result_dir)

    ### Trait Variables - Psychosis Proneness and Psychological Flexibility
    utils.log_msg(f'        *** Trait Variables - Psychosis Proneness and Psychological Flexibility ***')
    trait_data_file = f'{sourcedata_dir}/hierPrior_traitVariables.csv'
    trait_data = pd.read_csv(trait_data_file, header = 0, sep =',')
    trait_data = psych_prone(trait_data)


    ### Merge and save results
    df_total = pd.merge(df_results_total, trait_data, on='participant', how='left')

    df_total_file = f'{result_dir}/traits&results_hierprior.csv'
    df_total.to_csv(df_total_file, index=False)

    # logging
    timepoint_end = utils.log_msg(f'DONE:   Behavioral Analysis Module - Group Level')
    utils.log_save(log_df,f'{bidspath_processing.root}' ,'log_dataframe.csv')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end-timepoint_start)}\n\n')
# _____________________________logging___________________________________________

# _______________________________________________________________________________








# _____________________________LEGACY FUNCTIONS___________________________________________
#________________________Signal Detection Theory________________________
def behav_prior(df):#base, low_df, high_df
    """
    ...
    """
    #________________________Low-level prior________________________
    utils.log_msg(f"        LOW-LEVEL PRIOR - Signal Detection Measures")
    # low_test = low_df.loc[low_df['block_cond'] == 'test'
    low_df = df[(df['exp'] == 'lowlevel') & (df['block_cond'] == 'test')]
    
    # Compute c for left-prior trials
    low_c_left, low_dprime_left = SDT(low_df[low_df["prior"] == "left"])

    # Compute c for right-prior trials
    low_c_right, low_dprime_right = SDT(low_df[low_df["prior"] == "right"])
    utils.log_msg(f"        \tc criterion: left-bias: {low_c_left}; right-bias: {low_c_right}")


    # PriorRight = low_test.loc[low_test['prior'] == 'right']
    # print('Prior for RIGHT motion:')
    # print(f'\tMotion Direction: {PriorRight['motion_direction'].value_counts()}')
    # print(f'Response: \t{PriorRight['response'].value_counts()}')


    # PriorLeft = low_test.loc[low_test['prior'] == 'left']
    # print('Prior for LEFT motion:')
    # print(f'\tMotion Direction: {PriorLeft['motion_direction'].value_counts()}')
    # print(f'Response: \t{PriorLeft['response'].value_counts()}')

    # average over left and right cued d prime
    low_dprime_prior = (np.mean([low_dprime_left, low_dprime_right])).round(4)

    # combine c criterion for left- and right prior trials
    low_prior_bias = low_c_right - low_c_left
    utils.log_msg(f"        \tOVERALL SDT measures - c : {low_prior_bias}; d-prime: {low_dprime_prior}")

    low_df[['c_left', 'c_right', 'c_prior', 'dprime']] = [low_c_left, low_c_right, low_prior_bias, low_dprime_prior]


    #________________________High-level prior________________________
    utils.log_msg(f"        HIGH-LEVEL PRIOR - Signal Detection Measures")
    # high_test = high_df.loc[high_df['block_cond'] == 'test']
    high_df = df[(df['exp'] == 'highlevel') & (df['block_cond'] == 'test')]
    
    # Compute c for left-prior trials
    high_c_left, high_dprime_left = SDT(high_df[high_df["prior"] == "left"])

    # Compute c for right-prior trials
    high_c_right, high_dprime_right = SDT(high_df[high_df["prior"] == "right"])
    utils.log_msg(f"        \tc criterion: left-bias: {high_c_left}; right-bias: {high_c_right}")

    # average over left and right cued d prime
    high_dprime_prior = (np.mean([high_dprime_left, high_dprime_right])).round(4)

    # combine c criterion for left- and right prior trials
    high_prior_bias = (high_c_right - high_c_left).round(4)
    utils.log_msg(f"        \tOVERALL SDT measures - c : {high_prior_bias}; d-prime: {high_dprime_prior}")

    high_df[['c_left', 'c_right', 'c_prior', 'dprime']] = [high_c_left, high_c_right, high_prior_bias, high_dprime_prior]


    #______________Baseline Condition______________
    base = df[(df['block_cond'] == 'base')]
    c_base, dprime_base = SDT(base)

    utils.log_msg(f"        BASELINE - Signal Detection Measures")
    base[['c_prior', 'dprime']] = [c_base, dprime_base]
    utils.log_msg(f"        \tc criterion: {c_base}; d-prime: {dprime_base}")

    
    df = pd.concat([base, low_df, high_df], ignore_index=True)

    return df
def SDT(df):
    '''
    Computes Signal Detection Theory measures, namely d_prime (stimulus sensitivity) and c criterion (decision bias).
    "signal" is defined as leftwards moving dots. Therefore, c criterion is separately computed for trials with left prior (expecting negative c)
    and trials with right prior (expecting positive c). Computation is equivalent for low- and high-level prior.
    '''

    # make signal_present (left motion) and signal_absent (right motion) dfs
    signal_present = df[df["motion_direction"] == "left"]
    signal_absent = df[df["motion_direction"] == "right"]

    # compute hit and false alarm rate
    hit_rate = (signal_present["response"] == "left").mean()# hit: answering left in leftward trials.mean()
    false_alarm_rate = (signal_absent["response"] == "left").mean()# false alarm: answering left in rightward trials.mean()


    # compute hit and false alarm rate
    # hit_rate = ((df["motion_direction"] == "left") & (df["response"] == "left")).mean()# hit: answering left in leftward trials.mean()
    # false_alarm_rate = ((df["motion_direction"] == "right") & (df["response"] == "left")).mean()# false alarm: answering left in rightward trials.mean()


    # Apply correction for extreme values (avoid 0 or 1 probabilities)
    eps = 1e-6  # Small correction factor
    hit_rate = np.clip(hit_rate, eps, 1 - eps)
    false_alarm_rate = np.clip(false_alarm_rate, eps, 1 - eps)
    
    # logg
    utils.log_msg(f"        \thit_rate: {hit_rate:.2f}, false alarm rate: {false_alarm_rate:.2f}")


    # Compute d' (d-prime)
    d_prime = round(stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate), 4)

    # Compute criterion (c)
    criterion = round(-0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(false_alarm_rate)), 4)


    return criterion, d_prime

def compute_sdt_for_condition(df_subset, label):
    utils.log_msg(f"        {label.upper()} - Signal Detection Measures")

    c_left, d_left = SDT(df_subset[df_subset["prior"] == "left"])
    c_right, d_right = SDT(df_subset[df_subset["prior"] == "right"])

    d_avg = round(np.mean([d_left, d_right]), 4)
    c_bias = round(c_right - c_left, 4)

    utils.log_msg(f"        \tc criterion: left-bias: {c_left}; right-bias: {c_right}")
    utils.log_msg(f"        \tOVERALL SDT measures - c : {c_bias}; d-prime: {d_avg}")

    df_subset = df_subset.copy()
    df_subset.loc[:, 'c_left'] = c_left
    df_subset.loc[:, 'c_right'] = c_right
    df_subset.loc[:, 'c_prior'] = c_bias
    df_subset.loc[:, 'dprime'] = d_avg

    return df_subset

def compute_sdt_measures(df):
    """
    Computes Signal Detection Theory (SDT) measures (d', c) for different experimental conditions:
    - Low-Level Prior
    - High-Level Prior
    - Baseline

    Returns the input dataframe with additional columns for SDT results.
    """
    # remove trials with missing responses
    df = df.dropna(subset=['response'])

    # Low-Level Prior
    low_df = df[(df['exp'] == 'lowlevel') & (df['block_cond'] == 'test')]
    low_df = compute_sdt_for_condition(low_df, "low-level prior")

    # High-Level Prior
    high_df = df[(df['exp'] == 'highlevel') & (df['block_cond'] == 'test')]
    high_df = compute_sdt_for_condition(high_df, "high-level prior")

    # Baseline
    base_df = df[df['block_cond'] == 'base']
    c_base, d_base = SDT(base_df)
    utils.log_msg("        BASELINE - Signal Detection Measures")
    utils.log_msg(f"        \tc criterion: {c_base}; d-prime: {d_base}")

    base_df = base_df.copy()
    base_df.loc[:, 'c_prior'] = c_base
    base_df.loc[:, 'dprime'] = d_base

    # Combine all dfs
    df_out = pd.concat([base_df, low_df, high_df], ignore_index=True)

    return df_out

def compute_sdt_measures_new(df):
    """
    Computes Signal Detection Theory (SDT) measures (d', c) for different experimental conditions:
    - Low-Level Prior
    - High-Level Prior
    - Baseline

    Returns the input dataframe with additional columns for SDT results.
    """
    # remove trials with missing responses
    df = df.dropna(subset=['response'])

    # Low-Level Prior
    low_df = df[(df['exp'] == 'lowlevel') & (df['block_cond'] == 'test')]
    low_df = compute_sdt_for_condition(low_df, "low-level prior")

    # High-Level Prior
    high_df = df[(df['exp'] == 'highlevel') & (df['block_cond'] == 'test')]
    high_df = compute_sdt_for_condition(high_df, "high-level prior")

    # Baseline
    base_df = df[df['block_cond'] == 'base']
    c_base, d_base = SDT(base_df)
    utils.log_msg("        BASELINE - Signal Detection Measures")
    utils.log_msg(f"        \tc criterion: {c_base}; d-prime: {d_base}")

    base_df = base_df.copy()
    base_df.loc[:, 'c_prior'] = c_base
    base_df.loc[:, 'dprime'] = d_base

    # Combine all dfs
    df_out = pd.concat([base_df, low_df, high_df], ignore_index=True)

    return df_out