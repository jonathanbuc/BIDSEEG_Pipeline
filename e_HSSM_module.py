# _____________________________HSSM Analysis_______________________________________
# run with:
# python e_HSSM_module.py inputs.json
#
# Hierarchical Bayesian Sequential Sampling Model (HSSM) analysis.
# Reads group behavioral data produced by d_BehavAnalysis_module.py and fits a
# hierarchical DDM where drift rate varies by experimental condition
# (base / lowlevel / highlevel) with a random intercept per participant.
#
# Requires: pip install hssm
# Gated by: perform.compute_hssm in inputs.json
# _______________________________________________________________________________


# _____________________________Imports___________________________________________
import sys
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils_module as utils
# _______________________________________________________________________________


# _____________________________Load-time config__________________________________
inputs = utils.read_inputs(sys.argv[1])

compute_hssm  = inputs['perform']['compute_hssm']
condition_dict = inputs['Analysis']['conditions']
hssm_cfg       = inputs['Analysis'].get('hssm', {})

draws          = hssm_cfg.get('draws', 1000)
tune           = hssm_cfg.get('tune', 1000)
chains         = hssm_cfg.get('chains', 2)
cores          = hssm_cfg.get('cores', 2)
target_accept  = hssm_cfg.get('target_accept', 0.9)
sampler        = hssm_cfg.get('sampler', 'mcmc')
model_type     = hssm_cfg.get('model_type', 'ddm')
prior_settings = hssm_cfg.get('prior_settings', 'safe')
link_settings  = hssm_cfg.get('link_settings', 'log_logit')
formula_v      = hssm_cfg.get('formula_v', 'v ~ 1 + exp + (1|participant)')

bidspath_processing = utils.get_bidspath(inputs, 'bids_proc')
result_dir = os.path.join(bidspath_processing.root, 'results', 'groupBehavioral')
# _______________________________________________________________________________


# _____________________________Functions_________________________________________

def prep_hssm_data(df, cond_col, conditions):
    """
    Select and recode columns so the dataframe matches HSSM's expected format.
    Returns a copy with:
      - rt:       positive reaction times, RT-flagged outliers removed
      - response: upper boundary = 1 (prior-congruent), lower = -1 (prior-incongruent)
    """
    df = df.copy()
    if 'rt_flag' in df.columns:
        df = df[~df['rt_flag']]
    df = df[df['rt'] > 0].reset_index(drop=True)
    # HSSM upper/lower boundary encoding
    df['response'] = df['response_prior'].map({1: 1, 0: -1}).astype(int)
    # keep only conditions of interest
    df = df[df[cond_col].isin(conditions)].reset_index(drop=True)
    return df


def fit_hssm_hierarchical(df, condition_dict, formula_v, model_type,
                           prior_settings, link_settings,
                           draws, tune, chains, cores, target_accept, sampler):
    """
    Fit a hierarchical DDM: drift rate varies by condition with subject random
    intercepts; boundary, bias, and non-decision time have group-level priors.
    Sampling uses NUTS via PyMC ('mcmc') or the faster JAX backend ('nuts_numpyro').
    """
    import hssm  # deferred – PyMC/JAX stack is heavy and only needed when flag is on

    cond_col, conditions = list(condition_dict.items())[0]
    df_hssm = prep_hssm_data(df, cond_col, conditions)

    utils.log_msg(f'        model={model_type}, sampler={sampler}, formula="{formula_v}"')
    utils.log_msg(f'        N={len(df_hssm)} trials, '
                  f'{df_hssm["participant"].nunique()} subjects')

    model = hssm.HSSM(
        data=df_hssm,
        model=model_type,
        include=[{
            "name": "v",
            "formula": formula_v,
            "link": "identity",
        }],
        prior_settings=prior_settings,
        link_settings=link_settings,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sample(
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
        )

    return model


def save_hssm_diagnostics(model, result_dir):
    """Save trace plots (convergence diagnostics) to the results directory."""
    trace_path = os.path.join(result_dir, 'hssm_trace.png')
    model.plot_trace()
    plt.savefig(trace_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    utils.log_msg(f'        Trace plot: {trace_path}')

# _______________________________________________________________________________


# _____________________________Module exe________________________________________
if __name__ == '__main__':

    print('\n\n\n\n')
    timepoint_start = utils.log_msg('START:  HSSM Analysis Module')
    log_df = utils.log_load()

    if not compute_hssm:
        utils.log_msg('     -- HSSM analysis not performed (compute_hssm = false)')
    else:
        behav_data_file = os.path.join(result_dir, 'behavioraldata_hierprior.csv')
        if not os.path.exists(behav_data_file):
            raise FileNotFoundError(
                f'Group behavioral data not found: {behav_data_file}\n'
                f'Run d_BehavAnalysis_module.py first.'
            )

        df_group = pd.read_csv(behav_data_file)
        utils.log_msg(
            f'        Loaded {len(df_group)} trials from '
            f'{df_group["participant"].nunique()} subjects'
        )

        utils.log_msg('        *** Hierarchical Bayesian DDM (HSSM) ***')
        model = fit_hssm_hierarchical(
            df_group, condition_dict, formula_v, model_type,
            prior_settings, link_settings, draws, tune, chains, cores, target_accept, sampler
        )

        # Posterior summary -> CSV
        summary = model.summary()
        summary_path = os.path.join(result_dir, 'hssm_posterior_summary.csv')
        summary.to_csv(summary_path)
        utils.log_msg(f'        Posterior summary: {summary_path}')
        utils.log_msg(f'\n{summary.to_string()}\n')

        # Diagnostic trace plots
        save_hssm_diagnostics(model, result_dir)

        # Provenance log
        utils.log_update(log_df, 'hssm_model', model_type)
        utils.log_update(log_df, 'hssm_formula_v', formula_v)
        utils.log_update(log_df, 'hssm_draws', draws)
        utils.log_update(log_df, 'hssm_chains', chains)

    timepoint_end = utils.log_msg('DONE:   HSSM Analysis Module')
    utils.log_save(log_df, f'{bidspath_processing.root}', 'log_dataframe.csv')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end - timepoint_start)}\n\n')
# _______________________________________________________________________________
