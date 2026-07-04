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
import plotting_module as plotting
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
formula_v      = hssm_cfg.get('formula_v', 'v ~ 1 + exp * coh_level + (1|participant)')
# Optional per-parameter formulas (empty string = leave that parameter as a simple
# group-level estimate). z = starting-point bias, t = non-decision time, a = boundary.
# Modelling z with a prior term separates a pre-evidence (start-point) bias from the
# in-evidence drift bias carried by formula_v -- the Franzen-style dissociation.
formula_z      = hssm_cfg.get('formula_z', '')
formula_t      = hssm_cfg.get('formula_t', '')
formula_a      = hssm_cfg.get('formula_a', '')
# Fix non-decision time to a constant (s) instead of estimating it. The task blocks
# responses for ~500 ms, so that latency isn't decision time -- estimating t there
# invites artifacts. None -> estimate t normally.
fix_t          = hssm_cfg.get('fix_t', None)

bidspath_processing = utils.get_bidspath(inputs, 'bids_proc')
result_dir = os.path.join(bidspath_processing.root, 'results', 'groupBehavioral')
# _______________________________________________________________________________


# _____________________________Functions_________________________________________

def prep_hssm_data(df, cond_col, conditions, formula=None):
    """
    Select and recode columns so the dataframe matches HSSM's expected format.
    Returns a copy with:
      - rt:       positive reaction times, RT-flagged outliers removed
      - response: upper boundary = 1 (prior-congruent), lower = -1 (prior-incongruent)

    If `formula` is given, rows missing any per-trial alpha covariate that the
    formula actually uses are dropped (FOOOF leaves ~7% of trials without an alpha
    peak; the centre-of-gravity estimate has none missing).
    """
    df = df.copy()
    if 'rt_flag' in df.columns:
        df = df[~df['rt_flag']]
    df = df[df['rt'] > 0].reset_index(drop=True)
    # HSSM upper/lower boundary encoding
    df['response'] = df['response_prior'].map({1: 1, 0: -1}).astype(int)
    # Coherence predictors for the drift formula.
    #   coh       continuous, per-subject coherence (precision of sensory evidence)
    #   coh_level threshold-relative level; recode low/medium -> 0/1 so it enters
    #             the formula numerically (slope = medium minus low)
    if 'coh_level' in df.columns:
        df['coh_level'] = df['coh_level'].map({'low': 0, 'medium': 1})
        df = df.dropna(subset=['coh_level']).reset_index(drop=True)
        df['coh_level'] = df['coh_level'].astype(int)
    # Centered versions of continuous coherence. QUEST makes `coh` differ between
    # subjects, so grand-mean centering (coh_gc) blends within- and between-subject
    # variation, while subject-mean centering (coh_wc) isolates the within-subject
    # trial-level effect. coh_subjmean carries the between-subject part for an
    # optional within-between (Mundlak) decomposition.
    if 'coh' in df.columns:
        df = df.dropna(subset=['coh']).reset_index(drop=True)
        df['coh_gc'] = df['coh'] - df['coh'].mean()
        subj_mean = df.groupby('participant')['coh'].transform('mean')
        df['coh_wc'] = df['coh'] - subj_mean
        df['coh_subjmean'] = subj_mean - df['coh'].mean()
    # Per-trial individual alpha frequency (neural marker of expectation/inhibition).
    # Centered exactly like coherence: alpha differs between subjects (IAF ~9-11 Hz),
    # so grand-mean centering (_gc) blends within- and between-subject variation,
    # while subject-mean centering (_wc) isolates the trial-to-trial alpha shift --
    # the actual "does the alpha clock move evidence accumulation" question.
    for acol in ('alpha_cf_fooof', 'alpha_cf_cog'):
        if acol in df.columns:
            subj_mean = df.groupby('participant')[acol].transform('mean')
            df[f'{acol}_gc'] = df[acol] - df[acol].mean()
            df[f'{acol}_wc'] = df[acol] - subj_mean
            df[f'{acol}_subjmean'] = subj_mean - df[acol].mean()
    # keep only conditions of interest
    df = df[df[cond_col].isin(conditions)].reset_index(drop=True)
    # drop trials missing an alpha covariate the formula uses (centering means above
    # are computed first, on all available trials, so they're unaffected by this)
    if formula is not None:
        need = [c for c in df.columns if c.startswith('alpha_cf_') and c in formula]
        if need:
            before = len(df)
            df = df.dropna(subset=need).reset_index(drop=True)
            if before - len(df):
                utils.log_msg(f'        dropped {before - len(df)} trials missing '
                              f'alpha covariate(s) {need}')
    return df


def fit_hssm_hierarchical(df, condition_dict, formula_v, model_type,
                           prior_settings, link_settings,
                           draws, tune, chains, cores, target_accept, sampler,
                           formula_z='', formula_t='', formula_a='', fix_t=None):
    """
    Fit a hierarchical DDM. Drift rate (v) always gets a formula; z/t/a get one only
    if a non-empty formula is passed, otherwise they stay simple group-level
    parameters. Giving z its own formula (e.g. z ~ 1 + prior_dic) separates a
    pre-evidence start-point bias from the in-evidence drift bias in formula_v.
    Sampling uses NUTS via PyMC ('mcmc') or the faster JAX backend ('nuts_numpyro').
    """
    import hssm  # deferred – PyMC/JAX stack is heavy and only needed when flag is on

    cond_col, conditions = list(condition_dict.items())[0]
    # prep must see every covariate any formula uses, so it knows which alpha
    # columns to keep / drop-NaN on.
    all_formulas = ' '.join([formula_v, formula_z, formula_t, formula_a])
    df_hssm = prep_hssm_data(df, cond_col, conditions, formula=all_formulas)

    # Persist the exact matrix the sampler sees (recoded, centered, NaN-dropped) so the
    # model input is inspectable without re-deriving it.
    input_path = os.path.join(result_dir, 'hssm_input_data.csv')
    df_hssm.to_csv(input_path, index=False)
    utils.log_msg(f'        HSSM input data: {input_path} ({len(df_hssm)} rows)')

    # v keeps the identity link that produced the validated drift results; z/t/a are
    # left to link_settings (logit for the bounded z, log for the positive t/a).
    include = [{"name": "v", "formula": formula_v, "link": "identity"}]
    # Fix t to a constant when requested: HSSM treats a scalar `prior` as a fixed value
    # (a bambi ConstantComponent -- no sampled RV), unlike a `formula`. A fixed t will
    # therefore not appear in the posterior trace at all.
    if fix_t is not None:
        include.append({"name": "t", "prior": float(fix_t)})
    elif formula_t:
        include.append({"name": "t", "formula": formula_t})
    for name, f in (("z", formula_z), ("a", formula_a)):
        if f:
            include.append({"name": name, "formula": f})

    # t may be a fixed scalar (no "formula" key), so describe each spec generically.
    formula_msg = " | ".join(
        spec.get("formula", f'{spec["name"]}={spec.get("prior")}') for spec in include
    )
    utils.log_msg(f'        model={model_type}, sampler={sampler}, formulas="{formula_msg}"')
    utils.log_msg(f'        N={len(df_hssm)} trials, '
                  f'{df_hssm["participant"].nunique()} subjects')

    model = hssm.HSSM(
        data=df_hssm,
        model=model_type,
        include=include,
        prior_settings=prior_settings,
        link_settings=link_settings,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.sample(
                sampler=sampler,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
            )
        except ValueError as exc:
            # hssm 0.2.x raises a dimension mismatch when computing log likelihood
            # after numpyro sampling with random effects — posterior samples are already
            # stored in model._inference_obj before this happens, so we can proceed.
            if "different number of dimensions" not in str(exc):
                raise
            utils.log_msg("        (skipping log-likelihood: hssm 0.2.x/numpyro compat issue)")

    return model


def load_group_data(behav_file, formula_v, inputs):
    """
    Pick the trial table the drift formula needs.

    If the formula references a per-trial alpha covariate, load the per-trial alpha
    derivatives written by c_EEGAnalysis_module.py (extract_trial_alpha). Those CSVs
    are epochs.metadata with alpha columns appended, so they already carry every
    behavioural field -- no key-merge onto behavioraldata_hierprior.csv is needed
    (and that merge is unsafe anyway: block_cond/block_order/thisN are not a unique
    trial key). Otherwise fall back to the group behavioural CSV.
    """
    if 'alpha' not in formula_v:
        return pd.read_csv(behav_file)

    import glob
    proc = utils.get_bidspath(inputs, 'bids_proc')
    alpha_dir = os.path.join(proc.root, 'results', 'groupEEG', 'trial_alpha')
    files = sorted(glob.glob(os.path.join(alpha_dir, 'sub-*_trial_alpha.csv')))
    if not files:
        raise FileNotFoundError(
            f'Formula uses an alpha covariate but no per-trial alpha CSVs were found '
            f'in {alpha_dir}.\nRun c_EEGAnalysis_module.py with compute_fooof = true first.'
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    utils.log_msg(
        f'        Alpha covariate requested -> using per-trial alpha table '
        f'({len(df)} trials from {len(files)} subjects)'
    )
    return df


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

        df_group = load_group_data(behav_data_file, formula_v, inputs)
        utils.log_msg(
            f'        Loaded {len(df_group)} trials from '
            f'{df_group["participant"].nunique()} subjects'
        )

        utils.log_msg('        *** Hierarchical Bayesian DDM (HSSM) ***')
        model = fit_hssm_hierarchical(
            df_group, condition_dict, formula_v, model_type,
            prior_settings, link_settings, draws, tune, chains, cores, target_accept, sampler,
            formula_z=formula_z, formula_t=formula_t, formula_a=formula_a, fix_t=fix_t
        )

        # Posterior summary -> CSV
        # model.summary() triggers log-likelihood recomputation which has a dimension
        # mismatch bug in hssm 0.2.x with numpyro + random effects; go direct to ArviZ.
        import arviz as az
        idata = model._inference_obj
        summary = az.summary(idata)
        summary_path = os.path.join(result_dir, 'hssm_posterior_summary.csv')
        summary.to_csv(summary_path)
        utils.log_msg(f'        Posterior summary: {summary_path}')
        utils.log_msg(f'\n{summary.to_string()}\n')

        # Persist the InferenceData so posterior plots can be regenerated without refitting.
        idata_path = os.path.join(result_dir, 'hssm_idata.nc')
        idata.to_netcdf(idata_path)
        utils.log_msg(f'        InferenceData: {idata_path}')

        # Posterior plots (functions live in plotting_module to keep this module clean):
        # readable trace, Romei & Tarasi Fig 4C coefficient histograms, Franzen Fig 5
        # ridgelines per condition, and the DDM-anatomy schematic.
        cond_col = list(condition_dict.items())[0][0]
        plotting.hssm_trace(idata, result_dir)
        plotting.hssm_posterior_coefficients(idata, result_dir)
        plotting.hssm_posterior_ridgeline(idata, 'v', cond_col,
                                          condition_dict[cond_col], result_dir, link='identity')
        if 'z_Intercept' in idata.posterior.data_vars:
            # z now shares the exp factor (formula_z = z ~ 1 + exp), so the start-point
            # ridgeline runs over the same conditions as v; logit link back to [0,1].
            plotting.hssm_posterior_ridgeline(idata, 'z', cond_col,
                                              condition_dict[cond_col], result_dir, link='logit')
        plotting.hssm_ddm_schematic(idata, condition_dict, result_dir, t_value=fix_t)

        # Provenance log
        utils.log_update(log_df, 'hssm_model', model_type)
        utils.log_update(log_df, 'hssm_formula_v', formula_v)
        utils.log_update(log_df, 'hssm_draws', draws)
        utils.log_update(log_df, 'hssm_chains', chains)

    timepoint_end = utils.log_msg('DONE:   HSSM Analysis Module')
    utils.log_save(log_df, f'{bidspath_processing.root}', 'log_dataframe.csv')
    utils.log_msg(f'        Time elapsed: {str(timepoint_end - timepoint_start)}\n\n')
# _______________________________________________________________________________
