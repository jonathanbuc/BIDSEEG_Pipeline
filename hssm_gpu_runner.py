"""
Standalone GPU/CPU HSSM runner (WSL + jax).

Mirrors prep_hssm_data / fit_hssm_hierarchical from e_HSSM_module.py but imports only
the HSSM/jax/arviz stack -- NOT utils_module (which pulls mne/fooof and isn't installed
in the GPU env). Lets the hierarchical DDM sample on the GPU via jax[cuda12] under WSL,
where native-Windows jax is CPU-only.

Usage (inside the hssm-gpu WSL env, from the project root):
    python hssm_gpu_runner.py <config.json> [tag]

Paths are derived from the config's basic.bids_root_out, mirroring e_HSSM_module: it loads
the group behavioural CSV, or -- when formula_v names an alpha covariate -- the per-trial
alpha CSVs from results/groupEEG/trial_alpha (Stage 2). Outputs land next to module e's, with
`_<tag>` appended (default tag = the active jax backend, so gpu/cpu runs don't clobber).

Force CPU for a same-env baseline:  JAX_PLATFORMS=cpu python hssm_gpu_runner.py <config.json>
"""
import os, sys, json, time
import pandas as pd


def load_group_data(result_dir, proc_root, formula_v):
    """Mirror e_HSSM_module.load_group_data: alpha formula -> per-trial alpha table
    (EEG_iaf.csv), otherwise the group behavioural CSV."""
    if 'alpha' not in formula_v:
        return pd.read_csv(os.path.join(result_dir, 'behavioraldata_hierprior.csv'))
    iaf_file = os.path.join(proc_root, 'results', 'SpectralParameterization', 'EEG_iaf.csv')
    if not os.path.exists(iaf_file):
        raise FileNotFoundError(
            f'Formula uses an alpha covariate but the per-trial alpha table was not found at '
            f'{iaf_file}.\nRun c_EEGAnalysis_module.py with compute_fooof = true first.'
        )
    df = pd.read_csv(iaf_file)
    print(f'  alpha covariate -> per-trial table ({len(df)} trials from {df["participant"].nunique()} subjects)')
    return df


def prep_hssm_data(df, cond_col, conditions, formula=None):
    """Recode/center columns exactly as e_HSSM_module.prep_hssm_data does."""
    df = df.copy()
    if 'rt_flag' in df.columns:
        df = df[~df['rt_flag']]
    df = df[df['rt'] > 0].reset_index(drop=True)
    df['response'] = df['response_prior'].map({1: 1, 0: -1}).astype(int)
    if 'coh_level' in df.columns:
        df['coh_level'] = df['coh_level'].map({'low': 0, 'medium': 1})
        df = df.dropna(subset=['coh_level']).reset_index(drop=True)
        df['coh_level'] = df['coh_level'].astype(int)
    if 'coh' in df.columns:
        df = df.dropna(subset=['coh']).reset_index(drop=True)
        df['coh_gc'] = df['coh'] - df['coh'].mean()
        subj_mean = df.groupby('participant')['coh'].transform('mean')
        df['coh_wc'] = df['coh'] - subj_mean
        df['coh_subjmean'] = subj_mean - df['coh'].mean()
    for acol in ('alpha_cf_fooof', 'alpha_cf_cog', 'alpha_cf_hilbert',
                 'alpha_exp_fooof', 'alpha_offset_fooof'):
        if acol in df.columns:
            subj_mean = df.groupby('participant')[acol].transform('mean')
            df[f'{acol}_gc'] = df[acol] - df[acol].mean()
            df[f'{acol}_wc'] = df[acol] - subj_mean
            df[f'{acol}_subjmean'] = subj_mean - df[acol].mean()
    df = df[df[cond_col].isin(conditions)].reset_index(drop=True)
    if formula is not None:
        need = [c for c in df.columns
                if c.startswith(('alpha_cf_', 'alpha_exp_', 'alpha_offset_')) and c in formula]
        if need:
            before = len(df)
            df = df.dropna(subset=need).reset_index(drop=True)
            if before - len(df):
                print(f'  dropped {before - len(df)} trials missing alpha covariate(s) {need}')
    return df


def main():
    cfg_path = sys.argv[1]
    inputs = json.load(open(cfg_path))
    h = inputs['Analysis'].get('hssm', {})
    condition_dict = inputs['Analysis']['conditions']
    cond_col, conditions = list(condition_dict.items())[0]

    # Paths derived from config, exactly like e_HSSM_module (bids_proc = <out>/BIDSprocessed)
    proc_root = os.path.join(inputs['basic']['bids_root_out'], 'BIDSprocessed')
    result_dir = os.path.join(proc_root, 'results', 'groupBehavioral')

    formula_v = h.get('formula_v', 'v ~ 1 + exp * coh_level + (1|participant)')
    formula_z = h.get('formula_z', '')
    formula_t = h.get('formula_t', '')
    formula_a = h.get('formula_a', '')
    fix_t = h.get('fix_t', None)
    draws = h.get('draws', 1000); tune = h.get('tune', 1000)
    chains = h.get('chains', 2); cores = h.get('cores', 2)
    target_accept = h.get('target_accept', 0.9)
    sampler = h.get('sampler', 'nuts_numpyro')
    model_type = h.get('model_type', 'ddm')
    prior_settings = h.get('prior_settings', 'safe')
    link_settings = h.get('link_settings', 'log_logit')

    import jax
    backend = jax.default_backend()
    print(f'jax {jax.__version__} | backend={backend} | devices={jax.devices()}')
    tag = sys.argv[2] if len(sys.argv) > 2 else backend  # default tag = gpu/cpu

    df = load_group_data(result_dir, proc_root, formula_v)
    all_formulas = ' '.join([formula_v, formula_z, formula_t, formula_a])
    df_hssm = prep_hssm_data(df, cond_col, conditions, formula=all_formulas)
    print(f'N={len(df_hssm)} trials, {df_hssm["participant"].nunique()} subjects')
    print(f'formula_v: {formula_v}')

    import hssm
    include = [{"name": "v", "formula": formula_v, "link": "identity"}]
    if fix_t is not None:
        include.append({"name": "t", "prior": float(fix_t)})
    elif formula_t:
        include.append({"name": "t", "formula": formula_t})
    for name, f in (("z", formula_z), ("a", formula_a)):
        if f:
            include.append({"name": name, "formula": f})

    model = hssm.HSSM(data=df_hssm, model=model_type, include=include,
                      prior_settings=prior_settings, link_settings=link_settings)

    import warnings
    sample_kwargs = dict(sampler=sampler, draws=draws, tune=tune,
                         chains=chains, cores=cores, target_accept=target_accept)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # vectorized runs all chains in one GPU pass; fall back if unsupported
            try:
                model.sample(**sample_kwargs, chain_method='vectorized')
            except TypeError:
                model.sample(**sample_kwargs)
        except ValueError as exc:
            if "different number of dimensions" not in str(exc):
                raise
            print("  (skipping log-likelihood: hssm 0.2.x/numpyro compat issue)")
    elapsed = time.time() - t0
    print(f'SAMPLE_TIME_SEC={elapsed:.1f}  (tag={tag})')

    import arviz as az
    idata = model._inference_obj
    summary = az.summary(idata)
    os.makedirs(result_dir, exist_ok=True)
    sp = os.path.join(result_dir, f'hssm_posterior_summary_{tag}.csv')
    summary.to_csv(sp)
    idata.to_netcdf(os.path.join(result_dir, f'hssm_idata_{tag}.nc'))
    print(summary.to_string())
    print(f'WROTE {sp}')


if __name__ == '__main__':
    main()
