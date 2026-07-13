"""
Generate HSSM posterior plots from a saved InferenceData. Runs in the full pipeline
env (mne-env) — it completes the GPU workflow, where run_hssm_gpu.sh (WSL, slim env)
writes hssm_idata_<tag>.nc but can't import plotting_module (needs mne).

Draws the same four plots e_HSSM_module makes: readable trace, Romei & Tarasi Fig 4C
coefficient histograms, Franzen Fig 5 ridgelines (v and z per condition), and the
DDM-anatomy schematic. Outputs are tag-suffixed so runs don't clobber.

Usage:  python plot_hssm.py <config.json> [tag]     # tag matches hssm_idata_<tag>.nc
"""
import sys, os

config_path = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = [sys.argv[0], config_path]      # so utils_module sees the config at import

import arviz as az
import utils_module as utils
import plotting_module as plotting

inputs = utils.read_inputs(config_path)
condition_dict = inputs['Analysis']['conditions']
cond_col = list(condition_dict.items())[0][0]
hssm_cfg = inputs['Analysis'].get('hssm', {})
fix_t = hssm_cfg.get('fix_t', None)

proc = utils.get_bidspath(inputs, 'bids_proc')
result_dir = os.path.join(proc.root, 'results', 'groupBehavioral')

idata_name = f'hssm_idata_{tag}.nc' if tag else 'hssm_idata.nc'
idata = az.from_netcdf(os.path.join(result_dir, idata_name))
sfx = f'_{tag}' if tag else ''

plotting.hssm_trace(idata, result_dir, fname=f'hssm_trace{sfx}.png')
plotting.hssm_posterior_coefficients(idata, result_dir, fname=f'hssm_posterior_coefficients{sfx}.png')
plotting.hssm_posterior_ridgeline(idata, 'v', cond_col, condition_dict[cond_col],
                                  result_dir, link='identity', fname=f'hssm_ridgeline_v{sfx}.png')
if 'z_Intercept' in idata.posterior.data_vars:
    plotting.hssm_posterior_ridgeline(idata, 'z', cond_col, condition_dict[cond_col],
                                      result_dir, link='logit', fname=f'hssm_ridgeline_z{sfx}.png')
plotting.hssm_ddm_schematic(idata, condition_dict, result_dir, t_value=fix_t,
                            fname=f'hssm_ddm_schematic{sfx}.png')
print(f'HSSM plots ({tag or "default"}) written to {result_dir}')
