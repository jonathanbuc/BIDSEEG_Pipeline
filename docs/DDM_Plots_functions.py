import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Model summary of the following structure:
"""
                                     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
v_stimulus:task[social, causality] -0.308  0.064  -0.426   -0.186      0.001    0.001    8515.0    5934.0   1.00
v_stimulus:task[social, direction]  0.007  0.077  -0.129    0.162      0.001    0.001    9051.0    6080.0   1.00
v_Intercept                        -0.046  0.071  -0.183    0.081      0.001    0.001    4447.0    5508.0   1.00
a_stimulus:task[social, causality]  0.042  0.031  -0.020    0.097      0.000    0.000    4320.0    5854.0   1.00
a_stimulus:task[social, direction] -0.028  0.026  -0.076    0.023      0.000    0.000    4255.0    5294.0   1.00
a_Intercept                         0.916  0.041   0.839    0.991      0.001    0.001    1487.0    3031.0   1.00
t_stimulus:task[social, causality]  0.049  0.024   0.003    0.092      0.001    0.000    1993.0    4217.0   1.00
t_stimulus:task[social, direction]  0.067  0.017   0.036    0.102      0.000    0.000    2067.0    3609.0   1.00
t_Intercept                         1.127  0.177   0.772    1.428      0.007    0.004     597.0    1292.0   1.01
z                                   0.505  0.009   0.489    0.521      0.000    0.000    3614.0    5164.0   1.00
"""
# Function that takes the summary of a model; extracts two data frames, one for social-causality (the effect), one for physical-causality (the intercept)
# This needs to be adapted to the model-summary output!
def param_extraction(summary_df, variables, task):
    """
    Parameters:
    - summary_df: summary of a hssm model, e.g. called with arviz.summary()
    - variables: The variables of interest with which the summary was called, i.e. here = ['v_stimulus:task', 'v_Intercept', 'a_stimulus:task', 'a_Intercept', 't_stimulus:task', 't_Intercept', 'z']
    - task: Define which task to extract (here causality or direction)

    """ 
    #if (task != "causality") & (task != "direction"):
    #    raise Exception("Please specify a task (causality/direction)")

    #Prepare data frames
    df_intercept = pd.DataFrame(columns = ['mean','hdi_3%','hdi_97%'])
    df_effect = pd.DataFrame(columns = ['mean','hdi_3%','hdi_97%'])

    for var in variables:
        param = var[0] #Get the letter of the variables name (i.e., the DDM parameter)
        # PHYSICAL: Just the intercepts
        if var[1:] == "_Intercept":
            df_intercept.loc[param] = summary_df.loc[var, ['mean', 'hdi_3%', 'hdi_97%']]

        # SOCIAL: Intercept + effect (mean addition is straightforward)
        elif var[1:] == '_stimulus:task': #ADAPT to your variable names
            effect_row = f"{var}[social, {task}]" ## ADAPT to your variable names
            effect = summary_df.loc[effect_row, ['mean', 'hdi_3%', 'hdi_97%']]
            intercept_vals = summary_df.loc[f"{param}_Intercept"]
            df_effect.loc[param, 'mean'] = intercept_vals['mean'] + effect['mean']
            
            # For HDI: Keeps the HDI from the physical condition, since the other is the HDI of the change, rather than the new mean
            df_effect.loc[param, 'hdi_3%'] = intercept_vals['hdi_3%'] + effect['mean']
            df_effect.loc[param, 'hdi_97%'] = intercept_vals['hdi_97%'] + effect['mean']

        else:
            df_intercept.loc[param] = summary_df.loc[var, ['mean', 'hdi_3%', 'hdi_97%']]
            df_effect.loc[param] = summary_df.loc[var, ['mean', 'hdi_3%', 'hdi_97%']]
    return(df_intercept, df_effect)


def plot_ddm_structure(df_one, df_two, group_name, condition_names, figsize=(14, 6)):
    """
    Plot classic DDM representation for two conditions.
    
    Parameters:
    - df_one, df_two: DataFrames from param_extraction (rows: v,a,t,z; cols: mean, hdi_3%, hdi_97%)
    - group_name: i.e. "Control" or "Patient", for figure caption
    - conditions: Name of the conditions, in order of the dataframes given
    - figsize: figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    conditions = [(condition_names[0], df_one), (condition_names[1], df_two)]
    
    for ax, (cond_name, df) in zip(axes, conditions):
        # Extract parameters
        v_mean = df.loc['v', 'mean']
        v_hdi = [df.loc['v', 'hdi_3%'], df.loc['v', 'hdi_97%']]
        
        a_mean = df.loc['a', 'mean']
        a_hdi = [df.loc['a', 'hdi_3%'], df.loc['a', 'hdi_97%']]
        
        z_mean = df.loc['z', 'mean']
        z_hdi = [df.loc['z', 'hdi_3%'], df.loc['z', 'hdi_97%']]
        
        t_mean = df.loc['t', 'mean']
        t_hdi = [df.loc['t', 'hdi_3%'], df.loc['t', 'hdi_97%']]
        
        # Define Plot Size
        xlim_max = 2.5
        ylim_buffer = 1.15
        
        ax.set_xlim(-0.2, xlim_max)
        ax.set_ylim(-0.2, ylim_buffer)
        ax.set_aspect('equal')
        
        ## Draw decision thresholds with uncertainty
        # Upper boundary
        ax.axhline(y=a_mean, color='red', linewidth=2.5)
        ax.fill_between([-0.2, xlim_max], a_hdi[0], a_hdi[1], 
                        color='red', alpha=0.2)
        
        # Lower boundary
        ax.axhline(y=0, color='blue', linewidth=2.5)
        ax.fill_between([-0.2, xlim_max], a_hdi[0]-a_mean, a_hdi[1]-a_mean, 
                        color='blue', alpha=0.2)
        
        # Starting point with uncertainty lines (smaller dot with error bars)
        start_point = z_mean  # Normalized between boundaries
        start_point_hdi = [z_hdi[0], z_hdi[1]]
        
        # Draw vertical uncertainty lines
        ax.plot([0, 0], start_point_hdi, 'k-', linewidth=2, alpha=0.6)
        # Small dot at the mean
        ax.plot(0, start_point, 'ko', markersize=6)
        
        # Drift trajectory with uncertainty
        time_points = np.linspace(0, xlim_max, 100)
        drift_trajectory = start_point + v_mean * time_points
        
        # Upper and lower uncertainty bounds for trajectory
        v_upper = start_point + v_hdi[1] * (time_points)
        v_lower = start_point + v_hdi[0] * (time_points)
        
        ax.plot(time_points, drift_trajectory, '-', color='black', linewidth=2.5, label=f'Drift (v={v_mean:.3f})')
        ax.fill_between(time_points, v_lower, v_upper, color='gray', alpha=0.4)
        
        # Labels
        ax.text(0.05, start_point+0.1, f'z={z_mean:.3f}', fontsize=9, color='black', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.text(0.2, a_mean + 0.08, f'a={a_mean:.3f}', fontsize=9, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.text(xlim_max * 0.5, -0.1, f't={t_mean:.3f}', fontsize=9, color='purple',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.text(xlim_max * 0.4, a_mean/3, f'v={v_mean:.3f}', fontsize=9, color='purple',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Configure Axes
        ax.set_xlabel('Time (a.u.)', fontsize=10)
        ax.set_ylabel('Evidence', fontsize=10, labelpad = -30)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, a_mean], ['0','0.2', '0.4','0.6', '0.8', '1']) 

        # Plot title: e.g., Healthy Control - Causality
        ax.set_title(f'{group_name} - {cond_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig 

""" Example of Function call:
variables = ['v_stimulus:task', 'v_Intercept', 'a_stimulus:task', 'a_Intercept', 't_stimulus:task', 't_Intercept', 'z']
summary_model_HC = arviz.summary(model_HC.traces, var_names=variables)

df_intercept, df_effect = param_extraction(summary_model_HC, variables, "causality")
fig1 = plot_ddm_structure(df_intercept, df_effect, group_name = "Healthy Controls", conditions = ["Physical", "Social"])
plt.show()
"""