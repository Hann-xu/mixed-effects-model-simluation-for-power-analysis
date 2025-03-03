import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ncf

def compute_main_effect_power(df_vt, n_stimuli, n_participants_per_stimulus):
    # Print mean VT per condition
    print("Mean VT per Condition:\n", df_vt.groupby("condition")["vt"].mean())

    # Compute variance components following the NCC design equations
    sigma_E2 = np.var(df_vt["vt"] - df_vt.groupby("condition")["vt"].transform("mean"))
    sigma_P2 = np.var(df_vt.groupby("participant")["vt"].mean())
    sigma_T2 = np.var(df_vt.groupby("stimulus")["vt"].mean())
    sigma_P_C2 = np.var(df_vt.groupby(["participant", "condition"])["vt"].mean())
    sigma_T_C2 = np.var(df_vt.groupby(["stimulus", "condition"])["vt"].mean())
    sigma_P_T2 = np.var(df_vt.groupby(["participant", "stimulus"])["vt"].mean())
    sigma_P_T_C2 = np.var(df_vt.groupby(["stimulus", "participant", "condition"])["vt"].mean())
    
    # Print variance components for debugging
    print(f"sigma_E2: {sigma_E2:.6f}, sigma_P2: {sigma_P2:.6f}, sigma_T2: {sigma_T2:.6f}")
    print(f"sigma_P_C2: {sigma_P_C2:.6f}, sigma_T_C2: {sigma_T_C2:.6f}, sigma_P_T2: {sigma_P_T2:.6f}, sigma_P_T_C2: {sigma_P_T_C2:.6f}")
    
    # Compute total error variance (E)
    E = max(sigma_E2 + sigma_P_T2 + sigma_P_T_C2, 1e-10)  # Avoid zero division
    
    # Compute effect size (d0)
    mean_diff = df_vt.groupby("condition")["vt"].mean().diff().iloc[-1]
    print(f"Mean Difference: {mean_diff:.6f}")
    d0 = mean_diff / max(np.sqrt(sigma_P2 + sigma_P_C2 + sigma_T_C2 + E), 1e-10)
    
    # Compute noncentrality parameter (ncp)
    ncp = mean_diff / (2 * max(np.sqrt(E / (n_participants_per_stimulus * n_stimuli) + (sigma_P2 + sigma_P_C2) / n_participants_per_stimulus + sigma_T_C2 / n_stimuli), 1e-10))
    
    # Compute degrees of freedom (df) following NCC formula
    numerator = (E + n_stimuli * (sigma_P2 + sigma_P_C2) + 15 * sigma_T_C2 / 2) ** 2
    denominator = max((E ** 2) / ((15 - 2) * (n_stimuli - 1)) + ((E + n_stimuli * (sigma_P2 + sigma_P_C2)) ** 2) / (n_participants_per_stimulus - 2) + ((E + n_participants_per_stimulus * sigma_T_C2 / 2) ** 2) / (n_stimuli - 1), 1e-10)
    df_effect = numerator / denominator
    
    # Print degrees of freedom for debugging
    print(f"Degrees of Freedom: {df_effect:.6f}")
    
    # Compute F-statistic
    F_stat = (d0 ** 2) / max(df_effect, 1e-10)
    
    # Compute critical F-value for alpha = 0.05 (two-tailed)
    F_critical = stats.f.ppf(1 - 0.05, df_effect, df_effect) if df_effect > 0 else np.nan
    
    # Compute power using noncentral F-distribution
    power_main_effect = 1 - ncf.cdf(F_critical, df_effect, df_effect, ncp) if not np.isnan(F_critical) else np.nan
    
    # Print results
    print(f"Power for Main Effect of Condition: {power_main_effect:.3f}")
    return power_main_effect


# Example usage (replace with actual dataset loading)
df_vt_example = pd.read_csv("simulated_vt_40stim_5part.csv")
compute_main_effect_power(df_vt_example, n_stimuli=40, n_participants_per_stimulus=5)
