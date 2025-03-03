import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import nct

# Function to compute power analysis with corrected variance components
def compute_power_analysis_fixed(df_vt, n_stimuli, n_participants_per_stimulus):
    # Print mean VT per condition
    print("Mean VT per Condition:\n", df_vt.groupby("condition")["vt"].mean())

    # Variance Components for Condition 1 vs. 2
    sigma_E2_12 = np.var(df_vt[df_vt["condition"].isin([1, 2])]["vt"] - 
                          df_vt[df_vt["condition"].isin([1, 2])].groupby("condition")["vt"].transform("mean"))
    sigma_P_T2_12 = np.var(df_vt[df_vt["condition"].isin([1, 2])].groupby(["stimulus", "condition"])["vt"].mean() - 
                            df_vt[df_vt["condition"].isin([1, 2])].groupby("stimulus")["vt"].mean())
    sigma_P2_12 = np.var(df_vt[df_vt["condition"].isin([1, 2])].groupby("stimulus")["vt"].mean())
    sigma_P_C2_12 = np.var(df_vt[df_vt["condition"].isin([1, 2])].groupby(["stimulus", "condition"])["vt"].mean())
    sigma_T_C2_12 = np.var(df_vt[df_vt["condition"].isin([1, 2])]["vt"] - 
                            df_vt[df_vt["condition"].isin([1, 2])].groupby(["stimulus", "condition"])["vt"].transform("mean"))

    # Variance Components for Condition 1 vs. 3
    sigma_E2_13 = np.var(df_vt[df_vt["condition"].isin([1, 3])]["vt"] - 
                          df_vt[df_vt["condition"].isin([1, 3])].groupby("condition")["vt"].transform("mean"))
    sigma_P_T2_13 = np.var(df_vt[df_vt["condition"].isin([1, 3])].groupby(["stimulus", "condition"])["vt"].mean() - 
                            df_vt[df_vt["condition"].isin([1, 3])].groupby("stimulus")["vt"].mean())
    sigma_P_T_C2_13 = np.var(df_vt[df_vt["condition"] == 3].groupby("stimulus")["vt"].mean() - 
                              df_vt[df_vt["condition"].isin([1, 3])].groupby("stimulus")["vt"].mean())  # Extra variance for Condition 3
    sigma_P2_13 = np.var(df_vt[df_vt["condition"].isin([1, 3])].groupby("stimulus")["vt"].mean())
    sigma_P_C2_13 = np.var(df_vt[df_vt["condition"].isin([1, 3])].groupby(["stimulus", "condition"])["vt"].mean())
    sigma_T_C2_13 = np.var(df_vt[df_vt["condition"].isin([1, 3])]["vt"] - 
                            df_vt[df_vt["condition"].isin([1, 3])].groupby(["stimulus", "condition"])["vt"].transform("mean"))

# Variance Components for Condition 2 vs. 3
    sigma_E2_23 = np.var(df_vt[df_vt["condition"].isin([2, 3])]["vt"] - 
                          df_vt[df_vt["condition"].isin([2, 3])].groupby("condition")["vt"].transform("mean"))
    sigma_P_T2_23 = np.var(df_vt[df_vt["condition"].isin([2, 3])].groupby(["stimulus", "condition"])["vt"].mean() - 
                            df_vt[df_vt["condition"].isin([2, 3])].groupby("stimulus")["vt"].mean())
    sigma_P_T_C2_23 = np.var(df_vt[df_vt["condition"] == 3].groupby("stimulus")["vt"].mean() - 
                              df_vt[df_vt["condition"].isin([2, 3])].groupby("stimulus")["vt"].mean())  # Extra variance for Condition 3
    sigma_P2_23 = np.var(df_vt[df_vt["condition"].isin([2, 3])].groupby("stimulus")["vt"].mean())
    sigma_P_C2_23 = np.var(df_vt[df_vt["condition"].isin([2, 3])].groupby(["stimulus", "condition"])["vt"].mean())
    sigma_T_C2_23 = np.var(df_vt[df_vt["condition"].isin([2, 3])]["vt"] - 
                            df_vt[df_vt["condition"].isin([2, 3])].groupby(["stimulus", "condition"])["vt"].transform("mean"))



    # Print variance components
    print("\nVariance Components for Condition 1 vs. 2:")
    print(f"Residual Variance (σ_E2_12): {sigma_E2_12:.6f}")
    print(f"Stimulus-Condition Interaction Variance (σ_P_T2_12): {sigma_P_T2_12:.6f}")
    print(f"Stimulus Variance (σ_P2_12): {sigma_P2_12:.6f}")
    print(f"Condition-Stimulus Interaction Variance (σ_P_C2_12): {sigma_P_C2_12:.6f}")
    print(f"Condition Variance within Stimuli (σ_T_C2_12): {sigma_T_C2_12:.6f}")

    print("\nVariance Components for Condition 1 vs. 3:")
    print(f"Residual Variance (σ_E2_13): {sigma_E2_13:.6f}")
    print(f"Stimulus-Condition Interaction Variance (σ_P_T2_13): {sigma_P_T2_13:.6f}")
    print(f"Extra Variance for Condition 3 (σ_P_T_C2_13): {sigma_P_T_C2_13:.6f}")
    print(f"Stimulus Variance (σ_P2_13): {sigma_P2_13:.6f}")
    print(f"Condition-Stimulus Interaction Variance (σ_P_C2_13): {sigma_P_C2_13:.6f}")
    print(f"Condition Variance within Stimuli (σ_T_C2_13): {sigma_T_C2_13:.6f}")
    
    print("\nVariance Components for Condition 2 vs. 3:")
    print(f"Residual Variance (σ_E2_23): {sigma_E2_23:.6f}")
    print(f"Stimulus-Condition Interaction Variance (σ_P_T2_23): {sigma_P_T2_23:.6f}")
    print(f"Extra Variance for Condition 3 (σ_P_T_C2_23): {sigma_P_T_C2_23:.6f}")
    print(f"Stimulus Variance (σ_P2_23): {sigma_P2_23:.6f}")
    print(f"Condition-Stimulus Interaction Variance (σ_P_C2_23): {sigma_P_C2_23:.6f}")
    print(f"Condition Variance within Stimuli (σ_T_C2_23): {sigma_T_C2_23:.6f}")
    

    # Compute mean difference (effect size d)
    d_12 = df_vt[df_vt["condition"] == 2]["vt"].mean() - df_vt[df_vt["condition"] == 1]["vt"].mean()
    d_13 = df_vt[df_vt["condition"] == 3]["vt"].mean() - df_vt[df_vt["condition"] == 1]["vt"].mean()
    d_23 = df_vt[df_vt["condition"] == 3]["vt"].mean() - df_vt[df_vt["condition"] == 2]["vt"].mean()

    print(f"Effect Size (d) for Condition 1 vs. 2: {d_12:.3f}")
    print(f"Effect Size (d) for Condition 1 vs. 3: {d_13:.3f}")
    print(f"Effect Size (d) for Condition 2 vs. 3: {d_23:.3f}")


    # Compute noncentrality parameter (λ) for Condition 1 vs. 2
    lambda_12 = abs(d_12) / np.sqrt(
        (sigma_E2_12 + sigma_P_T2_12) / (n_participants_per_stimulus * n_stimuli) +
        (sigma_P2_12 + sigma_P_C2_12) / n_participants_per_stimulus +
        sigma_T_C2_12 / n_stimuli
    )

    # Compute noncentrality parameter (λ) for Condition 1 vs. 3 (includes sigma_P_T_C2_13)
    lambda_13 = abs(d_13) / np.sqrt(
        (sigma_E2_13 + sigma_P_T2_13 + sigma_P_T_C2_13) / (n_participants_per_stimulus * n_stimuli) +
        (sigma_P2_13 + sigma_P_C2_13) / n_participants_per_stimulus +
        sigma_T_C2_13 / n_stimuli
    )

    # Compute noncentrality parameter (λ) for Condition 2 vs. 3 (includes sigma_P_T_C2_23)
    lambda_23 = abs(d_23) / np.sqrt(
        (sigma_E2_23 + sigma_P_T2_23 + sigma_P_T_C2_23) / (n_participants_per_stimulus * n_stimuli) +
        (sigma_P2_23 + sigma_P_C2_23) / n_participants_per_stimulus +
        sigma_T_C2_23 / n_stimuli
    )


    print(f"Lambda for Condition 1 vs. 2: {lambda_12:.6f}")
    print(f"Lambda for Condition 1 vs. 3: {lambda_13:.6f}")
    print(f"Lambda for Condition 2 vs. 3: {lambda_23:.6f}")


    # Compute degrees of freedom correctly
    df_12 = (n_participants_per_stimulus - 1) * n_stimuli
    df_13 = (n_participants_per_stimulus - 1) * n_stimuli
    df_23 = (n_participants_per_stimulus - 1) * n_stimuli

    # Compute critical t-value for alpha = 0.05 (two-tailed)
    t_critical_12 = stats.t.ppf(1 - 0.05 / 2, df_12)
    t_critical_13 = stats.t.ppf(1 - 0.05 / 2, df_13)
    t_critical_23 = stats.t.ppf(1 - 0.05 / 2, df_23)

    # Compute power using the noncentral t-distribution (corrected)
    power_12 = 1 - nct.cdf(t_critical_12, df_12, lambda_12)
    power_13 = 1 - nct.cdf(t_critical_13, df_13, lambda_13)
    power_23 = 1 - nct.cdf(t_critical_23, df_23, lambda_23)

    # Print results
    print(f"Power Analysis for {n_stimuli} Stimuli, {n_participants_per_stimulus} Participants/Stimulus")
    print(f"Condition 1 vs. 2 - Power: {power_12:.3f}")
    print(f"Condition 1 vs. 3 - Power: {power_13:.3f}")
    print("-" * 50)

    return power_12, power_13, power_23

# Load the datasets if already generated
df_vt_60stim_5part = pd.read_csv("3ppl_simulated_vt_60stim_5part.csv")
df_vt_50stim_5part = pd.read_csv("3ppl_simulated_vt_50stim_5part.csv")
df_vt_40stim_5part = pd.read_csv("3ppl_simulated_vt_40stim_5part.csv")

# Compute power analysis for each dataset (fixed version)
compute_power_analysis_fixed(df_vt_60stim_5part, n_stimuli=60, n_participants_per_stimulus=3)
compute_power_analysis_fixed(df_vt_50stim_5part, n_stimuli=50, n_participants_per_stimulus=3)
compute_power_analysis_fixed(df_vt_40stim_5part, n_stimuli=40, n_participants_per_stimulus=3)
