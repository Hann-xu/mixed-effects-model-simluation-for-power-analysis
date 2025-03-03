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
    sigma_T2_12 = np.var(df_vt[df_vt["condition"].isin([1, 2])].groupby("stimulus")["vt"].mean())
    sigma_P_T_C2_12 = np.var(df_vt[df_vt["condition"] == 2].groupby("stimulus")["vt"].mean() - 
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
    sigma_T2_13 = np.var(df_vt[df_vt["condition"].isin([1, 3])].groupby("stimulus")["vt"].mean())
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
    sigma_T2_23 = np.var(df_vt[df_vt["condition"].isin([2, 3])].groupby("stimulus")["vt"].mean())
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


    # Calculate standardized effect sizes
    effect_size_12 = d_12 / np.sqrt(
        sigma_P2_12 + sigma_P_C2_12 + 
        sigma_T2_12 +  # σ²ᴛ (not explicitly calculated in your code in last version)
        sigma_T_C2_12 + 
        sigma_E2_12 + 
        sigma_P_T2_12 + 
        sigma_P_T_C2_12  
    )

    effect_size_13 = d_13 / np.sqrt(
        sigma_P2_13 + sigma_P_C2_13 + 
        sigma_T2_13 +  # σ²ᴛ (not explicitly calculated in your code)
        sigma_T_C2_13 + 
        sigma_E2_13 + 
        sigma_P_T2_13 + 
        sigma_P_T_C2_13  # This might be your σ²ₚ×ᴛ×ᴄ equivalent
    )

    effect_size_23 = d_23 / np.sqrt(
        sigma_P2_23 + sigma_P_C2_23 + 
        sigma_T2_23 +  # σ²ᴛ (not explicitly calculated in your code)
        sigma_T_C2_23 + 
        sigma_E2_23 + 
        sigma_P_T2_23 + 
        sigma_P_T_C2_23  # This might be your σ²ₚ×ᴛ×ᴄ equivalent
    )

    # Print effect sizes
    print("\nStandardized Effect Sizes:")
    print(f"Effect Size (d) for Condition 1 vs. 2: {effect_size_12:.4f}")
    print(f"Effect Size (d) for Condition 1 vs. 3: {effect_size_13:.4f}")
    print(f"Effect Size (d) for Condition 2 vs. 3: {effect_size_23:.4f}")


    # Compute noncentrality parameter (λ) for Condition 1 vs. 2
    lambda_12 = d_12 /(2*np.sqrt(
        (sigma_E2_12 + sigma_P_T2_12 + sigma_P_T_C2_12) / (n_participants_per_stimulus * n_stimuli) +
        (sigma_P2_12 + sigma_P_C2_12) / n_participants_per_stimulus +
        sigma_T_C2_12 / n_stimuli
    ))

    # Compute noncentrality parameter (λ) for Condition 1 vs. 3 (includes sigma_P_T_C2_13)
    lambda_13 = d_13 / (2*np.sqrt(
        (sigma_E2_13 + sigma_P_T2_13 + sigma_P_T_C2_13) / (n_participants_per_stimulus * n_stimuli) +
        (sigma_P2_13 + sigma_P_C2_13) / n_participants_per_stimulus +
        sigma_T_C2_13 / n_stimuli
    ))

    # Compute noncentrality parameter (λ) for Condition 2 vs. 3 (includes sigma_P_T_C2_23)
    lambda_23 = d_23 / (2*np.sqrt(
        (sigma_E2_23 + sigma_P_T2_23 + sigma_P_T_C2_23) / (n_participants_per_stimulus * n_stimuli) +
        (sigma_P2_23 + sigma_P_C2_23) / n_participants_per_stimulus +
        sigma_T_C2_23 / n_stimuli
    ))


    print(f"Lambda for Condition 1 vs. 2: {lambda_12:.6f}")
    print(f"Lambda for Condition 1 vs. 3: {lambda_13:.6f}")
    print(f"Lambda for Condition 2 vs. 3: {lambda_23:.6f}")


    # Calculate degrees of freedom using Satterthwaite approximation for Condition 1 vs. 2
    E_12 = sigma_E2_12 + sigma_P_T2_12 + sigma_P_T_C2_12  # Error components
    p = 45 

    numerator_12 = (E_12 + n_stimuli*(sigma_P2_12 + sigma_P_C2_12) + p*sigma_T_C2_12/2)**2
    denominator_12 = ((E_12)**2/((p-2)*(n_stimuli-1)) + 
                    (E_12 + n_stimuli*(sigma_P2_12 + sigma_P_C2_12))**2/(p-2) + 
                    (E_12 + p*sigma_T_C2_12/2)**2/(n_stimuli-1))
    df_12 = numerator_12 / denominator_12

    # Calculate degrees of freedom for Condition 1 vs. 3
    E_13 = sigma_E2_13 + sigma_P_T2_13 + sigma_P_T_C2_13  # Error components

    numerator_13 = (E_13 + n_stimuli*(sigma_P2_13 + sigma_P_C2_13) + p*sigma_T_C2_13/2)**2
    denominator_13 = ((E_13)**2/((p-2)*(n_stimuli-1)) + 
                    (E_13 + n_stimuli*(sigma_P2_13 + sigma_P_C2_13))**2/(p-2) + 
                    (E_13 + p*sigma_T_C2_13/2)**2/(n_stimuli-1))
    df_13 = numerator_13 / denominator_13

    # Calculate degrees of freedom for Condition 2 vs. 3
    E_23 = sigma_E2_23 + sigma_P_T2_23 + sigma_P_T_C2_23  # Error components

    numerator_23 = (E_23 + n_stimuli*(sigma_P2_23 + sigma_P_C2_23) + p*sigma_T_C2_23/2)**2
    denominator_23 = ((E_23)**2/((p-2)*(n_stimuli-1)) + 
                    (E_23 + n_stimuli*(sigma_P2_23 + sigma_P_C2_23))**2/(p-2) + 
                    (E_23 + p*sigma_T_C2_23/2)**2/(n_stimuli-1))
    df_23 = numerator_23 / denominator_23

    # Print the calculated degrees of freedom
    print(f"\nDegrees of Freedom:")
    print(f"df for Condition 1 vs. 2: {df_12:.2f}")
    print(f"df for Condition 1 vs. 3: {df_13:.2f}")
    print(f"df for Condition 2 vs. 3: {df_23:.2f}")


    # Compute critical t-value for alpha = 0.05 (two-tailed)
    t_critical_12 = stats.t.ppf(1 - 0.05 / 2, df_12)
    t_critical_13 = stats.t.ppf(1 - 0.05 / 2, df_13)
    t_critical_23 = stats.t.ppf(1 - 0.05 / 2, df_23)

    # Compute power using the noncentral t-distribution (corrected)
    power_12 = 1- nct.cdf(t_critical_12, df_12, abs(lambda_12))
    power_13 = 1- nct.cdf(t_critical_13, df_13, abs(lambda_13))
    power_23 = 1- nct.cdf(t_critical_23, df_23, abs(lambda_23))

    # Print results
    print(f"Power Analysis for {n_stimuli} Stimuli, {n_participants_per_stimulus} Participants/Stimulus")
    print(f"Condition 1 vs. 2 - Power: {power_12:.3f}")
    print(f"Condition 1 vs. 3 - Power: {power_13:.3f}")
    print(f"Condition 2 vs. 3 - Power: {power_23:.3f}")
    print("-" * 50)

    return power_12, power_13, power_23

# Load the datasets if already generated
df_vt_5stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_5stim_15part.csv")
df_vt_10stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_10stim_15part.csv")
df_vt_20stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_20stim_15part.csv")
df_vt_30stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_30stim_15part.csv")
df_vt_40stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_40stim_15part.csv")
df_vt_50stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_50stim_15part.csv")
df_vt_60stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_60stim_15part.csv")
df_vt_70stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_70stim_15part.csv")
df_vt_80stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_80stim_15part.csv")
df_vt_90stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_90stim_15part.csv")
df_vt_100stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_100stim_15part.csv")
df_vt_110stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_110stim_15part.csv")
df_vt_120stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_120stim_15part.csv")
df_vt_130stim_15part = pd.read_csv("main code: 15ppl per stimulus/simulated_vt_130stim_15part.csv")

# Compute power analysis for each dataset (fixed version)
compute_power_analysis_fixed(df_vt_5stim_15part, n_stimuli=5, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_10stim_15part, n_stimuli=10, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_20stim_15part, n_stimuli=20, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_30stim_15part, n_stimuli=30, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_40stim_15part, n_stimuli=40, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_50stim_15part, n_stimuli=50, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_60stim_15part, n_stimuli=60, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_70stim_15part, n_stimuli=70, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_80stim_15part, n_stimuli=80, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_90stim_15part, n_stimuli=90, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_100stim_15part, n_stimuli=100, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_110stim_15part, n_stimuli=110, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_120stim_15part, n_stimuli=120, n_participants_per_stimulus=15)
compute_power_analysis_fixed(df_vt_130stim_15part, n_stimuli=130, n_participants_per_stimulus=15)

# Create empty lists to store the results
stimuli_counts = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
power_12_list = []
power_13_list = []
power_23_list = []

# Load the datasets and run the power analysis
# (This section will collect the results returned by the compute_power_analysis_fixed function)
power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_5stim_15part, n_stimuli=5, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_10stim_15part, n_stimuli=10, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_20stim_15part, n_stimuli=20, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_30stim_15part, n_stimuli=30, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_40stim_15part, n_stimuli=40, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_50stim_15part, n_stimuli=50, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_60stim_15part, n_stimuli=60, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_70stim_15part, n_stimuli=70, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_80stim_15part, n_stimuli=80, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_90stim_15part, n_stimuli=90, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_100stim_15part, n_stimuli=100, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_110stim_15part, n_stimuli=110, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_120stim_15part, n_stimuli=120, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

power_12, power_13, power_23 = compute_power_analysis_fixed(df_vt_130stim_15part, n_stimuli=130, n_participants_per_stimulus=5)
power_12_list.append(power_12)
power_13_list.append(power_13)
power_23_list.append(power_23)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'n_stimuli': stimuli_counts,
    'power_12': power_12_list,
    'power_13': power_13_list,
    'power_23': power_23_list
})

# Save the results to a CSV file
results_df.to_csv('power_analysis_results_15ppl.csv', index=False)
print("\nPower analysis results saved to power_analysis_results_15ppl.csv")
