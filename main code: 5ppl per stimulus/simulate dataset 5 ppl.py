import numpy as np
import pandas as pd

# Function to simulate VT dataset
def simulate_vt_data(n_stimuli, target_effect_size=0.4, seed=42):
    np.random.seed(seed)  # Ensure reproducibility
    n_conditions = 3  # Three conditions (Control, Human-AI, AI-Human)
    n_total_participants = n_stimuli * 5 * n_conditions  # 5 participants per stimulus

    # Assign conditions (1: Control, 2: Human-AI, 3: AI-Human)
    conditions = np.repeat(np.arange(1, n_conditions + 1), n_stimuli * 5)

    # Assign stimuli
    stimuli = np.tile(np.arange(1, n_stimuli + 1), 5 * n_conditions)

    # Assign participants
    participants = np.arange(n_total_participants)

    # Likert scale bounds
    likert_min, likert_max = 1, 7

    # Define effect size adjustments
    mean_control = 4.0  # Baseline mean in the middle of the Likert scale
    sd_residual = 0.8   # Standard deviation
    d_target = target_effect_size * sd_residual  # Adjusted mean difference

    # Ensure correct direction of effect
    mean_human_ai = mean_control - d_target  # Condition 2 should be lower
    mean_ai_human = mean_control - d_target  # Condition 3 should be even lower

    # Generate item-level responses (5 items per participant)
    item_scores = np.zeros((n_total_participants, 5))
    for i in range(5):
        item_scores[:, i] = np.select(
            [
                conditions == 1,  # Control
                conditions == 2,  # Human-AI
                conditions == 3   # AI-Human
            ],
            [
                np.random.normal(mean_control, sd_residual,  n_total_participants),
                np.random.normal(mean_human_ai, sd_residual, n_total_participants),
                np.random.normal(mean_ai_human, sd_residual, n_total_participants)
            ]
        )

        # Ensure values stay within the Likert scale range and round to nearest integer
        item_scores[:, i] = np.clip(item_scores[:, i], likert_min, likert_max)
        item_scores[:, i] = np.round(item_scores[:, i])  

    # Compute VT as the average of the 5 items
    vt = item_scores.mean(axis=1)

    # Create DataFrame
    df_vt = pd.DataFrame({
        "participant": participants,
        "condition": conditions,
        "stimulus": stimuli,
        "item1": item_scores[:, 0],
        "item2": item_scores[:, 1],
        "item3": item_scores[:, 2],
        "item4": item_scores[:, 3],
        "item5": item_scores[:, 4],
        "vt": vt  # VT is the mean of 5 item scores
    })

    return df_vt

# Generate datasets for 60, 50, 40, 30, 20 stimuli
df_vt_130stim_5part = simulate_vt_data(n_stimuli=130)
df_vt_120stim_5part = simulate_vt_data(n_stimuli=120)
df_vt_110stim_5part = simulate_vt_data(n_stimuli=110)
df_vt_100stim_5part = simulate_vt_data(n_stimuli=100)
df_vt_90stim_5part = simulate_vt_data(n_stimuli=90)
df_vt_80stim_5part = simulate_vt_data(n_stimuli=80)
df_vt_70stim_5part = simulate_vt_data(n_stimuli=70)
df_vt_60stim_5part = simulate_vt_data(n_stimuli=60)
df_vt_50stim_5part = simulate_vt_data(n_stimuli=50)
df_vt_40stim_5part = simulate_vt_data(n_stimuli=40)
df_vt_30stim_5part = simulate_vt_data(n_stimuli=30)
df_vt_20stim_5part = simulate_vt_data(n_stimuli=20)
df_vt_10stim_5part = simulate_vt_data(n_stimuli=10)
df_vt_5stim_5part = simulate_vt_data(n_stimuli=5)

# Save datasets as CSV files
df_vt_130stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_130stim_5part.csv", index=False)
df_vt_120stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_120stim_5part.csv", index=False)
df_vt_110stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_110stim_5part.csv", index=False)
df_vt_100stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_100stim_5part.csv", index=False)
df_vt_90stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_90stim_5part.csv", index=False)
df_vt_80stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_80stim_5part.csv", index=False)
df_vt_70stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_70stim_5part.csv", index=False)
df_vt_60stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_60stim_5part.csv", index=False)
df_vt_50stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_50stim_5part.csv", index=False)
df_vt_40stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_40stim_5part.csv", index=False)
df_vt_30stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_30stim_5part.csv", index=False)
df_vt_20stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_20stim_5part.csv", index=False)
df_vt_10stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_10stim_5part.csv", index=False)
df_vt_5stim_5part.to_csv("main code: 5ppl per stimulus/simulated_vt_5stim_5part.csv", index=False)



print("✅ Datasets generated with **corrected effect size (0.3) between conditions!**")
