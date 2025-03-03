import numpy as np
import pandas as pd

# Function to simulate VT dataset with proper random effects structure
def simulate_vt_data(n_stimuli, n_participants_per_stimulus=5, target_effect_size=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Use different seeds for different simulations
    
    n_conditions = 3  # Three conditions (Control, Human-AI, AI-Human)
    
    # Generate random effects
    # Random stimulus effects
    stimulus_effects = np.random.normal(0, 0.3, n_stimuli)
    
    # Create participant effects
    participant_effects = np.random.normal(0, 0.4, n_stimuli * n_participants_per_stimulus)
    
    # Create stimulus-by-condition interaction effects
    stimulus_by_condition_effects = np.random.normal(0, 0.25, n_stimuli * n_conditions)
    
    # Likert scale bounds
    likert_min, likert_max = 1, 7
    
    # Baseline mean in the middle of the Likert scale
    mean_control = 4.0
    
    # Define condition effects
    condition_effects = [0, -target_effect_size, -target_effect_size * 1.05]
    
    # Create the data structure
    data = []
    participant_id = 0
    
    for stim_idx in range(n_stimuli):
        stimulus = stim_idx + 1
        
        for part_in_stim in range(n_participants_per_stimulus):
            participant = participant_id + 1
            
            for cond_idx in range(n_conditions):
                condition = cond_idx + 1
                
                # Get the base score with all random effects
                mean_score = (
                    mean_control +                              # Base score
                    condition_effects[cond_idx] +               # Fixed condition effect
                    stimulus_effects[stim_idx] +                # Random stimulus effect
                    participant_effects[participant_id] +       # Random participant effect
                    stimulus_by_condition_effects[stim_idx * n_conditions + cond_idx]  # Stimulus-by-condition interaction
                )
                
                # Generate 5 item scores with residual error
                item_scores = []
                for i in range(5):
                    # Add residual error to each item
                    score = mean_score + np.random.normal(0, 0.6)
                    # Ensure scores stay within Likert scale bounds
                    score = np.clip(score, likert_min, likert_max)
                    # Round to nearest integer for Likert scale
                    score = round(score)
                    item_scores.append(score)
                
                # Calculate mean VT score
                vt = np.mean(item_scores)
                
                # Add row to dataset
                data.append({
                    "participant": participant,
                    "condition": condition,
                    "stimulus": stimulus,
                    "item1": item_scores[0],
                    "item2": item_scores[1],
                    "item3": item_scores[2],
                    "item4": item_scores[3],
                    "item5": item_scores[4],
                    "vt": vt
                })
            
            participant_id += 1
    
    # Create DataFrame
    df_vt = pd.DataFrame(data)
    return df_vt

# Generate datasets for different numbers of stimuli with different random seeds
df_vt_60stim_5part = simulate_vt_data(n_stimuli=60, seed=100)
df_vt_50stim_5part = simulate_vt_data(n_stimuli=50, seed=200)
df_vt_40stim_5part = simulate_vt_data(n_stimuli=40, seed=300)
df_vt_30stim_5part = simulate_vt_data(n_stimuli=30, seed=400)
df_vt_20stim_5part = simulate_vt_data(n_stimuli=20, seed=500)

# Save datasets as CSV files
df_vt_60stim_5part.to_csv("s2_simulated_vt_60stim_5part.csv", index=False)
df_vt_50stim_5part.to_csv("s2_simulated_vt_50stim_5part.csv", index=False)
df_vt_40stim_5part.to_csv("s2_simulated_vt_40stim_5part.csv", index=False)
df_vt_30stim_5part.to_csv("s2_simulated_vt_30stim_5part.csv", index=False)
df_vt_20stim_5part.to_csv("s2_simulated_vt_20stim_5part.csv", index=False)

print("✅ Datasets generated with proper mixed-effects structure and effect size between conditions!")