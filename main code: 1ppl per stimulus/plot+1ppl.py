import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the power analysis results
results_df = pd.read_csv('power_analysis_results_1ppl.csv')

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the power values for each comparison
plt.plot(results_df['n_stimuli'], results_df['power_12'], 'o-', color='blue', linewidth=2, markersize=8, label='Condition 1 vs 2')
plt.plot(results_df['n_stimuli'], results_df['power_13'], 's-', color='red', linewidth=2, markersize=8, label='Condition 1 vs 3')
plt.plot(results_df['n_stimuli'], results_df['power_23'], '^-', color='green', linewidth=2, markersize=8, label='Condition 2 vs 3')

# Add a horizontal line at power = 0.8 (conventional threshold)
plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, label='Power = 0.8')

# Add labels and title
plt.xlabel('Number of Stimuli', fontsize=14)
plt.ylabel('Statistical Power', fontsize=14)
plt.title('1 participant per stimuli scenario', fontsize=16)

# Customize the plot
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Set the y-axis limits
plt.ylim(0, 1.05)

# Set the x-axis ticks
plt.xticks(results_df['n_stimuli'])

# Tight layout to ensure everything fits
plt.tight_layout()

# Save the figure
plt.savefig('power_analysis_by_stimuli_1ppl.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()