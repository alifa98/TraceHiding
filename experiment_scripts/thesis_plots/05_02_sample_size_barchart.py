import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch # Needed for custom legend for hatches
import os

# Values for each metric
metric_names = ['UA', 'RA', 'TA', 'MIA']
sample_sizes = [1, 5, 10, 20]
datasets = ['HO-Rome', 'HO-Geolife', 'HO-NYC']
methods = ['TraceHiding (Ent.)', 'NegGrad']


data = {
    'HO-Rome': {
        'UA': {
            'TraceHiding (Ent.)': [82.94, 86.51, 88.31, 90.47],
            'NegGrad': [89.51, 88.80, 92.64, 92.30],
        },
        'RA': {
            'TraceHiding (Ent.)': [38.39, 35.39, 32.17, 28.77],
            'NegGrad': [38.78, 32.79, 21.73, 20.21],
        },
        'TA': {
            'TraceHiding (Ent.)': [22.06, 20.93, 19.38, 16.61],
            'NegGrad': [22.56, 20.28, 13.72, 12.25],
        },
        'MIA': {
            'TraceHiding (Ent.)': [31.48, 22.56, 18.79, 14.24],
            'NegGrad': [34.64, 20.53, 19.43, 13.03],
        },
    },
    'HO-Geolife': {
        'UA': {
            'TraceHiding (Ent.)': [74.19, 64.62, 68.03, 65.54],
            'NegGrad': [77.23, 64.46, 59.12, 59.89],
        },
        'RA': {
            'TraceHiding (Ent.)': [81.88, 81.35, 79.62, 79.61],
            'NegGrad': [82.32, 83.48, 78.15, 75.55],
        },
        'TA': {
            'TraceHiding (Ent.)': [61.32, 61.74, 57.37, 55.41],
            'NegGrad': [61.28, 62.78, 57.10, 54.40],
        },
        'MIA': {
            'TraceHiding (Ent.)': [47.61, 43.44, 44.63, 42.86],
            'NegGrad': [42.64, 39.53, 45.18, 44.42],
        },
    },
    'HO-NYC': {
        'UA': {
            'TraceHiding (Ent.)': [46.24, 44.98, 62.89, 69.09],
            'NegGrad': [35.94, 45.73, 51.20, 47.06],
        },
        'RA': {
            'TraceHiding (Ent.)': [94.40, 93.81, 93.01, 92.72],
            'NegGrad': [94.16, 80.98, 66.64, 67.76],
        },
        'TA': {
            'TraceHiding (Ent.)': [79.68, 78.02, 73.71, 68.93],
            'NegGrad': [79.58, 67.76, 54.99, 54.74],
        },
        'MIA': {
            'TraceHiding (Ent.)': [49.71, 48.02, 45.41, 42.96],
            'NegGrad': [49.64, 47.97, 44.58, 39.44],
        },
    },
}

# Confidence interval data
error_data = {
    'HO-Rome': {
        'UA': {
            'TraceHiding (Ent.)': [13.44, 9.49, 4.80, 3.25],
            'NegGrad': [8.25, 6.08, 3.44, 2.59]
        },
        'RA': {
            'TraceHiding (Ent.)': [2.26, 2.95, 2.84, 3.42],
            'NegGrad': [1.56, 4.29, 2.24, 0.73]
        },
        'TA': {
            'TraceHiding (Ent.)': [1.16, 1.01, 1.41, 2.20],
            'NegGrad': [0.83, 2.51, 1.60, 0.31]
        },
        'MIA': {
            'TraceHiding (Ent.)': [16.77, 9.18, 6.18, 7.17],
            'NegGrad': [9.68, 10.13, 6.36, 3.03]
        }
    },
    'HO-Geolife': {
        'UA': {
            'TraceHiding (Ent.)': [19.38, 19.26, 17.47, 13.14],
            'NegGrad': [20.63, 21.42, 23.04, 12.69]
        },
        'RA': {
            'TraceHiding (Ent.)': [3.02, 2.92, 2.51, 3.77],
            'NegGrad': [3.18, 1.54, 7.58, 6.26]
        },
        'TA': {
            'TraceHiding (Ent.)': [3.53, 1.35, 3.65, 6.76],
            'NegGrad': [3.94, 0.85, 7.29, 7.31]
        },
        'MIA': {
            'TraceHiding (Ent.)': [3.18, 7.47, 3.91, 5.58],
            'NegGrad': [9.24, 9.32, 2.81, 2.76]
        }
    },
    'HO-NYC': {
        'UA': {
            'TraceHiding (Ent.)': [24.06, 8.33, 10.43, 10.02],
            'NegGrad': [27.92, 11.02, 9.39, 3.85]
        },
        'RA': {
            'TraceHiding (Ent.)': [0.26, 0.66, 1.17, 1.30],
            'NegGrad': [0.93, 6.96, 8.24, 2.79]
        },
        'TA': {
            'TraceHiding (Ent.)': [0.95, 0.72, 1.62, 2.67],
            'NegGrad': [2.08, 6.41, 6.38, 2.60]
        },
        'MIA': {
            'TraceHiding (Ent.)': [0.45, 0.97, 1.11, 1.80],
            'NegGrad': [0.51, 0.66, 1.08, 2.23]
        }
    }
}


# Define hatch patterns for datasets
# More patterns: ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
dataset_hatches = {
    'HO-Rome': '/',
    'HO-Geolife': 'x',
    'HO-NYC': '.'
}

method_colors = {
    'TraceHiding (Ent.)': '#0bb4ff',
    'NegGrad': '#ffa300'
}

# Number of datasets, sample sizes, and methods
n_datasets = len(datasets)
n_samples = len(sample_sizes)
n_methods = len(methods)

# Width of a single bar
bar_width = 0.35
# Spacing between groups of bars for different sample sizes within the same dataset
sample_group_padding = 0.1 # Space between (TH, NG) pairs for 1%, 5% etc.
# Spacing between groups of bars for different datasets
dataset_group_padding = 0.6 # Larger space between HO-Rome, HO-Geolife etc.

# Create directory if it doesn't exist for saving plots
output_dir = 'experiment_scripts/thesis_plots/outputs/'

legend_prop = {
    'UA': {
        'bbox_to_anchor': [(1, 1), (0.793, 1)],
        'loc': 'upper right',
    },
    'RA': {
        'bbox_to_anchor': [(0, 1), (0, 0.83)],
        'loc': 'upper left',
    },
    'TA': {
        'bbox_to_anchor': [(0, 1), (0, 0.83)],
        'loc': 'upper left',
    },
    'MIA': {
        'bbox_to_anchor': [(0, 1), (0.207, 1)],
        'loc': 'upper left',
    }
}

# --- Plotting ---
for metric in metric_names:
    fig, ax = plt.subplots(figsize=(15, 7)) # Adjusted figsize for better label display

    # Store positions for x-ticks and their labels
    x_tick_positions = []
    x_tick_labels = []
    dataset_label_positions = [] # For placing dataset names

    current_x_offset = 0 # Tracks the current starting x position for a new dataset group

    for i, dataset_name in enumerate(datasets):
        dataset_start_x = current_x_offset
        
        for j, sz in enumerate(sample_sizes):
            # Calculate x position for the pair of bars (TraceHiding, NegGrad)
            pos1 = current_x_offset
            pos2 = pos1 + bar_width

            val_th = data[dataset_name][metric]['TraceHiding (Ent.)'][j]
            val_ng = data[dataset_name][metric]['NegGrad'][j]

            # Plot bars
            ax.bar(pos1, val_th, bar_width,
                   label='TraceHiding (Ent.)' if i == 0 and j == 0 else "", # Avoid duplicate labels
                   color=method_colors['TraceHiding (Ent.)'],
                   hatch=dataset_hatches[dataset_name],
                   edgecolor='black') # Add edgecolor for better hatch visibility
            ax.bar(pos2, val_ng, bar_width,
                   label='NegGrad' if i == 0 and j == 0 else "", # Avoid duplicate labels
                   color=method_colors['NegGrad'],
                   hatch=dataset_hatches[dataset_name],
                   edgecolor='black') # Add edgecolor

            # Center of the current pair of bars for the sample size label
            pair_center = current_x_offset + bar_width / 2
            x_tick_positions.append(pair_center) # This will be center of the group (pos1, pos2)
            x_tick_labels.append(f"{sz}%")
            
            # Move x for the next pair of bars within the same dataset
            current_x_offset += (n_methods * bar_width) + sample_group_padding

        # Calculate the center position for the dataset label (underneath its group of bars)
        dataset_end_x = current_x_offset - sample_group_padding # End of the last bar group for this dataset
        dataset_center_x = (dataset_start_x + (dataset_end_x - bar_width )) / 2 # Adjusted for better centering
        dataset_label_positions.append(dataset_center_x)

        # Add larger padding before the next dataset group starts
        current_x_offset += dataset_group_padding


    ax.set_ylabel(f'{metric} Metric Value', fontsize=16)
    ax.set_title(f'{metric} Metric Comparison: Datasets, Sample Sizes, and Methods', fontsize=18)
    
    # Set primary x-ticks for sample sizes
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis='x', which='major', pad=7) # Add padding for primary labels

    # Add dataset names as a secondary layer of labels below the primary x-axis
    # This is a common way to create grouped x-axis labels.
    # We can iterate and place text or use a secondary axis if more complex.
    y_level_for_dataset_labels = -0.15
    if metric in ['RA', 'TA', 'MIA'] and max(ax.get_ylim()) < 50 :
         y_level_for_dataset_labels = -0.25
    elif metric == 'UA' and max(ax.get_ylim()) > 80:
         y_level_for_dataset_labels = -0.12


    for i, dataset_name in enumerate(datasets):
        ax.text(dataset_label_positions[i], y_level_for_dataset_labels, dataset_name,
                ha='center', va='top', fontsize=16, transform=ax.get_xaxis_transform())

    # --- Legends ---
    # Method legend (colors)
    method_legend_handles = [Patch(facecolor=method_colors[method], label=method) for method in methods]
    
    # Dataset legend (hatches)
    dataset_legend_handles = [Patch(facecolor='grey', hatch=dataset_hatches[ds_name], label=ds_name, edgecolor='black') for ds_name in datasets]

    
    
    # Place method legend
    leg1 = ax.legend(handles=method_legend_handles, title='Methods', loc=legend_prop[metric]['loc'], bbox_to_anchor=(legend_prop[metric]['bbox_to_anchor'][0]), fontsize=13)
    ax.add_artist(leg1) # Important to re-add the first legend

    # Place dataset (hatch) legend below the first one
    ax.legend(handles=dataset_legend_handles, title='Datasets', loc=legend_prop[metric]['loc'], bbox_to_anchor=(legend_prop[metric]['bbox_to_anchor'][1]), fontsize=13)

    plt.subplots_adjust(bottom=0.2, right=0.85)
    # plt.tight_layout(rect=[0, 0.05, 0.85, 1]) # rect=[left, bottom, right, top]
    
    plt.savefig(f'{output_dir}sample_size_bar_chart_{metric}_grouped_hatched.pdf', bbox_inches='tight')