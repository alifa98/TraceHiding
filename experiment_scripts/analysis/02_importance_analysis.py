import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utility.LengthImportance import TrajectoryLengthImportance
from utility.functions import custom_collate_fn
from utility.EntropyImportance import EntropyImportance
from utility.CoverageDiversityImportance import CoverageDiversityImportance
from utility.UserUniquenessImportance import UserUniquenessImportance
from torch.utils.data import DataLoader
import torch
import pandas as pd

DATASET_NAME = "HO_NYC_Res9"

os.makedirs(f"analysis/{DATASET_NAME}/importance_analysis/", exist_ok=True)

train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)


entropy_calculator = EntropyImportance()
coverage_calculator = CoverageDiversityImportance()
uniqueness_calculator = UserUniquenessImportance()
length_calculator = TrajectoryLengthImportance()

entropy_calculator.prepare(train_data + test_data)
coverage_calculator.prepare(train_data + test_data)
length_calculator.prepare(train_data + test_data)
uniqueness_calculator.prepare(train_data + test_data)


train_dloader = DataLoader(train_data, batch_size=len(train_data), collate_fn=custom_collate_fn, num_workers=24)
test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

all_entropy_importances = []
all_coverage_importances = []
all_length_importances = []
all_uniqueness_importances = []
all_users = []

for batch in train_dloader:
    
    batch_importance = entropy_calculator.calculate_importance(batch)
    all_entropy_importances.extend(batch_importance)
    
    batch_importance = coverage_calculator.calculate_importance(batch)
    all_coverage_importances.extend(batch_importance)
    
    batch_importance = length_calculator.calculate_importance(batch)
    all_length_importances.extend(batch_importance)
    
    batch_importance = uniqueness_calculator.calculate_importance(batch)
    all_uniqueness_importances.extend(batch_importance)
    
    all_users.extend(batch[1].tolist())
    
data = pd.DataFrame({
    'entropy': all_entropy_importances,
    'coverage': all_coverage_importances,
    'length': all_length_importances,
    'user_uniqueness': all_uniqueness_importances,
    'user': all_users
    })



import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

columns = ['length', 'entropy', 'coverage', 'user_uniqueness']
notations = [r'$\xi_{length}(x)$', r'$\xi_{entropy}(x)$', r'$\xi_{coverage}(x)$', r'$\xi_{unique}(x)$']
colors = ['red', 'green', 'blue', 'purple']

plt.figure(figsize=(20, 4))
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
for i, (col, color, notation) in enumerate(zip(columns, colors, notations), 1):
    plt.subplot(1, 4, i)
    
    # KDE plot
    sns.kdeplot(data[col], fill=True, common_norm=False, color=color)
    
    # Calculate mean and median
    mean_val = data[col].mean()
    median_val = data[col].median()
    
    # Plot mean and median lines
    plt.axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='orange', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
    
    plt.title(f'{notation} Distribution', fontsize=25)
    plt.xlabel('Importance Score', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.legend(prop={"size":18})
    
    # Thicken the axies:
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

plt.tight_layout()
plt.savefig(f"analysis/{DATASET_NAME}/importance_analysis/annotated_importances_density_{DATASET_NAME.lower()}.pdf", bbox_inches='tight', format='pdf')
plt.show()

print(f"The Plot has been saved to: analysis/{DATASET_NAME}/importance_analysis/annotated_importances_density_{DATASET_NAME.lower()}.pdf")



# ---- Correlation Heatmap ----
# plt.figure(figsize=(8, 6))
# corr_matrix = data[columns].corr()

# # Create a mask to hide only the upper triangle (excluding the diagonal)
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# # Plot the heatmap with the mask
# sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='YlGn', fmt='.2f', linewidths=0.5)
# # plt.title('Correlation Heatmap of Importance Scores')
# plt.tight_layout()
# plt.savefig(f"analysis/{DATASET_NAME}/importance_analysis/correlation_heatmap_{DATASET_NAME.lower()}.pdf", bbox_inches='tight', format='pdf')
# plt.show()

# print(f"The Plot has been saved to: analysis/{DATASET_NAME}/importance_analysis/correlation_heatmap_{DATASET_NAME.lower()}.pdf")

# # ---- Pair Plot ----
# ## Kind: scatter, kde, hist
# pair_plot = sns.pairplot(data[columns], kind='kde', diag_kind='kde', corner=True, plot_kws={'fill': True, 'color': 'darkcyan'}, diag_kws={'fill': True, 'color': 'green'})
# # pair_plot.fig.suptitle('Pairwise Scatter Plots of Importance Scores', y=1.02)
# pair_plot.savefig(f"analysis/{DATASET_NAME}/importance_analysis/pair_plot_{DATASET_NAME.lower()}.pdf", bbox_inches='tight', format='pdf')
# plt.show()

# print(f"The Plot has been saved to: analysis/{DATASET_NAME}/importance_analysis/pair_plot_{DATASET_NAME.lower()}.pdf")




## -- New Idea: combine both pair and heatmap for correlation because both have redundant empty upper trianlge ----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
mpl.rcParams["axes.labelsize"] = 25
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
   
# change labels to the mathematical notations
new_data = data.rename(columns=dict(zip(columns, notations)))

pairwise_corr_plot = sns.pairplot(new_data[notations], vars=notations,  kind='kde', diag_kind='kde', plot_kws={'fill': True, 'color': 'darkcyan'}, diag_kws={'fill': True, 'color': 'green'})
pairwise_corr_plot.map_upper(hide_current_axis)
(xmin, _), (_, ymax) = pairwise_corr_plot.axes[0, 0].get_position().get_points()
(_, ymin), (xmax, _) = pairwise_corr_plot.axes[-1, -1].get_position().get_points()
ax = pairwise_corr_plot.fig.add_axes([xmin, ymin, xmax - xmin, ymax - ymin], facecolor='none')

corr_matrix = new_data[notations].corr()
mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix,  mask=mask, cmap='YlGn', fmt='.2f',
            linewidths=.5, cbar=False, annot=True, annot_kws={'size': 20}, ax=ax)
ax.set_xticks([])
ax.set_yticks([])

pairwise_corr_plot.savefig(f"analysis/{DATASET_NAME}/importance_analysis/pariwise_analysis_{DATASET_NAME.lower()}.pdf", bbox_inches='tight', format='pdf')
plt.show()

print(f"The Pariwise Analysis has been saved to: analysis/{DATASET_NAME}/importance_analysis/pariwise_analysis_{DATASET_NAME.lower()}.pdf")



# Plotting density plots for each user (Randomly sample 10 users to visualize)
# users = data['user'].unique()
# sample_users = users[:10]
# data = data[data['user'].isin(sample_users)]
# plt.figure(figsize=(10, 6))
# sns.kdeplot(data=data, x='importance', hue='user', fill=True, common_norm=False)
# plt.title(f'Density Plot of {IMPORTANCE_NAME} importance Score by User')
# plt.xlabel('Importance Score')
# plt.ylabel('Density')

