import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import entropy
import seaborn as sns
import os
import sys

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utility.FrequencyOfVisitImportance import FrequencyImportance
from utility.functions import custom_collate_fn
from utility.EntropyImportance import EntropyImportance
from utility.CoverageDiversityImportance import CoverageDiversityImportance
from utility.UserUniquenessImportance import UserUniquenessImportance
from torch.utils.data import DataLoader
import torch
import pandas as pd

DATASET_NAME = "HO_Geolife_Res8"

os.makedirs(f"analysis/{DATASET_NAME}/score_optimize", exist_ok=True)

train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)


entropy_calculator = EntropyImportance()
coverage_calculator = CoverageDiversityImportance()

entropy_calculator.prepare(train_data + test_data)
coverage_calculator.prepare(train_data + test_data)


train_dloader = DataLoader(train_data, batch_size=len(train_data), collate_fn=custom_collate_fn, num_workers=24)
test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

all_entropy_importances = []
all_coverage_importances = []
all_length_importances = []
all_users = []

for batch in train_dloader:
    
    batch_importance = entropy_calculator.calculate_importance(batch)
    all_entropy_importances.extend(batch_importance)
    
    batch_importance = coverage_calculator.calculate_importance(batch)
    all_coverage_importances.extend(batch_importance)
    
    # length
    all_length_importances.extend([len([b for b in x if b!=0]) for x in batch[0]])
    
    all_users.extend(batch[1].tolist())


data = pd.DataFrame({
    'entropy': all_entropy_importances,
    'coverage': all_coverage_importances,
    'length': all_length_importances / np.max(all_length_importances),
    'user': all_users
    })


# Simulated data for normalized scores (replace with your data)
S1 = data['entropy'].values
S2 = data["coverage"].values
S3 = data["length"].values

data['entropy_norm'] = data['entropy'] / data['entropy'].std()
data['coverage_norm'] = data['coverage'] / data['coverage'].std()
data['length_norm'] = data['length'] / data['length'].std()


# Calculate variances and covariances
var1 = np.var(S1)
var2 = np.var(S2)
var3 = np.var(S3)
cov12 = np.cov(S1, S2)[0, 1]
cov13 = np.cov(S1, S3)[0, 1]
cov23 = np.cov(S2, S3)[0, 1]

# Variance function
def variance_to_maximize(coefs):
    alpha, beta, gamma = coefs
    return -(alpha**2 * var1 +
             beta**2 * var2 +
             gamma**2 * var3 +
             2 * alpha * beta * cov12 +
             2 * alpha * gamma * cov13 +
             2 * beta * gamma * cov23)  # Negative for maximization


from scipy.stats import skew, kurtosis

# Define a combined objective function
def combined_objective(coefs):
    alpha, beta, gamma = coefs

    # Compute the composite score
    data['composite_score'] = alpha * data['entropy'] + beta * data['coverage'] + gamma * data['length']
    
    # Group by user and calculate variance for each user
    user_composite = data.groupby('user')['composite_score']
    user_means = user_composite.mean()
    
    # Variance
    variance = user_means.var()
    
    # Skewness
    skewness = skew(user_means)
    
    # Kurtosis (excess kurtosis = kurtosis - 3)
    excess_kurtosis = kurtosis(user_means) - 3
    
    # Combined objective: maximize variance, minimize skewness and kurtosis
    # Adjust weights (w1, w2, w3) as needed
    w1, w2, w3 = 0.5, 0.7, 0.5
    return -(w1 * variance - w2 * abs(skewness) - w3 * excess_kurtosis)



# Constraints: coefficients sum to 1
constraints = {'type': 'eq', 'fun': lambda coefs: sum(coefs) - 1}
# Bounds: coefficients between 0 and 1
bounds = [(0, 1), (0, 1), (0, 1)]

# Initial guess
initial_guess = [1/2, 1/4, 1/4]


# Define entropy function
# def entropy_to_maximize(coefs):
#     alpha, beta, gamma = coefs
#     S = alpha * S1 + beta * S2 + gamma * S3  # Composite score
#     hist, _ = np.histogram(S, bins=100, density=True)  # Histogram estimate
#     prob_density = hist / np.sum(hist)  # Normalize to probabilities
#     prob_density = prob_density[prob_density > 0]  # Avoid log(0)
#     return -entropy(prob_density)  # Negative for maximization

# Optimization with entropy as the objective
result_optimized = minimize(combined_objective, initial_guess, bounds=bounds, constraints=constraints)
alpha_optimized, beta_optimized, gamma_optimized = result_optimized.x

# Composite score using entropy-optimized weights
S_composite_optimized = alpha_optimized * S1 + beta_optimized * S2 + gamma_optimized * S3

# Plot density of scores and composite score
plt.figure(figsize=(12, 8))

sns.kdeplot(S1, label="S1 (Entropy)", linewidth=2)
sns.kdeplot(S2, label="S2 (Coverage)", linewidth=2)
sns.kdeplot(S3, label="S3 (Length)", linewidth=2)
sns.kdeplot(S_composite_optimized, label="Composite Score", linewidth=2, linestyle="--")

title = ("Density Comparison of Scores and Composite Score \n"
         r"$\alpha = {:.2f}, \beta = {:.2f}, \gamma = {:.2f}$".format(alpha_optimized, beta_optimized, gamma_optimized))
plt.title(title, fontsize=16)
plt.xlabel("Score Value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.savefig(f"analysis/{DATASET_NAME}/score_optimize/composite_score_density.pdf", bbox_inches='tight')

# Output the coefficients for reference
print("Optimized coefficients:")
print("alpha = {:.5f}".format(alpha_optimized))
print("beta = {:.5f}".format(beta_optimized))
print("gamma = {:.5f}".format(gamma_optimized))