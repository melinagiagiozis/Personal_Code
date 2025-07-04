import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

# Load both evaluations
df = pd.read_csv("../../01_Data/Results/evaluation.csv")

# Filter only videos starting with 'SCI' (pilot study)
df = df[df["Video"].str.startswith("SCI")]

# Ensure numeric columns are properly typed
metrics = ["Coverage", "TemporalDistribution", 
           "Diversity", "Representativeness"]
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")

# Invert TemporalDistribution if needed so higher = better
df["TemporalDistribution"] = 1 - df["TemporalDistribution"]

# Ensure 'Split' is treated as categorical or integer-like
df["Split"] = pd.to_numeric(df["Split"], errors="coerce").astype("Int64")

# === Overall mean and std per algorithm ===
mean_scores = df.groupby("Algorithm")[metrics].mean().T
std_scores = df.groupby("Algorithm")[metrics].std().T

summary_df = mean_scores.round(4).astype(str) + " ± " + std_scores.round(4).astype(str)
summary_df.columns.name = None
print("Overall mean ± std across all videos (ignoring folds):")
# print(summary_df)

# === Plot overall mean scores ===
mean_scores.plot(kind='bar', rot=0, yerr=std_scores, capsize=4, figsize=(10, 5))
plt.ylabel("Mean Score")
plt.title("Mean Evaluation Scores per Algorithm (All Videos)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.close()

# === Plot metric comparisons per split ===
sns.set(style="whitegrid")
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Split", y=metric, hue="Algorithm", errorbar='sd', capsize=0.1)
    plt.title(f"{metric} Comparison per Split")
    plt.ylabel(metric)
    plt.xlabel("Split")
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.close()

# === Compute per-split (fold) averages ===
split_scores = df.groupby(["Split", "Algorithm"])[metrics].mean().reset_index()

# Compute mean ± std across splits
split_mean = split_scores.groupby("Algorithm")[metrics].mean().T
split_std = split_scores.groupby("Algorithm")[metrics].std().T
split_summary = split_mean.round(4).astype(str) + " ± " + split_std.round(4).astype(str)
split_summary.columns.name = None
print("Mean ± Std across CV Splits (5-fold):")
print(split_summary)

# === Boxplots: metric distributions across splits ===
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=split_scores, x="Algorithm", y=metric)
    plt.title(f"{metric} across CV Splits")
    plt.ylabel(metric)
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.close()

# === Plot metric comparisons per split ===
import seaborn as sns  # for better grouped plots
sns.set(style="whitegrid")

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Split", y=metric, hue="Algorithm", errorbar='sd', capsize=0.1)
    plt.title(f"{metric} Comparison per Split")
    plt.ylabel(metric)
    plt.xlabel("Split")
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.close()

# === Statstical test ===

# --- Setup ---
metrics = ["Coverage", "TemporalDistribution", "Diversity", "Representativeness"]
algorithms = df["Algorithm"].unique()
split_scores = df.groupby(["Split", "Algorithm"])[metrics].mean().reset_index()

# --- Step 1: Friedman test for each metric ---
raw_pvals = []
stat_results = []
n_folds = split_scores["Split"].nunique()
k_algos = len(algorithms)

for metric in metrics:
    data = [split_scores[split_scores["Algorithm"] == algo][metric].values for algo in algorithms]
    
    stat, p = friedmanchisquare(*data)
    kendall_w = stat / (n_folds * (k_algos - 1))
    raw_pvals.append(p)
    stat_results.append((metric, stat, p, kendall_w))

# --- Step 2: Bonferroni correction ---
corrected_pvals = multipletests(raw_pvals, method='bonferroni')[1]

# --- Step 3: Output + Conditional Nemenyi post hoc test ---
print("\nFriedman test results with Bonferroni correction:\n")
for (metric, stat, raw_p, kendall_w), adj_p in zip(stat_results, corrected_pvals):
    if raw_p is not None:
        print(f"{metric}: χ² = {stat:.4f}, raw p = {raw_p:.4f}, corrected p = {adj_p:.4f}, Kendall’s W = {kendall_w:.4f}")
        
        # Run Nemenyi post hoc test only if corrected p < 0.05
        if adj_p < 0.05:
            print(f"  → Performing Nemenyi post hoc test for {metric}:")
            data_long = split_scores.pivot(index="Split", columns="Algorithm", values=metric)
            nemenyi_result = sp.posthoc_nemenyi_friedman(data_long)
            print(nemenyi_result)
        else:
            print(f"  → No significant difference after correction; Nemenyi test skipped.")