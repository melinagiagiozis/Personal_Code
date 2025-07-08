"""
Script: human_evaluation.py

Description:
    This script evaluates the quality of video summaries generated 
    by different algorithms based on human evaluation using a Likert scale (1-5).

"""

import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = '../../01_Data/Results/Human_Evaluation.csv'
df = pd.read_csv(data_path)

# Compute mean score across all questions per row
df["MeanScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# ---------------- Consistency between raters ----------------

#  Compute ICC across all videos (i.e., consistency across participants)
icc = pg.intraclass_corr(data=df, targets='Video', raters='Participant ID', ratings='MeanScore')
# icc_result = icc[icc["Type"] == "ICC2"]  # absolute consistecy
icc_result = icc[icc["Type"] == "ICC3"]  # relative consistecy

print("Overall ICC (participant agreement across videos):")
print(icc_result)

# ---------------- Averages per algorithm ----------------

# Print descriptive stats per algorithm
print("\n=== Descriptive Statistics (MeanScore) ===")
print(df.groupby("Algorithm")["MeanScore"].describe())

# Plot using matplotlib's boxplot (Mean Scores)
plt.figure(figsize=(6, 4))
algos = df['Algorithm'].unique()
data_to_plot = [df[df['Algorithm'] == algo]['MeanScore'] for algo in algos]
plt.boxplot(data_to_plot, labels=algos)
plt.title("Human Evaluation: Mean Score per Algorithm")
plt.ylabel("Mean Score (1–5)")
plt.xlabel("Algorithm")
plt.tight_layout()
plt.close()

# Melt dataframe for per-question scores
melted = df.melt(id_vars=["Algorithm", "Video"], value_vars=["Q1", "Q2", "Q3", "Q4", "Q5"],
                 var_name="Question", value_name="Score")

# Plot using matplotlib's boxplot (Per-Question Scores)
plt.figure(figsize=(8, 5))
questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
positions = range(1, len(questions) + 1)

for i, algo in enumerate(algos):
    subset = melted[melted["Algorithm"] == algo]
    data_to_plot = [subset[subset["Question"] == q]["Score"] for q in questions]
    plt.boxplot(data_to_plot, positions=[p + i * 0.2 for p in positions], widths=0.15, patch_artist=True,
                boxprops=dict(facecolor='none', edgecolor='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))

plt.xticks(positions, questions)
plt.title("Per-Question Human Ratings by Algorithm")
plt.ylabel("Score (1–5)")
plt.xlabel("Question")
plt.tight_layout()
plt.close()

# Compute mean and std
mean_scores = df.groupby(["Video", "Algorithm"])["MeanScore"].mean().unstack()
std_scores = df.groupby(["Video", "Algorithm"])["MeanScore"].std().unstack()

# Reorder columns
ordered_columns = ['DR-DSN', 'CTVSUM', 'CA–SUM']
mean_scores = mean_scores[ordered_columns]
std_scores = std_scores[ordered_columns]

# Create annotations: "mean (std)"
annot = mean_scores.applymap(lambda x: f"{x:.2f}") + " (" + std_scores.applymap(lambda x: f"{x:.2f}") + ")"

# Plot
plt.figure(figsize=(12, 6))
ax = sns.heatmap(mean_scores, annot=annot, fmt='', cmap="Blues", vmin=1, vmax=5,
                 annot_kws={"size": 20}, cbar_kws={"ticks": [1, 2, 3, 4, 5]})
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=25)

# Set yticks
num_videos = mean_scores.shape[0]
plt.yticks(ticks=[i + 0.5 for i in range(num_videos)],
           labels=[f"Video {i+1}" for i in range(num_videos)],
           rotation=0, fontsize=24)
plt.xticks(fontsize=24)
plt.ylabel("")
plt.xlabel("")
plt.tight_layout()
plt.savefig("Figures/Heatmap.pdf", dpi=600)
plt.close()

# Compute stats
stats = df.groupby("Participant ID")["MeanScore"].agg(["mean", "std"]).reset_index()

# Sort by mean score for nicer layout (optional)
stats = stats.sort_values(by="mean", ascending=True)

# Plot
plt.figure(figsize=(10, 5))
plt.errorbar(
    x=stats["mean"],
    y=stats["Participant ID"],
    xerr=stats["std"],
    fmt='ok',
    capsize=5,
    elinewidth=2,
    ecolor='grey'
)

# Labels and layout
plt.xlabel("Mean Score")
plt.ylabel("Participant")
plt.title("Rater Bias and Consistency (Mean ± SD)")
plt.xlim(0.8, 5.2)  # Adjust based on your scoring range
plt.tight_layout()
plt.savefig("Figures/RaterBias.png", dpi=600)
plt.close()
