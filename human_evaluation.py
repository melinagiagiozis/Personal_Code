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
icc_result = icc[icc["Type"] == "ICC2"]

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

mean_scores = df.groupby(["Video", "Algorithm"])["MeanScore"].mean().unstack()
# Reorder algorithm columns
ordered_columns = ['DR-DSN', 'CTVSUM', 'CA–SUM']
mean_scores = mean_scores[ordered_columns]

plt.figure(figsize=(12, 6))
ax = sns.heatmap(mean_scores, annot=True, cmap="Blues", vmin=1, vmax=5, 
                 annot_kws={"size": 16}, cbar_kws={"ticks": [1, 2, 3, 4, 5]})
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=16)

# Set yticks with correct positions and labels
num_videos = mean_scores.shape[0]
plt.yticks(ticks=[i + 0.5 for i in range(num_videos)],
           labels=[f"Video {i+1}" for i in range(num_videos)],
           rotation=0, fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel("")
plt.xlabel("")
plt.tight_layout()
plt.savefig("Figures/Heatmap.pdf", dpi=600)
plt.close()
