"""
Script: human_evaluation.py

Description:
    This script evaluates the quality of video summaries generated 
    by different algorithms based on human evaluation using a Likert scale (1-5).

"""

import pandas as pd
import pingouin as pg
import krippendorff
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from PIL import Image


print_results = False


# Load data
data_path = '../../01_Data/Results/Human_Evaluation.csv'
df = pd.read_csv(data_path)

# Uniform notation
df['Algorithm'] = df['Algorithm'].str.replace('–', '-', regex=False)

# Compute mean score across all questions per row
df["MeanScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# ---------------- Consistency between raters ----------------

# #  Compute ICC across all videos (i.e., consistency across participants)
# icc = pg.intraclass_corr(data=df, targets='Video', raters='Participant ID', ratings='MeanScore')
# icc_result = icc[icc["Type"] == "ICC2"]

# print("Overall ICC (participant agreement across videos):")
# print(icc_result)

data = df.pivot_table(index='Video', columns='Participant ID', values='MeanScore', aggfunc='mean')
alpha = krippendorff.alpha(reliability_data=data.values, level_of_measurement='ordinal')
if print_results:
    print(f"Krippendorff's alpha: {alpha:.3f}")

questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]

for q in questions:
    if print_results:
        print(f"\n=== {q} ===")
    # Create a DataFrame for this question with required columns
    df_q = df[["Video", "Participant ID", q]].rename(columns={q: "Score"})
    
    # # Compute ICC
    # icc_q = pg.intraclass_corr(data=df_q, targets='Video', raters='Participant ID', ratings='Score')
    # icc2_q = icc_q[icc_q["Type"] == "ICC2"]
    
    # print(f"\nICC for {q}:")
    # print(icc2_q[["Type", "ICC", "CI95%", "F", "pval"]])
    # Prepare Data for Krippendorff's alpha
    
    df_pivot = df_q.pivot_table(index='Video', columns='Participant ID', values='Score', aggfunc='mean')
    alpha = krippendorff.alpha(reliability_data=df_pivot.values, level_of_measurement='ordinal')
    if print_results:
        print(f"Krippendorff's alpha: {alpha:.3f}")

# ---------------- Averages per algorithm ----------------

# Print descriptive stats per algorithm
if print_results:
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

# List of questions
questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# Generate boxplots with the line at the mean instead of the median
for q in questions:
    plt.figure(figsize=(6, 4))
    algos = df['Algorithm'].unique()
    data_to_plot = [df[df['Algorithm'] == algo][q] for algo in algos]

    # Prepare boxplot stats with the "median" replaced by the mean
    box_stats = []
    for group in data_to_plot:
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        mean = group.mean()
        iqr = q3 - q1
        lower_whisker = max(group.min(), q1 - 1.5 * iqr)
        upper_whisker = min(group.max(), q3 + 1.5 * iqr)
        fliers = group[(group < lower_whisker) | (group > upper_whisker)].tolist()

        box_stats.append({
            'label': '',
            'mean': mean,
            'med': mean,  # force center line to be the mean
            'q1': q1,
            'q3': q3,
            'whislo': lower_whisker,
            'whishi': upper_whisker,
            'fliers': fliers
        })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bxp(box_stats, showfliers=True, showmeans=False)
    ax.set_xticks(ticks=range(1, len(algos) + 1))
    ax.set_xticklabels(algos)
    ax.set_title(f"Human Evaluation: {q} Score per Algorithm (Line = Mean)")
    ax.set_ylabel("Rating (1–5)")
    ax.set_ylim(0.1, 5.9)
    plt.tight_layout()
    plt.savefig(f"Figures/HumanEvaluation/Boxplot_{q}.pdf", dpi=300)
    plt.close()



# Create a summary table: descriptive statistics per question and algorithm
question_stats = {}

for q in questions:
    stats = df.groupby("Algorithm")[q].describe()
    stats = stats[['mean', 'std']]
    stats.columns = [f"{q}_{col}" for col in stats.columns]
    question_stats[q] = stats

# Combine all question summaries into one table
combined_stats = pd.concat(question_stats.values(), axis=1)
# print(combined_stats)

# Compute the average of each column across all algorithms
average_stats = combined_stats.mean(axis=0)

# Optional: Reshape into a table with rows = questions and columns = mean, std
reshaped_avg = pd.DataFrame({
    'mean': [average_stats[f'{q}_mean'] for q in questions],
    'std':  [average_stats[f'{q}_std']  for q in questions]
}, index=questions)

if print_results:
    print(reshaped_avg)

# Melt to long format
long_df = df.melt(id_vars=["Participant ID", "Algorithm"], value_vars=questions,
                  var_name="Question", value_name="Rating")

# Now compute the overall mean and std from raw ratings
overall_mean = long_df["Rating"].mean()
overall_std = long_df["Rating"].std()

if print_results:
    print(f"Overall mean: {overall_mean:.2f} ± {overall_std:.2f}")


# ---------------- Heatmap averages (Video x Algorthm) ----------------

# Compute mean and std
mean_scores = df.groupby(["Video", "Algorithm"])["MeanScore"].mean().unstack()
std_scores = df.groupby(["Video", "Algorithm"])["MeanScore"].std().unstack()

# Reorder columns
ordered_columns = ['DR-DSN', 'CTVSUM', 'CA-SUM']
mean_scores = mean_scores[ordered_columns]
std_scores = std_scores[ordered_columns]

# Create annotations: "mean (std)"
annot = mean_scores.applymap(lambda x: f"{x:.2f}") + " (" + std_scores.applymap(lambda x: f"{x:.2f}") + ")"

# Plot
plt.figure(figsize=(12, 6))
ax = sns.heatmap(mean_scores, annot=annot, fmt='', cmap="rocket_r", vmin=1, vmax=5,
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
plt.savefig("Figures/HumanEvaluation/Heatmap.pdf", dpi=600)
plt.close()

# ---------------- Per Q: Heatmap averages (Video x Algorthm) ----------------

# Melt the DataFrame to long format for question-wise analysis
df_melted = pd.melt(
    df,
    id_vars=['Participant ID', 'Video', 'Summary', 'Algorithm'],
    value_vars=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
    var_name='Question',
    value_name='Rating'
)

# Group by Algorithm, Video, and Question
summary_stats = df_melted.groupby(['Algorithm', 'Video', 'Question']).agg(
    mean_rating=('Rating', 'mean'),
    std_rating=('Rating', 'std'),
    n=('Rating', 'count')
).reset_index()

# Get unique questions
questions = summary_stats['Question'].unique()

# Loop through each question and generate heatmaps
for q in questions:
    data_q = summary_stats[summary_stats['Question'] == q]

    # Pivot to get mean and std tables
    mean_scores = data_q.pivot_table(index='Video', columns='Algorithm', values='mean_rating')
    std_scores = data_q.pivot_table(index='Video', columns='Algorithm', values='std_rating')

    # Ensure consistent column order
    ordered_columns = ['DR-DSN', 'CTVSUM', 'CA-SUM']
    mean_scores = mean_scores.reindex(columns=ordered_columns)
    std_scores = std_scores.reindex(columns=ordered_columns)

    # Create annotation text: "mean (std)"
    annot = mean_scores.applymap(lambda x: f"{x:.2f}") + " (" + std_scores.applymap(lambda x: f"{x:.2f}") + ")"

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(mean_scores, annot=annot, fmt='', cmap="rocket_r", vmin=1, vmax=5,
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
    plt.title(f"Mean Ratings – {q}", fontsize=26)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"Figures/HumanEvaluation/Heatmap_{q}.pdf", dpi=600)
    plt.close()

# Prepare figure with one subplot per question
fig, axes = plt.subplots(nrows=1, ncols=len(questions), figsize=(24, 6), sharey=True)

# Loop through each question and corresponding axis
for i, q in enumerate(questions):
    data_q = summary_stats[summary_stats['Question'] == q]

    # Pivot to get mean and std tables
    mean_scores = data_q.pivot_table(index='Video', columns='Algorithm', values='mean_rating')
    std_scores = data_q.pivot_table(index='Video', columns='Algorithm', values='std_rating')

    # Ensure consistent column order
    ordered_columns = ['DR-DSN', 'CTVSUM', 'CA-SUM']
    mean_scores = mean_scores.reindex(columns=ordered_columns)
    std_scores = std_scores.reindex(columns=ordered_columns)

    # Create annotation text: "mean (std)"
    annot = mean_scores.applymap(lambda x: f"{x:.2f}") + "\n(" + std_scores.applymap(lambda x: f"{x:.2f}") + ")"

    # Plot heatmap in the subplot
    sns.heatmap(mean_scores, annot=annot, fmt='', cmap="rocket_r", vmin=1, vmax=5,
                annot_kws={"size": 10}, cbar=(i == len(questions) - 1), ax=axes[i])

    axes[i].set_title(q, fontsize=16)
    axes[i].set_xlabel("")

    if i == 0:
        axes[i].set_ylabel("Video", fontsize=14)
    else:
        axes[i].set_ylabel("")

    # Set y-axis labels
    axes[i].set_yticklabels([f"Video {v+1}" for v in range(mean_scores.shape[0])],
                            rotation=0, fontsize=12)

    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, fontsize=12)

plt.tight_layout()
plt.savefig("Figures/HumanEvaluation/All_Questions_Heatmap.pdf", dpi=600)
plt.close()

# ---------------- Rater bias ----------------

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
plt.savefig("Figures/HumanEvaluation/RaterBias.png", dpi=600)
plt.close()

# Define Likert scale labels (adjust if your labels are different)
likert_labels = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neither agree nor disagree",
    4: "Agree",
    5: "Strongly Agree"
}

# Define colors for each Likert category (must match your scale)
likert_colors = {
    1: "#E74C3C",  # Strongly Disagree
    2: "#F1948A",  # Disagree
    3: "#D5DBDB",  # Neutral
    4: "#85C1E9",  # Agree
    5: "#2874A6"   # Strongly Agree
}

# Melt the dataframe to long format
df_long = df.melt(id_vars=['Participant ID', 'Video', 'Algorithm'], 
                  value_vars=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                  var_name='Question', value_name='Rating')

# Loop over each (Video, Algorithm) pair
for (video, algo), group in df_long.groupby(['Video', 'Algorithm']):
    # Count % of each rating per question
    plot_data = (
        group.groupby(['Question', 'Rating'])
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / float(x.sum()))
        .unstack()
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)  # ← this fixes NaNs
    ).fillna(0)

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    left = pd.Series([0] * plot_data.shape[0], index=plot_data.index)

    for rating in sorted(likert_labels.keys()):
        values = plot_data[rating]
        bars = ax.barh(
            [str(i) for i in plot_data.index.get_level_values(0)],
            values,
            left=left,
            color=likert_colors[rating],
            label=likert_labels[rating]
        )

        # Add text labels centered in each bar segment
        for i, (v, l) in enumerate(zip(values, left)):
            if v > 5:  # Only label if the segment is big enough to read
                ax.text(
                    l + v / 2,
                    i,
                    f"{v:.0f}%",
                    va='center',
                    ha='center',
                    fontsize=10,
                    color='black'
                )
        left += plot_data[rating]

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Add legend below the chart
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False
    )

    # Adjust layout to make room for legend
    fig.subplots_adjust(bottom=0.25)
    ax.invert_yaxis()
    ax.set_title(f"Video {video} – {algo}", fontsize=14)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"Figures/HumanEvaluation/Video{video}_{algo}_Likert.pdf")
    plt.close()


# ---------------- Percentage bars Likert Scale (Per Video & Algorithm) ----------------


# Assuming df_melted exists from your script (melted from df to long format)
df_long = df_melted.copy()

# Ensure all questions Q1–Q5 are present for every (Participant, Video, Algorithm)
full_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
df_long['Question'] = pd.Categorical(df_long['Question'], categories=full_questions, ordered=True)

# Fill missing combinations with NaNs
df_long = (
    df_long
    .set_index(['Participant ID', 'Video', 'Algorithm', 'Question'])
    .unstack(fill_value=np.nan)
    .stack()
    .reset_index()
)

# Group to get mean and std
plot_data = df_long.groupby(['Video', 'Algorithm', 'Question'])['Rating'].agg(['mean', 'std']).reset_index()

# Make sure video is treated as a string for consistent matching
plot_data['Video'] = plot_data['Video'].astype(str)
videos = ['1', '2', '3', '4', '5']
algorithms = df_long['Algorithm'].unique()

# Define question order and algorithm order
questions = ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
ordered_algorithms = ['DR-DSN', 'CTVSUM', 'CA-SUM']
colors = sns.color_palette("rocket_r", n_colors=3)

# Plot per video
for vid in sorted(plot_data['Video'].unique()):
    video_data = plot_data[plot_data['Video'] == vid]
    if video_data.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, algo in enumerate(ordered_algorithms):
        algo_data = video_data[video_data['Algorithm'] == algo].set_index('Question').reindex(questions)
        if algo_data['mean'].isnull().all():
            continue
        means = algo_data['mean'].values
        stds = algo_data['std'].fillna(0).values
        ax.errorbar(means, questions, xerr=stds, label=algo, marker='o', linestyle='-', color=colors[i], capsize=4)

    # Custom legend: only markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=algo, markerfacecolor=colors[i], markersize=8)
        for i, algo in enumerate(ordered_algorithms)
    ]
    ax.legend(handles=legend_elements, title="Algorithm", fontsize=16, title_fontsize=16)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f"Video {vid}", fontsize=25)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    plt.tight_layout()
    plt.savefig(f"Figures/HumanEvaluation/Video{vid}.pdf")
    plt.close()


# ---------------- Plot per questions (showing each video) ----------------


# Group to get mean and std
plot_data = df_long.groupby(['Video', 'Algorithm', 'Question'])['Rating'].agg(['mean', 'std']).reset_index()

# Define algorithm order and colors
ordered_algorithms = ['DR-DSN', 'CTVSUM', 'CA-SUM']
videos = [1, 2, 3, 4, 5]
# colors = sns.color_palette("rocket_r", n_colors=3)
# colors = ['m', 'blue', 'green']
# colors = ['#c4a148', '#c86340', '#60adcd']
colors = ['#3c466d', '#a2a0c4', '#c69b33']

# Plot per question with videos on the y-axis
for q in questions:
    question_data = plot_data[plot_data['Question'] == q]
    if question_data.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, algo in enumerate(ordered_algorithms):
        algo_data = question_data[question_data['Algorithm'] == algo].set_index('Video').reindex(videos)
        means = algo_data['mean'].values
        stds = algo_data['std'].fillna(0).values
        ax.errorbar(means, videos, xerr=stds, label=algo, marker='o', linestyle='-', color=colors[i], capsize=4)

    # Custom legend: only markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=algo, markerfacecolor=colors[i], markersize=8)
        for i, algo in enumerate(ordered_algorithms)
    ]
    ax.legend(handles=legend_elements, title="Algorithm", fontsize=14, title_fontsize=16)

    # Remove spines and set titles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f"{q}", fontsize=25)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel('Videos', fontsize=25)
    ax.set_xlabel('Rating', fontsize=25)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    plt.tight_layout()
    plt.savefig(f"Figures/HumanEvaluation/{q}_Videos.pdf")
    plt.close()




######################## Figures for the Manuscript ########################


# ---------------- Figure 2 ----------------


# Colors
algo_colors = {'DR-DSN': '#3c466d', 'CTVSUM': '#a2a0c4', 'CA-SUM': '#c69b33'}

# Compute average over all videos for each (Question, Algorithm)
mean_data_q = (
    plot_data.groupby(['Question', 'Algorithm'])['mean']
    .mean()
    .reset_index()
)

std_data_q = (
    plot_data.groupby(['Question', 'Algorithm'])['std']
    .apply(lambda x: np.sqrt((x**2).mean()))
    .reset_index()
)

# Merge mean and std
summary_data_q = pd.merge(mean_data_q, std_data_q, on=['Question', 'Algorithm'])
summary_data_q.columns = ['Question', 'Algorithm', 'mean', 'std']
marker_dict = {'DR-DSN': 'D', 'CTVSUM': 'X', 'CA-SUM': 's'}
algo_colors = {'DR-DSN': '#3c466d', 'CTVSUM': '#a2a0c4', 'CA-SUM': '#c69b33'}

# Plot: Questions on y-axis, average rating on x-axis
fig, ax = plt.subplots(figsize=(10, 5))

for i, algo in enumerate(ordered_algorithms):
    algo_data = summary_data_q[summary_data_q['Algorithm'] == algo].set_index('Question').reindex(questions)
    means = algo_data['mean'].values
    stds = algo_data['std'].fillna(0).values
    
    # Apply vertical jitter: offset each algorithm slightly on the y-axis
    jitter = (i - len(ordered_algorithms) / 2) * 0.1  # Adjust 0.1 for spacing
    y_positions = np.arange(len(questions)) - jitter
    
    ax.errorbar(means, y_positions, xerr=stds, label=algo, marker=marker_dict[algo],
                color=algo_colors[algo], linestyle='-', markerfacecolor=algo_colors[algo], 
                markeredgecolor='white', capsize=4)

legend_elements = [
    Line2D([0], [0], marker=marker_dict[algo], color='w', label=algo,
           markerfacecolor=algo_colors[algo], markersize=8)
    for algo in ordered_algorithms
]
ax.legend(handles=legend_elements, title="Algorithm", fontsize=12, title_fontsize=12)

ax.set_yticks(np.arange(len(questions)))
# ax.set_yticklabels([
#     "Preservation of\nkey information", 
#     "Depiction of\nhand function", 
#     "Contextual\nclarity", 
#     "Visibility of difficulties or\n compensation strategies", 
#     "Representation of\nhand movements"
# ])
ax.set_yticklabels([
    "C5: Preservation of \nkey information", 
    "C4: Depiction of \nhand function", 
    "C3: Contextual\nclarity", 
    "C2: Visibility of \ndifficulties and\ncompensation", 
    "C1: Inclusion \n of hand \nmovements"
])

# Axis formatting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Participant rating", fontsize=16)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.5, 4.5)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=16)
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.8)

plt.tight_layout()
plt.savefig("Figures/HumanEvaluation/Average_Question.png", dpi=600)
# plt.savefig("Figures/Figure2.png", dpi=600)
plt.close()


# ---------------- Plot per video ----------------


# Get unique video names
videos = plot_data['Video'].unique()

for vid in videos:
    fig, ax = plt.subplots(figsize=(10, 5))

    vid_data = plot_data[plot_data['Video'] == vid]

    for i, algo in enumerate(ordered_algorithms):
        algo_data = vid_data[vid_data['Algorithm'] == algo].set_index('Question').reindex(questions)
        means = algo_data['mean'].values
        stds = algo_data['std'].fillna(0).values

        jitter = (i - len(ordered_algorithms) / 2) * 0.1  # vertical offset
        y_positions = np.arange(len(questions)) - jitter

        ax.errorbar(means, y_positions, xerr=stds, label=algo,  
                    marker=marker_dict[algo], color=algo_colors[algo],
                    linestyle='-', markerfacecolor=algo_colors[algo], 
                    markeredgecolor='white', capsize=4)

    # Set y-axis
    ax.set_yticks(np.arange(len(questions)))
    ax.set_yticklabels(['C5', 'C4', 'C3', 'C2', 'C1'])

    # Axis formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Participant rating", fontsize=16)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(f"Figures/HumanEvaluation/Average_Question_{vid}.png", dpi=600)
    plt.close()


# ---------------- Create combined plot: average per video ----------------


# Paths to the images
image_paths = [
    "Figures/HumanEvaluation/Average_Question.png",
    "Figures/HumanEvaluation/Average_Question_1.png",
    "Figures/HumanEvaluation/Average_Question_2.png",
    "Figures/HumanEvaluation/Average_Question_3.png",
    "Figures/HumanEvaluation/Average_Question_4.png",
    "Figures/HumanEvaluation/Average_Question_5.png",
]

# Load images
images = [Image.open(p) for p in image_paths]

# Create matplotlib figure
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
axs = axs.flatten()

# Plot each image
for ax, img, title in zip(axs, images, ["Overall", "Video 1", "Video 2", "Video 3", "Video 4", "Video 5"]):
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig("Figures/HumanEvaluation/Combined_Ratings_Grid.png", dpi=300)
plt.savefig("Figures/FigureA.png", dpi=300)
plt.close()


# ---------------- Plot with all the ratings ----------------


# ------------------- Configuration -------------------
questions = ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
question_labels = [
    "Preservation of\nkey information", 
    "Depiction of\nhand function", 
    "Contextual\nclarity", 
    "Visibility of difficulties or\ncompensatory strategies", 
    "Representation of\nhand movements"
]
ordered_algorithms = ['CA-SUM', 'CTVSUM', 'DR-DSN']
colors = ['crimson', 'cornflowerblue', 'k']
colors = ['#3c466d', '#a2a0c4', '#c69b33']

# ------------------- Melt to long format -------------------
df_long = df.melt(id_vars=['Participant ID', 'Video', 'Algorithm'],
                  value_vars=questions,
                  var_name='Question', value_name='Rating')

# Ensure consistent order for plotting
df_long['Question'] = pd.Categorical(df_long['Question'], categories=questions, ordered=True)
df_long['Algorithm'] = pd.Categorical(df_long['Algorithm'], categories=ordered_algorithms, ordered=True)

# ------------------- Plot -------------------
fig, ax = plt.subplots(figsize=(10, 5))

for i, algo in enumerate(ordered_algorithms):
    df_algo = df_long[df_long['Algorithm'] == algo]
    for pid in df_algo['Participant ID'].unique():
        df_part = df_algo[df_algo['Participant ID'] == pid]
    
        # Average ratings across videos per question
        df_part_avg = (
            df_part.groupby('Question')['Rating']
            .mean()
            .reindex(questions)
            .reset_index()
        )

        y_vals = np.arange(len(questions)) + (i - 1) * 0.1  # vertical jitter by algorithm
        ax.plot(df_part_avg['Rating'].values, y_vals, marker='o',
                color=colors[i], linewidth=1)

# Set labels and ticks
ax.set_yticks(np.arange(len(questions)))
ax.set_yticklabels(question_labels)

# Create the legend elements manually
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='DR-DSN', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='CTVSUM', markerfacecolor='cornflowerblue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='CA-SUM', markerfacecolor='crimson', markersize=8)
]

# Add the legend to your plot
ax.legend(handles=legend_elements, title="Algorithm", fontsize=12, title_fontsize=12)

# Formatting
ax.set_xlabel("Participant rating", fontsize=14)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.5, len(questions) - 0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.8)

plt.tight_layout()
plt.savefig("Figures/HumanEvaluation/Average_Question_IndividualRatings.png", dpi=600)
plt.close()

# ------------------- Plot per video -------------------
videos = df_long['Video'].unique()

for video in videos:
    fig, ax = plt.subplots(figsize=(10, 5))

    df_video = df_long[df_long['Video'] == video]

    for i, algo in enumerate(ordered_algorithms):
        df_algo = df_video[df_video['Algorithm'] == algo]

        for pid in df_algo['Participant ID'].unique():
            df_part = df_algo[df_algo['Participant ID'] == pid]

            # No averaging across videos now — keep all questions for this video
            df_part_sorted = (
                df_part.groupby('Question')['Rating']
                .mean()  # still average across possibly multiple entries per question
                .reindex(questions)
                .reset_index()
            )

            y_vals = np.arange(len(questions)) + (i - 1) * 0.1  # jitter
            ax.plot(df_part_sorted['Rating'].values, y_vals, marker='o',
                    color=colors[i], linewidth=1)

    # Set labels and ticks
    ax.set_yticks(np.arange(len(questions)))
    ax.set_yticklabels(question_labels)

    # Legend
    ax.legend(handles=legend_elements, title="Algorithm", fontsize=12, title_fontsize=12)

    # Formatting
    ax.set_xlabel("Participant rating", fontsize=14)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.5, len(questions) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(f"Figures/HumanEvaluation/Average_Question_IndividualRatings_{video}.png", dpi=600)
    plt.close()


# ---------------- Statistics ----------------


# ###### Are the evaluation questions (Q1–Q5) rated differently?

# # Average across videos and algorithms per participant per question
# participant_avg = (
#     df_long.groupby(['Participant ID', 'Question'])['Rating']
#     .mean()
#     .unstack()
# )

# # Friedman test
# stat, p = friedmanchisquare(*[participant_avg[q] for q in participant_avg.columns])

# # Calculate Kendall’s W
# k = len(participant_avg.columns)  # number of algorithms
# n = len(participant_avg)          # number of raters or participants
# kendalls_W = stat / (n * (k - 1))

# if print_results:
#     print(f"Friedman χ² = {stat:.3f}, p = {p:.4f}, Kendall’s W = {kendalls_W:.3f}")

# if p<0.05:
#     # Use pairwise_tests instead of deprecated pairwise_ttests
#     posthocs = pg.pairwise_tests(
#         data=df_long,
#         dv='Rating',
#         within='Question',
#         subject='Participant ID',
#         padjust='bonf'  # Bonferroni correction
#     )

#     # Display only relevant columns
#     if print_results:
#         print(posthocs[['A', 'B', 'T', 'p-unc', 'p-corr', 'p-adjust']])

# # Define significance stars based on p-value
# def p_to_stars(p):
#     if p < 0.001:
#         return '***'
#     elif p < 0.01:
#         return '**'
#     elif p < 0.05:
#         return '*'
#     else:
#         return None

# # Map each question to y-axis position
# question_to_y = {q: i for i, q in enumerate(questions)}  

# # Collect bars to plot (y1, y2, stars)
# bar_annotations = []
# for _, row in posthocs.iterrows():
#     q1, q2 = row['A'], row['B']
#     p_val = row['p-corr']
#     stars = p_to_stars(p_val)
#     if stars:
#         y1, y2 = question_to_y[q1], question_to_y[q2]
#         bar_annotations.append((min(y1, y2), max(y1, y2), stars))


# ###### Are the algoritms rated differently?


# # Average rating per (Participant, Algorithm)
# algo_avg = (
#     df_long.groupby(['Participant ID', 'Algorithm'])['Rating']
#     .mean()
#     .unstack()
# )

# # Friedman test
# stat, p = friedmanchisquare(*[algo_avg[algo] for algo in algo_avg.columns])

# # Calculate Kendall’s W
# k = len(algo_avg.columns)  # number of algorithms
# n = len(algo_avg)          # number of raters or participants
# kendalls_W = stat / (n * (k - 1))

# if print_results:
#     print(f"Friedman χ² = {stat:.3f}, p = {p:.4f}, Kendall’s W = {kendalls_W:.3f}")

# if p < 0.05:
#     posthocs_algo = pg.pairwise_tests(
#         data=df_long,
#         dv='Rating',
#         within='Algorithm',
#         subject='Participant ID',
#         padjust='bonf',
#         parametric=True
#     )

#     if print_results:
#         print(posthocs_algo[['A', 'B', 'T', 'p-unc', 'p-corr', 'p-adjust']])

# # Map algorithms to y-axis position
# algo_to_y = {algo: i for i, algo in enumerate(ordered_algorithms)}

# # Collect bars to plot (y1, y2, stars)
# bar_annotations_algo = []
# for _, row in posthocs_algo.iterrows():
#     a1, a2 = row['A'], row['B']
#     p_val = row['p-corr']
#     stars = p_to_stars(p_val)
#     if stars:
#         y1, y2 = algo_to_y[a1], algo_to_y[a2]
#         bar_annotations_algo.append((min(y1, y2), max(y1, y2), stars))


###### Do participant ratings differ depending on the algorithm, 
###### the evaluation criterion, or the interaction between them?


# Test normality per Algorithm × Question combination
df_long['Condition'] = df_long['Algorithm'].astype(str) + ' | ' + df_long['Question'].astype(str)
normality_results = pg.normality(data=df_long, dv='Rating', group='Condition')
# print(normality_results)

# Group by condition and apply skew and kurtosis functions
# from scipy.stats import skew, kurtosis
# print(df_long.groupby('Condition')['Rating'].agg([skew, kurtosis]))

# Two-way repeated-measures ANOVA
anova = pg.rm_anova(
    data=df_long,
    dv='Rating',
    within=['Algorithm', 'Question'],
    subject='Participant ID',
    detailed=True
)

if print_results:
    print(anova)

# Check if the p-value for 'Algorithm' is significant
if anova['p-unc'][0] < 0.05:  # If Algorithm is significant
    # Perform Tukey's HSD for 'Algorithm'
    tukey_algo = pairwise_tukeyhsd(df_long['Rating'], df_long['Algorithm'], alpha=0.05)
    print("Post-hoc testing for Algorithm:")
    print(tukey_algo.summary())

# Check if the p-value for 'Question' is significant
if anova['p-unc'][1] < 0.05:  # If Question is significant
    # Perform Tukey's HSD for 'Question'
    tukey_question = pairwise_tukeyhsd(df_long['Rating'], df_long['Question'], alpha=0.05)
    print("Post-hoc testing for Question:")
    print(tukey_question.summary())

# If the interaction effect (Algorithm * Question) is significant
if anova['p-unc'][2] < 0.05:  # If Algorithm * Question interaction is significant
    # Tukey's HSD for the interaction (if relevant)
    tukey_interaction = pairwise_tukeyhsd(df_long['Rating'], df_long['Algorithm'] + df_long['Question'], alpha=0.05)
    print("Post-hoc testing for Algorithm * Question interaction:")
    print(tukey_interaction.summary())

# Define significance stars based on p-value
def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return None

# Map algorithms or groups to y-axis position
algo_to_y = {algo: i for i, algo in enumerate(ordered_algorithms)}

# Collect bars to plot (y1, y2, stars) from Tukey's HSD results
bar_annotations_algo = []

# Iterate over Tukey's HSD results
for row in tukey_algo.summary().data[1:]:  # Skip the header row
    a1, a2 = row[0], row[1]  # group1 and group2
    p_val = row[3]  # p-value
    stars = p_to_stars(p_val)  # Define the p_to_stars function (e.g., based on p-value)

    if stars:
        y1, y2 = algo_to_y[a1], algo_to_y[a2]
        bar_annotations_algo.append((min(y1, y2), max(y1, y2), stars))

# Collect bars to plot (y1, y2, stars) from Tukey's HSD results
question_to_y = {question: i for i, question in enumerate(questions)}

# Create a list to store bar annotations (y1, y2, stars)
bar_annotations_question = []

# Iterate over Tukey's HSD results for 'Question' (post-hoc results)
for row in tukey_question.summary().data[1:]:  # Skip the header row
    q1, q2 = row[0], row[1]  # group1 and group2 (questions)
    p_val = row[3]  # p-value
    stars = p_to_stars(p_val)  # Get significance stars from p-value
    
    if stars:
        y1, y2 = question_to_y[q1], question_to_y[q2]
        bar_annotations_question.append((min(y1, y2), max(y1, y2), stars))

# ---------------- Figure 2 – Significances ----------------


# Average per question (Figure 2A)
avg_data = (
    summary_data_q
    .groupby('Question')
    .agg(mean=('mean', 'mean'), std=('mean', 'std'))
    .reindex(questions)
)

# Average per algorithm (Figure 2B)
avg_per_algo = (
    summary_data_q
    .groupby('Algorithm')
    .agg(mean=('mean', 'mean'), std=('mean', 'std'))
    .reindex(ordered_algorithms)
)

# Figure with A and B above C
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
marker_dict = {'DR-DSN': 'D', 'CTVSUM': 'X', 'CA-SUM': 's'}
color_dict = {'DR-DSN': '#3c466d', 'CTVSUM': '#a2a0c4', 'CA-SUM': '#c69b33'}

# -------------- Plot 2A – Average per Question (top left) --------------
y_positions = np.arange(len(questions))

axes[0].errorbar(avg_data['mean'], y_positions, xerr=avg_data['std'], marker='o',
                 linestyle='-', color='black', markerfacecolor='black', markeredgecolor='white',
                 markersize=8, capsize=0, label='Average')

# Add individual participant lines in the background
participant_matrix = (
    df_long
    .groupby(['Participant ID', 'Question'])['Rating']
    .mean()
    .unstack()
    .reindex(columns=questions)
)
for _, row in participant_matrix.iterrows():
    axes[0].plot(row.values, np.arange(len(questions)), color='gray', 
                 alpha=0.3, linewidth=1, label='Individual')

axes[0].set_yticks(y_positions)
axes[0].set_yticklabels(["C5", "C4", "C3", "C2", "C1"])
axes[0].set_ylim(-0.5, 4.5)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[-2:], labels[-2:])

# Add significance bars to Plot 2A
x_base = 4.95  # adjust based on axis scale
offset = 0.08  # spacing between bars

# bar_annotations_question[2], bar_annotations_question[3] = bar_annotations_question[3], bar_annotations_question[2]
# bar_annotations_question[1], bar_annotations_question[4] = bar_annotations_question[4], bar_annotations_question[1]

for i, (y1, y2, stars) in enumerate(bar_annotations_question):
    x = x_base + i * offset
    y_top = y2
    y_bot = y1
    axes[0].plot([x, x], [y_bot, y_top], color='black', linewidth=1)
    axes[0].text(x + 0.004, (y_bot + y_top) / 2, stars, 
                 ha='left', va='center', fontsize=10, 
                 rotation=90, clip_on=False)

# -------------- Plot 2B – Average per Algorithm (top right) --------------
y_positions = np.arange(len(ordered_algorithms))
for i, algo in enumerate(ordered_algorithms):
    row = avg_per_algo.loc[algo]
    axes[1].errorbar(row['mean'], i, xerr=row['std'],
                     fmt=marker_dict[algo], markersize=8,
                     color=color_dict[algo],
                     markerfacecolor=color_dict[algo],
                     markeredgecolor='white', linestyle='-', capsize=0)
    
# Compute individual participant averages per algorithm
participant_algo_avg = (
    df_long
    .groupby(['Participant ID', 'Algorithm'])['Rating']
    .mean()
    .unstack()  # rows = Participant ID, columns = Algorithm
    .reindex(columns=ordered_algorithms)  # ensure order matches plot
)

# Plot faint individual participant averages with vertical jitter
for _, row in participant_algo_avg.iterrows():
    for i, algo in enumerate(ordered_algorithms):
        jitter = np.random.uniform(-0.1, 0.1)  # adjust the range for desired spacing
        y_pos = i + jitter
        axes[1].plot(row[algo], y_pos, marker=marker_dict[algo], markeredgecolor='white',
                     color=color_dict[algo], alpha=0.2, zorder=0)

axes[1].set_yticks(y_positions)
axes[1].set_yticklabels(ordered_algorithms)
axes[1].set_ylim(-0.5, len(ordered_algorithms)-0.5)

# Add significance bars to Plot 2B
x_base = 5.1  # adjust based on axis scale
offset = 0.11  # spacing between bars

for i, (y1, y2, stars) in enumerate(bar_annotations_algo):
    x = x_base + i * offset
    axes[1].plot([x, x], [y1, y2], color='black', linewidth=1)
    axes[1].text(x + 0.01, (y1 + y2) / 2, stars, ha='left', va='center', 
                 fontsize=10, rotation=90, clip_on=False)

# -------------- Plot 2C – Ratings per Algorithm (bottom, spans both columns) --------------
for i, algo in enumerate(ordered_algorithms[::-1]):
    algo_data = summary_data_q[summary_data_q['Algorithm'] == algo].set_index('Question').reindex(questions)
    means = algo_data['mean'].values
    stds = algo_data['std'].fillna(0).values
    jitter = (i - len(ordered_algorithms) / 2) * 0.1
    y_positions = np.arange(len(questions)) - jitter
    axes[2].errorbar(means, y_positions, xerr=stds, label=algo, marker=marker_dict[algo],
                     linestyle='-', color=color_dict[algo], markeredgecolor='w',
                     markerfacecolor=color_dict[algo], capsize=0)
axes[2].set_yticks(np.arange(len(questions)))
axes[2].set_yticklabels([
    "C5: Preservation of \nkey information", 
    "C4: Depiction of \nhand function", 
    "C3: Contextual\nclarity", 
    "C2: Visibility of \ndifficulties and\ncompensation", 
    "C1: Inclusion \n of hand \nmovements"
])
# axes[2].set_yticklabels([
#     "C5: Key \ninformation", 
#     "C4: Hand \nfunction", 
#     "C3: Context", 
#     "C2: Compensation", 
#     "C1: Hand \nmovements"
# ])
axes[2].set_ylim(-0.5, 4.5)
axes[2].legend(title="Algorithm", fontsize=14, title_fontsize=14, loc='upper left')

# # Add (C1)–(C5) labels
# short_labels = ["C5:", "C4:", "C3", "C2:", "C1:"]
# for i, label in enumerate(short_labels):
#     axes[2].text(-0.15, i, label, va='center', ha='right', fontsize=14,
#                  transform=axes[2].get_yaxis_transform())

# Shared styling
for ax in axes:
    ax.set_xlim(0.89, 5.51)
    ax.set_xlabel("Participant rating", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space above the plot
plt.subplots_adjust(hspace=0.3)  # Increase spacing between the two rows
plt.savefig("Figures/Figure2.png", dpi=600)
plt.close()



# ---------------- Computational evaluation ----------------


# Load data
data_path = '../../01_Data/Results/Computational_Evaluation.csv'
comp_data = pd.read_csv(data_path)

# Load human ratings
human_data_path = '../../01_Data/Results/Human_Evaluation.csv'
human_df = pd.read_csv(human_data_path)

# Compute mean human score across Q1–Q5
human_df['MeanRating'] = human_df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Aggregate mean ratings by video and algorithm
human_avg = (
    human_df.groupby(['Video', 'Algorithm'])['MeanRating']
    .mean()
    .reset_index()
)

# Filter by split and specific videos
target_videos = ['SCI02_01', 'SCI03_06', 'SCI08_32', 'SCI17_10', 'SCI21_09']
comp_data = comp_data[(comp_data['Split'] == 0) & (comp_data['Video'].isin(target_videos))]

# Adjust names
comp_data['Algorithm'] = comp_data['Algorithm'].replace({
    'pytorch-CTVSUM': 'CTVSUM',
    'pytorch-vsumm-reinforce': 'DR-DNS',
})

# Adjust names
comp_data['Video'] = comp_data['Video'].replace({
    'SCI02_01': '1',
    'SCI03_06': '2',
    'SCI08_32': '3',
    'SCI17_10': '4',
    'SCI21_09': '5'
})

# Ensure both 'Video' columns are of the same type
comp_data['Video'] = comp_data['Video'].astype(str)
human_avg['Video'] = human_avg['Video'].astype(str)

# Merge with computational data
merged_df = pd.merge(
    comp_data,
    human_avg,
    on=['Video', 'Algorithm'],
    how='inner'
)

# Drop 'Split' column
merged_df = merged_df.drop(columns=['Split'])

# # Correlation analysis
# correlation_results = merged_df.corr(numeric_only=True)['MeanRating'].sort_values(ascending=False)
# print("Correlation with Mean Human Rating:\n", correlation_results)


for col in ['Representativeness', 'InformationLoss', 'Coverage', 'TemporalDistribution', 'Diversity']:
    X = sm.add_constant(merged_df[[col]])
    y = merged_df['MeanRating']
    model = sm.OLS(y, X).fit()
    if print_results:
        print(f"\nModel with {col}:\n", model.summary())
