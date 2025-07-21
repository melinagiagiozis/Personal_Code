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
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare


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

questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# for q in questions:
#     # Create a DataFrame for this question with required columns
#     df_q = df[["Video", "Participant ID", q]].rename(columns={q: "Score"})
    
#     # Compute ICC
#     icc_q = pg.intraclass_corr(data=df_q, targets='Video', raters='Participant ID', ratings='Score')
#     icc2_q = icc_q[icc_q["Type"] == "ICC2"]
    
#     print(f"\nICC for {q}:")
#     print(icc2_q[["Type", "ICC", "CI95%", "F", "pval"]])

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
    # stats = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats = stats[['mean', 'std']]
    stats.columns = [f"{q}_{col}" for col in stats.columns]
    question_stats[q] = stats

# Combine all question summaries into one table
combined_stats = pd.concat(question_stats.values(), axis=1)
print(combined_stats)

# ---------------- Heatmap averages (Video x Algorthm) ----------------

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
    ordered_columns = ['DR-DSN', 'CTVSUM', 'CA–SUM']
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
    ordered_columns = ['DR-DSN', 'CTVSUM', 'CA–SUM']
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
ordered_algorithms = ['DR-DSN', 'CTVSUM', 'CA–SUM']
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
ordered_algorithms = ['DR-DSN', 'CTVSUM', 'CA–SUM']
videos = [1, 2, 3, 4, 5]
# colors = sns.color_palette("rocket_r", n_colors=3)
# colors = ['m', 'blue', 'green']
# colors = ['#c4a148', '#c86340', '#60adcd']
colors = ['k', 'cornflowerblue', 'crimson']

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


# ---------------- Average over everything to show results per Q ----------------


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

# Plot: Questions on y-axis, average rating on x-axis
fig, ax = plt.subplots(figsize=(10, 5))

for i, algo in enumerate(ordered_algorithms):
    algo_data = summary_data_q[summary_data_q['Algorithm'] == algo].set_index('Question').reindex(questions)
    means = algo_data['mean'].values
    stds = algo_data['std'].fillna(0).values
    # ax.errorbar(means, questions, xerr=stds, label=algo, marker='o',
    #             linestyle='-', color=colors[i], capsize=4)
    for i, algo in enumerate(ordered_algorithms):
        algo_data = summary_data_q[summary_data_q['Algorithm'] == algo].set_index('Question').reindex(questions)
        means = algo_data['mean'].values
        stds = algo_data['std'].fillna(0).values
        
        # Apply vertical jitter: offset each algorithm slightly on the y-axis
        jitter = (i - len(ordered_algorithms) / 2) * 0.1  # Adjust 0.1 for spacing
        y_positions = np.arange(len(questions)) + jitter
        
        ax.errorbar(means, y_positions, xerr=stds, label=algo, marker='o',
                    linestyle='-', color=colors[i], capsize=4)

# Custom legend: only markers
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=algo, markerfacecolor=colors[i], markersize=8)
    for i, algo in enumerate(ordered_algorithms)
]
ax.legend(handles=legend_elements, title="Algorithm", fontsize=12, title_fontsize=12)

ax.set_yticks(np.arange(len(questions)))
ax.set_yticklabels([
    "Preservation of\nkey information", 
    "Depiction of\nhand function", 
    "Contextual\nclarity", 
    "Visibility of difficulties or\n compensation strategies", 
    "Representation of\nhand movements"
])

# Axis formatting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Participant Rating", fontsize=14)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.5, 4.5)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig("Figures/HumanEvaluation/Average_Question.png", dpi=600)
plt.close()


# ---------------- Average over everything to show results per Q with individual ratings ----------------


# 1. Define full labels for evaluation criteria
question_map = {
    'Q5': "Preservation of\nkey information",
    'Q4': "Depiction of\nhand function",
    'Q3': "Contextual\nclarity",
    'Q2': "Visibility of difficulties or\n compensation strategies",
    'Q1': "Representation of\nhand movements"
}

# 2. Prepare list of full question labels (for correct order)
questions = [
    "Preservation of\nkey information", 
    "Depiction of\nhand function", 
    "Contextual\nclarity", 
    "Visibility of difficulties or\n compensation strategies", 
    "Representation of\nhand movements"
]

# 3. Prepare participant_ratings in long format
participant_ratings = df.melt(
    id_vars=['Participant ID', 'Video', 'Summary', 'Algorithm'],
    value_vars=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
    var_name='QuestionCode',
    value_name='Rating'
)
participant_ratings['Question'] = participant_ratings['QuestionCode'].map(question_map)
participant_ratings = participant_ratings[['Algorithm', 'Question', 'Rating']]

# 4. Set plotting order and colors
ordered_algorithms = ['DR-DSN', 'CTVSUM', 'CA–SUM']
colors = ['k', 'cornflowerblue', 'crimson']

# 5. Plotting
fig, ax = plt.subplots(figsize=(10, 5))

for i, algo in enumerate(ordered_algorithms):
    # Get means and stds for the current algorithm
    algo_data = summary_data_q[summary_data_q['Algorithm'] == algo].set_index('Question').reindex(questions)
    means = algo_data['mean'].values
    stds = algo_data['std'].fillna(0).values

    # Vertical jitter to avoid overlapping
    jitter = (i - len(ordered_algorithms) / 2) * 0.1
    y_positions = np.arange(len(questions)) + jitter

    # Plot group-level error bars
    ax.errorbar(means, y_positions, xerr=stds, label=algo, marker='o',
                linestyle='-', color=colors[i], capsize=4)

    # Plot individual participant ratings
    algo_ratings = participant_ratings[participant_ratings['Algorithm'] == algo]
    for j, q in enumerate(questions):
        ratings = algo_ratings[algo_ratings['Question'] == q]['Rating'].values
        ax.scatter(ratings, np.full_like(ratings, j + jitter),
                   color=colors[i], alpha=0.7, s=10)

# 6. Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=algo, markerfacecolor=colors[i], markersize=8)
    for i, algo in enumerate(ordered_algorithms)
]
ax.legend(handles=legend_elements, title="Algorithm", fontsize=12, title_fontsize=12)

# 7. Y-axis ticks
ax.set_yticks(np.arange(len(questions)))
ax.set_yticklabels(questions)

# 8. Styling and labels
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Participant Rating", fontsize=14)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.5, len(questions) - 0.5)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig("Figures/HumanEvaluation/Average_Question_IndividualRatings.png", dpi=600)
plt.show()


# ---------------- Statistics ----------------


###### Are the evaluation questions (Q1–Q5) rated differently?

# Average across videos and algorithms per participant per question
participant_avg = (
    df_long.groupby(['Participant ID', 'Question'])['Rating']
    .mean()
    .unstack()
)

# Friedman test
stat, p = friedmanchisquare(*[participant_avg[q] for q in participant_avg.columns])

# Calculate Kendall’s W
k = len(participant_avg.columns)  # number of algorithms
n = len(participant_avg)          # number of raters or participants
kendalls_W = stat / (n * (k - 1))

print(f"Friedman χ² = {stat:.3f}, p = {p:.4f}, Kendall’s W = {kendalls_W:.3f}")

if p<0.05:
    # Use pairwise_tests instead of deprecated pairwise_ttests
    posthocs = pg.pairwise_tests(
        data=df_long,
        dv='Rating',
        within='Question',
        subject='Participant ID',
        padjust='bonf'  # Bonferroni correction
    )

    # Display only relevant columns
    print(posthocs[['A', 'B', 'T', 'p-unc', 'p-corr', 'p-adjust']])


###### Are the algoritms rated differently?


# Average rating per (Participant, Algorithm)
algo_avg = (
    df_long.groupby(['Participant ID', 'Algorithm'])['Rating']
    .mean()
    .unstack()
)

# Friedman test
stat, p = friedmanchisquare(*[algo_avg[algo] for algo in algo_avg.columns])

# Calculate Kendall’s W
k = len(algo_avg.columns)  # number of algorithms
n = len(algo_avg)          # number of raters or participants
kendalls_W = stat / (n * (k - 1))

print(f"Friedman χ² = {stat:.3f}, p = {p:.4f}, Kendall’s W = {kendalls_W:.3f}")

if p < 0.05:
    posthocs_algo = pg.pairwise_tests(
        data=df_long,
        dv='Rating',
        within='Algorithm',
        subject='Participant ID',
        padjust='bonf'
    )

    print(posthocs_algo[['A', 'B', 'T', 'p-unc', 'p-corr', 'p-adjust']])

