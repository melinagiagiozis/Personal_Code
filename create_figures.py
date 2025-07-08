"""
Script: create_figure.py

Description:
    This script creates figures to visualize the video summaries and
    saves then in Personal_Code/Figures.

"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from glob import glob
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as colors
import json
import h5py


# --------------------- plot importance scores ---------------------

# for video_id in ['SCI02_01', 'SCI02_07', 'SCI02_16',
#                  'SCI03_06', 'SCI08_32', 'SCI12_06',
#                  'SCI15_14', 'SCI15_15', 'SCI16_01',
#                  'SCI17_01', 'SCI17_07', 'SCI17_10', 
#                  'SCI21_09']:

for video_id in ['SCI02_01']:


    # ---------- Load CA-SUM scores ----------
    ca_sum_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code/02_ca-sum_40epochs_egocentric/importance_scores_split_0.json"
    with open(ca_sum_path, "r") as f:
        ca_sum_scores = np.array(json.load(f).get(video_id, []))

    # ---------- Load DR-DSN scores ----------
    h5_path = "/Users/melinagiagiozis/Desktop/result_split_0.h5"
    json_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code/03_pytorch-vsumm-reinforce_40epochs_egocentric/importance_scores_split_0.json"

    # # Create json from h5 file (save h5 to desktop!)
    # output = {}
    # with h5py.File(h5_path, "r") as f:
    #     for video_id in f.keys():
    #         if "score" in f[video_id]:
    #             scores = f[video_id]["score"][...]
    #             output[video_id] = scores.tolist()  # convert numpy array to list
    # with open(json_path, "w") as f:
    #     json.dump(output, f, indent=2)

    rl_sum_path = json_path
    
    with open(rl_sum_path, "r") as f:
        rl_sum_scores = np.array(json.load(f).get(video_id, []))

    # ---------- Load CTVSUM scores ----------
    cl_sum_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code/01_pytorch-ctvsum_40epochs_egocentric/importance_scores_split_0.json"
    with open(cl_sum_path, "r") as f:
        cl_sum_scores = np.array(json.load(f).get(video_id, []))

    # ---------- Plot using scatter with colormap ----------
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    cmap = plt.get_cmap('Reds')  # or 'hot', 'viridis', etc.

    def plot_colored_scatter(ax, scores, title, y_label):
        x = np.arange(len(scores))
        norm = colors.Normalize(vmin=scores.min(), vmax=scores.max())  # scale per subplot
        sc = ax.scatter(x, scores, c=scores, cmap=cmap, norm=norm, s=5)
        ax.set_title(title)
        ax.set_ylabel(y_label, size=9)
        ax.grid(True)

    plot_colored_scatter(axs[1], rl_sum_scores, 'DR-DSN', r'$\mathrm{Importance\ Score}_{DR-DSN}$')
    plot_colored_scatter(axs[2], cl_sum_scores, 'CTVSUM', r'$\mathrm{Importance\ Score}_{CTVSUM}$')
    plot_colored_scatter(axs[0], ca_sum_scores, 'CA-SUM', r'$\mathrm{Importance\ Score}_{CA-SUM}$')
    axs[2].set_xlabel("Video frame")

    plt.tight_layout()
    plt.savefig('Figures/' + video_id + '_summary_comparison_v1.png', dpi=300, bbox_inches='tight')
    plt.close()


    # --------------------- plot importance scores (highlighted summary frames) ---------------------


    # ---------- Helper to extract frame indices from filenames ----------
    def load_summary_frame_indices(frame_dir):
        frames = sorted(glob(os.path.join(frame_dir, '*.jpg')))
        return [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in frames]

    # ---------- Define paths to summary folders ----------
    summary_dirs = {
        'DR-DSN': f'/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/pytorch-vsumm-reinforce/split_0/{video_id.split("_")[0]}/{video_id}',
        'CA-SUM': f'/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/CA-SUM/split_0/{video_id.split("_")[0]}/{video_id}',
        'CTVSUM': f'/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/pytorch-CTVSUM/split_0/{video_id.split("_")[0]}/{video_id}'
    }

    # ---------- Load importance scores ----------
    with open("/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code/01_pytorch-ctvsum_40epochs_egocentric/importance_scores_split_0.json", "r") as f:
        cl_sum_scores = np.array(json.load(f).get(video_id, []))

    with open("/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code/02_ca-sum_40epochs_egocentric/importance_scores_split_0.json", "r") as f:
        ca_sum_scores = np.array(json.load(f).get(video_id, []))

    with open("/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code/03_pytorch-vsumm-reinforce_40epochs_egocentric/importance_scores_split_0.json", "r") as f:
        rl_sum_scores = np.array(json.load(f).get(video_id, []))

    # ---------- Prepare summary indices ----------
    summary_positions = {
        'DR-DSN': set(load_summary_frame_indices(summary_dirs['DR-DSN'])),
        'CTVSUM': set(load_summary_frame_indices(summary_dirs['CTVSUM'])),
        'CA-SUM': set(load_summary_frame_indices(summary_dirs['CA-SUM']))
    }

    # ---------- Plotting ----------
    def plot_highlighted(ax, scores, selected_ids, title, y_label):
        x = np.arange(len(scores))
        selected = np.array([i in selected_ids for i in x])
        
        # Plot data
        ax.scatter(x[~selected], scores[~selected], color='lightgrey', s=20, label='Full video frames')
        ax.scatter(x[selected], scores[selected], color='crimson', s=20, label='Summary frames')
        
        # Title below plot
        ax.text(0.5, -0.05, title, transform=ax.transAxes, ha='center', va='top', fontsize=28)
        # ax.set_title(title, size=18, pad=12)
        # ax.set_ylabel(y_label, size=13)
        # ax.tick_params(axis='y', labelsize=14)
        # ax.tick_params(axis='x', labelsize=14)
        # if title == '(C) DR-DSN':
        #     ax.set_yticks([0.4, 0.5, 0.6])

        # Compute custom y-ticks: min, 1/3, 2/3, max
        y_min = scores.min()
        y_max = scores.max()
        yticks = np.linspace(y_min, y_max, 4)

        # Set y-ticks and hide labels
        ax.set_yticks(yticks)
        ax.set_yticklabels([])

        ax.grid(True)

    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    plot_highlighted(axs[0], rl_sum_scores, summary_positions['DR-DSN'], 
                     '(A) DR-DSN', r'$\mathrm{Importance\ Score}_{DR-DSN}$')
    plot_highlighted(axs[1], cl_sum_scores, summary_positions['CTVSUM'], 
                     '(B) CTVSUM', r'$\mathrm{Importance\ Score}_{CTVSUM}$')
    plot_highlighted(axs[2], ca_sum_scores, summary_positions['CA-SUM'], 
                     '(C) CA-SUM', r'$\mathrm{Importance\ Score}_{CA-SUM}$')
    # axs[2].set_xlabel('Frames', size=13)

    # fig.align_ylabels(axs)

    # axs[0].set_ylim(0.7, 0.9)
    # axs[1].set_ylim(0.0, 1.02)
    # axs[2].set_ylim(0.4, 0.6)

    # Remove top and right spines
    for i in [0, 1, 2]:
        # axs[i].spines['top'].set_visible(False)
        # axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, bottom=0.15)
    plt.savefig(f'Figures/{video_id}_summary_comparison_v2.png', dpi=300, bbox_inches='tight')
    plt.close()


# --------------------- visualize summaries ---------------------

for video in ['SCI02_01', 'SCI02_07', 'SCI02_16',
              'SCI03_06', 'SCI08_32', 'SCI12_06',
              'SCI15_14', 'SCI15_15', 'SCI16_01',
              'SCI17_01', 'SCI17_07', 'SCI17_10', 
              'SCI21_09']:

    subject_id = video.split('_')[0]

    # Paths
    full_video_dir = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Frames/' + subject_id + '/' + video
    summary_dirs = {
        'DR-DSN': '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/pytorch-vsumm-reinforce/split_0/' + subject_id + '/' + video,
        'CTVSUM': '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/pytorch-CTVSUM/split_0/' + subject_id + '/' + video,
        'CA-SUM': '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/CA-SUM/split_0/' + subject_id + '/' + video
    }

    # Helper to load frame indices
    def load_frame_indices(frame_dir):
        frames = sorted(glob(os.path.join(frame_dir, '*.jpg')))
        indices = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in frames]
        return frames, indices

    # Load full video frames
    full_frames, full_indices = load_frame_indices(full_video_dir)
    n_total = len(full_frames)
    hist = np.ones(n_total)

    # Load summary frame indices
    summary_indices = {}
    for key, path in summary_dirs.items():
        _, indices = load_frame_indices(path)
        summary_indices[key] = indices

    # Plotting
    fig, axs = plt.subplots(len(summary_dirs) + 1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [5, 1]})

    # Full video row
    axs[0, 0].bar(range(n_total), hist, color='lightgray')
    axs[0, 0].set_title('(A) Example frames from an egocentric video')
    axs[0, 1].imshow(mpimg.imread(full_frames[n_total // 2]))
    axs[0, 1].axis('off')

    # Summary rows
    numerate = ['(B)', '(C)', '(D)']
    colors = ['green', 'blue', 'purple']
    for i, (key, indices) in enumerate(summary_indices.items(), start=1):
        bars = np.zeros(n_total)
        bars[indices] = 1
        axs[i, 0].bar(range(n_total), bars, color=colors[i - 1])
        axs[i, 0].set_title(numerate[i - 1] + ' ' + key)
        
        if indices:
            mid_idx = indices[len(indices) // 2]
            axs[i, 1].imshow(mpimg.imread(full_frames[mid_idx]))
        axs[i, 1].axis('off')

    for ax in axs[:, 0]:
        ax.set_xlim(0, n_total)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig('Figures/' + video + '_summary_comparison_v3.png', dpi=300, bbox_inches='tight')
    plt.close()


# --------------------- visualize summaries with thumbnails ---------------------

for video in ['SCI02_01', 'SCI02_07', 'SCI02_16',
              'SCI03_06', 'SCI08_32', 'SCI12_06',
              'SCI15_14', 'SCI15_15', 'SCI16_01',
              'SCI17_01', 'SCI17_07', 'SCI17_10', 
              'SCI21_09']:

    subject_id = video.split('_')[0]

    # Paths
    full_video_dir = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Frames/' + subject_id + '/' + video
    summary_dirs = {
        'DR-DSN': '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/pytorch-vsumm-reinforce/split_0/' + subject_id + '/' + video,
        'CTVSUM': '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/pytorch-CTVSUM/split_0/' + subject_id + '/' + video,
        'CA-SUM': '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries/CA-SUM/split_0/' + subject_id + '/' + video
    }


    # Helper to get evenly spaced indices
    def get_evenly_spaced_indices(indices, n, offset_ratio):
        if not indices or n == 0:
            return []
        start = int(len(indices) * offset_ratio)
        end = len(indices)
        effective_indices = indices[start:end]
        if not effective_indices:
            effective_indices = indices
        step = max(1, len(effective_indices) // n)
        selected = [effective_indices[i] for i in range(0, len(effective_indices), step)]
        return selected[:n]

    # Load full video frames
    full_frames, full_indices = load_frame_indices(full_video_dir)
    n_total = len(full_frames)
    id_to_position = {fid: idx for idx, fid in enumerate(full_indices)}

    # Load summary frame indices
    summary_indices = {}
    for key, path in summary_dirs.items():
        _, indices = load_frame_indices(path)
        summary_indices[key] = indices

    # Plotting setup
    n_thumbs = 5
    colors = {
        'Full Video': 'lightgray',
        'CA-SUM': 'green',
        'CTVSUM': 'blue',
        'DR-DSN': 'm'
    }

    # Step 1: Create list of rows to plot (hist or thumb)
    row_configs = []
    row_configs.append(("thumb", 'Full Video', None))  # just thumbnails
    row_configs.append(("title", 'Full Video', '(A) Example frames from an egocentric video'))  # title below

    method_titles = ['(B) DR-DSN', '(C) CTVSUM', '(D) CA-SUM']
    for i, key in enumerate(summary_dirs.keys()):
        row_configs.append(("hist", key, None))
        row_configs.append(("thumb", key, None))
        row_configs.append(("title", key, method_titles[i]))

    # Step 2: Create GridSpec
    fig = plt.figure(figsize=(18, 1.5 * len(row_configs)), constrained_layout=True)
    gs = gridspec.GridSpec(len(row_configs), 1, height_ratios=[
        1 if kind == 'hist' else 0.6 if kind == 'thumb' else 0.2
        for kind, _, _ in row_configs
    ])
    gs.update(hspace=1.2)

    # Step 3: Plot
    for i, (kind, label, title) in enumerate(row_configs):
        if label == 'Full Video':
            indices = list(range(n_total))
        else:
            indices = summary_indices[label]
        
        thumb_ids = get_evenly_spaced_indices(indices, n_thumbs, offset_ratio=0.13)
        thumb_indices = [id_to_position[i] for i in thumb_ids if i in id_to_position]

        if kind == 'hist':
            ax = fig.add_subplot(gs[i])

            # Importance scores
            if label == 'CA-SUM':
                scores = ca_sum_scores
                summary_ids = summary_indices['CA-SUM']
            elif label == 'CTVSUM':
                scores = cl_sum_scores
                summary_ids = summary_indices['CTVSUM']
            elif label == 'DR-DSN':
                scores = rl_sum_scores
                summary_ids = summary_indices['DR-DSN']
            else:
                scores = np.zeros(n_total)
                summary_ids = []

            bars = np.zeros(n_total)
            bars[:len(scores)] = scores
            bar_colors = ['lightgray'] * n_total

            # Color summary frames in method color
            for idx in summary_ids:
                if 0 <= idx < n_total:
                    bar_colors[idx] = colors[label]

            # # Color thumbnail frames black (on top)
            # for idx in thumb_indices:
            #     if 0 <= idx < n_total:
            #         bar_colors[idx] = 'k'

            if label == 'DR-DSN':
                y_pos = max(scores) - 0.04 * (max(scores) - min(scores))  # 2% lower
            else:
                y_pos = max(scores) - 0.1 * (max(scores) - min(scores))  # 2% lower
            for idx in thumb_indices:
                if 0 <= idx < n_total:
                    ax.text(idx, y_pos, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')

            ax.bar(range(n_total), bars, color=bar_colors)
            ax.set_xlim(0, n_total)
            ax.set_ylim(min(scores), max(scores))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            if title:
                ax.text(0.5, -0.3, title, fontsize=26, ha='center', va='top', transform=ax.transAxes)

        elif kind == 'thumb':
            ax = fig.add_subplot(gs[i])
            ax.axis('off')
            for j, idx in enumerate(thumb_indices):
                img = mpimg.imread(full_frames[idx])
                imagebox = OffsetImage(img, zoom=0.1)
                ab = AnnotationBbox(imagebox, (j + 0.5, 0.5), frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
            ax.set_xlim(0, n_thumbs)
            ax.set_ylim(0, 1)

        elif kind == 'title':
            ax = fig.add_subplot(gs[i])
            ax.axis('off')
            ax.text(0.5, 0.5, title, fontsize=26, ha='center', va='center', transform=ax.transAxes)

    # Save the figure
    plt.savefig('Figures/' + video + '_summary_comparison_v4.pdf', dpi=300, bbox_inches='tight')
    plt.close()
