import os
import time
import numpy as np
import cv2
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.pyplot as plt
from config import *
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_frames_from_folder(folder_path):
    frames = sorted(os.listdir(folder_path))
    return [os.path.join(folder_path, frame) for frame in frames if frame.endswith(('.png', '.jpg', '.jpeg'))]


def compute_coverage(summary_frames, full_frames):
    print(f"  - Computing coverage...")
    return len(summary_frames) / len(full_frames)


def compute_temporal_distribution(summary_frames, full_frames):
    print("  - Computing temporal distribution...")

    # Extract filenames only
    summary_names = set(os.path.basename(f) for f in summary_frames)
    full_names = [os.path.basename(f) for f in full_frames]

    # Find indices in full video that correspond to summary frames
    summary_indices = [i for i, name in enumerate(full_names) if name in summary_names]

    normalized_summary = np.array(summary_indices) / len(full_frames)
    uniform_grid = np.linspace(0, 1, len(summary_indices))
    deviation = normalized_summary - uniform_grid
    return np.std(deviation)


def compute_histograms(frames):
    histograms = []
    for frame in frames:
        img = cv2.imread(frame)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        histograms.append(hist.flatten())
    return np.array(histograms)


def compute_diversity(summary_frames):
    print(f"  - Computing diversity...")
    histograms = compute_histograms(summary_frames)
    distances = pairwise_distances(histograms, metric="cosine")
    return np.mean(distances)


def compute_representativeness(summary_frames, full_frames):
    print(f"  - Computing representativeness...")
    summary_histograms = compute_histograms(summary_frames)
    full_histograms = compute_histograms(full_frames)
    if len(summary_histograms) == 0 or len(full_histograms) == 0:
        return 0
    distances = pairwise_distances(full_histograms, summary_histograms, metric="cosine")
    min_distances = np.min(distances, axis=1)
    return 1 - np.mean(min_distances)


def compute_entropy(frames):
    entropies = []
    for frame in frames:
        img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-9))
        entropies.append(entropy)
    return np.mean(entropies) if entropies else 0


def compute_information_loss(summary_frames, full_frames):
    print(f"  - Computing information loss...")

    summary_entropy = compute_entropy(summary_frames)
    original_entropy = compute_entropy(full_frames)

    if original_entropy == 0:
        return 1  # Avoid division by zero

    info_loss = 1 - (summary_entropy / original_entropy)
    return info_loss


def evaluate_summary(summary_folder, full_folder):
    print(f"> Evaluating summary: {summary_folder}")
    summary_frames = get_frames_from_folder(summary_folder)
    full_frames = get_frames_from_folder(full_folder)

    return {
        "coverage": compute_coverage(summary_frames, full_frames),
        "temporal_distribution": compute_temporal_distribution(summary_frames, full_frames),
        "diversity": compute_diversity(summary_frames),
        "representativeness": compute_representativeness(summary_frames, full_frames),
        "information_loss": compute_information_loss(summary_frames, full_frames),
    }


def evaluate_all_patients(summary_root, full_root, output_csv_path=None):
    results = defaultdict(dict)
    start_all = time.time()

    # If writing to CSV, open it here
    csv_writer = None
    csv_data = pd.DataFrame(columns=["Algorithm", "Video", "Split", "Coverage", "TemporalDistribution", 
                                     "Diversity", "Representativeness", "InformationLoss"])

    if output_csv_path:
        file_exists = os.path.exists(output_csv_path)

        if file_exists:
            csv_data = pd.read_csv(output_csv_path)
            csv_file = open(output_csv_path, mode='a', newline='')  # Append mode
            csv_writer = csv.writer(csv_file)
        else:
            csv_file = open(output_csv_path, mode='w', newline='')  # New file
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Algorithm", "Video", "Split", "Coverage", "TemporalDistribution", 
                                 "Diversity", "Representativeness", "InformationLoss"])
            csv_file.flush()

    for patient in os.listdir(summary_root):
        if not patient.startswith('.'):
            print(f"\n=== Processing patient: {patient} ===")
            patient_summary_path = os.path.join(summary_root, patient)
            patient_full_path = os.path.join(full_root, patient)

            if not os.path.isdir(patient_summary_path) or not os.path.isdir(patient_full_path):
                print("  - Invalid directory. Skipping...")
                continue

            for video in os.listdir(patient_summary_path):
                if not video.startswith('.'):

                    existing_entries = set(zip(csv_data['Algorithm'], csv_data['Video'], csv_data['Split'].astype(str)))
                    if (algorithm, video, split) in existing_entries:
                        print(f"--- Skipping {video} with {algorithm} split {split} (already in CSV)")
                        continue

                    print(f"\n--- Processing video: {video} ---")
                    summary_video_path = os.path.join(patient_summary_path, video)
                    full_video_path = os.path.join(patient_full_path, video)

                    start_time = time.time()
                    scores = evaluate_summary(summary_video_path, full_video_path)
                    results[patient][video] = scores
                    print(f"--- Finished {video} in {time.time() - start_time:.2f}s")

                    # Save result to CSV immediately
                    if csv_writer:
                        csv_writer.writerow([
                            algorithm,
                            video,
                            split,
                            scores["coverage"],
                            scores["temporal_distribution"],
                            scores["diversity"],
                            scores["representativeness"],
                            scores["information_loss"]
                        ])
                        csv_file.flush()

    if csv_writer:
        csv_file.close()


    print(f"\n=== All evaluations completed in {time.time() - start_all:.2f}s ===")
    return results



# Run
algorithms = ['pytorch-CTVSUM', 'CA-SUM', 'pytorch-vsumm-reinforce']
for split in ['0', '1', '2', '3', '4']:
    for algorithm in algorithms:
        frames_path = '../../01_Data/SCI_HOME_Frames'
        summaries_path = '../../01_Data/SCI_HOME_Summaries/' + algorithm + '/split_' + split
        output_csv_path = '../../01_Data/Results/evaluation.csv'

        results = evaluate_all_patients(summaries_path, frames_path, output_csv_path)