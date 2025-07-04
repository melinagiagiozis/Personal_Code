"""
Script: main.py

Description:
    This script provides a complete pipeline for preprocessing 
    egocentric video data for summarization tasks. It supports 
    multiple steps, including:

    - Frame extraction from videos (1 FPS)
    - Feature extraction using GoogLeNet (Pool5 layer)
    - Aggregation of features into HDF5 or NumPy format
    - Creation of training/test splits
    - Assembly of training/testing datasets per split

Note: High-level visual features are computed using the GoogleNet model 
pre-trained on the ImageNet dataset [29], which is a a common approach 
in video summarization tasks.

CA-SUM and CTVSUM require HDF5 format, while DR-DSN uses both HDF5 
and NumPy formats.

"""


import os
import time
import h5py
import numpy as np
from functions import *
from config import *
import shutil
import json
import random

# Boolean flags to control which steps to run.
run_extract_frames = False
run_extract_features = False
run_combine_features = False
assemble_selected_features = False  # for training
create_splits = False  # test/train splits
set_up_test_features = False  # combine features for test split
set_up_train_features = False  # combine features for train split


# Loop through each patient folder
for patient in sorted(os.listdir(data_path)):
    patient_path = os.path.join(data_path, patient)
    
    # Check if it's a directory
    if os.path.isdir(patient_path):
    
        # Create corresponding patient folder in frames_path
        patient_frames_path = os.path.join(frames_path, patient)
        os.makedirs(patient_frames_path, exist_ok=True)

        # Create corresponding patient folder in features_path
        patient_features_path = os.path.join(features_path, patient)
        os.makedirs(patient_features_path, exist_ok=True)

        # Loop through each video file in the patient's folder
        for video in sorted(os.listdir(patient_path)):
            video_path = os.path.join(patient_path, video)

            # Ensure it's a file and has a video extension
            if os.path.isfile(video_path) and video.lower().endswith((".mp4")) and not video.startswith((".")):

                # Create a folder for the video inside the patient’s folder in frames_path
                video_frames_path = os.path.join(patient_frames_path, os.path.splitext(video)[0])
                os.makedirs(video_frames_path, exist_ok=True)

                # Create a folder for the video inside the patient’s folder in features_path
                video_features_path = os.path.join(patient_features_path, os.path.splitext(video)[0])
                os.makedirs(video_features_path, exist_ok=True)

                # Extract frames
                if run_extract_frames:
                    if not os.listdir(video_frames_path):
                        start_time = time.time()
                        extract_frames(video_path, video_frames_path, frame_rate=1)
                        end_time = time.time()
                        elapsed_time = (end_time - start_time)/60
                        print(f"Frame extraction completed in {elapsed_time:.2f} minutes.")
                    else:
                        print(f"Frames already extracted from {video}.")

                # Extract GoogleNet features
                if run_extract_features:

                    video_name = os.path.splitext(video)[0]  # Remove file extension
                    feature_h5_file = os.path.join(video_features_path, f"{video_name}.h5")
                    feature_npy_file = os.path.join(video_features_path, "selected_features")

                    if not os.listdir(video_features_path):
                        start_time = time.time()
                        features = extract_features(video_frames_path)
                        end_time = time.time()
                        elapsed_time = (end_time - start_time)/60
                        print(f"Feature extraction completed in {elapsed_time:.2f} minutes.")

                        # Save extracted features
                        save_to_h5(features, feature_h5_file, video_name)
                        save_to_npy(features, feature_npy_file, f"{video_name}.npy")
                    else:
                        print(f"Features already extracted from {video}.")

                if run_extract_features or run_extract_frames:
                    print(f"Extracted frames and features from {video}.")

                    print('------------------------------------------------')

if run_extract_features or run_extract_frames:
    print("Frame and feature extraction completed.")



### === Combine All Features into a Single HDF5 File === ###
if run_combine_features:
    output = '../../01_Data/egocentric_googlenet_all.h5'

    # Collect all feature files across patient folders
    feature_files = []
    for patient in sorted(os.listdir(features_path)):
        patient_features_path = os.path.join(features_path, patient)
        patient_frames_path = os.path.join(frames_path, patient)
        if os.path.isdir(patient_features_path):
            for video_folder in sorted(os.listdir(patient_features_path)):
                video_features_path = os.path.join(patient_features_path, video_folder)
                video_frames_path = os.path.join(patient_frames_path, video_folder)
                if os.path.isdir(video_features_path) and os.path.isdir(video_frames_path):
                    # Check if the number of frames is at least 60 (1 minute)
                    num_frames = len([f for f in os.listdir(video_frames_path)
                                      if f.endswith(('.jpg', '.png')) and not f.startswith('.')])
                    if num_frames >= 60:
                        for feature_file in os.listdir(video_features_path):
                            if feature_file.endswith(".h5") and not feature_file.startswith("."):
                                feature_files.append(os.path.join(video_features_path, feature_file))

    print(f"Found {len(feature_files)} feature files for merging.")

    # Merge features into a single HDF5 file
    with h5py.File(output, "w") as h5f:
        for idx, feature_file in enumerate(feature_files):
            video_name = os.path.splitext(os.path.basename(feature_file))[0]  # Extract video name without extension

            # Only include videos starting with SCI (not Pilot videos)
            if not video_name.startswith("SCI"):
                continue

            # Check if the feature file exists
            if os.path.exists(feature_file):
                video_key = video_name  # Create a group for this video

                with h5py.File(feature_file, "r") as pool5_h5:
                    subgroup_name = list(pool5_h5.keys())[0]  # Extract subgroup (e.g., "video1")
                    video_group = pool5_h5[subgroup_name]
                    
                    # Check if "features" dataset has at least 60 frames (1 minute)
                    if "features" in video_group:
                        video_key = video_name
                        grp = h5f.create_group(video_key)

                        for dataset_name in video_group.keys():
                            grp.create_dataset(dataset_name, data=video_group[dataset_name][...])

                        # Include the video name
                        grp.create_dataset("video_name", data=np.string_(video_name))
                    else:
                        print(f"Skipping {video_name} — fewer than 60 frames.")

    print(f"Merged Pool5 HDF5 saved: {output}")



### === Copy .npy files for training into a single selected_features folder === ###

if assemble_selected_features:

    # Destination folder for all .npy files
    destination_folder = os.path.join(features_path, "selected_features_all")
    os.makedirs(destination_folder, exist_ok=True)

    # Walk through patient and video folders
    for patient in sorted(os.listdir(features_path)):
        patient_path = os.path.join(features_path, patient)
        if not os.path.isdir(patient_path) or patient.startswith("."):
            continue

        for video_folder in sorted(os.listdir(patient_path)):

            # Only include videos starting with SCI (not Pilot videos)
            if not video_folder.startswith("SCI"):
                continue

            selected_path = os.path.join(patient_path, video_folder, "selected_features")
            if os.path.isdir(selected_path):
                for file in os.listdir(selected_path):
                    if file.endswith(".npy"):
                        full_path = os.path.join(selected_path, file)
                        destination_path = os.path.join(destination_folder, file)
                        
                        # Handle possible name collisions
                        if os.path.exists(destination_path):
                            base, ext = os.path.splitext(file)
                            destination_path = os.path.join(destination_folder, f"{base}{ext}")

                        shutil.copy(full_path, destination_path)

    print(f"All .npy files copied to: {destination_folder}")
        

# Five 80/20 splits of the egocentric data
if create_splits:

    # Collect valid video names with more than 60 frames
    valid_videos = []

    for patient_folder in os.listdir(frames_path):
        patient_path = os.path.join(frames_path, patient_folder)
        if os.path.isdir(patient_path):
            for video_folder in os.listdir(patient_path):
                video_path = os.path.join(patient_path, video_folder)
                if os.path.isdir(video_path):
                    frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                    if len(frame_files) > 60 and video_folder.startswith("SCI"):
                        valid_videos.append(video_folder)

    valid_videos.sort()  # Optional, for reproducibility
    random.seed(42)      # Ensures consistent results across runs
    random.shuffle(valid_videos)

    # Create 5 train-test splits (80-20)
    num_splits = 5
    num_test = len(valid_videos) // num_splits
    splits = []

    for i in range(num_splits):
        test_keys = valid_videos[i * num_test:(i + 1) * num_test]
        train_keys = [v for v in valid_videos if v not in test_keys]
        splits.append({
            "test_keys": test_keys,
            "train_keys": train_keys
        })

    # Save to JSON
    output_path = os.path.join("../../01_Data/egocentric_splits.json")
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Saved splits to {output_path}")

for split in [0, 1, 2, 3, 4]:
    if set_up_test_features:
        print('------ Split ', str(split), ' ------')
        ### === Combine test features into a single HDF5 File === ###

        output = '../../01_Data/egocentric_googlenet_test_split_' + str(split) + '.h5'

        if os.path.exists(output):
            print(f"Split {split} already exists at {output}. Skipping.")
            continue

        # Load first split from existing splits JSON
        splits_path = os.path.join("../../01_Data/egocentric_splits.json")
        with open(splits_path, "r") as f:
            splits = json.load(f)
        
        split_0_keys = splits[split]['test_keys']
        selected_videos = set(split_0_keys)

        # Collect all feature files for selected videos
        feature_files = []
        for patient in sorted(os.listdir(features_path)):
            patient_features_path = os.path.join(features_path, patient)
            patient_frames_path = os.path.join(frames_path, patient)
            if os.path.isdir(patient_features_path):
                for video_folder in sorted(os.listdir(patient_features_path)):
                    if video_folder not in selected_videos:
                        continue  # Skip videos not in first split

                    video_features_path = os.path.join(patient_features_path, video_folder)
                    video_frames_path = os.path.join(patient_frames_path, video_folder)

                    if os.path.isdir(video_features_path) and os.path.isdir(video_frames_path):
                        # Check if the number of frames is at least 60
                        num_frames = len([
                            f for f in os.listdir(video_frames_path)
                            if f.endswith(('.jpg', '.png')) and not f.startswith('.')
                        ])
                        if num_frames >= 60:
                            for feature_file in os.listdir(video_features_path):
                                if feature_file.endswith(".h5") and not feature_file.startswith("."):
                                    feature_files.append(os.path.join(video_features_path, feature_file))

        print(f"Found {len(feature_files)} feature files for merging.")

        # Merge features into a single HDF5 file
        with h5py.File(output, "w") as h5f:
            for idx, feature_file in enumerate(feature_files):
                video_name = os.path.splitext(os.path.basename(feature_file))[0]

                if os.path.exists(feature_file):
                    with h5py.File(feature_file, "r") as pool5_h5:
                        subgroup_name = list(pool5_h5.keys())[0]
                        video_group = pool5_h5[subgroup_name]

                        # Only if "features" exist and has >= 60 frames
                        if "features" in video_group and len(video_group["features"]) >= 60:
                            grp = h5f.create_group(video_name)
                            for dataset_name in video_group.keys():
                                grp.create_dataset(dataset_name, data=video_group[dataset_name][...])
                            grp.create_dataset("video_name", data=np.string_(video_name))
                        else:
                            print(f"Skipping {video_name} — 'features' missing or < 60 frames.")

        print(f"Merged Pool5 HDF5 saved: {output}")


    if set_up_train_features:
        print('------ Split ', str(split), ' ------')
        # Destination folder for all .npy files
        destination_folder = os.path.join(features_path, "selected_features_train_" + str(split))
        os.makedirs(destination_folder, exist_ok=True)

        # Load first split from existing splits JSON
        splits_path = os.path.join("../../01_Data/egocentric_splits.json")
        with open(splits_path, "r") as f:
            splits = json.load(f)
        
        split_0_keys = splits[split]['train_keys']
        selected_videos = set(split_0_keys)

        # Walk through patient and video folders
        for patient in sorted(os.listdir(features_path)):
            patient_path = os.path.join(features_path, patient)
            if not patient.startswith("."):
                for video_folder in sorted(os.listdir(patient_path)):
                    if video_folder not in selected_videos:
                        continue
                    selected_path = os.path.join(patient_path, video_folder, "selected_features")
                    if os.path.isdir(selected_path):
                        for file in os.listdir(selected_path):
                            if file.endswith(".npy"):
                                full_path = os.path.join(selected_path, file)
                                destination_path = os.path.join(destination_folder, file)
                                
                                # Handle possible name collisions
                                if os.path.exists(destination_path):
                                    base, ext = os.path.splitext(file)
                                    destination_path = os.path.join(destination_folder, f"{base}{ext}")

                                shutil.copy(full_path, destination_path)

        print(f"All .npy files copied to: {destination_folder}")