"""
Script: config.py

Description:
    This script defines the absolute paths for data storage.
    
    - data_path: full videos
    - frames_path: extracted frames for each video (1 FPS)
    - features_path: extracted features for each video
    - summaries_path: generated summaries for each video

"""

import os
import numpy as np

# data_path = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Data'

# frames_path = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Frames'

# features_path = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Features'

# summaries_path = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries'

data_path = '../../01_Data/SCI_HOME_Data'

frames_path = '../../01_Data/SCI_HOME_Frames'

features_path = '../../01_Data/SCI_HOME_Features'

summaries_path = '../../01_Data/SCI_Home_Summaries'


# # Check number of frames per video per patient
# for patient_id in sorted(os.listdir(frames_path)):
#     patient_path = os.path.join(frames_path, patient_id)
#     if not os.path.isdir(patient_path):
#         continue

#     frame_counts = []
#     for video_id in os.listdir(patient_path):
#         video_path = os.path.join(patient_path, video_id)
#         if not os.path.isdir(video_path):
#             continue
#         num_frames = len([f for f in os.listdir(video_path) if f.endswith('.jpg')])
#         frame_counts.append(num_frames)

#    # Filter to only include videos with at least 60 frames (1 minute)
#     filtered_frame_counts = [count for count in frame_counts if count >= 60]

#     if filtered_frame_counts:
#         mean_frames = np.mean(filtered_frame_counts)
#         std_frames = np.std(filtered_frame_counts)
#         num_videos = len(filtered_frame_counts)
#         print(f"{patient_id}: {num_videos} videos, {mean_frames / 60:.2f} ({std_frames / 60:.2f}) min")
#     else:
#         print(f"{patient_id}: No videos with >= 60 frames.")
