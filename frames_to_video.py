import os
import cv2
from glob import glob

# # Base path to the summaries
base_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Summaries"
# base_path = '/Users/melinagiagiozis/Desktop'

# Two algorithm folders
algorithms = ["CA-SUM", "pytorch-CTVSUM", "pytorch-vsumm-reinforce"]

# Frames to repeat per image to slow down video
# frame_repeat = 7
frame_repeat = 5

# Video settings
fps = 5
frame_size = None

for algo in algorithms:
    algo_path = os.path.join(base_path, algo)
    split_path = os.path.join(algo_path, "split_0")  # Only use split_0
    
    # Iterate over patient folders
    for patient in sorted(os.listdir(split_path)):
        patient_path = os.path.join(split_path, patient)
        if not os.path.isdir(patient_path):
            continue

        # Iterate over video summary folders
        for video_folder in sorted(os.listdir(patient_path)):
            video_path = os.path.join(patient_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            # Output video path
            output_video = os.path.join(video_path, f"{video_folder}.mp4")
            if os.path.exists(output_video):
                print(f"Video already exists: {output_video}")
                continue

            # Gather and sort frame paths
            frame_paths = sorted(glob(os.path.join(video_path, "frame_*.jpg")))
            if not frame_paths:
                print(f"No frames found in {video_path}")
                continue

            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                print(f"Could not read frame in {video_path}")
                continue

            height, width, _ = first_frame.shape
            frame_size = (width, height)

            # Define the video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

            print(f"Creating video: {output_video}")
            for frame_file in frame_paths:
                frame = cv2.imread(frame_file)
                if frame is None:
                    print(f"Warning: could not read {frame_file}")
                    continue
                for _ in range(frame_repeat):
                    out.write(frame)

            out.release()
            print(f"Saved: {output_video}")
