import cv2
import os
from glob import glob

# Mapping of video sources to destination folders
video_map = {
    "SCI02/SCI02_01": "Video_1",
    "SCI03/SCI03_06": "Video_2",
    "SCI08/SCI08_32": "Video_3",
    "SCI17/SCI17_10": "Video_4",
    "SCI21/SCI21_09": "Video_5",
}

# Base directory for frames
base_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/01_Data/SCI_HOME_Frames"
evaluation_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/05_HumanEvaluation"

# Settings
frame_rate = 1       # 1 FPS input
speed_factor = 3     # speed-up
output_fps = frame_rate * speed_factor

for sub_path, video_folder in video_map.items():
    frame_dir = os.path.join(base_path, sub_path)
    output_dir = os.path.join(evaluation_path, video_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Get all frame paths in order
    frame_paths = sorted(glob(os.path.join(frame_dir, 'frame_*.jpg')))

    if not frame_paths:
        print(f"No frames found in {frame_dir}")
        continue

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    output_path = os.path.join(output_dir, f"Full_Video_SpedUp.MP4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Saved: {output_path}")

# # Input and output paths
# input_path = "/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/05_HumanEvaluation/Video_02/Video_02_Full.mp4"
# output_path = input_path.replace(".mp4", "_Rotated180.mp4")

# # Open video
# cap = cv2.VideoCapture(input_path)
# if not cap.isOpened():
#     raise IOError(f"Cannot open video: {input_path}")

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # Define the output writer
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Flip 180°: flip vertically and horizontally
#     flipped = cv2.flip(frame, -1)
#     out.write(flipped)

# # Cleanup
# cap.release()
# out.release()
# print(f"Saved rotated video to: {output_path}")


# import os
# from moviepy.editor import VideoFileClip, vfx

# # Path to the folder containing the videos
# folder_path = '/Volumes/NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/05_HumanEvaluation/00_Admin/Full Videos'

# # Speed factor
# speed_factor = 2.0

# # Process each video in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.MP4') and not filename.endswith('_fast.mp4'):
#         file_path = os.path.join(folder_path, filename)
#         output_path = os.path.join(folder_path, filename.replace('.MP4', '_fast.mp4'))

#         # Load and speed up video (without audio)
#         clip = VideoFileClip(file_path).without_audio()
#         fast_clip = clip.fx(vfx.speedx, factor=speed_factor)

#         # Write sped-up video (no audio)
#         fast_clip.write_videofile(output_path, codec='libx264', audio=False)

#         # Free resources
#         clip.close()
#         fast_clip.close()

