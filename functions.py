"""
Script: functions.py

Description:
    This script contains utility functions for preprocessing videos 
    and extracting frame-level features.
    - Extract frames from videos at a specified frame rate.
    - Extract deep features from frames using the GoogLeNet model (Pool5 layer).
    - Save extracted features in HDF5 or in NumPy format (depending on the algorithm used).

"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import glob
import cv2
import os
import h5py
import imageio.v3 as iio
import imageio

# # For Mac
# def extract_frames(video_path, output_folder, frame_rate):
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)

#     fps = int(cap.get(cv2.CAP_PROP_FPS))  
#     frame_interval = fps // frame_rate  # one frame per second if frame_rate = 1
#     # frame_interval = fps * frame_rate # one frame every 5 seconds

#     count = 0
#     frame_id = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % frame_interval == 0:
#             frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
#             cv2.imwrite(frame_path, frame)
#             frame_id += 1
#         count += 1
#     cap.release()


# For Linux
def extract_frames(video_path, output_folder, frame_rate):
    os.makedirs(output_folder, exist_ok=True)

    # Get FPS of the video
    meta = iio.immeta(video_path, plugin="pyav")
    fps = meta.get("fps")
    frame_interval = fps // frame_rate

    # Read video frames
    frames = iio.imiter(video_path, plugin="pyav")

    frame_id = 0
    for i, frame in enumerate(frames):
        if i % frame_interval == 0:  # Extract frame at correct interval
            frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
            imageio.imwrite(frame_path, frame)
            frame_id += 1


def extract_features(image_folder):
    """ Extracts Google Pool5 features from GoogLeNet. """

    # Transformation for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load GoogLeNet model and remove fully connected layers to get Pool5 features
    googlenet_model = models.googlenet(pretrained=True)
    googlenet_model = torch.nn.Sequential(*(list(googlenet_model.children())[:-2]))  # Keep up to Pool5 layer
    googlenet_model.eval()

    features = []
    image_paths = sorted(glob.glob(image_folder + "/*.jpg"))

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0)  
            feature_vector = googlenet_model(img).squeeze().numpy()  
            features.append(feature_vector)

    return np.array(features)

# Save features for testing (i.e. evaluation)
def save_to_h5(features, output_h5, video_name):
    with h5py.File(output_h5, "w") as f:
        f.create_dataset(video_name + "/features", data=features)
        f.create_dataset(video_name + "/gtscore", data=np.zeros(features.shape[0]))  # Placeholder
        f.create_dataset(video_name + "/user_summary", data=np.zeros((1, features.shape[0])))  # Placeholder
        f.create_dataset(video_name + "/change_points", data=np.array([[0, features.shape[0]-1]]))  # Single segment
        f.create_dataset(video_name + "/n_frame_per_seg", data=np.array([features.shape[0]]))
        f.create_dataset(video_name + "/n_frames", data=features.shape[0])
        f.create_dataset(video_name + "/picks", data=np.arange(features.shape[0]))
        f.create_dataset(video_name + "/n_steps", data=np.array([features.shape[0]]))
        f.create_dataset(video_name + "/gtsummary", data=np.zeros(features.shape[0]))  # Placeholder

# Save features for training
def save_to_npy(features, output_npy, video_name):
    
    os.makedirs(output_npy, exist_ok=True)  # Ensure output directory exists
    file_path = os.path.join(output_npy, video_name)
    np.save(file_path, features)
