# Quantification of lower limb kinematics during swimming in individuals with spinal cord injury

🎯 Objectives: The aim of this project is to apply unsupervised video summarization algorithms to egocentric, home-based videos recorded by individuals with spinal cord injury (SCI) and evaluate their performance.

## Summary 

🔍 Authors: Melina Giagiozis, José Zariffa

📝 Summary: Wearable cameras provide a means to assess hand function in individuals with spinal cord injury (SCI) beyond clinical settings. Previous studies have found that clinicians acknowledge the potential of egocentric video for monitoring and informing rehabilitation, however, the large volumes of video data poses a challenge for its efficient integration into clinical practice. To address this, we implemented and evaluated three video summarization algorithms.

🗝️ Keywords: egocentric video, home monitoring, video summarization, spinal cord injury, upper limb rehabilitation, wearable technology

## Getting Started

All code in this repository is written in Python 3.8.20 and the dependencies listed in `requirements.txt`. The file paths in this repository use absolute paths to the UHN M: drive.

Path to this repo on the drive: 'NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code'.

📁 Script Overview

config.py: Defines absolute paths to video data, extracted frames, features, and summaries.
functions.py: Contains preprocessing utilities for frame extraction and GoogLeNet-based feature computation.
adjust_video_speed.py: Creates sped-up full-length videos from extracted frames for human evaluation.
frames_to_video.py: Converts frame sequences into .mp4 video summaries produced by different summarization algorithms.
evaluation.py: Evaluates summary quality using coverage, temporal distribution, diversity, representativeness, and information loss.
data_analysis.py: Performs statistical comparisons of summarization algorithms across cross-validation splits.
create_figure.py: Generates visualizations of video summaries for use in reports or publications.

## Contact 

📧 For comments or questions related to this repository or the manuscript contact [Melina Giagiozis](Melina.Giagiozis@balgrist.ch).
