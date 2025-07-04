# Quantification of lower limb kinematics during swimming in individuals with spinal cord injury

🎯 Objectives: The aim of this project is to apply unsupervised video summarization algorithms to egocentric, home-based videos recorded by individuals with spinal cord injury (SCI) and evaluate their performance.

## Summary 

🔍 Authors: Melina Giagiozis, José Zariffa

📝 Summary: Wearable cameras provide a means to assess hand function in individuals with spinal cord injury (SCI) beyond clinical settings. Previous studies have found that clinicians acknowledge the potential of egocentric video for monitoring and informing rehabilitation, however, the large volumes of video data poses a challenge for its efficient integration into clinical practice. To address this, we implemented and evaluated three video summarization algorithms. A dataset comprising 316 egocentric videos from 20 individuals with cervical SCI was used. Participants wore head-mounted cameras to record daily activities in their home environment. Three unsupervised video summarization algorithms were applied to the data (CTVSUM [1](#1), CA-SUM [2](#2), and DR-DSN [3](#3)). Using 5-fold cross-validation, the summaries were evaluated based on computational metrics. In addition, a human evaluation was conducted to determine whether the summaries adequately captured upper limb use in the home environment following SCI.

🗝️ Keywords: egocentric video, home monitoring, video summarization, spinal cord injury, upper limb rehabilitation, wearable technology

References:

<a id="1">[1]</a> Pang, Z., Nakashima, Y., Otani, M., & Nagahara, H. (2023). Contrastive losses are natural criteria for unsupervised video summarization. 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). https://doi.org/10.1109/wacv56688.2023.00205
<br>

<a id="2">[2]</a> Apostolidis, E., Balaouras, G., Mezaris, V., & Patras, I. (2022). Summarizing videos using concentrated attention and considering the uniqueness and diversity of the video frames. Proceedings of the 2022 International Conference on Multimedia Retrieval (ICMR). https://doi.org/10.1145/3512527.3531404
<br>

<a id="3">[3]</a> Zhou, K., Qiao, Y., & Xiang, T. (2018). Deep reinforcement learning for unsupervised video summarization with diversity-representativeness reward. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). https://doi.org/10.1609/aaai.v32i1.12255


## Getting Started

All code in this repository is written in Python 3.8.20 and the dependencies listed in `requirements.txt`. The file paths in this repository use absolute paths to the UHN M: drive.

Path to this repo on the drive: 'NET/Wearable Hand Monitoring/Centralized_Datasets/SUMMARIZATION/02_Code/Personal_Code'.


📁 Script Overview
- config.py: Defines paths to video data, extracted frames, features, and summaries.
- functions.py: Contains preprocessing utilities for frame extraction and GoogLeNet-based feature computation.
- adjust_video_speed.py: Creates sped-up full-length videos from extracted frames for human evaluation.
- frames_to_video.py: Converts frame sequences into .mp4 video summaries produced by different summarization algorithms.
- evaluation.py: Evaluates summary quality using coverage, temporal distribution, diversity, representativeness, and information loss.
- data_analysis.py: Performs statistical comparisons of summarization algorithms across cross-validation splits.
- create_figure.py: Generates visualizations of video summaries for use in reports or publications.

📂 Summarization Algorithm Outputs

The following folders contain scripts to generate summaries with each algorithm for all videos in the dataset. Furthermore, they contain extracted frame-level importance scores and other model-specific outputs, each summarization algorithm trained for 40 epochs on egocentric video data:

- 01_pytorch-ctvsum_40epochs_egocentric: CTVSUM
- 02_ca-sum_40epochs_egocentric: CA-SUM
- 03_pytorch-vsumm-reinforce_40epochs_egocentric: DR-DSN


## Contact 

📧 For comments or questions related to this repository or the manuscript contact [Melina Giagiozis](Melina.Giagiozis@balgrist.ch).
