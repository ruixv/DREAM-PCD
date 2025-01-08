
# DREAM-PCD: Deep Reconstruction and Enhancement of mmWave Radar Pointcloud

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://img.shields.io/badge/Project%20Status-WIP-yellow)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of *DREAM-PCD: Deep Reconstruction and Enhancement of  mmWave Radar Pointcloud*


## Project Status: Under Construction 🚧


**Note**: This repository is currently under active development. The code and complete documentation will be released soon.

---

## Dataset

- Our RadarEyes Dataset
> https://github.com/ruixv/RadarEyes


> Put the dataset into `Datasets`
> e.g., `./Datasets/2023_06_04_22_48_54_A409_hall_4_50s`

## Installation

### Step 1: Install IPLab_mmwavePCD

> IPLab_mmwavePCD is a comprehensive Python library for mmWave radar point cloud processing that provides:
> - **Point Cloud Reconstruction**: 
>   - Converts raw ADC data to point clouds
>   - Implements TI's official MATLAB algorithms in Python
>   - Supports various radar configurations
> - **Key Features**:
>   - Point cloud visualization tools
>   - Coordinate transformation utilities
>   - Data fusion capabilities
>   - Real-time processing support
> This library can be used in two ways:
> 1. As a prerequisite for DREAM-PCD project
> 2. As a standalone tool for general mmWave radar processing
> **Primary Support**: RadarEyes Dataset. Recommended to download RadarEyes first, also can be developed to support other dataset (e.g., [Coloradar](https://arpg.github.io/coloradar/) ).

#### 1.1 Environment Setup
```bash
# Create and activate conda environment
conda create -n IPLabmmWave python=3.9
conda activate IPLabmmWave

cd IPLab_mmwavePCD

# Install PyTorch (example for CUDA 11.6)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

# Install this package
pip install -e .
```

#### 1.2 Point Cloud Visualization Demo
```python
# Visualize fused mmWave radar point clouds
python ./FuseData/new_azi_radar_fuse.py

# Visualize fused LiDAR point clouds 
python ./FuseData/new_lidar_fuse.py
```

### 1.3 Generate Point Cloud from Raw ADC Data
In `new_azi_radar_fuse.py`, uncomment the section "Example: How to load ADC files and generate point cloud":

```python
"""
Example: How to load ADC files and generate point cloud
Note: This process may take some time
You can customize the point cloud generation parameters
"""
```

> Note: Change the scene name in the code to visualize different scenarios


### Step 

```python
# comming soon
```


## Citation

```bibtex
@article{gengDREAMPCDDeepReconstruction2024,
  title = {{{DREAM-PCD}}: {{Deep Reconstruction}} and {{Enhancement}} of {{mmWave Radar Pointcloud}}},
  author = {Geng, Ruixu and Li, Yadong and Zhang, Dongheng and Wu, Jincheng and Gao, Yating and Hu, Yang and Chen, Yan},
  year = {2024},
  volumes = {},
  number = {},
  pages = {},
  journal = {IEEE Transactions on Image Processing}
}
```
