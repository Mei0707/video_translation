## Project Overview

This project is based on the [Video Subtitle Remover](https://github.com/YaoFANGUK/video-subtitle-remover) developed by YaoFANGUK. It expands upon the original work by integrating additional features such as automated subtitle translation and further improvements in AI-based subtitle removal techniques.

The original project laid the groundwork for efficiently removing hardcoded subtitles, and this version builds on that with enhancements like support for various algorithms (**STTN, LAMA, PROPAINTER**) and integration with **ComfyUI** for improved usability.

# Video-Subtitle-Remover (VSR)

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![Python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Support OS](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

Video-Subtitle-Remover (VSR) is an AI-based software that removes hardcoded subtitles from videos. It implements the following functionalities:

- **Lossless resolution**: Removes hardcoded subtitles from videos and generates files without subtitles.
- Fills the removed subtitle area using a powerful AI algorithm (non-adjacent pixel filling and mosaic removal).
- Supports custom subtitle positions by removing subtitles in a defined location (input position).
- Supports automatic removal of all text throughout the video (without inputting a position).
- Supports multi-selection of images for batch removal of watermark text.

## Key Features

- **Subtitle Translation Workflow**: This project integrates automated subtitle translation. By adding a custom node to integrate this translation into ComfyUI, it allows users to translate subtitles to different languages based on input.
- **Subtitle Detection and Removal**: Advanced subtitle detection algorithms like STTN, LAMA, and PROPAINTER are supported to remove subtitles from different types of videos.
- **Custom Node for ComfyUI**: This project integrates with ComfyUI for enhanced control over subtitle translation.

## Demonstration

- **GUI Example**:

<p style="text-align:center;">
    <img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="GUI Example"/>
</p>

- **CLI Example**:

```
python ./backend/main.py
```


## New Subtitle Translation Feature


In addition to subtitle removal, this project supports **automated subtitle translation** using the **Youdao Translation API**. The workflow includes:

1. **Input**: Video URL and target language.__
2. **Subtitle** Detection: Extract hardcoded subtitles.__
3. **Translation**: Translates extracted subtitles to the desired language.__
4. **Output**: A video with translated subtitles.

### Setup for Translation


1. Add your Youdao API credentials in config.py:__

```
YOUDAO_API_ID = "your_api_id"
YOUDAO_API_KEY = "your_api_key"
```

2. Install Youdao SDK:__

```
pip install youdao-python-sdk
```

3. Use the translation feature via CLI:__

```
python ./backend/subtitleTranslation.py --video_url <video_url> --target_language <language_code>
```

## Integration with ComfyUI
The subtitle translation workflow is integrated into **ComfyUI**. Follow these steps:

1. Install **ComfyUI** (refer to the ComfyUI documentation).
2. Access the custom node for subtitle translation.
3. Upload the video, select the target language, and click "Translate."


## Common Issues
1. **Slow Processing Speed**: Adjust the configuration in backend/config.py to use the **STTN** algorithm for faster processing:__
```
MODE = InpaintMode.STTN
STTN_SKIP_DETECTION = True
```
2. **Unsatisfactory Removal Results**:
* Use different algorithms like **LAMA or PROPAINTER** in the `config.py` file.
* Adjust `STTN_NEIGHBOR_STRIDE` and `STTN_REFERENCE_LENGTH` for better results.

3. **CondaHTTPError**:Place the `.condarc` file from the project in the user directory (e.g., `C:/Users/<your_username>`).

4. **7z File Extraction Error**: Upgrade the 7-zip extraction program.
```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```


## Installation Guide
### 1. Download and Install Miniconda
* **Windows**: Miniconda3-py38_4.11.0-Windows-x86_64.exe
* **Linux**: Miniconda3-py38_4.11.0-Linux-x86_64.sh
### 2. Create and Activate a Virtual Environment
Switch to the source code directory:
```
cd <source_code_directory>
```
For example, if your source code is in the `tools` folder on drive D, and the source code folder is `video-subtitle-remover`:
```
cd D:/tools/video-subtitle-remover-main
```
Create and activate the Conda environment:
```
conda create -n videoEnv python=3.8
conda activate videoEnv
```

### 3. Install Dependencies
Ensure Python 3.8+ is installed, and use conda to create and activate the environment.

* **Install CUDA** and **cuDNN**:
**Linux Installatio** 
1.Download CUDA 11.7:
```
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
```
2. Install CUDA 11.7:
```
sudo sh cuda_11.7.0_515.43.04_linux.run
```
Add the following content to ~/.bashrc:
```
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
3. Install cuDNN
```
tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
sudo cp ./cuda/include/* /usr/local/cuda-11.7/include/
sudo cp ./cuda/lib/* /usr/local/cuda-11.7/lib64/
sudo chmod a+r /usr/local/cuda-11.7/lib64/*
sudo chmod a+r /usr/local/cuda-11.7/include/*
```

**Windows Installation**
1. Download CUDA 11.7:

Download CUDA

2. Install cuDNN:

* Unzip `cudnn-windows-x64-v8.2.4.15.zip`, then move all files in "bin, include, lib" to:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\
```

* **Install PaddlePaddle (GPU)**:
```
# Windows
python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Linux
python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

* **Install Pytorch (GPU)**:
```
# Conda installation
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# pip installation
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

* **Install Other Dependencies**:
```
pip install -r requirements.txt
```

4. Run the Program
* **Run GUI**:
```
python gui.py
```
* **Run CLI**:
```
python ./backend/main.py
```