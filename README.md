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

```bash
python ./backend/main.py


## New Subtitle Translation Feature


In addition to subtitle removal, this project supports **automated subtitle translation** using the **Youdao Translation API**. The workflow includes:__

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
1. **Slow Processing Speed**: Adjust the configuration in backend/config.py to use the STTN algorithm for faster processing:__
```
MODE = InpaintMode.STTN
STTN_SKIP_DETECTION = True
```
2. Unsatisfactory Removal Results:__
* Use different algorithms like **LAMA or PROPAINTER** in the config.py file.
* Adjust STTN_NEIGHBOR_STRIDE and STTN_REFERENCE_LENGTH for better results.

3. CondaHTTPError:Place the .condarc file from the project in the user directory (e.g., 
```
C:/Users/<your_username>).
```


