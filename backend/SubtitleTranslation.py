import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.tools.common_tools import is_video_or_image, is_image_file, is_video_file
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.video_inpaint import VideoInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator
import importlib
import platform
import tempfile
import torch
import multiprocessing
from shapely.geometry import Polygon
import time
from tqdm import tqdm
from tools.infer import utility
from tools.infer.predict_det import TextDetector
import llm.LLMAPI as LLMAPI
import re
from collections import Counter

from main import *

import requests
import hashlib
import random
import json

YOUDAO_API_KEY = 'YOUDAO_API_KEY'
YOUDAO_APP_SECRET = 'YOUDAO_APP_SECRET'

def remove_extension(file_path):
    # Get the file name (with extension)
    file_name_with_extension = os.path.basename(file_path)
    
    # Remove the extension
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    
    # Get the file directory
    file_directory = os.path.dirname(file_path)
    
    # Return the full path without the extension
    return os.path.join(file_directory, file_name_without_extension)

def merge_audio_to_video(video_temp_file, video_path, video_out_name):
    is_successful_merged = 0
    # Create temporary audio object; on Windows, delete=True may cause a permission denied error
    temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
    audio_extract_command = [config.FFMPEG_PATH,
                             "-y", "-i", video_path,
                             "-acodec", "copy",
                             "-vn", "-loglevel", "error", temp.name]
    use_shell = True if os.name == "nt" else False
    try:
        subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
    except Exception:
        print('Failed to extract audio')
        return
    else:
        if os.path.exists(video_temp_file.name):
            audio_merge_command = [config.FFMPEG_PATH,
                                   "-y", "-i", video_temp_file.name,
                                   "-i", temp.name,
                                   "-vcodec", "libx264" if config.USE_H264 else "copy",
                                   "-acodec", "copy",
                                   "-loglevel", "error", video_out_name]
            try:
                subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell)
            except Exception:
                print('Failed to merge audio')
                return
        if os.path.exists(temp.name):
            try:
                os.remove(temp.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'Failed to delete temp file {temp.name}')
        is_successful_merged = 1
    finally:
        temp.close()
        if not is_successful_merged:
            try:
                shutil.copy2(video_temp_file.name, video_out_name)
            except IOError as e:
                print("Unable to copy file. %s" % e)
        video_temp_file.close()

def parse_srt_file(srt_file):
    """Parse an SRT file and return a list of timestamps and subtitle content"""
    subtitles = []
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                timestamps = lines[1]
                text = ' '.join(lines[2:])
                subtitles.append((timestamps, text))
    return subtitles

def split_text_by_punctuation(text):
    """Split text based on common punctuation marks"""
    # Define punctuation list
    punctuation = r'[。！？；：，——……,.、]'
    
    # Use regular expressions to split, keeping punctuation
    phrases = re.split(f'({punctuation})', text)
    
    # Combine phrases with the punctuation that follows
    combined_phrases = []
    for i in range(0, len(phrases) - 1, 2):
        combined_phrases.append(phrases[i] + phrases[i + 1])
    
    # If the last phrase doesn't have punctuation, manually add it
    if len(phrases) % 2 != 0:
        combined_phrases.append(phrases[-1])

    # Use list comprehension to remove empty strings
    combined_phrases = [element for element in combined_phrases if element != '']
    
    return combined_phrases

def replace_subtitles(subtitles, phrases):
    """Replace subtitles in the SRT file with the given phrases"""
    replaced_subtitles = []
    num_phrases = len(phrases)
    num_subtitles = len(subtitles)
    
    if num_phrases > num_subtitles:
        last_index = num_subtitles - 1
        while num_phrases > num_subtitles:
            phrases[last_index] += phrases[-1]
            phrases.pop()
            num_phrases -= 1
    
    for i in range(num_phrases):
        phrase = phrases[i]
        timestamps = subtitles[i][0]
        replaced_subtitles.append((timestamps, phrase))
        print(subtitles[i])
        print(phrases[i])
    for i in range(num_phrases, num_subtitles):
        phrase = phrases[-1]
        timestamps = subtitles[i][0]
        replaced_subtitles.append((timestamps, phrase))
        print(subtitles[i])
        print(phrases[-1])

    return replaced_subtitles

def generate_new_srt(replaced_subtitles, output_file):
    """Generate a new SRT file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (timestamps, text) in enumerate(replaced_subtitles):
            f.write(f"{i+1}\n")
            f.write(f"{timestamps}\n")
            f.write(f"{text}\n\n")

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

def createYoudaoRequest(q, lang_from, lang_to):
    url = 'https://openapi.youdao.com/api'
    
    data = {
        'appKey': YOUDAO_API_KEY,
        'q': q,
        'from': lang_from,
        'to': lang_to,
        'salt': str(random.randint(1, 65536)),
        'signType': 'v3',
        'curtime': str(int(time.time())),
    }
    
    signStr = YOUDAO_API_KEY + truncate(q) + data['salt'] + data['curtime'] + YOUDAO_APP_SECRET
    data['sign'] = encrypt(signStr)
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=data, headers=headers)
    return json.loads(response.text)['translation'][0]

def translate_subtitle(subtitle_src_file_path, subtitle_destination_file_path, target_language="zh-CHS", model="youdao"):
    with open(subtitle_src_file_path, 'r', encoding='utf-8') as file:
        src_content = file.read()
        
    # Remove timestamps and serial numbers, keeping only the text
    text_blocks = re.split(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', src_content)
    text = " ".join(text_blocks)
    
    # Remove extra newlines
    text = re.sub(r'\n', ' ', text).strip()

    mm = LLMAPI.ModelManager()

    if model == "youdao":
        ans = createYoudaoRequest(q=text, lang_from="auto", lang_to=target_language)
    else:
        mm = LLMAPI.ModelManager()
        if model == "gpt4":
            content = "Translate the following text into " + target_language + ": *" + text + "*"
            message = [{"role": "user", "content": content}]
            ans = mm.Get_Response_OpenAI_GPT4(messages=message)["content"]
        else:
            content_spark = "Translate the following text into " + target_language + ": *" + text + "*"
            message = [{"role": "user", "content": content_spark}]
            ans = mm.Get_Response_Spark(messages=message)

    # Step 1: Parse the SRT file
    subtitles = parse_srt_file(subtitle_src_file_path)

    # Step 2: Split text by punctuation
    phrases = split_text_by_punctuation(ans)

    # Step 3: Replace subtitle content in the SRT file with the given phrases
    replaced_subtitles = replace_subtitles(subtitles, phrases)

    # Step 4: Write the new subtitles to a new SRT file
    generate_new_srt(replaced_subtitles, subtitle_destination_file_path)

def add_subtitle_to_video(video_path, subtitle_path, output_path):
    # Replace backslashes with forward slashes in the paths
    video_path = video_path.replace('\\', '/')
    # subtitle_path = subtitle_path.replace('\\', '/')
    output_path = output_path.replace('\\', '/')

    command = [
        config.FFMPEG_PATH,
        '-i', video_path,
        '-vf', f"subtitles={subtitle_path}",
        '-c:a', 'copy',
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print("Subtitles added successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while adding subtitles: {e}")

def SubtitleTranslation():
    # Input video path, subtitle area, source language, and target language
    video_path = ''
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    sub_area = [xmin, xmax, ymin, ymax]
    src_language = 'en'
    target_language = 'cn'

    # Get the video name from the video path
    vd_name = Path(video_path).stem
    video_out_name = os.path.join(os.path.dirname(video_path), f'{vd_name}_no_sub.mp4')

    # Check if the file is a valid video
    if is_video_or_image(video_path):
        print(f'Valid video path: {video_path}')
        sys.exit()
    else:
        print(f'Invalid video path: {video_path}')

    # Remove subtitles and audio from the video
    sd = SubtitleRemover(vd_path=video_path, sub_area=sub_area, add_audio=False)
    sd.run()

    # Translate the subtitle file
    srt_input_path = remove_extension(video_path) + ".srt"
    srt_output_path = remove_extension(video_path) + "_translated.srt"
    translate_subtitle(subtitle_src_file_path=srt_input_path, subtitle_destination_file_path=srt_output_path)

    # Merge the translated subtitles into the video
    video_final_name = os.path.join(os.path.dirname(video_path), f'{vd_name}_no_sub_add_subtitle.mp4')
    add_subtitle_to_video(video_path=video_out_name, subtitle_path=srt_output_path, output_path=video_final_name)


def videoCombineTest():
    # There's an issue: unable to properly read the SRT file
    video_path = "D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.mp4"
    subtitle_path = "D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.srt"
    output_path = "D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5_trans.mp4"
    add_subtitle_to_video(video_path, subtitle_path, output_path)


def srtTranTest():
    # Works fine, just need to adjust the path
    # Calls the Spark/OpenAI API
    # For the OpenAI API, you need to add the key in the LLMAPI file and configure different models
    path = "../test/test5/test5.srt"
    output_path = "../test/test5/test5_trans.srt"
    translate_subtitle(subtitle_src_file_path=path, subtitle_destination_file_path=output_path, model="Spark")


def translatedTextToSrtTest():
    # Test for matching translated text segments to the number of original SRT subtitles.
    # Add/remove punctuation from the `ans` to improve text segmentation.
    subtitle_src_file_path = "D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.srt"
    subtitle_destination_file_path = "D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5_trans.srt"

    ans = "Curiosity is one of humanity's driving forces, you know, modern people, the kind of people who wanted to explore the world, knowing nothing, they got on boats, crossing oceans."

    # Step 1: Parse the SRT file
    subtitles = parse_srt_file(subtitle_src_file_path)

    # Step 2: Split the translated text by punctuation
    phrases = split_text_by_punctuation(ans)

    # Step 3: Replace the subtitles in the SRT file with the translated phrases
    replaced_subtitles = replace_subtitles(subtitles, phrases)

    # Step 4: Generate a new SRT file
    generate_new_srt(replaced_subtitles, subtitle_destination_file_path)


if __name__ == '__main__':
    srtTranTest()
