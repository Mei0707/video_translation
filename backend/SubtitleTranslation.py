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
from backend.tools.common_tools import is_video_or_image, is_image_file,is_video_file
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

YOUDAO_API_KEY = '75a7f5c5babf556e'
YOUDAO_APP_SECRET = 'zi306pwC76MkUOa5kcvKw1deLkeNVTxX'

def remove_extension(file_path):
    # 获取文件名（带扩展名）
    file_name_with_extension = os.path.basename(file_path)
    
    # 去掉扩展名
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    
    # 获取文件路径
    file_directory = os.path.dirname(file_path)
    
    # 返回去掉扩展名后的完整路径
    return os.path.join(file_directory, file_name_without_extension)

def merge_audio_to_video(video_temp_file,video_path,video_out_name):
        is_successful_merged=0
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception:
            print('fail to extract audio')
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
                    print('fail to merge audio')
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    if platform.system() in ['Windows']:
                        pass
                    else:
                        print(f'failed to delete temp file {temp.name}')
            is_successful_merged=1
        finally:
            temp.close()
            if not is_successful_merged:
                try:
                    shutil.copy2(video_temp_file.name, video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            video_temp_file.close()

def parse_srt_file(srt_file):
    """解析 SRT 文件，返回时间戳和字幕内容的列表"""
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
    """根据常见标点符号拆分文本"""
    # 定义标点符号列表
    punctuation = r'[。！？；：，——……,.、]'
    
    # 使用正则表达式进行拆分，保留标点符号
    phrases = re.split(f'({punctuation})', text)
    
    # 将短语与其后的标点符号拼接起来
    combined_phrases = []
    for i in range(0, len(phrases) - 1, 2):
        combined_phrases.append(phrases[i] + phrases[i + 1])
    
    # 如果原文本的最后一个短语没有标点符号，则需要手动添加
    if len(phrases) % 2 != 0:
        combined_phrases.append(phrases[-1])

    # 使用列表推导式移除空字符串
    combined_phrases = [element for element in combined_phrases if element != '']
    
    return combined_phrases

def replace_subtitles(subtitles, phrases):
    """用短语替换 SRT 文件中的字幕内容"""
    replaced_subtitles = []
    num_phrases = len(phrases)
    num_subtitles = len(subtitles)
    
    if num_phrases>num_subtitles:
        last_index=num_subtitles-1
        while num_phrases>num_subtitles:
            phrases[last_index]+=phrases[-1]
            phrases.pop()
            num_phrases=num_phrases-1
    
    for i in range(num_phrases):
            phrase=phrases[i]
            timestamps=subtitles[i][0]
            replaced_subtitles.append((timestamps, phrase))
            print(subtitles[i])
            print(phrases[i])
    for i in range(num_phrases,num_subtitles):
            phrase=phrases[-1]
            timestamps=subtitles[i][0]
            replaced_subtitles.append((timestamps, phrase))
            print(subtitles[i])
            print(phrases[-1])
    #print(subtitles[i])
    #print(phrases[i])



    return replaced_subtitles

def generate_new_srt(replaced_subtitles, output_file):
    """生成新的 SRT 文件"""
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

def translate_subtitle(subtitle_src_file_path,subtitle_destination_file_path,target_language="zh-CHS",model="youdao"):
    
    with open(subtitle_src_file_path, 'r', encoding='utf-8') as file:
        src_content = file.read()
        
        
    # 去除时间戳和序号，只保留文本
    text_blocks = re.split(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', src_content)
    text = " ".join(text_blocks)
    
    # 去除多余的换行符
    text = re.sub(r'\n', ' ', text).strip()

    mm=LLMAPI.ModelManager()

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
            
    
    # if(model=="gpt4"):
    #     content="把星号中的文本*"+text+"*翻译成"+target_language+',按照下一个星号中的json格式返回，不要有多余输出,返回json中的引号应当为英文双引号 * {"content":"翻译后内容"} *'
    #     message=[{
    #     "role":"user",
    #     "content":content
    #     }]
    #     ans=mm.Get_Response_OpenAI_GPT4(messages=message)["content"]
    # else:
    #     content_spark="你是一个很棒的翻译官，把星号中的文本信、达、雅地*"+text+"*翻译成"+target_language+",直接返回翻译结果"
    #     message=[{
    #         "role":"user",
    #         "content":content_spark
    #     }]
    #     ans=mm.Get_Response_Spark(messages=message)

    
    # 步骤 1: 解析 SRT 文件
    subtitles = parse_srt_file(subtitle_src_file_path)

    # 步骤 2: 拆分带有标点符号的文本
    phrases = split_text_by_punctuation(ans)

    # 步骤 3: 替换 SRT 文件中的字幕内容
    replaced_subtitles = replace_subtitles(subtitles, phrases)

    # 步骤 4: 生成新的 SRT 文件
    generate_new_srt(replaced_subtitles, subtitle_destination_file_path)

def add_subtitle_to_video(video_path, subtitle_path, output_path):
    # 替换路径中的反斜杠为正斜杠
    video_path = video_path.replace('\\', '/')
    #subtitle_path = subtitle_path.replace('\\', '/')
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
        print("字幕添加成功!")
    except subprocess.CalledProcessError as e:
        print(f"添加字幕时出错: {e}")



def SubtitleTranslation():
    # 传入视频地址，字幕框位置，原语言与目标语言
    video_path = ''
    xmin,xmax,ymin,ymax = 0,0,0,0
    sub_area=[xmin,xmax,ymin,ymax]
    src_language='en'
    target_language='cn'

    # 通过视频路径获取视频名称
    vd_name = Path(video_path).stem
    video_out_name = os.path.join(os.path.dirname(video_path), f'{vd_name}_no_sub.mp4')

    
    if is_video_or_image(video_path):
        print(f'Valid video path: {video_path}')
        sys.exit()
    else:
        print(f'Invalid video path: {video_path}')


    #视频擦除为无字幕无声视频
    sd = SubtitleRemover(vd_path=video_path, sub_area=sub_area,add_audio=False)
    sd.run()
    
    # 字幕文件翻译
    srt_input_path=remove_extension(video_path) + ".srt"
    srt_output_path=remove_extension(video_path) + "_translated.srt"
    translate_subtitle(subtitle_src_file_path= srt_input_path,subtitle_destination_file_path=srt_output_path)

    # 合成为新字幕无声视频
    video_final_name = os.path.join(os.path.dirname(video_path), f'{vd_name}_no_sub_add_subtitle.mp4')
    add_subtitle_to_video(video_path=video_out_name,subtitle_path=srt_output_path,output_path=video_final_name)

def videoCombineTest():
    # 有问题 无法正常读取SRT文件
    video_path="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.mp4"
    subtitle_path="D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.srt"
    output_path="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5_trans.mp4"
    add_subtitle_to_video(video_path, subtitle_path, output_path)

def srtTranTest():
    # 没问题，改一下路径就可以
    # 调用星火/openai接口
    # 如果用openai接口需要取LLMAPI文件中添加key，可以设置不同的模型
    path ="../test/test5/test5.srt"
    output_path ="../test/test5/test5_trans.srt"
    translate_subtitle(subtitle_src_file_path=path,subtitle_destination_file_path=output_path,model="星火")

def translatedTextToSrtTest():
    # 测试翻译文本片段-原SRT文件数目匹配，给ans加/减一些标点符号分隔
    subtitle_src_file_path="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.srt"
    subtitle_destination_file_path="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5_trans.srt"

    ans="好奇心是人类的驱动力之一，你知道，现代人，就是那种,想要探索世界、毫无所知,就登上船只、跨越海洋的人。"
    
    # 步骤 1: 解析 SRT 文件
    subtitles = parse_srt_file(subtitle_src_file_path)

    # 步骤 2: 拆分带有标点符号的文本
    phrases = split_text_by_punctuation(ans)

    # 步骤 3: 替换 SRT 文件中的字幕内容
    replaced_subtitles = replace_subtitles(subtitles, phrases)

    # 步骤 4: 生成新的 SRT 文件
    generate_new_srt(replaced_subtitles, subtitle_destination_file_path)
    

if __name__ == '__main__':
    srtTranTest()

    
    
    



