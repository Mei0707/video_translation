import subprocess
import os
import json
import re

def read_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # 去除时间戳和序号，只保留文本
    text_blocks = re.split(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', content)
    text = " ".join(text_blocks)
    
    # 去除多余的换行符
    text = re.sub(r'\n', ' ', text).strip()
    
    return text

def count_subtitles_in_srt(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 检查该行是否为字幕编号（只包含数字）
            if line.strip().isdigit():
                count += 1
    return count

def extract_audio(video_path, audio_path):
    """
    使用 ffmpeg 将视频中的音频提取并保存为指定格式。

    :param video_path: 输入视频文件路径
    :param audio_path: 输出音频文件路径（包括文件名和后缀）
    """
    command = [
        'D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\backend\\ffmpeg\\win_x64\\ffmpeg.exe',
        '-i', video_path,    # 输入视频文件
        '-vn',               # 不处理视频部分
        '-acodec', 'pcm_s16le', # 音频编码格式
        '-ar', '44100',      # 音频采样率
        '-ac', '2',          # 音频通道数
        audio_path           # 输出音频文件
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Audio extracted and saved to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def extract_subtitles(video_path, subtitle_path, subtitle_stream=0):
    """
    使用 ffmpeg 将视频中的软链接字幕提取并保存为指定文件。

    :param video_path: 输入视频文件路径
    :param subtitle_path: 输出字幕文件路径（包括文件名和后缀）
    :param subtitle_stream: 要提取的字幕流编号（默认为第一个字幕流）
    """
    command = [
        'D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\backend\\ffmpeg\\win_x64\\ffmpeg.exe',            # ffmpeg 可执行文件路径
        '-i', video_path,     # 输入视频文件
        '-map', f'0:s:{subtitle_stream}', # 选择字幕流
        subtitle_path         # 输出字幕文件
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Subtitles extracted and saved to {subtitle_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def parse_asr_result(asr_result):

    data = asr_result
    
    # 初始化一个结果列表
    result = []
    
    # 遍历lattice数组
    for item in data['content']['orderResult']['lattice']:
        json_1best = json.loads(item['json_1best'])
        rt = json_1best['st']['rt']
        
        for segment in rt:
            words = [cw['w'] for ws in segment['ws'] for cw in ws['cw']]
            sentence = ''.join(words)
            start_time = int(segment['ws'][0]['wb'])
            end_time = int(segment['ws'][-1]['we'])
            
            result.append({
                'sentence': sentence,
                'start_time': start_time,
                'end_time': end_time
            })
    
    return result

def has_subtitles(video_path):
    """
    检查视频文件是否包含软链接字幕。

    :param video_path: 输入视频文件路径
    :return: 如果包含字幕返回 True，否则返回 False
    """
    command = [
        'D:\\superADS\\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\backend\\ffmpeg\\win_x64\\ffmpeg.exe',            # ffmpeg 可执行文件路径
        '-i', video_path
    ]

    try:
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output = result.stderr

        # 正则表达式匹配字幕流信息
        subtitle_info_pattern = re.compile(r'Stream #(\d+:\d+)(\[.*?\])?: Subtitle: (\w+)', re.IGNORECASE)
        subtitle_info = subtitle_info_pattern.findall(output)

        # 提取字幕流的格式信息
        subtitles = []
        for stream, lang, format in subtitle_info:
            subtitles.append({
                'stream': stream,
                'language': lang.strip('[]') if lang else 'unknown',
                'format': format
            })

        return subtitles

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []

def video_extract_test():
    video_path = "D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test.mp4"
    audio_path = "D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test.wav"
    extract_audio(video_path, audio_path)

def a2t_test():
    file_path="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\backend\\result.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            asr_result = json.load(file)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("请检查JSON文件格式是否正确。")
            exit(1)
    parsed_result = parse_asr_result(asr_result)
    for segment in parsed_result:
        print(f"Sentence: {segment['sentence']}, Start Time: {segment['start_time']}, End Time: {segment['end_time']}")

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
    
    return combined_phrases

def subtitle_distract_test():
    video_path ="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test.mp4"
    subtitles_path = "D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test.srt"
    extract_subtitles(video_path, subtitles_path)

def has_subtitle_test():
    video_path ="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test2.mp4"
    ans=has_subtitles(video_path)
    print(ans)


if __name__ == "__main__":
    # 可以在这里检查脚本是否正常读取了srt文件
    path ="D:\superADS\VideoTranslation\\video-subtitle-remover\\video-subtitle-remover\\test\\test5\\test5.srt"
    print(read_srt(path))
    
    num_subtitles = count_subtitles_in_srt(path)
    print(f"该SRT文件共有 {num_subtitles} 条字幕。")


