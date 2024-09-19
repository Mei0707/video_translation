import os
import sys
from pathlib import Path
from backend.main import SubtitleDetect, SubtitleRemover
from backend.SubtitleTranslation import translate_subtitle, add_subtitle_to_video
from backend.tools.common_tools import is_video_or_image

def remove_extension(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    file_directory = os.path.dirname(file_path)
    return os.path.join(file_directory, file_name_without_extension)

def workflow(video_url, target_language):
    video_url = video_url.replace('\\', '/')
    video_path = video_url
    video_name = Path(video_path).stem
    video_out_name = os.path.join(os.path.dirname(video_path), f'{video_name}_no_sub.mp4')

    # Check if the video path is valid
    if is_video_or_image(video_path):
        print(f'Valid video path: {video_path}')
    else:
        print(f'Invalid video path: {video_path}')
        return

    # Step 1: Detect and remove subtitles
    print("Detecting and removing subtitles from video...")
    sd = SubtitleDetect(video_path=video_path)
    sd.detect_subtitles()  # Assuming this method detects and processes subtitles
    sr = SubtitleRemover(vd_path=video_path, sub_area=[0, 0, 0, 0], add_audio=False)
    sr.run()


    # Step 2: Translate subtitles
    print("Translating subtitles...")
    srt_input_path = remove_extension(video_path) + ".srt"
    srt_output_path = remove_extension(video_path) + "_translated.srt"
    translate_subtitle(subtitle_src_file_path=srt_input_path, subtitle_destination_file_path=srt_output_path, target_language=target_language)

    # Step 3: Add translated subtitles to the video
    print("Adding translated subtitles to video...")
    video_final_name = os.path.join(os.path.dirname(video_path), f'{video_name}_no_sub_add_subtitle.mp4')
    add_subtitle_to_video(video_path=video_out_name, subtitle_path=srt_output_path, output_path=video_final_name)

    print(f"Translation complete. Output video: {video_final_name}")

    return video_final_name

if __name__ == '__main__':
    video_url = 'path/to/your/video.mp4'
    target_language = 'zh'  # Example: 'zh' for Chinese
    translated_video_url = workflow(video_url, target_language)
    print(f"Translated video URL: {translated_video_url}")
