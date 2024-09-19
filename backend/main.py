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
from backend.tools.common_tools import is_video_or_image, is_image_file
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


class SubtitleDetect:
    """
    Textbox detection class, used to detect the presence of textboxes in video frames.
    """

    def __init__(self, video_path, sub_area=None):
        # Load configuration parameters
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)
        self.video_path = video_path
        self.sub_area = sub_area

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        Get coordinates from the detected bounding boxes.
        :param dt_box Detection box results
        :return list List of coordinate points
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        print('[Processing] start finding subtitles...')
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # If reading the video frame fails (reached the last frame) 
            if not ret:
                break
            # Successfully read video frame
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if self.sub_area is not None:
                        s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                if len(temp_list) > 0:
                    subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        # if config.UNITE_COORDINATES:
        #     subtitle_frame_no_box_dict = self.get_subtitle_frame_no_box_dict_with_united_coordinates(subtitle_frame_no_box_dict)
        #     if sub_remover is not None:
        #         try:
        #             # When the number of frames is greater than 1, it indicates that it is not an image or a single frame
        #             if sub_remover.frame_count > 1:
        #                 subtitle_frame_no_box_dict = self.filter_mistake_sub_area(subtitle_frame_no_box_dict,
        #                                                                           sub_remover.fps)
        #         except Exception:
        #             pass
        #     subtitle_frame_no_box_dict = self.prevent_missed_detection(subtitle_frame_no_box_dict)
        print('[Finished] Finished finding subtitles...')
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # Ensure the list of discrete values is ordered
        points.sort()
        # List to store the resulting intervals
        result_intervals = []
        # Traverse intervals
        for start, end in intervals:
            # Points within the current interval
            current_points = [p for p in points if start <= p <= end]

            # Traverse the discrete points within the current interval
            for p in current_points:
                # If the current discrete point is not the starting point of the interval, add an interval from the start to the point before
                if start < p:
                    result_intervals.append((start, p - 1))
                # Update the interval start to the current discrete point
                start = p
            # Add the interval from the last discrete point or interval start to the end of the interval
            result_intervals.append((start, end))
        # Output the result
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        Get the frame numbers where scene changes occur
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """Check if two regions are similar."""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.PIXEL_TOLERANCE_X and abs(xmax1 - xmax2) <= config.PIXEL_TOLERANCE_X and \
            abs(ymin1 - ymin2) <= config.PIXEL_TOLERANCE_Y and abs(ymax1 - ymax2) <= config.PIXEL_TOLERANCE_Y

    def unify_regions(self, raw_regions):
        """Unify similar regions, preserving list structure."""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  #Unify similar regions, preserving list structure.
            unified_regions = {}

            # Initialize
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # New list to store matched standard intervals
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # If the current interval is similar to the previous key's corresponding interval, unify them
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # Update unify_value_map with the latest interval values
                unify_value_map[key] = new_unify_values
                last_key = key

            # Pass the final unified result to unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        Get the start and end frame numbers where subtitles appear
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # Initial range start value

        for i in range(1, len(numbers)):
            # If the current number is more than 1 interval from the previous number,
            # the previous interval ends, record the start and end of the current interval
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # The number is the end of the current continuous interval
                ranges.append((start, end))
                start = numbers[i]  # Start the next continuous interval
        # Add the last interval
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # Initial range start value
        for i in range(1, len(numbers)):
            # If the current frame number is more than 1 interval from the previous frame number,
            # the previous interval ends, record the start and end of the current interval
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # The number is the end of the current continuous interval
                ranges.append((start, end))
                start = numbers[i]  # Start the next continuous interval
            # If the current frame number is 1 interval from the previous frame number,
            # but the subtitle box regions differ, the previous interval ends, record the start and end of the current interval
            elif subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                end = numbers[i - 1]  # The number is the end of the current continuous interval
                ranges.append((start, end))
                start = numbers[i]  # Start the next continuous interval
        # Add the last interval
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        Convert subtitle area to a polygon.
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    @staticmethod
    def expand_and_merge_intervals(intervals, expand_size=config.STTN_NEIGHBOR_STRIDE*config.STTN_REFERENCE_LENGTH, max_length=config.STTN_MAX_LOAD_NUM):
        # Initialize the output interval list
        expanded_intervals = []

        # Expand each original interval
        for interval in intervals:
            start, end = interval

            # Expand to at least 'expand_size' units, but not exceed 'max_length' units
            expansion_amount = max(expand_size - (end - start + 1), 0)

            # Split expansion amount as evenly as possible before and after the interval while ensuring the interval is included
            expand_start = max(start - expansion_amount // 2, 1)  # Ensure start is not less than 1
            expand_end = end + expansion_amount // 2

            # Adjust if the expanded interval exceeds maximum length
            if (expand_end - expand_start + 1) > max_length:
                expand_end = expand_start + max_length - 1

            # Handle single-point intervals to ensure at least 'expand_size' length
            if start == end:
                if expand_end - expand_start + 1 < expand_size:
                    expand_end = expand_start + expand_size - 1

            # Check and merge with the previous interval if they overlap
            if expanded_intervals and expand_start <= expanded_intervals[-1][1]:
                previous_start, previous_end = expanded_intervals.pop()
                expand_start = previous_start
                expand_end = max(expand_end, previous_end)

            # Add the expanded interval to the result list
            expanded_intervals.append((expand_start, expand_end))

        return expanded_intervals

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length=config.STTN_REFERENCE_LENGTH):
        """
        Merge input subtitle start intervals, ensuring that intervals are at least STTN_REFERENCE_LENGTH in size
        """
        expanded = []
        # First handle single-point intervals to expand them
        for start, end in intervals:
            if start == end:  # Single-point interval
                # Expand to near the target length while ensuring no overlap before and after
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # Find the starting point of the next interval
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # Determine new expansion start and end points
                new_start = max(start - (target_length - 1) // 2, prev_end + 1)
                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # If new end is before the start, no space for expansion
                if new_end < new_start:
                    new_start, new_end = start, start  # Keep original
                expanded.append((new_start, new_end))
            else:
                # Non-single-point intervals are retained, and overlap will be handled later
                expanded.append((start, end))
        # Sort to merge overlapping intervals due to expansion
        expanded.sort(key=lambda x: x[0])
        # Merge overlapping intervals, but only if they overlap and are less than target length
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # Check for overlap
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # Merge intervals
                merged[-1] = (last_start, max(last_end, end))  # Merge interval
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # Merge adjacent intervals as well
                merged[-1] = (last_start, end)
            else:
                # If no overlap and both are greater than target length, retain directly
                merged.append((start, end))
        return merged

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # Find the largest area text box for the interval
            current_no = start_no
            # Find the largest area rectangle in the current interval
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # Extract coordinates of each text box
                    xmin, xmax, ymin, ymax = coord
                    # Calculate area of the current text box coordinates
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # If the max box list is empty, the current area is the maximum area for the interval
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # If the list is not empty, check if the current text box is in the same region as any of the max boxes
                    else:
                        has_same_position = False
                        # Iterate over each max box to check if the current text box is in the same line and intersects
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.THRESHOLD_HEIGHT_DIFFERENCE <= ymin
                                    and ymax <= area_max_box['ymax'] + config.THRESHOLD_HEIGHT_DIFFERENCE):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # If height difference is different
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # If on the same line, check if the current area is larger
                                    # Update the maximum area for the row if the current area is larger
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # If after checking all max boxes, it's a new line, add it directly
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        Unify the text region coordinates of multiple video frames
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        添加额外的文本框，防止漏检
        """
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        Filter out incorrect subtitle areas
        """
        sub_frame_no_list_continuous = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None, gui_mode=False, add_audio=False):
        importlib.reload(config)
        # Thread lock
        self.lock = threading.RLock()
        # User-specified subtitle area position
        self.sub_area = sub_area
        # Whether running in GUI mode, which requires displaying preview
        self.gui_mode = gui_mode
        # Check if it's an image
        self.is_picture = False
        if is_image_file(str(vd_path)):
            self.sub_area = None
            self.is_picture = True
        # Video path
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # Get video name from path
        self.vd_name = Path(self.video_path).stem
        # Total number of frames in the video
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # Video frame rate
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # Video dimensions
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Create subtitle detection object
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # Create temporary video object; windows will encounter permission denied error if delete=True
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # Create video writer object
        self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')
        if torch.cuda.is_available():
            print('Use GPU for acceleration')
        # Overall progress
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # Preview frame
        self.preview_frame = None
        # Whether to embed audio
        self.add_audio = add_audio
        # Whether to merge original audio into the video with subtitles removed
        self.is_successful_merged = False

    @staticmethod
    def get_coordinates(dt_box):
        """
        Get coordinates from the detection box returned
        :param dt_box Detection box return result
        :return list List of coordinate points
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        Check if the given frame number is the start of a range, if so return the end frame number, otherwise return -1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        Check if the given frame number is the end of a range, if so return the end frame number, otherwise return -1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1
    
    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover

    def propainter_mode(self, tbar):
        print('Using ProPainter mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(continuous_frame_no_list, scene_div_points)
        self.video_inpaint = VideoInpaint(config.PROPAINTER_MAX_LOAD_NUM)
        print('[Processing] Start removing subtitles...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            # If the current frame does not have a watermark/text, write it directly
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                print(f'Write frame: {index}')
                self.update_progress(tbar, increment=1)
                continue
            # If there is a watermark, check if the frame is the starting frame
            else:
                # If it is the starting frame, batch process until the ending frame
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    start_frame_no = index
                    print(f'Find start: {start_frame_no}')
                    # Find the end frame
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # Check if the current frame number is the subtitle start position
                    if end_frame_no != -1:
                        print(f'Find end: {end_frame_no}')
                        # ************ Read all frames in the interval start ************
                        temp_frames = list()
                        # Add the starting frame to the processing list
                        temp_frames.append(frame)
                        inner_index = 0
                        # Read until the ending frame
                        while index < end_frame_no:
                            ret, frame = self.video_cap.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ Read all frames in the interval end ************
                        if len(temp_frames) < 1:
                            # No frames to process, skip
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            if self.lama_inpaint is None:
                                self.lama_inpaint = LamaInpaint()
                            inpainted_frame = self.lama_inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            print(f'Write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # Batch process the video frames
                            # 1. Get the mask used for the current batch
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            for batch in batch_generator(temp_frames, config.PROPAINTER_MAX_LOAD_NUM):
                                # 2. Perform batch inference
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    if self.lama_inpaint is None:
                                        self.lama_inpaint = LamaInpaint()
                                    inpainted_frame = self.lama_inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    print(f'Write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = self.video_inpaint.inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        print(f'Write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        if self.gui_mode:
                                            self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                                self.update_progress(tbar, increment=len(batch))

    def sttn_mode_with_no_detection(self, tbar):
        """
        Use STTN to repaint the selected area without subtitle detection
        """
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen. As a result, the final outcome might be suboptimal.')
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def sttn_mode(self, tbar):
        # Whether to skip subtitle frame detection
        if config.STTN_SKIP_DETECTION:
            # If skipped, use STTN mode
            self.sttn_mode_with_no_detection(tbar)
        else:
            print('Using STTN mode')
            sttn_inpaint = STTNInpaint()
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
            print(continuous_frame_no_list)
            continuous_frame_no_list = self.sub_detector.filter_and_merge_intervals(continuous_frame_no_list)
            print(continuous_frame_no_list)
            start_end_map = dict()
            for interval in continuous_frame_no_list:
                start, end = interval
                start_end_map[start] = end
            current_frame_index = 0
            print('[Processing] Start removing subtitles...')
            while True:
                ret, frame = self.video_cap.read()
                # End if no more frames
                if not ret:
                    break
                current_frame_index += 1
                # Check if the current frame number is the start of a subtitle range; if not, write it directly
                if current_frame_index not in start_end_map.keys():
                    self.video_writer.write(frame)
                    print(f'Write frame: {current_frame_index}')
                    self.update_progress(tbar, increment=1)
                    if self.gui_mode:
                        self.preview_frame = cv2.hconcat([frame, frame])
                # If it's the start of a subtitle range, find the end of the range
                else:
                    start_frame_index = current_frame_index
                    end_frame_index = start_end_map[current_frame_index]
                    print(f'Processing frames from {start_frame_index} to {end_frame_index}')
                    # Store frames that need inpainting
                    frames_need_inpaint = list()
                    frames_need_inpaint.append(frame)
                    inner_index = 0
                    # Read frames until the end of the range
                    for j in range(end_frame_index - start_frame_index):
                        ret, frame = self.video_cap.read()
                        if not ret:
                            break
                        current_frame_index += 1
                        frames_need_inpaint.append(frame)
                    mask_area_coordinates = []
                    # 1. Get the mask coordinates for the current batch
                    for mask_index in range(start_frame_index, end_frame_index):
                        if mask_index in sub_list.keys():
                            for area in sub_list[mask_index]:
                                xmin, xmax, ymin, ymax = area
                                # Check if it's a non-subtitle area (if width is greater than height, it's considered incorrect detection)
                                if (ymax - ymin) - (xmax - xmin) > config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE:
                                    continue
                                if area not in mask_area_coordinates:
                                    mask_area_coordinates.append(area)
                    # 1. Create the mask for the current batch
                    mask = create_mask(self.mask_size, mask_area_coordinates)
                    print(f'Inpainting with mask: {mask_area_coordinates}')
                    for batch in batch_generator(frames_need_inpaint, config.STTN_MAX_LOAD_NUM):
                        # 2. Perform batch inference
                        if len(batch) >= 1:
                            inpainted_frames = sttn_inpaint(batch, mask)
                            for i, inpainted_frame in enumerate(inpainted_frames):
                                self.video_writer.write(inpainted_frame)
                                print(f'Write frame: {start_frame_index + inner_index} with mask')
                                inner_index += 1
                                if self.gui_mode:
                                    self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                        self.update_progress(tbar, increment=len(batch))

    def lama_mode(self, tbar):
        print('Using Lama mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        if self.lama_inpaint is None:
            self.lama_inpaint = LamaInpaint()
        index = 0
        print('[Processing] Start removing subtitles...')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if index in sub_list.keys():
                mask = create_mask(self.mask_size, sub_list[index])
                if config.LAMA_SUPER_FAST:
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    frame = self.lama_inpaint(frame, mask)
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, frame])
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self.video_writer.write(frame)
            tbar.update(1)
            self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            self.progress_total = 50 + self.progress_remover

    def run(self):
        # Record start time
        start_time = time.time()
        # Reset progress bar
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self.lama_inpaint = LamaInpaint()
            original_frame = cv2.imread(self.video_path)
            if len(sub_list):
                mask = create_mask(original_frame.shape[0:2], sub_list[1])
                inpainted_frame = self.lama_inpaint(original_frame, mask)
            else:
                inpainted_frame = original_frame
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, inpainted_frame])
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_name)
            tbar.update(1)
            self.progress_total = 100
        else:
            # For precise mode, get scene split frame numbers and further segment
            if config.MODE == config.InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.MODE == config.InpaintMode.STTN:
                self.sttn_mode(tbar)
            else:
                self.lama_mode(tbar)
        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # Merge original audio into the new video file
            if self.add_audio:
                self.merge_audio_to_video()
            print(f"[Finished] Subtitle successfully removed, video generated at: {self.video_out_name}")
        else:
            print(f"[Finished] Subtitle successfully removed, picture generated at: {self.video_out_name}")
        print(f'Time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'Failed to delete temp file {self.video_temp_file.name}')

        return self.video_out_name

    def merge_audio_to_video(self):
        # Create a temporary audio file, on Windows, delete=True may cause permission denied error
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                "-y", "-i", self.video_path,
                                "-acodec", "copy",
                                "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception:
            print('Failed to extract audio')
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [config.FFMPEG_PATH,
                                    "-y", "-i", self.video_temp_file.name,
                                    "-i", temp.name,
                                    "-vcodec", "libx264" if config.USE_H264 else "copy",
                                    "-acodec", "copy",
                                    "-loglevel", "error", self.video_out_name]
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
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            self.video_temp_file.close()

    if __name__ == '__main__':
        multiprocessing.set_start_method("spawn")
        # 1. Prompt user to input video path
        video_path = input(f"Please input video or image file path: ").strip()
        # Determine if the video path is a directory; if it is, process all video files in the directory
        # 2. Provide subtitle area in the following order
        # sub_area = (ymin, ymax, xmin, xmax)
        # 3. Create a subtitle extraction object
        if is_video_or_image(video_path):
            sd = SubtitleRemover(video_path, sub_area=None)
            sd.run()
        else:
            print(f'Invalid video path: {video_path}')
