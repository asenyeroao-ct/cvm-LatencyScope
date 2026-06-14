import re
import subprocess
from typing import List, Optional, Tuple

import cv2


SUPPORTED_CAPTURE_FORMATS = ["MJPG", "NV12", "YUYV", "YUY12", "BGR24"]
FOURCC_ALIASES = {
    "MJPG": "MJPG",
    "NV12": "NV12",
    "YUYV": "YUY2",
    "YUY2": "YUY2",
    "YUY12": "YV12",
    "YV12": "YV12",
    "BGR24": "BGR3",
}


def normalize_capture_format(value: str) -> str:
    format_name = str(value or "MJPG").upper()
    if format_name not in FOURCC_ALIASES:
        raise ValueError(f"Unsupported capture format: {value}")
    return format_name


def list_dshow_capture_devices(max_indices: int = 10) -> List[dict]:
    names = _list_dshow_video_device_names()
    devices = []
    if names:
        for device_index, name in enumerate(names[: max_indices + 1]):
            devices.append({"index": device_index, "name": name})
        return devices
    for device_index in range(max_indices + 1):
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            devices.append({"index": device_index, "name": f"DShow Device {device_index}"})
        cap.release()
    return devices


def _list_dshow_video_device_names() -> List[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        names = []
        for line in combined.splitlines():
            match = re.search(r'"(.+?)"\s+\((?:audio,\s*)?video\)', line)
            if match:
                names.append(match.group(1))
        return names
    except Exception:
        return []


class CaptureCardCamera:
    def __init__(self, config, region=None):
        self.frame_width = int(getattr(config, "capture_width", 1920))
        self.frame_height = int(getattr(config, "capture_height", 1080))
        self.target_fps = float(getattr(config, "capture_fps", 240))
        self.device_index = int(getattr(config, "capture_device_index", 0))
        self.capture_buffer_mb = int(getattr(config, "capture_buffer_mb", 64))
        self.capture_format = normalize_capture_format(getattr(config, "capture_format", "MJPG"))
        self.fourcc_pref = [self.capture_format]
        self.config = config
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        self.running = True

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open DShow capture device {self.device_index}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.capture_buffer_mb))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.frame_width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.frame_height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        fourcc = FOURCC_ALIASES[self.capture_format]
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    def get_latest_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        base_w = int(getattr(self.config, "capture_width", 1920))
        base_h = int(getattr(self.config, "capture_height", 1080))
        range_x = int(getattr(self.config, "capture_range_x", 0))
        range_y = int(getattr(self.config, "capture_range_y", 0))
        if range_x <= 0:
            range_x = getattr(self.config, "region_size", 200)
        if range_y <= 0:
            range_y = getattr(self.config, "region_size", 200)

        offset_x = int(getattr(self.config, "capture_offset_x", 0))
        offset_y = int(getattr(self.config, "capture_offset_y", 0))

        left = (base_w - range_x) // 2 + offset_x
        top = (base_h - range_y) // 2 + offset_y
        right = left + range_x
        bottom = top + range_y

        left = max(0, min(left, base_w))
        top = max(0, min(top, base_h))
        right = max(left, min(right, base_w))
        bottom = max(top, min(bottom, base_h))
        return frame[top:bottom, left:right]

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None


def get_capture_card_region(config) -> Tuple[int, int, int, int]:
    base_w = int(getattr(config, "capture_width", getattr(config, "screen_width", 1920)))
    base_h = int(getattr(config, "capture_height", getattr(config, "screen_height", 1080)))
    range_x = int(getattr(config, "capture_range_x", 0))
    range_y = int(getattr(config, "capture_range_y", 0))
    if range_x <= 0:
        range_x = getattr(config, "region_size", 200)
    if range_y <= 0:
        range_y = getattr(config, "region_size", 200)
    offset_x = int(getattr(config, "capture_offset_x", 0))
    offset_y = int(getattr(config, "capture_offset_y", 0))
    left = (base_w - range_x) // 2 + offset_x
    top = (base_h - range_y) // 2 + offset_y
    right = left + range_x
    bottom = top + range_y
    left = max(0, min(left, base_w))
    top = max(0, min(top, base_h))
    right = max(left, min(right, base_w))
    bottom = max(top, min(bottom, base_h))
    return (left, top, right, bottom)


def validate_capture_card_config(config) -> Tuple[bool, Optional[str]]:
    try:
        device_index = int(getattr(config, "capture_device_index", 0))
        if device_index < 0 or device_index > 10:
            return False, f"Device index {device_index} is out of valid range (0-10)"
        width = int(getattr(config, "capture_width", 1920))
        height = int(getattr(config, "capture_height", 1080))
        if width < 320 or width > 7680:
            return False, f"Capture width {width} is out of valid range (320-7680)"
        if height < 240 or height > 4320:
            return False, f"Capture height {height} is out of valid range (240-4320)"
        fps = float(getattr(config, "capture_fps", 240))
        if fps < 1 or fps > 300:
            return False, f"Capture FPS {fps} is out of valid range (1-300)"
        normalize_capture_format(getattr(config, "capture_format", "MJPG"))
        return True, None
    except Exception as e:
        return False, f"Configuration validation error: {str(e)}"


def create_capture_card_camera(config, region=None):
    is_valid, error_msg = validate_capture_card_config(config)
    if not is_valid:
        raise ValueError(f"Invalid capture card configuration: {error_msg}")
    return CaptureCardCamera(config, region)


def get_default_capture_card_config() -> dict:
    return {
        "capture_width": 1920,
        "capture_height": 1080,
        "capture_fps": 240,
        "capture_device_index": 0,
        "capture_format": "MJPG",
        "capture_buffer_mb": 64,
        "capture_fourcc_preference": ["MJPG"],
        "capture_range_x": 0,
        "capture_range_y": 0,
        "capture_offset_x": 0,
        "capture_offset_y": 0,
        "capture_center_offset_x": 0,
        "capture_center_offset_y": 0,
    }


def apply_capture_card_config(config, **kwargs):
    valid_keys = {
        "capture_width", "capture_height", "capture_fps",
        "capture_device_index", "capture_format", "capture_buffer_mb",
        "capture_fourcc_preference", "capture_range_x", "capture_range_y",
        "capture_offset_x", "capture_offset_y",
        "capture_center_offset_x", "capture_center_offset_y",
    }
    for key, value in kwargs.items():
        if key in valid_keys:
            setattr(config, key, value)
