"""Read-only camera preview decoding and freshness checks."""
import math

import cv2
import numpy as np

VIEWS = {'main': '主相机', 'left': '左夹爪相机', 'right': '右夹爪相机'}
PREVIEW_FILES = {'main': 'camera.jpg', 'left': 'camera_left.jpg', 'right': 'camera_right.jpg'}


def decode_raw(msg):
    """Decode ROS Image with row padding; never reinterpret padding as pixels."""
    formats = {'rgb8': (3, cv2.COLOR_RGB2BGR), 'bgr8': (3, None),
               'rgba8': (4, cv2.COLOR_RGBA2BGR), 'bgra8': (4, cv2.COLOR_BGRA2BGR),
               'mono8': (1, cv2.COLOR_GRAY2BGR), 'yuv422': (2, cv2.COLOR_YUV2BGR_UYVY)}
    if msg.encoding not in formats:
        raise ValueError('Unsupported preview encoding: ' + msg.encoding)
    channels, conversion = formats[msg.encoding]
    width, height, step = msg.width, msg.height, msg.step
    if width <= 0 or height <= 0 or step < width * channels or len(msg.data) < step * height:
        raise ValueError('Invalid image dimensions or payload')
    rows = np.frombuffer(msg.data, dtype=np.uint8, count=step * height).reshape(height, step)
    pixels = rows[:, :width * channels].reshape(height, width, channels)
    return cv2.cvtColor(pixels, conversion) if conversion is not None else pixels


def write_preview(directory, view, frame):
    height, width = frame.shape[:2]
    scale = min(768 / width, 432 / height, 1.)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError('JPEG preview encoding failed')
    path = directory / PREVIEW_FILES[view]
    temp = path.with_suffix('.tmp')
    temp.write_bytes(encoded.tobytes())
    temp.replace(path)


def preview_status(state, view, now):
    """An old JPEG must not be displayed as a live camera after disconnect."""
    age = now - state.get('updated_unix_s', 0)
    if not math.isfinite(age) or not -.1 <= age <= .8:
        return False, '监视未连接'
    row = state.get('previews', {}).get(view, {})
    if view == 'main' and not row:
        row = state.get('main_camera', {})
    if view != 'main':
        health = state.get('wrist_cameras', {}).get(view, {})
        if health.get('health') not in ('ok', 'warning'):
            return False, health.get('label', '等待图像')
    if row.get('decode_error'):
        return False, '图像解码失败'
    if not row.get('frames_decoded'):
        return False, '等待图像'
    ages = (row.get('source_age_s'), row.get('received_age_s'))
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and -.1 <= v <= .8 for v in ages):
        return False, '图像中断 / 过期'
    return True, '实时画面'
