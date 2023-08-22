import os
import sys
import time
import json
from loguru import logger

import cv2
from PIL import Image

logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


def dump_image_to_dir(image: Image.Image, output_dir: str, name=''):
    timestamp = str(int(time.time()))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, timestamp + '_%s.png' % name)
    if isinstance(image, Image.Image):
        image.save(output_path)
    else:
        cv2.imwrite(output_path, image)


def dump_request_to_file(task, output_dir: str, name='task'):
    timestamp = str(int(time.time()))
    obj = task.json
    if 'condition_img_str' in obj:
        del obj['condition_img_str']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, timestamp + '_%s.json' % name)
    with open(output_path, 'w') as f:
        json.dump(obj, f, indent=4)