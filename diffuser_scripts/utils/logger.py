import os
import sys
import time
import json
from loguru import logger

import cv2
from PIL import Image

# logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add('./lc-{time:YYYY-MM-DD}.log', rotation='12:00', format="{time} {level} {message}")


def dump_image_to_dir(image: Image.Image, output_dir: str, name=''):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    if isinstance(image, Image.Image):
        image.save(output_path)
    else:
        cv2.imwrite(output_path, image)


def dump_request_to_file(task, output_dir: str, name='task'):
    timestamp = str(int(time.time()))
    request_id = task.uniq_id
    obj = task.json
    if 'condition_img_str' in obj:
        del obj['condition_img_str']
    if 'id_reference_img' in obj:
        del obj['id_reference_img']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, request_id + '_%s.json' % name)
    with open(output_path, 'w') as f:
        json.dump(obj, f, indent=4)