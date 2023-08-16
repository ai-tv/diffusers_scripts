import os
import time
import json
import logging
from PIL import Image
from .tasks import LatentCoupleWithControlTaskParams

logging.basicConfig(
    format='[%(asctime)s %(name)-8s'
           '%(levelname)s %(process)d '
           '%(filename)s:%(lineno)-5d]'
           ' %(message)s',
    level=logging.INFO
)

logger = logging#.getLogger()


def dump_image_to_dir(image: Image.Image, output_dir: str, name=''):
    timestamp = str(int(time.time()))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, timestamp + '_%s.png' % name)
    image.save(output_path)

def dump_request_to_file(task: LatentCoupleWithControlTaskParams, output_dir: str, name='task'):
    timestamp = str(int(time.time()))
    obj = task.json
    if 'condition_img_str' in obj:
        del obj['condition_img_str']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, timestamp + '_%s.json' % name)
    with open(output_path, 'w') as f:
        json.dump(obj, f, indent=4)