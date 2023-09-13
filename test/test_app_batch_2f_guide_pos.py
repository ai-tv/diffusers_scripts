import os
import json
import time
import glob

import cv2
import fire
import torch
import requests
import numpy as np
from PIL import Image

from diffuser_scripts.tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams
from diffuser_scripts.utils.decode import encode_image_b64, decode_image_b64
from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig
from diffuser_scripts.pipelines.couple import latent_couple_with_control


# a = '/mnt/lg104/character_dataset/preprocessed_v0/张译/77.png'
# a = '/mnt/lg104/character_dataset/preprocessed_v0/刘德华/212.png'
a = '/mnt/lg104/character_dataset/preprocessed_v0/杨幂/2300.png'
b = '/mnt/lg104/character_dataset/preprocessed_v0/林心如/4.png'
a = Image.open(a)
b = Image.open(b)

@fire.Fire
def test_pipeline(test_name="2f_guide", port=1234, random_seed=0):
    tasks = sorted(glob.glob("test_images/2women/*.json"))
    tasks_conditions = sorted(glob.glob("test_images/2women/*.jpg"))
    # tasks = [tasks[26] for t in tasks]
    # tasks_conditions = [tasks_conditions[26] for t in tasks_conditions]
    for i, (task, condition) in enumerate(zip(tasks, tasks_conditions)):
        print(task)
        output_name = 'test_result/%s/%d_%d.jpg' % (test_name, i // 2, i % 2)
        if os.path.exists(output_name):
            continue
        obj = json.load(open(task))
        obj['lora_configs'] = [
            {"051_mix_with_vista-000002": 1},
            {"051_mix_with_vista-000002": 1},
            {"051_mix_with_vista-000002": 1}
        ]
        obj['ad_lora_configs'] = [
            {"035_mix-000008": 1},
            {"035_mix-000008": 1},
        ]
        obj['random_seed'] = i
        obj['sampler'] = 'dpm++'
        obj['add_pos_encode'] = [True, True, True]
        prompts = obj['prompt']

        if prompts[1] == 'yangmi':
            obj['id_reference_img'] = [None, encode_image_b64(b), encode_image_b64(a)]
        else:
            obj['id_reference_img'] = [None, encode_image_b64(a), encode_image_b64(b)]
        obj['add_id_feature'] = [True, True, True]
        obj['request_id'] = 'test_%s_%d' % (test_name, i)
        prompts[1] = prompts[1].replace('yangmi', '1woman').replace('ouyangnana', '1woman')
        prompts[2] = prompts[2].replace('yangmi', '1woman').replace('ouyangnana', '1woman')
        condition = encode_image_b64(Image.open(condition))
        obj['condition_img_str'] = condition
        # obj['control_guidance_scale'] = 0
        start = time.time()
        response = requests.post(
            "http://192.168.110.104:%s/get_latent_couple" % port, 
            data=json.dumps(obj)
        )
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_name), exist_ok=True)
            result = response.json()
            cost = time.time() - start
            print("200 ok, cost %.3fs, save result to %s" % (cost, output_name, ))
            result = ImageGenerationResult.from_json(result)
            image = result.get_generated_images(channel_reverse=True)
            image.save(output_name.replace('.png', '.jpg'))
            debugs = result.get_intermediate_states(channel_reverse=True)
            print("image saved")
        else:
            print("Error:", response.text)

# if __name__ == '__main__':
#     test_post_app()
