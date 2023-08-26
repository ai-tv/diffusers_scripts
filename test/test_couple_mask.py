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
from diffuser_scripts.annotators.dwpose import DWposeDetector
from diffuser_scripts.utils.image_process import detectmap_proc

from dm.nn.yolo_mask_annotator import YoloPredictor
    
yolo = YoloPredictor("/mnt/lg102/zwshi/model_zoo/yolov8x-seg.pt")

def process_masks(masks, h, w, weights=0.7, device='cuda'):
    new_masks = []
    xs = []
    intersect = None
    for mask in masks:
        y, x = np.where(mask)
        xs.append(x.mean())
        mask = cv2.resize(mask, (w//8, h//8), )
        intersect = mask > 0 if intersect is None else intersect & (mask > 0)
        new_masks.append(mask)

    for i, mask in enumerate(new_masks):
        mask = np.where(
            mask > 0, 
            mask * weights, 
            np.where(intersect, np.zeros_like(mask), np.ones_like(mask) * (1 - weights) / 2))
        new_masks[i] = mask

    xs, new_masks = zip(*sorted(zip(xs, new_masks)))
    new_masks = [1-sum(new_masks)] + list(new_masks)
    return [
        torch.FloatTensor(mask[None, ...]).to(device)
        for mask in new_masks
    ]


@fire.Fire
def test_pipeline(output_name="default", random_seed=0):
    os.makedirs('test_result/%s' % output_name, exist_ok=True)
    default_model_path = "configs/default_model_infos.json"
    default_pipeline_path = "configs/latent_couple_config_canny02.json"
    pipelin_config = LatentCoupleConfig.from_json(default_pipeline_path)
    if os.path.exists(default_model_path):
        model_config = json.load(open(default_model_path))
    model_manager = LatentCouplePipelinesManager(config=pipelin_config, model_config=model_config)

    lora_configs = [
        {"couple_0614_001-000001": 1},
        {"couple_0614_001-000001": 1},
        {"couple_0614_001-000001": 1}
    ]
    lora_configs = [{os.path.join('/mnt/2T/zwshi/model_zoo/%s.safetensors' % k): v for k, v in config.items()} for config in lora_configs]
    model_manager.load_loras(lora_configs)
    # model_manager.set_controlnet(['lllyasviel/control_v11p_sd15_canny', 'lllyasviel/control_v11p_sd15_openpose'])
    model_manager.set_controlnet('lllyasviel/control_v11p_sd15_canny')
    # dw_pose = DWposeDetector()

    tasks = sorted(glob.glob("test_images/*.json"))
    tasks_conditions = sorted(glob.glob("test_images/*.png"))
    tasks_ori = sorted(glob.glob("test_images/original/*.png"), key=lambda x: (len(x), x))
    tasks_ori = [i for t in tasks_ori for i in [t, t]]
    for i, (task, condition, o) in enumerate(zip(tasks, tasks_conditions, tasks_ori)):
        params = LatentCoupleWithControlTaskParams.from_json(task)
        control_image = cv2.imread(condition)
        o = cv2.imread(o)
        _, original_image = detectmap_proc(o, params.height, params.width)
        _, masks = yolo.draw_person_masks(original_image)
        # masks = process_masks(masks, params.height, params.width, weights=0.9)
        result, _ = latent_couple_with_control(
            pipes = model_manager.pipelines,
            prompts = params.prompt,
            image = control_image, #Image.fromarray(cv2.imread('image.png', cv2.IMREAD_COLOR)),
            # couple_pos = params.latent_pos,
            # couple_weights = params.latent_mask_weight, 
            couple_mask_list=masks,
            negative_prompts = params.negative_prompt, 
            height = params.height,
            width = params.width,
            guidance_scale = params.guidance_scale,
            num_inference_steps = params.num_inference_steps,
            control_mode = 'prompt',
            control_guidance_start = params.control_guidance_start,
            control_guidance_end = params.control_guidance_end,
            controlnet_conditioning_scale = params.control_guidance_scale,
            main_prompt_decay = 0.03,
            latent_couple_max_ratio = 0.9,
            latent_couple_min_ratio = 0.1,
            control_preprocess_mode = params.control_preprocess_mode,
            generator=torch.Generator(device='cuda').manual_seed(random_seed + i)
        )
        result.save('test_result/%s/%d.jpg' % (output_name, i + random_seed))

# if __name__ == '__main__':
#     test_post_app()
