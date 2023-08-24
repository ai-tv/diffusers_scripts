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
    model_manager.set_controlnet('lllyasviel/control_v11p_sd15_openpose')
    dw_pose = DWposeDetector()

    tasks = sorted(glob.glob("test_images/*.json"))
    tasks_conditions = sorted(glob.glob("test_images/*.png"))
    tasks_ori = sorted(glob.glob("test_images/original/*.png"), key=lambda x: (len(x), x))
    tasks_ori = [i for t in tasks_ori for i in [t, t]]
    for i, (task, condition, o) in enumerate(zip(tasks, tasks_conditions, tasks_ori)):
        params = LatentCoupleWithControlTaskParams.from_json(task)
        control_image = cv2.imread(condition)
        o = cv2.imread(o)
        control_image_pose = dw_pose(o)
        annotation = cv2.addWeighted(o, 0.3, control_image_pose, 0.7, 0)
        cv2.imwrite('test_result/%s/%d_cond.jpg' % (output_name, i + random_seed), annotation)

        result = latent_couple_with_control(
            pipes = model_manager.pipelines,
            prompts = params.prompt,
            image = control_image_pose, #Image.fromarray(cv2.imread('image.png', cv2.IMREAD_COLOR)),
            couple_pos = params.latent_pos,
            couple_weights = params.latent_mask_weight, 
            negative_prompts = params.negative_prompt, 
            height = params.height,
            width = params.width,
            guidance_scale = params.guidance_scale,
            num_inference_steps = params.num_inference_steps,
            control_mode = 'balance',
            control_guidance_start = params.control_guidance_start,
            control_guidance_end = params.control_guidance_end,
            controlnet_conditioning_scale = params.control_guidance_scale,
            main_prompt_decay = params.latent_mask_weight_decay,
            control_preprocess_mode = params.control_preprocess_mode,
            generator=torch.Generator(device='cuda').manual_seed(random_seed + i)
        )
        result.save('test_result/%s/%d.jpg' % (output_name, i + random_seed))

# if __name__ == '__main__':
#     test_post_app()
