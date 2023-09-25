import os
import json
import time

import fire
import torch
import requests
import numpy as np
from PIL import Image
from sbp.vision.transform import resize_shortest

from diffuser_scripts.tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams
from diffuser_scripts.utils.decode import encode_image_b64, decode_image_b64
from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig
from diffuser_scripts.pipelines.couple import latent_couple_with_control


@fire.Fire
def test_pipeline(random_seed=-1):
    import cv2
    image = np.array(Image.open("images/lc_ball_canny_2.png"))[..., None]
    image = np.array(Image.open("images/lc_test_canny.png"))
    # image = cv2.imread('ori.jpg')
    # image = cv2.resize(image, (1024, 768))
    # image = cv2.Canny(image, 100, 200)[..., None]
    # image = image
    print(image.shape)
    # image = np.array(Image.open("ball_canny.png"))[..., None]

    data = {
        'prompt': [
            '1woman and 1man, drinking',
            'baijingting',
            'yangmi',
        ],
        "negative_prompt": [
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
        ],
        "base_model": ['chilloutmix', 'chilloutmix', 'chilloutmix'],
        "width": 1024,
        "height": 768,
        "lora_configs": [
            {"couple_0614_001-000001": 1},
            {"couple_0614_001-000001": 1},
            {"couple_0614_001-000001": 1}
        ],
        "condition_img_str": encode_image_b64(image),
        "control_guidance_scale": 1.0,
        "control_guidance_end": 0.5,
        "latent_mask_weight": [0.7, 0.3, 0.3],
        "latent_pos": ["1:1-0:0","1:32-0:0-14","1:32-0:14-32"],
        # "latent_pos": ["1:1-0:0","1:2-0:0","1:2-0:1"],
        "random_seed": random_seed,
        "latent_mask_weight_decay": 0.03,
        "control_mode": "prompt"
    }
    # data1 = {
    #     'prompt': [
    #         '1woman and 1man, working',
    #         'baijingting',
    #         'yangmi',
    #     ],
    #     "negative_prompt": [
    #         'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
    #         'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
    #         'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
    #     ],
    #     "base_model": ['chilloutmix', 'chilloutmix', 'chilloutmix'],
    #     "width": 1024,
    #     "height": 768,
    #     "lora_configs": [
    #         {"couple_0614_001-000001": 1},
    #         {"couple_0614_001-000001": 1},
    #         {"couple_0614_001-000001": 1}
    #     ],
    #     "condition_img_str": encode_image_b64(image),
    #     "control_guidance_scale": 1.0,
    #     "control_guidance_end": 0.5,
    #     "latent_mask_weight": [0.7, 0.3, 0.3],
    #     "latent_pos": ["1:1-0:0","1:32-0:0-17","1:32-0:17-32"],
    #     # "latent_pos": ["1:1-0:0","1:2-0:0","1:2-0:1"],
    #     "random_seed": random_seed,
    #     "latent_mask_weight_decay": 0.03,
    #     "control_mode": "prompt"
    # }
    # data2 = {
    #     'prompt': [
    #         '1woman and 1man, playing ball',
    #         'baijingting',
    #         'yangmi',
    #     ],
    #     "negative_prompt": [
    #         'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
    #         'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
    #         'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
    #     ],
    #     "base_model": ['chilloutmix', 'chilloutmix', 'chilloutmix'],
    #     "width": 1024,
    #     "height": 768,
    #     "lora_configs": [
    #         {"couple_0614_001-000001": 1},
    #         {"couple_0614_001-000001": 1},
    #         {"couple_0614_001-000001": 1}
    #     ],
    #     "condition_img_str": encode_image_b64(image),
    #     "control_guidance_scale": 1.0,
    #     "control_guidance_end": 0.5,
    #     "latent_mask_weight": [0.7, 0.3, 0.3],
    #     "latent_pos": ["1:1-0:0","1:32-0:0-15","1:32-0:15-32"],
    #     # "latent_pos": ["1:1-0:0","1:2-0:0","1:2-0:1"],
    #     "random_seed": random_seed,
    #     "latent_mask_weight_decay": 0.03,
    #     "control_mode": "prompt"
    # }
    default_model_path = "configs/default_model_infos.json"
    default_pipeline_path = "configs/latent_couple_config.json"
    pipelin_config = LatentCoupleConfig.from_json(default_pipeline_path)
    if os.path.exists(default_model_path):
        model_config = json.load(open(default_model_path))
    model_manager = LatentCouplePipelinesManager(config=pipelin_config, model_config=model_config)

    params = LatentCoupleWithControlTaskParams(**data)
    control_image = params.condition_image
    lora_configs = [{os.path.join('/mnt/2T/zwshi/model_zoo/%s.safetensors' % k): v for k, v in config.items()} for config in params.lora_configs]
    model_manager.load_loras(lora_configs)
    for i in range(4):
        result = model_manager.pipelines[0](
            prompt = params.prompt[0],
            image = control_image, #Image.fromarray(cv2.imread('image.png', cv2.IMREAD_COLOR)),
            # couple_pos = params.latent_pos,
            # couple_weights = params.latent_mask_weight, 
            negative_prompt = params.negative_prompt[0], 
            height = params.height,
            width = params.width,
            guidance_scale = params.guidance_scale,
            num_inference_steps = params.num_inference_steps,
            # control_mode = params.control_mode,
            control_guidance_start = params.control_guidance_start,
            control_guidance_end = params.control_guidance_end,
            controlnet_conditioning_scale = params.control_guidance_scale,
            # main_prompt_decay = params.latent_mask_weight_decay,
            generator=torch.Generator(device='cuda').manual_seed(100+i)
        )
        result.images[0].save('result_control_%d.jpg' % i)

# if __name__ == '__main__':
#     test_post_app()