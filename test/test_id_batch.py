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

from sbp.io import SequenceFileReader, read_list
from sbp.io.common.encoder_decoder import decode_b64
from sbp.vision.image_viz import cat_pil, resize_pil
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from diffuser_scripts.tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams
from diffuser_scripts.utils.decode import encode_image_b64, decode_image_b64
from diffuser_scripts.utils.long_prompt_weighting import get_weighted_text_embeddings
from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig, LoraLoader
from diffuser_scripts.pipelines.couple import latent_couple_with_control


def decode_feature(s):
    obj = json.loads(s)
    if len(obj) == 0:
        return np.zeros(512, dtype=np.float16)
    else:
        return decode_b64(obj[0]['embedding'].encode())  


SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


@fire.Fire
def test_pipeline(output_name="default", random_seed=0):
    os.makedirs('test_result/%s' % output_name, exist_ok=True)
    default_model_path = "configs/default_model_infos.json"
    default_pipeline_path = "configs/latent_couple_config.json"
    pipelin_config = LatentCoupleConfig.from_json(default_pipeline_path)
    if os.path.exists(default_model_path):
        model_config = json.load(open(default_model_path))
    model_manager = LatentCouplePipelinesManager(config=pipelin_config, model_config=model_config)
    
    id_lora_path = '/mnt/lg102/zwshi/projects/core/lora-scripts/tasks/output/022_larger/022_larger-000029.safetensors'
    id_mlp_path = id_lora_path + '_mlp.pt'
    lora_loader = LoraLoader()
    lora_loader.load_lora_for_pipelines(model_manager.pipelines, [{id_lora_path: 1} for _ in range(3)])
    id_mlp = torch.load(id_mlp_path).to('cuda').eval()

    ssf = SequenceFileReader('/mnt/lg104/zwshi/data/character/stars/prossessed_v2/result.list')
    tasks = sorted(glob.glob("test_images/*.json"))
    tasks_conditions = sorted(glob.glob("test_images/*.png"))
    negative_prompt = 'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)'

    scheduler = DPMSolverMultistepScheduler(        
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        algorithm_type='dpmsolver++'
    )
    # pipe = model_manager.get_sd(0)
    for pipe in model_manager.pipelines:
        pipe.scheduler = scheduler

    # k1 = "/mnt/lg104/character_dataset/preprocessed_v0/万茜/34.png" #ids[i % len(ids)]
    k1 = "/mnt/lg104/character_dataset/preprocessed_v0/娄艺潇/4.png"
    # k2 = "/mnt/lg104/character_dataset/preprocessed_v0/张国荣/8.png"
    k2 = "/mnt/lg104/character_dataset/preprocessed_v0/马嘉祺/141.png"

    f1 = torch.FloatTensor(decode_feature(ssf.read(k1)))[None, ...].cuda()
    f1 = id_mlp(f1)
    f2 = torch.FloatTensor(decode_feature(ssf.read(k2)))[None, ...].cuda()
    f2 = id_mlp(f2)
    ref1 = Image.open(k1)
    ref2 = Image.open(k2)

    for i, (task, condition) in enumerate(zip(tasks, tasks_conditions)):
        control_image = cv2.imread(condition)
        control_image_pil = Image.fromarray(control_image)
        params = LatentCoupleWithControlTaskParams.from_json(task)
        params.prompt[1] = params.prompt[1].replace('yangmi', '1woman').replace('baijingting', '1man')
        params.prompt[2] = params.prompt[2].replace('yangmi', '1woman').replace('baijingting', '1man')
        if False:
            text_embedding, uncond_embedding = get_weighted_text_embeddings(model_manager.pipelines[0], params.prompt[0], negative_prompt)
            text_embedding = torch.cat([f[None, ], text_embedding, ], dim=1)
            uncond_embedding = torch.cat([f[None, ], uncond_embedding], dim=1)
            result = model_manager.pipelines[0](
                prompt_embeds = text_embedding,
                negative_prompt_embeds = uncond_embedding,
                # prompt = prompt.replace('man', s),
                # negative_prompt = negative_prompt,
                image = control_image_pil,
                height = 768,
                width = 1024,
                guidance_scale = 8.0,
                num_inference_steps = 30,
                controlnet_conditioning_scale = 0.8,
                control_guidance_end = 0.5,
                generator=torch.Generator(device='cuda').manual_seed(random_seed + i),
            ).images[0]
        result = latent_couple_with_control(
            pipes = model_manager.pipelines,
            prompts = params.prompt,
            image = control_image, #Image.fromarray(cv2.imread('image.png', cv2.IMREAD_COLOR)),
            couple_pos = params.latent_pos,
            couple_weights = params.latent_mask_weight, 
            negative_prompts = params.negative_prompt, 
            id_features=[torch.zeros_like(f1), f1, f2] if params.prompt[1] == '1woman' else [torch.zeros_like(f1), f2, f1],
            height = params.height,
            width = params.width,
            guidance_scale = params.guidance_scale,
            num_inference_steps = params.num_inference_steps,
            control_mode = params.control_mode,
            control_guidance_start = params.control_guidance_start,
            control_guidance_end = params.control_guidance_end,
            controlnet_conditioning_scale = params.control_guidance_scale,
            main_prompt_decay = params.latent_mask_weight_decay,
            control_preprocess_mode = params.control_preprocess_mode,
            generator=torch.Generator(device='cuda').manual_seed(random_seed + i)
        )
        result = cat_pil([resize_pil(ref1, 768), resize_pil(ref2, 768), result])
        result.save('test_result/%s/%d.jpg' % (output_name, i + random_seed))

# if __name__ == '__main__':
#     test_post_app()
