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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from sbp.io import SequenceFileReader, read_list
from sbp.io.common.encoder_decoder import decode_b64
from sbp.vision.image_viz import cat_pil, resize_pil

# from diffuser_scripts.tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams
# from diffuser_scripts.utils.decode import encode_image_b64, decode_image_b64
from diffuser_scripts.utils.long_prompt_weighting import get_weighted_text_embeddings
# from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig, LoraLoader
from diffuser_scripts.utils.lora_loader import LoraLoader
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline


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
    pipeline = StableDiffusionPipeline.from_pretrained(
        "/home/zwshi/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/",
        # torch_dtype=torch.float16,
        load_safety_checker=False,
        local_files_only=True,
        safety_checker = None
    ).to('cuda')
    # pipeline = StableDiffusionLongPromptWeightingPipeline(
    #     text_encoder=pipeline.text_encoder,
    #     vae=pipeline.vae,
    #     unet=pipeline.unet,
    #     tokenizer=pipeline.tokenizer,
    #     scheduler=pipeline.scheduler,
    #     clip_skip=1,
    #     safety_checker=None,
    #     feature_extractor=None,
    #     requires_safety_checker=False,
    # ).to('cuda')
    scheduler = DPMSolverMultistepScheduler(        
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        algorithm_type='dpmsolver++'
    )
    # scheduler.config.algorithm_type = 'sde-dpmsolver++'
    pipeline.scheduler = scheduler

    id_mlp_path = '/mnt/lg102/zwshi/projects/core/lora-scripts/tasks/output/033_cropface_relax1.0/033_cropface_relax1.0-000009.safetensors'
    lora_loader = LoraLoader()
    lora_loader.load_lora_for_pipelines([pipeline], [{id_mlp_path: 1}])
    id_mlp = torch.load(id_mlp_path + '_mlp.pt').to('cuda').eval()
    ssf = SequenceFileReader('/mnt/lg104/zwshi/data/character/stars/prossessed_v2/result.list')

    tasks = sorted(glob.glob("test_images/*.json"))
    tasks_conditions = sorted(glob.glob("test_images/*.png"))
    prompts = [
        'a photo of a man, facing front, colored, beautilful, full body',
        'a photo of a man, facing camera, colored, beautilful, full body',
        'a photo of a man, frontal face, colored, beautilful, full body',
        'a photo of a man, facing left, colored, beautilful, full body',
        'a photo of a man, facing right, colored, beautilful, full body',
        'a photo of a man, facing up, colored, beautilful, full body',
        'a photo of a man, facing down, colored, beautilful, full body',
    ]
    negative_prompt = 'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)'
    
    rows = []
    with torch.no_grad():
        for k, s in [
            ('/mnt/lg104/character_dataset/preprocessed_v0/万茜/34.png', 'woman'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/林心如/4.png', 'woman'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/杨幂/2300.png', 'woman'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/张雨绮/2.png', 'woman'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/胡歌/99.png', 'man'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/张国荣/8.png', 'man'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/谢霆锋/340.png', 'man'),
            ('/mnt/lg104/character_dataset/preprocessed_v0/周杰伦/269.png', 'man'),
        ]:
            f = torch.FloatTensor(decode_feature(ssf.read(k)))[None, ...].cuda()
            f = id_mlp(f)
            # print(f)
            results = []
            for i, prompt in enumerate(prompts):        
                text_embedding, uncond_embedding = get_weighted_text_embeddings(pipeline, prompt, negative_prompt)
                text_embedding = torch.cat([f[None, ], text_embedding, ], dim=1)
                uncond_embedding = torch.cat([f[None, ], uncond_embedding], dim=1)
                result = pipeline(
                    prompt_embeds = text_embedding,
                    negative_prompt_embeds = uncond_embedding,
                    # prompt = prompt.replace('man', s),
                    # negative_prompt = negative_prompt,
                    height = 768,
                    width = 512,
                    guidance_scale = 8.0,
                    num_inference_steps = 30,
                    # id_features=f,
                    generator=torch.Generator(device='cuda').manual_seed(random_seed + i),
                ).images[0]
                results.append(result)
                
            ref = Image.open(k)
            result = cat_pil([resize_pil(ref, 768)] + results)
            result.save('test_result/%s/%s.jpg' % (output_name, k.split('/')[-2]))
            rows.append(result)
    result = cat_pil(rows, vertical=True)
    result.save('test_result/%s/grid.jpg' % (output_name, ))

# if __name__ == '__main__':
#     test_post_app()
