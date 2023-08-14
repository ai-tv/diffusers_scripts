import os
import sys

import fire
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from sbp.io.common.utils import read_img_from_list
from diffuser_scripts.utils import LoraLoader
from diffuser_scripts.pipelines.couple import latent_couple_with_control


def _load_default_pipeline():
    control_model = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
    pipe_ = StableDiffusionPipeline.from_pretrained(
        '/home/zwshi/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/',
        torch_dtype=torch.float16,
        load_safety_checker=False,
        local_files_only=True,
        safety_checker = None
    )
    scheduler = DPMSolverMultistepScheduler.from_config(pipe_.scheduler.config, use_karras_sigmas=True)
    scheduler.config.algorithm_type = 'sde-dpmsolver++'
    pipes = []
    for i in range(3):
        if i == 0:
            unet = pipe_.unet
            text_encoder = pipe_.text_encoder
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                '/home/zwshi/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/text_encoder/',
                torch_dtype=torch.float16,
            )
            unet = UNet2DConditionModel.from_pretrained(
                "/home/zwshi/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/unet/", 
                torch_dtype=torch.float16
            )
        pipe = StableDiffusionControlNetPipeline(
            tokenizer = pipe_.tokenizer,
            text_encoder = text_encoder,
            vae = pipe_.vae,
            unet = unet,
            controlnet = control_model,
            scheduler = scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False
        )
        pipe.to('cuda')
        pipes.append(pipe)
    
    del pipe_
    return pipes


@fire.Fire
def _test(
    input_list,
    output_dir,
    lora_name: str = 'yangmi_ouyangnana'
):
    # cond = Image.open("lc_silence_canny_2.png")
    lora_configs_dict = {
        "baijingting_ouyangnana": [
            {
                "/mnt/2T/zwshi/model_zoo/couple_0614_001.safetensors": 0.5,
                "/mnt/2T/zwshi/model_zoo/0807_OYNN_YM_2female-000001.safetensors": 0.5
            },
            {'/mnt/2T/zwshi/model_zoo/couple_0614_001.safetensors': 1},
            {'/mnt/2T/zwshi/model_zoo/0807_OYNN_YM_2female-000001.safetensors': 1},
        ],
        "yangmi_ouyangnana": [
            {
                # "/mnt/2T/zwshi/model_zoo/couple_0614_001.safetensors": 0.5,
                "/mnt/2T/zwshi/model_zoo/0807_oynn_bjt_autotag-000001.safetensors": 1
            },
            {'/mnt/2T/zwshi/model_zoo/0807_oynn_bjt_autotag-000001.safetensors': 1},
            {'/mnt/2T/zwshi/model_zoo/0807_oynn_bjt_autotag-000001.safetensors': 1},

        ]
    }
    pipes = _load_default_pipeline()
    lora_loader = LoraLoader()
    lora_loader.load_lora_for_pipelines(pipes, lora_configs_dict['baijingting_ouyangnana'])

    os.makedirs(output_dir, exist_ok=True)
    for i, (k, cond) in enumerate(read_img_from_list(input_list, return_type='pil')):
        output_path = os.path.join(output_dir, os.path.basename(k))
        result = latent_couple_with_control(
            pipes, image=cond, height=768, width=1024,
            prompts = [
                '1man and 1woman',
                'baijingting',
                'ouyangnana',
            ],
            negative_prompts = 'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
            control_guidance_end=0.5,
            main_prompt_decay=0.01,
            generator=torch.Generator('cuda').manual_seed(i)
        )
        result.save(output_path.replace('.jpg', f'_${i}.jpg').replace('.png', f'_{i}.png'))
