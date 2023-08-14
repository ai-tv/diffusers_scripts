import os
from threading import Lock

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel
from fastapi import FastAPI, Request

from diffuser_scripts.utils import LoraLoader
from diffuser_scripts.pipelines.couple import latent_couple_with_control
from diffuser_scripts.tasks import LatentCoupleWithControlTaskParams, ImageGenerationResult


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


app = FastAPI()
pipelines = _load_default_pipeline()
lora_loader = LoraLoader()
_lock = Lock()


@app.post("/get_latent_couple")
async def handle_latent_couple (request: Request):
    data = await request.json()
    params = LatentCoupleWithControlTaskParams(**data)
    control_image = params.condition_image
    lora_configs = [{os.path.join('/mnt/2T/zwshi/model_zoo/%s.safetensors' % k): v for k, v in config.items()} for config in params.lora_configs]
    with _lock:
        lora_loader.load_lora_for_pipelines(pipelines, lora_configs)
        result = latent_couple_with_control(
            pipes = pipelines,
            prompts = params.prompt,
            image = control_image,
            couple_pos = params.latent_pos,
            couple_weights = params.latent_mask_weight, 
            negative_prompts = params.negative_prompt, 
            height = params.height,
            width = params.width,
            guidance_scale = params.guidance_scale,
            num_inference_steps = params.num_inference_steps,
            control_guidance_start = params.control_guidance_start,
            control_guidance_end = params.control_guidance_end,
            controlnet_conditioning_scale = params.control_guidance_scale,
            main_prompt_decay = params.latent_mask_weight_decay
        )
        lora_loader.unload_lora_for_pipelines(pipelines, lora_configs)
        return ImageGenerationResult.from_task_and_image(params, result).json
