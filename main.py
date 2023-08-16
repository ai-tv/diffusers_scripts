import os
import json
import random
from threading import Lock
import traceback

import torch
from fastapi import FastAPI, Request, HTTPException

from diffuser_scripts.logger import logger, dump_image_to_dir, dump_request_to_file
from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig
from diffuser_scripts.pipelines.couple import latent_couple_with_control
from diffuser_scripts.tasks import LatentCoupleWithControlTaskParams, ImageGenerationResult


app = FastAPI()
default_model_path = "configs/default_model_infos.json"
default_pipeline_path = "configs/latent_couple_config.json"
pipelin_config = LatentCoupleConfig.from_json(default_pipeline_path)
if os.path.exists(default_model_path):
    model_config = json.load(open(default_model_path))
model_manager = LatentCouplePipelinesManager(config=pipelin_config, model_config=model_config)


@app.post("/get_latent_couple")
async def handle_latent_couple (request: Request):
    data = await request.json()
    params = LatentCoupleWithControlTaskParams(**data)
    control_image = params.condition_image
    lora_configs = [{os.path.join('/mnt/2T/zwshi/model_zoo/%s.safetensors' % k): v for k, v in config.items()} for config in params.lora_configs]
    random_seed = random.randrange(0, 1<<63) if params.random_seed < 0 else params.random_seed

    dump_request_to_file(params, 'log')
    dump_image_to_dir(control_image, 'log', name='condition')
    if len(params.prompt) != len(model_manager.pipelines) or len(params.negative_prompt) != len(model_manager.pipelines):
        raise HTTPException(status_code=400, detail="prompt or negative prompt must be a list of %d" % (len(model_manager.pipelines), ))

    with model_manager.lock:
        try:
            model_manager.load_loras(lora_configs)
            result = latent_couple_with_control(
                pipes = model_manager.pipelines,
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
                main_prompt_decay = params.latent_mask_weight_decay,
                generator=torch.Generator(device='cuda').manual_seed(random_seed)
            )
            dump_image_to_dir(result, 'log', name='result')
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            model_manager.unload_loras()

    if result is None:
        return {"result": result}
    else:
        return ImageGenerationResult.from_task_and_image(params, result).json
