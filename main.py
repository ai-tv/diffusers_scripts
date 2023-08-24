import os
import json
import random
from threading import Lock
import traceback

import torch
from fastapi import FastAPI, Request, HTTPException

from diffuser_scripts.utils.logger import logger, dump_image_to_dir, dump_request_to_file
from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig
from diffuser_scripts.pipelines.couple import latent_couple_with_control
from diffuser_scripts.tasks import LatentCoupleWithControlTaskParams, ImageGenerationResult


log_dir = 'log'
app = FastAPI()
default_model_path = "configs/default_model_infos.json"
default_pipeline_path = "configs/latent_couple_config.json"
pipelin_config = LatentCoupleConfig.from_json(default_pipeline_path)
if os.path.exists(default_model_path):
    model_config = json.load(open(default_model_path))
model_manager = LatentCouplePipelinesManager(config=pipelin_config, model_config=model_config)


def get_lora_path(name):
    for lora_config in model_config['loras']:
        lora_prefix = lora_config['prefix']
        lora_suffix = lora_config['suffix']
        lora_path = f'{lora_prefix}/{name}{lora_suffix}'.format()
        if os.path.exists(lora_path):
            return lora_path
    else:
        raise ValueError("%s cannot be found under config %s" % (name, model_config['loras']))


def get_control_images(params):
    image = params.condition_image_np
    annotators = model_manager.annotators
    annotator_names = params.control_annotators
    if params.control_image_type == 'processed':
        control_image = image
        dump_image_to_dir(control_image, log_dir, name='condition')
    elif params.control_image_type == 'original':
        if isinstance(annotator_names, str):
            control_image = annotators[annotator_names](image)
            dump_image_to_dir(control_image, log_dir, name='condition')
        elif isinstance(annotator_names, list):
            control_image = [annotators[n](image) for n in annotator_names]
            for c in control_image:
                dump_image_to_dir(c, log_dir, name='condition')
        else:
            raise ValueError(f"bad value for annotator={annotator_names} or control_type={params.control_image_type}")
    else:
        raise ValueError(f"bad value for annotator={annotator_names} or control_type={params.control_image_type}")
    return control_image


@app.post("/get_latent_couple")
async def handle_latent_couple (request: Request):
    data = await request.json()
    params = LatentCoupleWithControlTaskParams(**data)
    logger.info("got request, %s" % (params.prompt, ))
    lora_configs = [{ get_lora_path(k): v for k, v in config.items()} for config in params.lora_configs]
    params.random_seed = random.randrange(0, 1<<63) if params.random_seed < 0 else params.random_seed
    dump_request_to_file(params, 'log')
    control_image = get_control_images(params)

    if len(params.prompt) != len(model_manager.pipelines) or len(params.negative_prompt) != len(model_manager.pipelines):
        raise HTTPException(status_code=400, detail="prompt or negative prompt must be a list of %d" % (len(model_manager.pipelines), ))

    with model_manager.lock:
        try:
            logger.info("loading loras: %s" % lora_configs, )
            model_manager.load_loras(lora_configs)
            model_manager.set_controlnet(controlnet_path=params.control_model_name)
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
                control_mode = params.control_mode,
                control_guidance_start = params.control_guidance_start,
                control_guidance_end = params.control_guidance_end,
                controlnet_conditioning_scale = params.control_guidance_scale,
                main_prompt_decay = params.latent_mask_weight_decay,
                control_preprocess_mode = params.control_preprocess_mode,
                generator = torch.Generator(device='cuda').manual_seed(params.random_seed),
                control_scale_decay_ratio = params.control_scale_decay_ratio
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            logger.info("unload loras ...")
            model_manager.unload_loras()

    logger.info("generation succeed for %s" % (params.prompt, ))
    dump_image_to_dir(result, 'log', name='result')
    if result is None:
        return {"result": result}
    else:
        return ImageGenerationResult.from_task_and_image(params, result).json
