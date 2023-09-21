import os
import copy
import json
import random
from threading import Lock
import traceback

import torch
import numpy as np
from fastapi import FastAPI, Request, HTTPException

from diffuser_scripts.utils.logger import logger, dump_image_to_dir, dump_request_to_file
from diffuser_scripts.model_manager import LatentCouplePipelinesManager, LatentCoupleConfig
from diffuser_scripts.tasks import LatentCoupleWithControlTaskParams, ImageGenerationResult
from diffuser_scripts.handler import handle_latent_couple


log_dir = 'log'
app = FastAPI()
default_model_path = "configs/default_model_infos.json"
default_pipeline_path = "configs/latent_couple_config_ad.json"
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
    raise ValueError("%s cannot be found under config %s" % (name, model_config['loras']))


@app.post("/get_latent_couple")
async def handle_get_latent_couple(request: Request):
    try:
        ### 1. setup generation params
        # torch.cuda.empty_cache()
        data = await request.json()
        params = LatentCoupleWithControlTaskParams(**data)
        logger.info("ad_loras: %s" % (params.ad_lora_configs, ))
        lora_configs = [{ get_lora_path(k): v for k, v in config.items()} for config in params.lora_configs]
        ad_lora_configs = [{ get_lora_path(k): v for k, v in config.items()} for config in params.ad_lora_configs]
        params.random_seed = random.randrange(0, 1<<63) if params.random_seed < 0 else params.random_seed
        request_obj = params.json
        for k in ('id_reference_img', 'condition_img_str', ):
            del request_obj[k]
        logger.info("%s got request, %s" % (params.uniq_id, json.dumps(request_obj), ))
        dump_request_to_file(params, 'log')
        if len(params.prompt) != len(model_manager.pipelines) or len(params.negative_prompt) != len(model_manager.pipelines):
            raise HTTPException(status_code=400, detail="prompt or negative prompt must be a list of %d" % (len(model_manager.pipelines), ))
        return handle_latent_couple(
            model_manager, 
            params, 
            lora_configs,
            ad_lora_configs, 
            log_dir=log_dir
        ).json
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/get_txt2img_control")
async def handle_control(request: Request):
    try:
        ### 1. setup generation params
        data = await request.json()
        params = LatentCoupleWithControlTaskParams(**data)
        lora_configs = [{ get_lora_path(k): v for k, v in config.items()} for config in params.lora_configs]
        params.random_seed = random.randrange(0, 1<<63) if params.random_seed < 0 else params.random_seed
        request_obj = params.json
        for k in ('id_reference_img', 'condition_img_str', ):
            del request_obj[k]
        logger.info("%s got request, %s" % (params.uniq_id, json.dumps(request_obj), ))
        dump_request_to_file(params, 'log')
        if len(params.prompt) != len(model_manager.pipelines) or len(params.negative_prompt) != len(model_manager.pipelines):
            raise HTTPException(status_code=400, detail="prompt or negative prompt must be a list of %d" % (len(model_manager.pipelines), ))
        return handle_control(model_manager, params, lora_configs, log_dir=log_dir)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/get_status")
async def is_alive(request: Request):
    return {'status': 'ok' if model_config.check_correctness else 'undefined'}
