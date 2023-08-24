import os
import copy
import json
import collections
import typing as T
from dataclasses import dataclass
from threading import Lock

import torch
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler, DPMSolverSDEScheduler, DPMSolverSinglestepScheduler

from diffuser_scripts.utils.lora_loader import LoraLoader
from diffuser_scripts.utils.logger import logger


@dataclass
class LatentCoupleConfig:

    model_names: T.List
    use_id_mlp: bool = False
    use_controlnet: bool = True
    default_id_mlp: str = None
    default_controlnet_name: str = 'lllyasviel/control_v11p_sd15_canny'

    @staticmethod
    def from_json(json_path):
        obj = json.load(open(json_path))
        for k in list(obj.keys()):
            if k.startswith('_'):
                del obj[k]
        return LatentCoupleConfig(**obj)


def retry(func, max_trial=-1):
    count = 0
    def _func(*args, **kw):
        nonlocal count
        try:
            return func(*args, **kw)    
        except Exception as e:
            if count == max_trial:
                raise e
            else:
                print(e)
                count += 1
                return _func(*args, **kw) 
    return _func



def load_latent_couple_pipeline(
    latent_couple_config: LatentCoupleConfig,
    model_infos: T.Dict
):
    config = latent_couple_config
    if config.use_controlnet:
        logger.info("loading controlnet ...")
        control_model = retry(ControlNetModel.from_pretrained)(
            config.default_controlnet_name, torch_dtype=torch.float16)

    load_method = {
        'pretrained': StableDiffusionPipeline.from_pretrained,
        'single_file': StableDiffusionPipeline.from_single_file
    }

    model_info = model_infos['base_models'][config.model_names[0]]

    logger.info("loading main pipe ...")
    main_pipe = load_method[model_info['load_method']](
        model_info['local_path'],
        torch_dtype=torch.float16,
        load_safety_checker=False,
        local_files_only=True,
        safety_checker = None
    )
        
    scheduler = DPMSolverMultistepScheduler.from_config(main_pipe.scheduler.config, use_karras_sigmas=True)
    scheduler.config.algorithm_type = 'sde-dpmsolver++'
    # scheduler = DPMSolverSinglestepScheduler.from_config(main_pipe.scheduler.config, use_karras_sigmas=True)
    pipes = []
    for i, name in enumerate(config.model_names):
        logger.info("setting pipe %d" % (i, ))
        if i == 0:
            unet = main_pipe.unet
            text_encoder = main_pipe.text_encoder
        else:
            model_info = model_infos['base_models'][name]
            text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(model_info['local_path'], 'text_encoder'),
                torch_dtype=torch.float16,
            )
            unet = UNet2DConditionModel.from_pretrained(
                os.path.join(model_info['local_path'], "unet"),
                torch_dtype=torch.float16
            )
        pipe = StableDiffusionControlNetPipeline(
            tokenizer = main_pipe.tokenizer,
            text_encoder = text_encoder,
            vae = main_pipe.vae,
            unet = unet,
            controlnet = control_model,
            scheduler = copy.deepcopy(scheduler),
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False
        )
        pipe.to('cuda')
        if i == 0 and config.use_id_mlp:
            from diffuser_scripts.net.id_mlp import ResMlp
            pipe.id_mlp = torch.load(config.default_id_mlp).cuda().eval()
        pipes.append(pipe)
    
    del main_pipe
    return pipes


class LatentCouplePipelinesManager:

    def __init__(self, config: LatentCoupleConfig, model_config: T.Dict):
        self.pipelines = load_latent_couple_pipeline(config, model_config)
        self.lora_loader = LoraLoader()
        self.lora_status = [collections.defaultdict(lambda : 0.0) for _ in self.pipelines]
        self.lock = Lock()

    def get_sd(self, i=0):
        p = self.pipelines[i]
        return StableDiffusionPipeline(
            vae = p.vae,
            text_encoder = p.text_encoder,
            tokenizer = p.tokenizer,
            unet = p.unet,
            scheduler = p.scheduler,
            safety_checker = p.safety_checker,
            feature_extractor = p.feature_extractor,
            requires_safety_checker = p.requires_safety_checker
        ).to(p.device)

    def set_controlnet(self, controlnet_path: T.Union[str, list]):
        logger.info("setting controlnet as %s ... " % (controlnet_path, ))
        if isinstance(controlnet_path, str):
            control_model = retry(ControlNetModel.from_pretrained)(controlnet_path, torch_dtype=torch.float16)
        else:
            multicontrol = []
            for path in controlnet_path:
                control_model = retry(ControlNetModel.from_pretrained)(path, torch_dtype=torch.float16)
                multicontrol.append(control_model)
            control_model = MultiControlNetModel(multicontrol)
        control_model.to(self.pipelines[0].device)
        for pipe in self.pipelines:
            pipe.controlnet = control_model

    def load_lora(self, i, lora, weight=1.0):
        pipe = self.pipelines[i]
        self.lora_status[i][lora] += weight
        self.lora_loader.load_lora_weights(pipe, lora, weight, 'cuda', torch.float32)
        if abs(self.lora_status[i][lora]) < 1e-6:
            del self.lora_status[i][lora]
    
    def load_loras(self, lora_configs):
        for i, lora_config in enumerate(lora_configs):
            for k, v in lora_config.items():
                self.load_lora(i, k, v)

    def unload_loras(self):
        unload_configs = []
        for i, lora_status in enumerate(self.lora_status):
            unload_configs.append({k: -v for k, v in lora_status.items()})
        self.load_loras(unload_configs)

