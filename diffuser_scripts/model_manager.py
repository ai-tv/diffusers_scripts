import json
import typing as T
from dataclasses import dataclass
from threading import Lock

import torch
from diffuser_scripts.utils.lora_loader import LoraLoader


default_model_infos = {
    'chilloutmix': {
        'load_method': 'pretrained',
        'local_path': '~/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/',
        'path': 'emilianJR/chilloutmix_NiPrunedFp32Fix'
    },
}


@dataclass
class LatentCoupleConfig:

    model_names: T.List 
    use_controlnet: bool = True
    default_controlnet_name: str = 'lllyasviel/control_v11p_sd15_canny'

    @staticmethod
    def from_json(json_path):
        obj = json.load(open(json_path))
        return LatentCoupleConfig(**obj)    


def load_latent_couple_pipeline(
    latent_couple_config: LatentCoupleConfig,
    model_infos: T.Dict = default_model_infos
):
    config = latent_couple_config
    if config.use_controlnet:
        control_model = ControlNetModel.from_pretrained(
            config.default_controlnet_name, torch_dtype=torch.float16)

    load_method = {
        'pretrained': StableDiffusionPipeline.from_pretrained,
        'single_file': StableDiffusionPipeline.from_single_file
    }

    model_info = model_infos[config.model_names[0]]
    main_pipe = load_method[model_info['load_method']](
        model_info['local_path'],
        torch_dtype=torch.float16,
        load_safety_checker=False,
        local_files_only=True,
        safety_checker = None
    )
    scheduler = DPMSolverMultistepScheduler.from_config(main_pipe.scheduler.config, use_karras_sigmas=True)
    scheduler.config.algorithm_type = 'sde-dpmsolver++'
    pipes = []
    for i, name in enumerate(config.model_names):
        if i == 0:
            unet = main_pipe.unet
            text_encoder = main_pipe.text_encoder
        else:
            model_info = model_infos[name]
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
            scheduler = scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False
        )
        pipe.to('cuda')
        pipes.append(pipe)
    
    del main_pipe
    return pipes


class LatentCouplePipelinesManager:

    def __init__(self, config: LatentCoupleConfig):
        self.pipelines = load_latent_couple_pipeline(config)
        self.lora_loader = LoraLoader()
        self.lora_status = [{} for _ in self.pipelines]
        self.lock = Lock()

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
        for i, lora_status in self.lora_status.items():
            unload_configs.append({k: -v for k, v in lora_status.items()})
        self.load_loras(unload_configs)

