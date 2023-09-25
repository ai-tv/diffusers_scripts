import os
import copy
import json
import collections
import typing as T
from dataclasses import dataclass, field
from threading import Lock

import torch
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler, SchedulerMixin

from diffuser_scripts.utils.lora_loader import LoraLoader
from diffuser_scripts.utils.logger import logger
from diffuser_scripts.annotators import GuidanceProcessor


default_samplers = {
    'dpm++': DPMSolverMultistepScheduler(        
        num_train_timesteps = 1000,
        beta_start = 8.5e-4,
        beta_end = 1.2e-2,
        beta_schedule = 'scaled_linear',
        algorithm_type = 'dpmsolver++',
    ),
    'sde-dpm++': DPMSolverMultistepScheduler(
        beta_start = 8.0e-4,
        beta_end = 1.1e-2,
        beta_schedule = "scaled_linear",
        use_karras_sigmas = True,
        algorithm_type = 'sde-dpmsolver++'
    )
}


@dataclass
class GuidanceProcessConfig:

    control_annotator_names: T.List[str]
    control_annotator_args: T.Dict 
    subject_locator: str
    face_service_host: str = '192.168.110.102'
    face_id_mlp: str = None


@dataclass
class LatentCoupleConfig:

    model_names: T.List
    use_id_mlp: bool = False
    use_controlnet: bool = True
    default_controlnet_name: str = 'lllyasviel/control_v11p_sd15_canny'
    preprocessor_config: GuidanceProcessConfig = None
    ad_pipeline: str = None

    @staticmethod
    def from_json(json_path):
        obj = json.load(open(json_path))
        for k in list(obj.keys()):
            if k.startswith('_'):
                del obj[k]
        preprocessor_config = GuidanceProcessConfig(**obj.get('preprocessor', {}))
        del obj['preprocessor']
        return LatentCoupleConfig(preprocessor_config=preprocessor_config, **obj)


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


def load_preprocessor(latent_couple_config: LatentCoupleConfig):
    config = latent_couple_config.preprocessor_config
    preprocessor = GuidanceProcessor(
        config.control_annotator_names,
        control_annotator_args = config.control_annotator_args,
        subject_locator = config.subject_locator,
        face_service_host = config.face_service_host,
        face_id_mlp = config.face_id_mlp + '_mlp.pt'
    )
    return preprocessor


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
    scheduler = DPMSolverMultistepScheduler(        
        num_train_timesteps = 1000,
        beta_start = 8.5e-4,
        beta_end = 1.2e-2,
        beta_schedule = 'scaled_linear',
        algorithm_type = 'dpmsolver++',
        # use_karras_sigmas = True
    )
    # scheduler.config.algorithm_type = 'sde-dpmsolver++'
    # scheduler = DPMSolverMultistepScheduler.from_config(main_pipe.scheduler.config, use_karras_sigmas=True)
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
        pipes.append(pipe)
    
    del main_pipe
    return pipes


class LatentCouplePipelinesManager:

    def __init__(self, config: LatentCoupleConfig, model_config: T.Dict):
        self.preprocessor = load_preprocessor(config)
        self.pipelines = load_latent_couple_pipeline(config, model_config)
        if config.ad_pipeline is not None:
            from asdff import AdPipeline
            logger.info("loading detailer ...")
            self.ad_pipeline = AdPipeline.from_pretrained(
                model_config['base_models'][config.ad_pipeline]['local_path'], 
                torch_dtype=torch.float16,
                load_safety_checker=False,
                local_files_only=True,
                safety_checker = None
            ).to(self.pipelines[0].device)
            self.ad_pipeline.scheduler = copy.deepcopy(self.pipelines[0].scheduler)
            self.use_ad_pipeline = True
        else:
            self.use_ad_pipeline = False
        self.controlnet_names = config.default_controlnet_name
        self.lora_loader = LoraLoader()
        self.lora_status = [collections.defaultdict(lambda : 0.0) for _ in self.pipelines]
        self.ad_lora_status = collections.defaultdict(lambda : 0.0)
        self.id_mlp_names = {'detailer': None, 'main': None}
        self.id_mlp = None
        self.detailer_id_mlp = None
        self.lock_lc = Lock()
        self.lock_ad = Lock()

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

    def is_controlnet_same(self, controlnet_path):
        if isinstance(controlnet_path, str):
            return self.controlnet_names == controlnet_path
        elif isinstance(controlnet_path, list):
            return isinstance(self.controlnet_names, list) and \
                all([a == b for a, b in zip(controlnet_path, self.controlnet_names)])

    def set_controlnet(self, controlnet_path: T.Union[str, list]):
        if self.is_controlnet_same(controlnet_path):
            return

        logger.info("setting controlnet as %s ... " % (controlnet_path, ))
        if isinstance(controlnet_path, str):
            control_model = retry(ControlNetModel.from_pretrained)(controlnet_path, torch_dtype=torch.float16)
        else:
            multicontrol = []
            for path in controlnet_path:
                control_model = retry(ControlNetModel.from_pretrained)(path, torch_dtype=torch.float16)
                multicontrol.append(control_model)
            control_model = MultiControlNetModel(multicontrol)
        self.controlnet_names = controlnet_path
        control_model.to(self.pipelines[0].device)
        for pipe in self.pipelines:
            pipe.controlnet = control_model

    def set_sampler(self, sampler: str):
        if sampler in default_samplers:
            logger.info("setting sampler %s" % (sampler, ))
            for pipe in self.pipelines:
                pipe.scheduler = copy.deepcopy(default_samplers[sampler])
        elif isinstance(sampler, SchedulerMixin):
            for pipe in self.pipelines:
                pipe.scheduler = copy.deepcopy(sampler)
        else:
            raise ValueError(sampler)

    def set_ad_sampler(self, sampler: str):
        if sampler in default_samplers:
            logger.info("setting ad sampler %s" % (sampler, ))
            self.ad_pipeline.scheduler = copy.deepcopy(default_samplers[sampler])
        elif isinstance(sampler, SchedulerMixin):
            self.ad_pipeline.scheduler = copy.deepcopy(sampler)
        else:
            raise ValueError(sampler)

    def load_lora(self, i, lora, weight=1.0):
        if isinstance(i, int):
            pipe = self.pipelines[i]
            self.lora_status[i][lora] += weight
            self.lora_loader.load_lora_weights(pipe, lora, weight, 'cuda', torch.float32)
            if abs(self.lora_status[i][lora]) < 1e-6:
                del self.lora_status[i][lora]
        else:
            self.lora_loader.load_lora_weights(i, lora, weight, 'cuda', torch.float32)
    
    def load_loras_for_pipelines(self, lora_configs):
        mlp_path = [k + '_mlp.pt' for lora in lora_configs for k in lora if os.path.exists(k + '_mlp.pt')]
        mlp_path = list(set(mlp_path))
        if len(mlp_path) not in (0, 1):
            raise ValueError("different multi-person lora not supported yet")
        elif len(mlp_path) == 1:
            if mlp_path[0] != self.id_mlp_names['main']:
                logger.info("loading idmlp %s for main pipe" % (mlp_path[0], ))
                self.id_mlp_names['main'] = mlp_path[0]
                self.id_mlp = torch.load(self.id_mlp_names['main']).cuda().eval()
        for i, lora_config in enumerate(lora_configs):
            for k, v in lora_config.items():
                self.load_lora(i, k, v)

    def unload_loras_for_pipelines(self):
        unload_configs = []
        for i, lora_status in enumerate(self.lora_status):
            unload_configs.append({k: -v for k, v in lora_status.items()})
        self.load_loras_for_pipelines(unload_configs)

    def load_lora_for_detailer(self, lora_config):
        mlp_path = [k + '_mlp.pt' for k in lora_config if os.path.exists(k + '_mlp.pt')]
        mlp_path = list(set(mlp_path))
        if len(mlp_path) not in (0, 1):
            raise ValueError("different multi-person lora not supported yet")
        elif len(mlp_path) == 1:
            if mlp_path[0] != self.id_mlp_names['detailer']:
                logger.info("loading idmlp %s for detailer" % (mlp_path[0], ))
                self.id_mlp_names['detailer'] = mlp_path[0]
                self.detailer_id_mlp = torch.load(self.id_mlp_names['detailer']).cuda().eval()
        for k, v in lora_config.items():
            if os.path.exists(k + '_mlp.pt'):
                self.ad_pipeline.ad_id_mlp = torch.load(k + '_mlp.pt').cuda().eval()
            assert abs(self.ad_lora_status[k]) < 1e-6
            self.ad_lora_status[k] += v
            self.load_lora(self.ad_pipeline, k, v)

    def unload_lora_for_detailer(self):
        for k, v in copy.deepcopy(self.ad_lora_status).items():
            del self.ad_lora_status[k]
            self.load_lora(self.ad_pipeline, k, -v)

    def check_correctness(self):
        return len(self.lora_status) == 0


