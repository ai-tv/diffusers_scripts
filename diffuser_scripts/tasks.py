""" data structure for generation tasks """

import typing as T
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from diffuser_scripts.utils.decode import decode_image_b64, encode_image_b64


@dataclass
class LoraConfig:
    """ config for lora """

    lora_name: str
    lora_path: str
    lora_charactors: T.List[str]
    lora_default_model: str = "chilloutmix"
    lora_base_version: str = "sd1.5"
    is_character_lora: bool = True


@dataclass
class SDModelConfig:
    """ config for sd base model """

    model_name: str
    model_path: str
    model_type: str = "realistic"
    model_description: T.Optional[str] = ""
    model_default_negative: T.Optional[str] = "paintings, sketches, (worst quality:2), "\
        "(low quality:2),(normal quality:2),lowres,normal quality, ((monochrome)), "\
        "((grayscale)), skin spots, acnes, skin blemishes, age spot,  glans, lowres," \
        "bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits,"\
        "cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark," \
        "username,blurry,bad feet, cropped,poorly drawn hands,poorly drawn face,mutation,"\
        "deformed,worst quality,low quality,normal quality, jpeg artifacts,watermark,"\
        "extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs, "\
        "fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,"\
        "bad body,bad proportions, gross proportions,text,error,missing fingers,missing arms,"\
        "missing legs,extra digit,(nsfw:1.5),(sexy)"


@dataclass
class Txt2ImageParams:
    """ request for txt2image """

    prompt: str
    negative_prompt: str
    base_model: str
    lora_configs: T.Dict[str, float]
    guidance_scale: float = 8.0
    num_image_per_prompt: int = 1
    height: int = 768
    width: int = 512
    sampler: str = 'sde-dmpsolver++'
    num_inference_steps: int = 30
    extra_params: T.Optional[T.Dict] = None
    random_seed: int = -1

    @property
    def json(self):
        return asdict(self)


@dataclass
class Txt2ImageWithControlParams(Txt2ImageParams):
    """ request for txt2image with controlnet """

    condition_img_str: str = None
    control_model_name: str = 'lllyasviel/control_v11p_sd15_canny'
    control_mode: str = 'prompt' # [prompt, balance, control] as in webui
    control_guidance_scale: T.Union[float, T.List[float]] = 1.0
    control_guidance_start: T.Union[float, T.List[float]] = 0.0
    control_guidance_end: T.Union[float, T.List[float]] = 0.5
    control_preprocess_mode: str = "webui"

    @property
    def condition_image(self):
        return Image.fromarray(self.condition_image_np)

    @property
    def condition_image_np(self):
        return decode_image_b64(self.condition_img_str)

@dataclass
class LatentCoupleWithControlTaskParams(Txt2ImageWithControlParams):
    """ request for latent couple pipeline with controlnet """

    prompt: T.List[str]
    negative_prompt: T.List[str]
    base_model: T.List[str]
    lora_configs: T.List[T.Dict[str, float]]
    latent_mask_weight: T.List[float] = (0.7, 0.3, 0.3)
    latent_mask_weight_decay: T.List[float] = 0.03
    latent_pos: T.List[str] = None
    latent_mask: T.List[str] = None

    @property
    def json(self):
        return asdict(self)



@dataclass
class ImageGenerationResult:
    """ image generation result """

    task: Txt2ImageParams
    result_image_str: str

    @staticmethod
    def from_task_and_image(task: Txt2ImageParams, image: T.Union[Image.Image, np.ndarray]):
        return ImageGenerationResult(task, encode_image_b64(image))

    @staticmethod
    def from_json(obj):
        task = obj['task']
        return ImageGenerationResult(
            LatentCoupleWithControlTaskParams(**task),
            result_image_str=obj['result_image_str']
        )

    @property
    def json(self):
        return asdict(self)

    @property
    def generated_image(self):
        return Image.fromarray(decode_image_b64(self.result_image_str))
