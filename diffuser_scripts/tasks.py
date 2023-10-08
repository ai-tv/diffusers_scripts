""" data structure for generation tasks """

import json
import typing as T
from dataclasses import dataclass, asdict, field

import numpy as np
from PIL import Image
from diffuser_scripts.utils.decode import decode_image_b64, encode_image_b64
from diffuser_scripts import __version__

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
    ad_lora_configs: T.Dict[str, float] = None
    id_reference_img: T.List = None
    add_id_feature: T.List = None
    add_pos_encode: T.List = None
    guidance_scale: float = 8.0
    num_image_per_prompt: int = 1
    height: int = 768
    width: int = 512
    sampler: str = 'dpmsolver++'
    num_inference_steps: int = 30
    extra_params: T.Optional[T.Dict] = None
    random_seed: int = -1
    debug_steps: T.List[int] = field(default_factory=list)
    request_id: str = 'none'
    image_index : int = 0
    sampler: str = None

    @property
    def json(self):
        return asdict(self)

    @property
    def uniq_id(self):
        return self.request_id + '-' + str(self.image_index)


@dataclass
class Txt2ImageWithControlParams(Txt2ImageParams):
    """ request for txt2image with controlnet """

    condition_img_str: str = None
    control_image_type: str = 'processed'
    control_annotators: str = 'canny'
    control_model_name: T.Union[str, T.List] = '/mnt/lg102/zwshi/.cache/huggingface/hub/models--lllyasviel--control_v11p_sd15_canny/snapshots/115a470d547982438f70198e353a921996e2e819/'
    control_mode: str = 'balance' # [prompt, balance, control] as in webui
    control_guidance_scale: T.Union[float, T.List[float]] = 1.0
    control_guidance_start: T.Union[float, T.List[float]] = 0.0
    control_guidance_end: T.Union[float, T.List[float]] = 0.5
    control_preprocess_mode: str = "webui"
    control_scale_decay_ratio: float = 0.825

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
    use_main_prompt_for_branches: bool = True
    ad_lora_configs: T.List[T.Dict[str, float]] = field(default_factory=list)
    id_reference_img: T.List = field(default_factory=lambda:[None, None, None])
    add_id_feature: T.List = field(default_factory=lambda:[False, False, False])
    add_pos_encode: T.List = field(default_factory=lambda:[False, False, False])
    latent_mask_weight: T.List[float] = (0.7, 0.3, 0.3)
    latent_mask_weight_decay: T.List[float] = 0.03
    latent_pos: T.List[str] = None
    latent_mask: T.List[str] = None
    latent_couple_min_ratio: float = 0.1
    latent_couple_max_ratio: float = 0.9
    latent_cls_weight: float = 0.6
    latent_neg_cls_weight: float = 0.1
    latent_bg_weight: float = 0.3
    latent_mode: str = "division"

    @property
    def json(self):
        return asdict(self)

    @staticmethod
    def from_json(obj):
        if isinstance(obj, str):
            with open(obj) as f:
                obj = json.load(f)
        return LatentCoupleWithControlTaskParams(**obj)


@dataclass
class ExamineResult:

    is_pass: bool
    face_count: int
    unpass_reasons: T.List = field(default_factory=list)
    face_sim_scores: T.List = field(default_factory=list)
    body_quality_scores: T.List = field(default_factory=list)
    sfw_score: float = 1.0
    extras: T.Dict = field(default_factory=dict)

    @property
    def json(self):
        return asdict(self)

@dataclass
class ImageGenerationResult:
    """ image generation result """

    task: Txt2ImageParams
    result_image_str: str
    examine_result: ExamineResult
    app_verion: str = __version__
    status: int = 0
    status_message: str = "ok"
    intermediate_states: T.List[str] = None

    @staticmethod
    def from_task_and_image(
        task: Txt2ImageParams,
        image: T.Union[Image.Image, np.ndarray],
        intermediate_states: T.List[T.Union[Image.Image, np.ndarray]] = [],
        examine_result: ExamineResult = None
    ):
        return ImageGenerationResult(
            task = task,
            result_image_str = encode_image_b64(image),
            intermediate_states = [encode_image_b64(i) for i in intermediate_states],
            examine_result = examine_result
        )

    @staticmethod
    def from_json(obj):
        task = obj['task']
        examine_result = obj['examine_result']
        return ImageGenerationResult(
            LatentCoupleWithControlTaskParams(**task),
            examine_result=ExamineResult(**examine_result),
            result_image_str = obj['result_image_str'],
            intermediate_states = obj['intermediate_states']
        )

    @property
    def json(self):
        return asdict(self)

    @property
    def generated_image(self):
        return Image.fromarray(decode_image_b64(self.result_image_str))

    def get_intermediate_states(self, channel_reverse=True):
        return [Image.fromarray(decode_image_b64(i)[..., ::(-1 if channel_reverse else 1)]) for i in self.intermediate_states]

    def get_generated_images(self, channel_reverse=True):
        i = decode_image_b64(self.result_image_str)
        if channel_reverse:
            i = i[..., ::-1]
        return Image.fromarray(i)