import traceback
import torch
from fastapi import HTTPException

from sbp.nn.app.client import FaceClient
from asdff.yolo import create_mask_from_bbox
from .model_manager import LatentCoupleConfig, LatentCouplePipelinesManager
from .utils.logger import logger, dump_image_to_dir
from .utils.decode import encode_image_b64
from .utils.long_prompt_weighting import get_weighted_text_embeddings
from .pipelines.couple import latent_couple_with_control
from .tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams


def get_guidance_result(model_manager, params, log_dir='log'):
    guidance_processor = model_manager.preprocessor
    image = params.condition_image_np
    if params.control_image_type == 'processed':
        control_image = image
        dump_image_to_dir(control_image, log_dir, name='condition')
    elif params.control_image_type == 'original':
        annotator_names = params.control_annotators
        image_result = guidance_processor.infer_guidance_image(image)
        if isinstance(annotator_names, str):
            control_image = image_result.annotation_maps[annotator_names]
            dump_image_to_dir(control_image, log_dir, name='condition')
        elif isinstance(annotator_names, list):
            control_image = [image_result.annotation_maps[n] for n in annotator_names]
            for c in control_image:
                dump_image_to_dir(c, log_dir, name='condition')
        else:
            raise ValueError(f"bad value for annotator={annotator_names} or control_type={params.control_image_type}")
    else:
        raise ValueError(f"bad value for annotator={annotator_names} or control_type={params.control_image_type}")
    return control_image


def get_face_feature(model_manager, params):
    results = []
    for image in params.id_reference_img:
        image_result = model_manager.preprocessor.infer_reference_image(image)
        results.append(image_result.extra['main_face_encode'])
    return results


def handle_latent_couple(
    model_manager: LatentCouplePipelinesManager,
    params: LatentCoupleWithControlTaskParams,
    lora_configs: dict,
    log_dir: str = 'log'
):
    ### 2. preprocess
    control_image = get_guidance_result(model_manager, params, log_dir=log_dir)
    features = get_face_feature(model_manager, params) if params.id_reference_img is not None else None
    if params.add_id_feature is not None or features is not None:
        features = [f if is_add else None for f, is_add in zip(features, params.add_id_feature)]

    ### 3. latent couple preprocess
    with model_manager.lock_lc:
        try:
            logger.info("loading loras: %s" % lora_configs, )
            model_manager.load_loras(lora_configs)
            model_manager.set_controlnet(controlnet_path=params.control_model_name)
            if params.sampler is not None:
                model_manager.set_sampler(params.sampler)
            result, debugs = latent_couple_with_control(
                pipes = model_manager.pipelines,
                prompts = params.prompt,
                image = control_image,
                couple_pos = params.latent_pos,
                couple_weights = params.latent_mask_weight, 
                negative_prompts = params.negative_prompt, 
                height = params.height,
                width = params.width,
                id_features=features,
                guidance_scale = params.guidance_scale,
                num_inference_steps = params.num_inference_steps,
                control_mode = params.control_mode,
                control_guidance_start = params.control_guidance_start,
                control_guidance_end = params.control_guidance_end,
                controlnet_conditioning_scale = params.control_guidance_scale,
                main_prompt_decay = params.latent_mask_weight_decay,
                control_preprocess_mode = params.control_preprocess_mode,
                generator = torch.Generator(device='cuda').manual_seed(params.random_seed),
                control_scale_decay_ratio = params.control_scale_decay_ratio,
                debug_steps=params.debug_steps
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            logger.info("unload loras ...")
            model_manager.unload_loras()

    ### 4. detailing around face
    with model_manager.lock_ad:
        logger.info("detailing ...")
        common = {
            "num_inference_steps": 30,
            'strength': 0.5
        }
        client = FaceClient('192.168.110.102')
        image_result = client.request_face(encode_image_b64(result))
        dets = image_result.get_detection('face', topk=2, topk_order='area', return_order='center_x')
        for i, (f, lora) in enumerate(zip(features[1:],
            [list(lora.keys())[0] for lora in lora_configs[1:]])):
            model_manager.load_lora(model_manager.ad_pipeline, lora)
            try:
                text_embedding, uncond_embedding = get_weighted_text_embeddings(
                    model_manager.ad_pipeline,
                    prompt="a photo of young thin face, good-looking, best quality",
                    uncond_prompt="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), wrinkle, skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)"
                )
                if f is not None:
                    logger.info("activating multi lora ...")
                    p = torch.cat([f[None, ], text_embedding, ], dim=1)
                    u = torch.cat([f[None, ], uncond_embedding], dim=1)
                else:
                    p = text_embedding
                    u = uncond_embedding
                inpaint_args = [{
                    "prompt_embeds": p,
                    "negative_prompt_embeds": u,
                }]
                result = model_manager.ad_pipeline(
                    common = common,
                    images = result,
                    inpaint_only = inpaint_args,
                    detectors = [lambda image: create_mask_from_bbox([dets[i].bbox], image.size)]
                ).images[0]
            finally:
                model_manager.load_lora(model_manager.ad_pipeline, lora, -1.0)

    ### 4. parse and save output and return
    logger.info("generation succeed for %s" % (params.prompt, ))
    dump_image_to_dir(result, 'log', name='result')
    if result is None:
        return {"result": result}
    else:
        return ImageGenerationResult.from_task_and_image(params, result, debugs).json
