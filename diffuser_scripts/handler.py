import traceback
import torch
from fastapi import HTTPException

from .model_manager import LatentCoupleConfig, LatentCouplePipelinesManager
from .utils.logger import logger, dump_image_to_dir
from .pipelines.couple import latent_couple_with_control
from .tasks import ImageGenerationResult


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
    params: LatentCoupleConfig,
    lora_configs: dict,
    log_dir: str = 'log'
):
    ### 2. preprocess
    control_image = get_guidance_result(model_manager, params, log_dir=log_dir)
    features = get_face_feature(model_manager, params) if params.id_reference_img is not None else None
    if params.add_id_feature is not None or features is not None:
        features = [f if is_add else None for f, is_add in zip(features, params.add_id_feature)]

    ### 3. latent couple preprocess
    with model_manager.lock:
        try:
            logger.info("loading loras: %s" % lora_configs, )
            model_manager.load_loras(lora_configs)
            model_manager.set_controlnet(controlnet_path=params.control_model_name)
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

    ### 4. parse and save output and return
    logger.info("generation succeed for %s" % (params.prompt, ))
    dump_image_to_dir(result, 'log', name='result')
    if result is None:
        return {"result": result}
    else:
        return ImageGenerationResult.from_task_and_image(params, result, debugs).json
