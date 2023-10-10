import copy
import traceback
import collections

import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import HTTPException

from sbp.nn.app.client import FaceClient
from sbp.nn.utils import encode_pos_scale
from asdff.yolo import create_mask_from_bbox
from .model_manager import LatentCoupleConfig, LatentCouplePipelinesManager
from .utils.logger import logger, dump_image_to_dir
from .utils.decode import encode_image_b64
from .utils.long_prompt_weighting import get_weighted_text_embeddings
from .pipelines.couple import latent_couple_with_control
from .tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams, Txt2ImageParams, ExamineResult


@torch.no_grad()
def handle_latent_couple(
    model_manager: LatentCouplePipelinesManager,
    params: LatentCoupleWithControlTaskParams,
    lora_configs: list,
    ad_lora_configs: list,
    log_dir: str = 'log'
):
    request_id = params.uniq_id

    ### 2. preprocess
    guidance_results = model_manager.preprocessor.get_guidance_result(params, log_dir='log')
    if params.latent_pos is None:
        r = guidance_results.guidance_image_results
        faces = r.get_detection('face', topk=2)
        mid = np.round((faces[0].center_x + faces[1].center_x) / 2 / r.width * 32)
        mid = int(mid)
        params.latent_pos = [
            '1:1-0:0',
            '1:32-0:0-%s' % (mid, ),
            '1:32-0:%s-32' % (mid, )
        ]
        logger.info("use pos %s" % (params.latent_pos, ))
        couple_mask_list = None #guidance_results.latent_masks
    else:
        couple_mask_list = None

    ### 3. latent couple preprocess
    with model_manager.lock_lc:
        try:
            ## 3.1 load lora
            logger.info("%s loading loras: %s" % (request_id, lora_configs, ))
            model_manager.load_loras_for_pipelines(lora_configs)

            ## 3.2 load controlnet
            model_manager.set_controlnet(controlnet_path=params.control_model_name)

            ## 3.3 set sampler
            # if params.sampler is not None:
            #     model_manager.set_sampler(params.sampler)

            ## 3.4 set extra conditions
            r = guidance_results.guidance_image_results
            guided_face_dets = r.get_detection('face', topk=3) if r is not None else []
            features = []
            for i, result in enumerate(guidance_results.id_reference_results):
                if params.add_pos_encode[i]:
                    ph = torch.zeros([3, 512+64*3]).cuda()
                    if i > 0 and r is not None:
                        for j, det in enumerate(guided_face_dets):
                            face = result.extra['main_face_rec'].cuda()
                            pos = encode_pos_scale(det.bbox, r.height, r.width)
                            pos = torch.tensor(pos[None, ...]).cuda()
                            face = torch.cat([pos, face], dim=1)
                            ph[j] = face
                    feature = model_manager.id_mlp['main'][i](ph)
                else:
                    if params.add_id_feature[i]:
                        face = result.extra['main_face_rec']
                        feature = model_manager.id_mlp['main'][i](face.cuda())
                    else:
                        feature = None
                features.append(feature)
            
            prompts = copy.deepcopy(params.prompt)
            if params.use_main_prompt_for_branches:
                for i, f in enumerate(features):
                    if f is not None and i > 0:
                        prompts[i] = params.prompt[0]
                    elif f is None and i > 0:
                        prompts[i] += params.prompt[0]

            ## 3.5 run pipelines
            result, debugs = latent_couple_with_control(
                pipes = model_manager.pipelines,
                prompts = prompts,
                image = guidance_results.annotations,
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
                latent_couple_min_ratio=params.latent_couple_min_ratio,
                latent_couple_max_ratio=params.latent_couple_max_ratio,
                couple_mask_list=couple_mask_list,
                debug_steps=params.debug_steps,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            logger.info("%s unload loras ..." % (request_id, ))
            model_manager.unload_loras_for_pipelines()
    dump_image_to_dir(result, 'log', name='%s_before_detailer.jpg' % request_id)

    ### 4. after detailer
    features = []
    for i, r in enumerate(guidance_results.id_reference_results[1:]):
        if params.add_id_feature[1:][i]:
            face = r.extra['main_face_rec']
            face = torch.tensor(face).cuda()
        else:
            face = None
        features.append(face)
    if len(ad_lora_configs) == 0:
        ad_lora_configs = lora_configs[1:]

    client = model_manager.preprocessor.face_detector
    image_result = client.request_face(encode_image_b64(result))
    dets = image_result.get_detection('face', topk=5, topk_order='area', return_order='center_x')
    result = detailer(model_manager, params, init_image=result, features=features, lora_configs=ad_lora_configs, dets=dets[:2])
    examine_result = examine_image(result, guidance_results, client)
    logger.info("%s examine: %s" % (request_id, examine_result))

    ### 5. parse and save output and return
    logger.info("%s generation succeed for %s" % (request_id, params.prompt, ))
    dump_image_to_dir(result, 'log', name='%s.jpg' % request_id)
    if result is None:
        return {"result": result}
    else:
        return ImageGenerationResult.from_task_and_image(
            params, result, debugs, examine_result)


def examine_image(
    image,
    guidance_results,
    face_client,
):
    is_pass = True
    face_scores = []
    unpass_reasons = []
    extras = collections.defaultdict(list)
    image_result = face_client.request_face(encode_image_b64(image))
    dets = image_result.get_detection('face', topk=3, topk_order='area', return_order='center_x')
    if len(dets) != 2:
        is_pass = False
        unpass_reasons.append('missing or extra face')
    for i, (det, result, gdet) in enumerate(zip(
        dets[:2],
        guidance_results.id_reference_results[1:],
        guidance_results.guidance_image_results.get_detection('face', topk=2)
    )):
        f = result.extra['main_face_rec'].cpu().numpy()[0]
        fnorm = (f ** 2).sum() ** 0.5
        if fnorm == 0: # given `None` as ref will provide zero feature 
            continue
        f = f / fnorm
        s = float(det.norm_feature.dot(f))
        face_scores.append(s)
        if s < 0.4:
            is_pass = False
            unpass_reasons.append('face %d low similarity' % i)
        iou = det.compute_iou(gdet)
        extras['face_iou'].append(iou)
        if iou < 0.6:
            is_pass = False
            unpass_reasons.append('face %d low iou with guidance' % i)
        
    examine = ExamineResult(
        face_count=len(dets),
        face_sim_scores=face_scores,
        is_pass = is_pass,
        unpass_reasons = unpass_reasons,
        extras = dict(extras)
    )
    return examine


def detailer(
    model_manager: LatentCouplePipelinesManager, 
    params: Txt2ImageParams, 
    init_image: Image, 
    features, 
    lora_configs: list,
    dets: list = []
):
    request_id = params.uniq_id
    with model_manager.lock_ad:
        common = {
            "num_inference_steps": 30,
            'strength': 0.5
        }
        result = init_image
        logger.info("%s detailing %d faces..." % (request_id, len(dets) ))
        # if params.sampler is not None:
        #     model_manager.set_ad_sampler(params.sampler)
        for i, (f, lora) in enumerate(zip(features, lora_configs)):
            if i >= len(dets):
                break
            model_manager.load_lora_for_detailer(lora)
            logger.info("%s activateing lora %s ..." % (request_id, lora ))
            try:
                text_embedding, uncond_embedding = get_weighted_text_embeddings(
                    model_manager.ad_pipeline,
                    prompt="%s, young, good-looking, best quality" % params.prompt[1+i], 
                    uncond_prompt="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), wrinkle, skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)"
                )
                if f is not None:
                    f = model_manager.detailer_id_mlp(f)
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
                model_manager.unload_lora_for_detailer()
        return result