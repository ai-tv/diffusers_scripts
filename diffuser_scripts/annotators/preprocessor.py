import base64
import typing as T
from dataclasses import dataclass

import cv2
import torch
import numpy as np
from PIL import Image
from controlnet_aux import LineartDetector

from sbp.vision.protocal import ImageResult
from sbp.utils.functions import retry
from sbp.nn.app.ultralytics.segment import YoloMaskPredictor
from sbp.nn.app.client import FaceClient
from sbp.nn.utils import encode_pos_scale

from diffuser_scripts.annotators.dwpose import DWposeDetector
from diffuser_scripts.annotators.canny import canny
from diffuser_scripts.annotators.hand import Hand
from diffuser_scripts.utils.logger import dump_image_to_dir, dump_request_to_file
from diffuser_scripts.utils.decode import decode_image_b64, encode_image_b64, encode_bytes_b64, decode_image
from diffuser_scripts.utils.image_process import detectmap_proc
from diffuser_scripts.utils.logger import logger


def pil2np_wrapper(f):
    def initializer(**args):
        processor = f(**args)
        def new_f(image):
            image = Image.fromarray(image[..., ::-1])
            image = processor(image)
            image = np.array(image)
            return image
        return new_f
    return initializer



def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def process_masks(masks, h, w, cls_weight=0.6, neg_cls_weight=0.1, bg_weight=0.3, device='cuda'):
    new_masks = []
    xs = []
    union = None
    for mask in masks:
        mask = cv2.resize(mask, (w//8, h//8), )
        union = mask > 0 if union is None else (union | (mask > 0))
        new_masks.append(mask)

    for i, mask in enumerate(new_masks):
        mask = np.where(mask > 0, 
            mask * cls_weight, 
            np.where(union > 0, 
                np.ones_like(mask) * neg_cls_weight, 
                np.ones_like(mask) * bg_weight))
        new_masks[i] = mask

    new_masks = [1-sum(new_masks)] + list(new_masks)
    for i, m in enumerate(new_masks):
        new_masks[i] = np.tile(m[None, ..., 0], (4, 1, 1))
    return [
        torch.FloatTensor(mask[None, ...]).to(device)
        for mask in new_masks
    ]


def create_control_guidance_mask(image_result):
    from sbp.vision.common import relax_bbox
    faces = image_result.get_detection('face', topk=2)
    h, w = image_result.height, image_result.width
    final_mask = 0
    for face in faces:
        rect = face.bbox
        x1, y1, x2, y2 = relax_bbox(rect, h, w, 0.3, 0.2, 0.4, return_integer=True)
        large_mask = np.zeros((h, w))
        large_mask[y1:y2, x1:x2] = 1.0
        x1, y1, x2, y2 = relax_bbox(rect, h, w, -0.25, -0.1, -0.05, return_integer=True)
        small_mask = np.zeros((h, w))
        small_mask[y1:y2, x1:x2] = 1.0
        mask = (large_mask - small_mask) > 0
        final_mask = mask | final_mask
    return final_mask.astype(np.uint8) * 255


@dataclass
class GuidanceInfo:

    guidance_image_results: ImageResult
    id_reference_results: T.List[ImageResult]
    annotations: T.List
    latent_masks: T.List
    controlnet_face_reweight_mask: np.ndarray = None
    controlnet_hand_reweight_mask: np.ndarray = None


class GuidanceProcessor:
    _annotators = {
        'canny': (lambda : canny),
        'dwpose': DWposeDetector,
        'hand': Hand,
        'lineart': pil2np_wrapper(retry()(LineartDetector.from_pretrained))
    }  

    def __init__(
        self, 
        control_annotator_names, 
        control_annotator_args,
        subject_locator = "/mnt/lg102/zwshi/model_zoo/yolov8x-seg.pt",
        face_service_host = '192.168.110.102',
        face_id_mlp = None
    ):
        self.face_detector = FaceClient(face_service_host)
        self.control_annotator_names = control_annotator_names
        self.control_annotators = [
            self._annotators[n](**control_annotator_args.get(n, {})) 
            for n in control_annotator_names
        ]
        self.subject_locator = YoloMaskPredictor(subject_locator)
        self.caption_model = None
        # self.id_mlp = torch.load(face_id_mlp).cuda().eval()
        # self.ad_id_mlp = torch.load(face_id_mlp).cuda().eval()

    @torch.no_grad()
    def infer_guidance_image(self, image: T.Union[str, bytes, np.ndarray, Image.Image], height, width):
        if isinstance(image, bytes):
            image = decode_image(image)
        elif isinstance(image, str):
            image = decode_image_b64(image)
        else:
            raise ValueError
        _, image = detectmap_proc(image, height, width)
        image_b64 = encode_image_b64(image)

        logger.info("request guidance face ...")
        image_result = self.face_detector.request_face(image_b64)

        annotation_maps = {}
        for n, annotator in zip(
            self.control_annotator_names, self.control_annotators):
            logger.info("do controlnet annotation %s ..." % n)
            annotation_maps[n] = annotator(image)

        logger.info("do person segmentation ...")
        segment_results = self.subject_locator.get_obj_segment_result(image)
        image_result.detection_results.extend(segment_results)
        image_result.annotation_maps = annotation_maps
        image_result.extra['positional_encoding'] = []
        h, w = np.array(image).shape[:2]
        for face in image_result.get_detection('face', topk=2):
            image_result.extra['positional_encoding'].append(
                encode_pos_scale(face.bbox, h, w, dim=64)[None, ...]
            )
        image_result.height = h
        image_result.width = w
        return image_result

    @torch.no_grad()
    def infer_reference_image(self, image: T.Union[str, bytes, np.ndarray, Image.Image]):
        if isinstance(image, (np.ndarray, Image.Image)):
            image_b64 = encode_image_b64(image)
        elif isinstance(image, bytes):
            image_b64 = encode_bytes_b64(image)
        elif isinstance(image, str):
            image_b64 = image
        elif image is None:
            rec = torch.zeros((1, 512), dtype=torch.float32).cuda()
            return ImageResult(extra={'main_face_rec': rec})
        else:
            raise ValueError

        logger.info("request reference face ...")        
        image_result = self.face_detector.request_face(image_b64)
        detection = image_result.get_max_detection('face')
        rec = torch.FloatTensor(detection.rec_feature).cuda()[None, ...]
        image_result.extra['main_face_rec'] = rec
        logger.info("got face encode ...")        
        # image_result.extra['main_face_encode'] = self.id_mlp(rec)
        # image_result.extra['main_face_encode_ad'] = self.ad_id_mlp(rec)
        return image_result

    def get_guidance_result(self, params, log_dir='log'):
        id_reference_results = self.get_face_feature(params)
        image = params.condition_image_np
        request_id = params.uniq_id
        if params.control_image_type == 'processed':
            control_image = image
            image_result = None
            latent_mask = None
            dump_image_to_dir(control_image, log_dir, name='%s_cond.jpg' % request_id)
        elif params.control_image_type == 'original':
            annotator_names = params.control_annotators
            image_result = self.infer_guidance_image(params.condition_img_str, params.height, params.width)
            segments = image_result.get_detection('person', topk=2)        
            # controlnet_face_mask = create_control_guidance_mask(image_result)
            latent_mask = process_masks(
                [seg.mask for seg in segments], 
                params.height, params.width, 
                cls_weight = params.latent_cls_weight,
                neg_cls_weight = params.latent_neg_cls_weight,
                bg_weight = params.latent_bg_weight
            )
            for m in latent_mask:
                logger.info("latent mask: %s" %(m.shape, ))
            dump_image_to_dir(image, log_dir, name='%s_guidance.jpg' % request_id)
            if isinstance(annotator_names, str):
                control_image = image_result.annotation_maps[annotator_names]
                dump_image_to_dir(control_image, log_dir, name='%s_cond.jpg' % request_id)
            elif isinstance(annotator_names, list):
                control_image = [image_result.annotation_maps[n] for n in annotator_names]
                for i, c in enumerate(control_image):
                    dump_image_to_dir(c, log_dir, name='%s_cond%d.jpg' % (request_id, i))
            else:
                raise ValueError(f"bad value for annotator={annotator_names} or control_type={params.control_image_type}")
        else:
            raise ValueError(f"bad value for annotator={annotator_names} or control_type={params.control_image_type}")

        return GuidanceInfo(
            guidance_image_results = image_result,
            id_reference_results = id_reference_results,
            annotations = control_image,
            latent_masks = latent_mask,
            # controlnet_face_reweight_mask = controlnet_face_mask
        )

    def get_face_feature(self, params):
        results = []
        for image in params.id_reference_img:
            image_result = self.infer_reference_image(image)
            results.append(image_result)
        return results


if __name__ == '__main__':
    preprocessor = GuidanceProcessor(
        ['canny', 'dwpose'],
        {
            'dwpose': dict(
                onnx_det = 'annotator/ckpts/yolox_l.onnx',
                onnx_pose = 'annotator/ckpts/dw-ll_ucoco_384.onnx'
            )
        },
        face_id_mlp="/mnt/lg102/zwshi/projects/core/lora-scripts/tasks/output/035_mix/035_mix-000008.safetensors_mlp.pt"
    )
    
    image = file2base64('/mnt/2T/zwshi/projects/playground/lc/test_images/original/下载 (2).png')
    result = preprocessor.infer_guidance_image(image, 768, 1024)
    print(result)