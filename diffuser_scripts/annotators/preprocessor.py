import base64
import typing as T
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from sbp.vision.protocal import ImageResult
from sbp.nn.app.ultralytics.segment import YoloMaskPredictor
from sbp.nn.app.client import FaceClient

from diffuser_scripts.annotators.dwpose import DWposeDetector
from diffuser_scripts.annotators.canny import canny
from diffuser_scripts.utils.decode import decode_image_b64, encode_image_b64, encode_bytes_b64


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


class GuidanceProcessor:
    _annotators = {
        'canny': (lambda : canny),
        'dwpose': DWposeDetector,
    }  

    def __init__(
        self, 
        control_annotator_names, 
        control_annotator_args,
        subject_locator = "/mnt/lg102/zwshi/model_zoo/yolov8x-seg.pt",
        face_service_host = '192.168.110.102',
        face_id_mlp = None
    ):
        self.face_detecor = FaceClient(face_service_host)
        self.control_annotator_names = control_annotator_names
        self.control_annotators = [
            self._annotators[n](**control_annotator_args.get(n, {})) 
            for n in control_annotator_names
        ]
        self.subject_locator = YoloMaskPredictor(subject_locator)
        self.id_mlp = torch.load(face_id_mlp).cuda().eval()

    @torch.no_grad()
    def infer_guidance_image(self, image: T.Union[str, bytes, np.ndarray, Image.Image]):
        if isinstance(image, (np.ndarray, Image.Image)):
            image_b64 = encode_image_b64(image)
        elif isinstance(image, bytes):
            image_b64 = encode_bytes_b64(image)
            image = decode_image_b64(image_b64)
        elif isinstance(image, str):
            image_b64 = image
            image = decode_image_b64(image_b64)
        else:
            raise ValueError

        image_result = self.face_detecor.request_face(image_b64)
        annotation_maps = {}
        for n, annotator in zip(
            self.control_annotator_names, self.control_annotators):
            annotation_maps[n] = annotator(image)
        segment_results = self.subject_locator.get_obj_segment_result(image)
        image_result.detection_results.extend(segment_results)
        image_result.annotation_maps = annotation_maps
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
            rec = torch.zeros((1, 512)).cuda()
            rec = self.id_mlp(rec)
            return ImageResult(extra={'main_face_encode': rec})
        else:
            raise ValueError        
        image_result = self.face_detecor.request_face(image_b64)
        detection = image_result.get_max_detection('face')
        rec = torch.FloatTensor(detection.rec_feature).cuda()[None, ...]
        image_result.extra['main_face_rec'] = detection.rec_feature
        image_result.extra['main_face_encode'] = self.id_mlp(rec)
        return image_result


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
    result = preprocessor.infer_guidance_image(image)
    print(result)