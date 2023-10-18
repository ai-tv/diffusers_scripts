import cv2
import numpy as np
from PIL import Image

from sbp.vision.protocal import DetResult
from sbp.nn.app.dwpose import DWposeDetector
from sbp.nn.app.mediapipe.hand_annotator import HandAnnotator


def relax_bbox2(bbox, h, w, target_size, min_relax=3.0):
    x1, y1, x2, y2 = bbox
    dw, dh = x2 - x1, y2 - y1
    size = max(dw * w, dh * h)
    relax = max(target_size / size, min_relax)
    target_size = min(int(relax * size), h, w)
    x = (x1 + x2) / 2 * w
    y = (y1 + y2) / 2 * h
    x_start = max(int(x - (target_size / 2)), 0)
    y_start = max(int(y - (target_size / 2)), 0)
    if x_start + target_size > w:
        x_start = w - target_size
    if y_start + target_size > h:
        y_start = h - target_size
    x_end = x_start + target_size
    y_end = y_start + target_size
    cropped_bbox = x_start, y_start, x_end, y_end
    return cropped_bbox




def resize(detected_map, h, w):
    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    k = max(k0, k1)
    detected_map = cv2.resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
    new_h, new_w, _ = detected_map.shape
    pad_h = max(0, (new_h - h) // 2)
    pad_w = max(0, (new_w - w) // 2)
    detected_map = detected_map[pad_h:pad_h+h, pad_w:pad_w+w]
    return detected_map


class Hand:

    def __init__(self, hand_model, pose_model):
        self.hand_model = HandAnnotator(hand_model)
        self.pose = DWposeDetector(**pose_model)
    
    def __call__(self, image):
        image = resize(image, 768, 1024)
        ph = np.zeros_like(image)
        h, w = image.shape[:2]
        dets = []
        for person in self.pose.infer_pose(image, return_type='image_result').get_detection("person", topk=2):
            for bbox, viz in [person.left_hand_bbox, person.right_hand_bbox]:
                if viz > 0.4:
                    bbox = relax_bbox2(bbox, h, w, 100, 1.2)
                    dets.append(DetResult("hand", viz, bbox=bbox))
        dets = self.nms(dets)
        masked_annotations = np.zeros_like(image)
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            mask = np.zeros_like(image)
            mask[y1:y2, x1:x2] = 1
            relaxed_bbox = x1, y1, x2, y2 = relax_bbox2(det.bbox, h, w, 384, 2.2)
            cropped_input = Image.fromarray(image[..., ::-1]).crop(relaxed_bbox)
            cropped_input = np.array(cropped_input)
            cropped_annotation, hand_result = self.hand_model.draw_annotation(cropped_input)
            annotation = np.zeros_like(image)
            annotation[y1:y2, x1:x2] = np.array(cropped_annotation)
            annotation = mask * annotation
            masked_annotations = np.maximum(masked_annotations, annotation)
        return masked_annotations
        
            
    def nms(self, bboxes):
        n_bbox = len(bboxes)
        bboxes = sorted(bboxes, key=lambda x: -x.area)
        ious = []
        for i in range(n_bbox):
            ious.append([])
            for j in range(n_bbox):
                iou = bboxes[i].compute_iou(bboxes[j])
                ious[-1].append(iou)
        ious = np.array(ious)
        print(ious)
        excludes = set()
        for i, iou in enumerate(ious):
            for j, _iou in enumerate(iou):
                if j > i and _iou > 0.2:
                    excludes.add(j)
        bboxes = [bboxes[i] for i in range(n_bbox) if i not in excludes]
        return bboxes