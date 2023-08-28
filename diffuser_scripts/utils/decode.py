""" utils for encoding image and arrs for saving and communicating """

import base64

import cv2
import numpy as np
from PIL import Image


def encode_image_b64(image, ext='.png'):
    if isinstance(image, Image.Image):
        image = np.array(image)[..., ::-1]
    _, data = cv2.imencode(ext, image)        
    return base64.b64encode(data).decode()
    

def decode_image_b64(encoded):
    s = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(s, cv2.IMREAD_COLOR)
    return img


def encode_arrfp_b64(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    arr = arr.astype(np.float16)
    s = base64.b64encode(arr).decode()
    return s


def decode_arrfp_b64(s, dtype=np.float16):
    r = base64.decodebytes(s)
    q = np.frombuffer(r, dtype=dtype)
    return q


def encode_bytes_b64(bdata, encoding='ascii'):
    return base64.b64encode(bdata).decode(encoding)
