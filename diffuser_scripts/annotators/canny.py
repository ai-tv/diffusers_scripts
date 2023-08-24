import cv2
from diffuser_scripts.utils.image_process import resize_shortest


def canny(image, size=512, t1=100, t2=200):
    image = resize_shortest(image, size)
    return cv2.Canny(image, t1, t2)
