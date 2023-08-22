import json
import time

import fire
import requests
import numpy as np
from PIL import Image

from diffuser_scripts.tasks import ImageGenerationResult, LatentCoupleWithControlTaskParams
from diffuser_scripts.utils.decode import encode_image_b64, decode_image_b64


@fire.Fire
def test_post_app(port='1234', random_seed=-1):
    image = np.array(Image.open("images/image.png"))[..., None]

    obj = {
        'prompt':[
            "1woman and 1man, drink",
            "baijingting",
            "ouyangnana"
        ],
        "negative_prompt": [
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)',
        ],
        "base_model": ['chilloutmix', 'chilloutmix', 'chilloutmix'],
        "width": 1024,
        "height": 768,
        "lora_configs": [
            {
                "exp0814_baijingting_single-000005": 0.5,
                "exp0814_ouyangnana_single-000005": 0.5
            },
            {
                "exp0814_baijingting_single-000005": 1
            },
            {
                "exp0814_ouyangnana_single-000005": 1
            }
        ],
        "condition_img_str": encode_image_b64(image),
        "control_guidance_scale": 1.0,
        "control_guidance_end": 0.5,
        "latent_mask_weight": [0.7, 0.3, 0.3],
        "latent_pos": ["1:1-0:0",
            "1:32-0:0-14",
            "1:32-0:14-32"],
        "random_seed": random_seed,
        "latent_mask_weight_decay": 0.03,
        "control_mode": "prompt"
    }
    start = time.time()
    response = requests.post("http://192.168.110.102:%s/get_latent_couple" % port, data=json.dumps(obj))
    if response.status_code == 200:
        output_name = 'result.png'
        result = response.json()
        cost = time.time() - start
        result = ImageGenerationResult.from_json(result)
        image = result.generated_image
        image.save(output_name)
        print("200 ok, cost %.3fs, save result to %s" % (cost, output_name, ))
    else:
        print("Error:", response.text)


# if __name__ == '__main__':
#     test_post_app()
