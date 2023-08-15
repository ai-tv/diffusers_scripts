## Usage

### install
`pip install -r requirements.txt`
`pip install -e . `
(Not actually tested, #TODO ensure)

### start server

* config the model path in `configs/default_model_infos.json`
* config the pipeline in `configs/latent_couple_config.json`
* start the server with fastapi
``` 
CUDA_VISIBLE_DEVICES=0 uvicorn main:app --port 1234 --workers 1 --host 0.0.0.0
```

### request api

see following example in `test/test_app.py`

```{python}
import json
import time
import requests
from PIL import Image

from diffuser_scripts.tasks import ImageGenerationResult
from diffuser_scripts.utils.decode import encode_image_b64, decode_image_b64


def test_post_app():
    image = Image.open("images/lc_silence_canny_2.png")

    obj = {
        'prompt': [
            '1man and 1woman',
            'baijingting',
            'ouyangnana',
        ],
        "negative_prompt": [
            'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ... ... ,(nsfw:1.5), (sexy)',
        ],
        "base_model": ['chilloutmix', 'chilloutmix', 'chilloutmix'],
        "width": 1024,
        "height": 768,
        "lora_configs": [
            {"couple_0614_001": 0.5, "0807_OYNN_YM_2female-000001": 0.5},
            {"couple_0614_001": 1},
            {"0807_OYNN_YM_2female-000001": 1}
        ],
        "condition_img_str": encode_image_b64(image),
        "control_guidance_scale": 1.0,
        "control_guidance_end": 0.5,
        "latent_mask_weight": [0.7, 0.3, 0.3],
        "latent_pos": ["1:1-0:0","1:2-0:0","1:2-0:1"],
        "random_seed": 1234
    }
    start = time.time()
    response = requests.post("http://127.0.0.1:1234/get_latent_couple", data=json.dumps(obj))
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
```