import os
import glob
import json
import aiohttp
import requests

import numpy as np
import gradio as gr
from sbp.utils.parallel import parallel_imap
from sbp.io.common.encoder_decoder import encode_image_b64
from diffuser_scripts.tasks import ImageGenerationResult

NEG_PROMPT = "paintings, sketches, (worst quality:2), "\
"(low quality:2),(normal quality:2),lowres,normal quality, ((monochrome)), "\
"((grayscale)), skin spots, acnes, skin blemishes, age spot,  glans, lowres," \
"bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits,"\
"cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark," \
"username,blurry,bad feet, cropped,poorly drawn hands,poorly drawn face,mutation,"\
"deformed,worst quality,low quality,normal quality, jpeg artifacts,watermark,"\
"extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs, "\
"fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,"\
"bad body,bad proportions, gross proportions,text,error,missing fingers,missing arms,"\
"missing legs,extra digit,(nsfw:1.5),(sexy)"


def request_app(
    guidance_img: np.ndarray,
    prompt: str,
    width: int,
    ref_img1: np.ndarray,
    prompt1: str,
    lora: str,
    lora1: str,
    lora1_face: str = None,
    ref_img2 = None,
    prompt2 = None,
    lora2 = None,
    lora2_face: str = None,
    random_seed: str = -1
):
    img_b64 = encode_image_b64(guidance_img[..., ::-1])
    request_json = {
        "base_model": ['chilloutmix'] * 3,
        'lora_configs': [
            {lora1 or lora: 1},
            {lora1 or lora: 1},
            {lora2 or lora: 1},
        ],
        'ad_lora_configs': [
            {lora1_face or lora1 or lora: 1},
            {lora2_face or lora2 or lora: 1},
        ],
        'prompt': [
            prompt,
            prompt1,
            prompt2
        ],
        'width': width,
        'condition_img_str': img_b64,
        'control_image_type': 'original',
        'control_model_name': 'lllyasviel/control_v11p_sd15_canny',
        "add_id_feature": [False, True, True],
        'sampler': 'dpm++',
        'control_mode': 'balance',
        'id_reference_img': [
            None, 
            encode_image_b64(ref_img1[..., ::-1]), 
            encode_image_b64(ref_img2[..., ::-1]),
        ],
        'negative_prompt': [NEG_PROMPT] * 3,
        'random_seed': int(random_seed)
    }

    def post(task):
        url, request_json = task
        response = requests.post(url, data=request_json)
        return json.loads(response.content)

    def tasks():
        for seed in (0, 1):
            obj = request_json.copy()
            obj['random_seed'] = int(random_seed) + seed
            data = json.dumps(obj)
            task = ('http://127.0.0.1:%d/get_latent_couple' % (1234 + seed), data)
            yield task

    images = []
    for result in parallel_imap(post, tasks()):
        result = ImageGenerationResult.from_json(result)
        image = np.array(result.generated_image)[..., ::-1]
        images.append(image)
    return images


loras = sorted(glob.glob('/data/model_zoo/lora/*.safetensors'))
loras = [os.path.basename(lora).rsplit('.', maxsplit=1)[0] for lora in loras]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    with gr.Group():
                        gr.Markdown("&nbsp;&nbsp;guidance image")
                        guidance_image = gr.Image()
                with gr.Row():
                    prompt_textbox = gr.Textbox(value='a couple, 1man and 1woman, suit and wedding dress, 4k, best quality, highly detailed, cinematic, (simple background)', label="prompt")
                with gr.Row():
                    with gr.Column(scale=1):
                        lora_dropdown = gr.Dropdown(choices=loras, label='lora (set all lora in couple and adface to this lora)')
                    with gr.Column(scale=1):
                        generate = gr.Button("Generate")
                        refresh_btn = gr.Button("refresh")

                with gr.Row():
                    with gr.Column(scale=1):
                        ref_img1 = gr.Image(label='ref image 1')
                        prompt1_textbox = gr.Textbox(value='a couple', label="prompt1")
                        lora1_dropdown = gr.Dropdown(choices=loras, label='lora1')
                        lora1_face_dropdown = gr.Dropdown(choices=loras, label='lora1 face')
                        # refresh1 = gr.Button("Refresh")

                    with gr.Column(scale=1):
                        ref_img2 = gr.Image(label='ref image 2')
                        prompt2_textbox = gr.Textbox(value='a couple', label="prompt2")
                        lora2_dropdown = gr.Dropdown(choices=loras, label='lora2')
                        lora2_face_dropdown = gr.Dropdown(choices=loras, label='lora2 face')
                        # refresh2 = gr.Button("Refresh")

        with gr.Column():
            with gr.Group():
                gr.Markdown("&nbsp;&nbsp;Output Image")
                output1 = gr.Image()
                output2 = gr.Image()
            random_seed = gr.Textbox(value=0, label="Random Seed")
            width = gr.Slider(label='width', minimum=512, maximum=2048, step=128, value=1024)
    
    def refresh():
        loras = sorted(glob.glob('/mnt/2T/zwshi/model_zoo/*.safetensors'))
        loras = [os.path.basename(lora).rsplit('.', maxsplit=1)[0] for lora in loras]
        return [gr.Dropdown.update(choices=loras)] * 5

    refresh_btn.click(fn=refresh, outputs=[lora_dropdown, lora1_dropdown, lora2_dropdown, lora1_face_dropdown, lora2_face_dropdown])
    # refresh2.click(fn=refresh, outputs=lora2_dropdown)
    generate.click(
        fn = request_app,
        inputs = [
            guidance_image, 
            prompt_textbox,
            width,
            ref_img1,
            prompt1_textbox,
            lora_dropdown,
            lora1_dropdown,
            lora1_face_dropdown,
            ref_img2,
            prompt2_textbox,
            lora2_dropdown,
            lora2_face_dropdown,
            random_seed
        ],
        outputs = [
            output1,
            output2
        ]
    )
    
    
demo.launch(server_name='0.0.0.0')
