import json
import torch
from PIL import Image
from asdff import AdPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

from sbp.io import SequenceFileReader
from sbp.io.common.encoder_decoder import decode_b64
from diffuser_scripts.utils.lora_loader import LoraLoader
from diffuser_scripts.utils.long_prompt_weighting import get_weighted_text_embeddings

SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def decode_feature(s):
    obj = json.loads(s)
    if len(obj) == 0:
        return np.zeros(512, dtype=np.float16)
    else:
        return decode_b64(obj[0]['embedding'].encode())  


pipe = AdPipeline.from_pretrained(
    "/home/zwshi/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/", 
    torch_dtype=torch.float16
)
pipe.safety_checker = None
pipe.to("cuda")
scheduler = DPMSolverMultistepScheduler(        
    num_train_timesteps=SCHEDULER_TIMESTEPS,
    beta_start=SCHEDULER_LINEAR_START,
    beta_end=SCHEDULER_LINEAR_END,
    beta_schedule=SCHEDLER_SCHEDULE,
    algorithm_type='dpmsolver++'
)
pipe.scheduler = scheduler

id_mlp_path = '/data/model_zoo/lora/038_mix_with_me.safetensors'
lora_loader = LoraLoader()
lora_loader.load_lora_for_pipelines(
    [pipe], 
    [{id_mlp_path: 1}]
)

prompt = "a photo of young thin face, good-looking, best quality"
negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)"
text_embedding, uncond_embedding = get_weighted_text_embeddings(pipe, prompt, negative_prompt)
common = {
    "num_inference_steps": 30,
    'strength': 0.5,
    'width': 640,
    'height': 640
}

ssf = SequenceFileReader('../../diffusers_scripts/result.list')
id_mlp = torch.load(id_mlp_path + '_mlp.pt').to('cuda').eval()

id_batch = [
    ('/data/raw/characters/stars/samples/万茜/34.png', 'woman'),
    ('/data/raw/characters/stars/samples/胡歌/99.png', 'man'),
    ('/data/raw/characters/stars/samples/林心如/4.png', 'woman'),
    ('/data/raw/characters/stars/samples/张国荣/8.png', 'man'),
    ('/data/raw/characters/stars/samples/陈妍希/272.png', 'woman'),
    ('/data/raw/characters/stars/samples/刘德华/212.png', 'man'),
    ('/data/raw/characters/stars/samples/娄艺潇/4.png', 'woman'),
    ('/data/raw/characters/stars/samples/张译/77.png', 'man'),
    ('/data/raw/characters/stars/samples/唐嫣/43.png', 'woman'),
    ('/data/raw/characters/stars/samples/彭于晏/73.png', 'man'),
    ('/data/raw/characters/stars/samples/姚晨/753.png', 'woman'),
    ('/data/raw/characters/stars/samples/王俊凯/17.png', 'man'),
    ('/data/raw/characters/stars/samples/杨幂/2300.png', 'woman'),
    ('/data/raw/characters/stars/samples/周杰伦/269.png', 'man'),
    ('/data/raw/characters/stars/samples/张雨绮/2.png', 'woman'),
    ('/data/raw/characters/stars/samples/谢霆锋/340.png', 'man'),
    ('/data/raw/characters/stars/samples/陈乔恩/4605.png', 'woman'),
    ('/data/raw/characters/stars/samples/易烊千玺/7.png', 'man'),
]
reverse = [-1, 1, -1, 1, 1, 1, 1, -1, -1, -1]
init_dir = '/home/zwshi/projects/diffusers_scripts/test_result/default/'
for pair in zip(id_batch[::2], id_batch[1::2]):
    (n1, _), (n2, _) = pair
    n1 = n1.split('/')[-2]
    n2 = n2.split('/')[-2]
    for i in range(10):
        inpaint_args = []
        image = Image.open(init_dir + "%s_%s_%d.jpg" % (n1, n2, i))
        for k, s in pair[::reverse[i]]:
            k = k.replace(
                '/data/raw/characters/stars/samples/',
                '/mnt/lg104/character_dataset/preprocessed_v0/'
            )
            f = torch.FloatTensor(decode_feature(ssf.read(k)))[None, ...].cuda()
            f = id_mlp(f)
            p = torch.cat([f[None, ], text_embedding, ], dim=1)
            u = torch.cat([f[None, ], uncond_embedding], dim=1)
            inpaint_args.append({
                "prompt_embeds": p,
                "negative_prompt_embeds": u,
            })

        result = pipe(
            common = common, 
            images = image,
            inpaint_only = inpaint_args
        )

        im = result.init_images[0]
        # im.save("init_%s_%s_%d.jpg" % (n1, n2, i))
        im = result.images[0]
        im.save("%s_%s_%d.jpg" % (n1, n2, i))