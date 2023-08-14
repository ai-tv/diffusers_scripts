
from fastapi import FastAPI, Request
from diffuser_scripts.tasks import LatentCoupleWithControlTaskParams
from diffuser_scripts.handle import handle_latent_couple_controlnet
from diffuser_scripts.model_manager import LatentCoupleConfig, LatentCouplePipelinesManager


model_config = LatentCoupleConfig.from_json('default_lc.json')
model_manager = LatentCouplePipelinesManager(model_config)

app = FastAPI()


@app.post("/task")
def handle(requst: Request):
    task = await requst.json()
    task = LatentCoupleWithControlTaskParams(**task)
    result = handle_latent_couple_controlnet(model_manager, task)
    return result.json()
