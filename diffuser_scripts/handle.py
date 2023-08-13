from .tasks import LatentCoupleWithControlTaskParams
from .pipelines.couple import latent_couple_with_control


def handle_latent_couple_controlnet(
    pipelines,
    task: LatentCoupleWithControlTaskParams,
):
    for 