from diffuser_scripts.pipelines.couple import latent_couple_with_control
from diffuser_scripts.model_manager import LatentCouplePipelinesManager
from diffuser_scripts.tasks import LatentCoupleWithControlTaskParams, ImageGenerationResult


def handle_latent_couple_controlnet(
    model_manager: LatentCouplePipelinesManager,
    task: LatentCoupleWithControlTaskParams,
) -> ImageGenerationResult:
    pipelines = model_manager.pipelines

    with model_manager.lock:
        model_manager.load_loras(task.lora_configs)
        output_image = latent_couple_with_control(
            pipelines,
            task.prompt,
            image=task.condition_image,
            couple_pos=task.couple_pos,
            couple_weights=task.couple_weights,
            negative_prompts=task.negative_prompt,
            height=task.height,
            width=task.width
        )
        model_manager.unload_loras()
    return ImageGenerationResult.from_task_and_image(task, output_image)
