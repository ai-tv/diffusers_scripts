import os
from typing import Optional, Callable, Union, List

import torch
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet import MultiControlNetModel

from diffuser_scripts.utils import get_weighted_text_embeddings


def make_mask_list(
    pos=["1:1-0:0","1:2-0:0","1:2-0:1"], 
    weights = [0.7, 0.3, 0.3],
    width: int = 1024,
    height: int = 768
):
    mask_list = []
    batch_size = 1
    device = 'cuda'
    for idx in range(len(pos)):
        pos_base = pos[idx].split("-")
        pos_dev = pos_base[0].split(":")
        pos_pos = pos_base[1].split(":")
        one_filter = None
        zero_f = False                                      
        # dynamic weighting      
        weight = weights[idx]
        for y in range(int(pos_dev[0])):
            one_line = None
            zero = False
            for x in range(int(pos_dev[1])):
                if(y == int(pos_pos[0]) and x == int(pos_pos[1])):
                    if zero:
                        one_block = torch.ones(batch_size, 4, (height//8) // int(pos_dev[0]), (width//8) // int(pos_dev[1])).to(device).to(torch.float16) * weight
                        one_line = torch.cat((one_line, one_block), 3)
                    else:
                        zero = True
                        one_block = torch.ones(batch_size, 4, (height//8) // int(pos_dev[0]), (width//8) // int(pos_dev[1])).to(device).to(torch.float16) * weight
                        one_line = one_block
                else:
                    if zero:
                        one_block = torch.zeros(batch_size, 4, (height//8) // int(pos_dev[0]), (width//8) // int(pos_dev[1])).to(device).to(torch.float16)
                        one_line = torch.cat((one_line, one_block), 3)
                    else:
                        zero = True
                        one_block = torch.zeros(batch_size, 4, (height//8) // int(pos_dev[0]), (width//8) // int(pos_dev[1])).to(device).to(torch.float16)
                        one_line = one_block
            one_block = torch.zeros(batch_size, 4, (height//8) // int(pos_dev[0]), (width//8) - one_line.size()[3]).to(device).to(torch.float16)
            one_line = torch.cat((one_line, one_block), 3)
            if zero_f:
                one_filter = torch.cat((one_filter, one_line), 2)
            else:
                zero_f = True
                one_filter = one_line
        mask_list.append(one_filter)
    return mask_list


def prepare_image(
    self,
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    num_prompts,
    device,
    dtype,
    do_classifier_free_guidance=False,
    guess_mode=False,
):
    image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * (num_prompts + 1))

    return image



@torch.no_grad()
def latent_couple_with_control(
    pipes,
    prompts: str,
    image: Image.Image,
    couple_pos: List[str] = ["1:1-0:0","1:2-0:0","1:2-0:1"],
    couple_weights: List[float] = [0.7, 0.3, 0.3],
    negative_prompts = None,
    height: int = None,
    width: int = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
    guidance_scale: float = 8,
    num_inference_steps: int = 30,
    num_images_per_prompt: int = 1,
    latents = None,
    generator = None,
    eta: float = 0.0,
    cross_attention_kwargs = None,
    guess_mode = False,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 0.5,
    controlnet_conditioning_scale = 1.0,
    main_prompt_decay = 0.01,
):
    mask_list = make_mask_list(couple_pos, weights=couple_weights, width=width, height=height)
    prompt_embeddings = []
    for i, prompt in enumerate(prompts):
        text_embedding, uncond_embedding = get_weighted_text_embeddings(pipes[i], prompt=prompt, uncond_prompt=negative_prompts[i], )
        prompt_embeddings.append(text_embedding)
    prompt_embeds = prompt_embeddings
    negative_prompt_embed = uncond_embedding

    height = height or pipes[0].unet.config.sample_size * pipes[0].vae_scale_factor
    width = width or pipes[0].unet.config.sample_size * pipes[0].vae_scale_factor
    controlnet = pipes[0].controlnet._orig_mod if is_compiled_module(pipes[0].controlnet) else pipes[0].controlnet
    device = pipes[0].device

    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
            control_guidance_end
        ]
        
    # 1. Check inputs. Raise error if not correct
    pipes[0].check_inputs(None, image, callback_steps, None, prompt_embeds[0], negative_prompt_embed, controlnet_conditioning_scale, control_guidance_start, control_guidance_end)
    
    # 2. Define call parameters
    batch_size = prompt_embeds[0].shape[0]
    
    devices = [pipe._execution_device for pipe in pipes]
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    
    # 3. Encode input prompt
    if prompt_embeds is not None:
        text_embeddings = []
        for prompt_embed in prompt_embeddings:
            text_embeddings.append(torch.cat([negative_prompt_embed, prompt_embed], 0))    
    
    # 4. Prepare image
    if isinstance(controlnet, ControlNetModel):
        image = prepare_image(
            pipes[0],
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_prompts=len(prompts),
            device=pipes[0].device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        # height, width = image.shape[-2:]
    elif isinstance(controlnet, MultiControlNetModel):
        images = []
        for image_ in image:
            image_ = prepare_image(
                pipes[0],
                image=image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_prompts=len(prompts),
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
    
            images.append(image_)
    
        image = images
        # height, width = image[0].shape[-2:]
    else:
        assert False
    
    # 5. Prepare timesteps
    timesteps_list = []
    for pipe in pipes:
        pipe.scheduler.set_timesteps(num_inference_steps-1, device=pipes[0].device)
        timesteps = pipe.scheduler.timesteps
        timesteps_list.append(timesteps)
    
    # 6. Prepare latent variables
    num_channels_latents = pipes[0].unet.config.in_channels
    latents = pipes[0].prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        text_embeddings[0].dtype,
        devices[0],
        generator,
        latents,
    )
    
    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipes[0].prepare_extra_step_kwargs(generator, eta)
    
    # 7.1 Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
    # print(controlnet_keep)
    
    
    # 8. Denoising loop
    # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    for i, t in pipes[0].progress_bar(enumerate(timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * (len(text_embeddings) + 1)) if do_classifier_free_guidance else latents
        latent_model_input = pipes[0].scheduler.scale_model_input(latent_model_input, t)
    
        # predict the noise residual, with control net
        noise_preds = []                
        controlnet_prompt_embeds = []
        for idx in range(len(prompts)):
            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = pipes[0].scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = text_embeddings[idx].chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds.append(text_embeddings[idx].chunk(2)[1])
            # control_model_input = latents
            # controlnet_prompt_embeds = text_embeddings[idx][0:1]
    
        #inference negative prompt
        controlnet_prompt_embeds.append(text_embeddings[0].chunk(2)[0])
        controlnet_prompt_embeds = torch.cat(controlnet_prompt_embeds)
        
        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

        # print(control_model_input.shape, controlnet_prompt_embeds.shape, image.shape)
        down_block_res_samples, mid_block_res_sample = pipes[0].controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )
        down_block_res_samples_list, mid_block_res_sample_list = [], []
        for j in range(len(text_embeddings)):
            down_block_res_samples_list.append([torch.stack([d[-1], d[j]], dim=0) for d in down_block_res_samples])
            mid_block_res_sample_list.append(torch.stack([mid_block_res_sample[-1], mid_block_res_sample[j]], dim=0))
    
        if guess_mode and do_classifier_free_guidance:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        latent_couple = 0
        for j, (embed, ds, m) in enumerate(zip(text_embeddings, down_block_res_samples_list, mid_block_res_sample_list)):
            # print(embed.shape, latent_model_input.shape, t, m.shape)
            noise_pred = pipes[j].unet(
                latent_model_input[:2], 
                t,
                encoder_hidden_states=embed,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=ds,
                mid_block_additional_residual=m,
                return_dict=False,                        
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            this_mask =  torch.where(mask_list[j] > 0, mask_list[j] + 0.01 * i * (-1 if j == 0 else 1), mask_list[j])
            latent_couple += noise_pred * this_mask
        latents = pipes[0].scheduler.step(latent_couple, t, latents, **extra_step_kwargs).prev_sample

    output_image = pipes[0].decode_latents(latents)
    output_image = pipes[0].numpy_to_pil(output_image)
    return output_image[0]