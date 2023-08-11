from collections import defaultdict

import torch
from safetensors.torch import load_file


class LoraLoader:
    """
        LoRA loader that supports Kohya-ss format LoRA which cannot be properly loaded by diffusers
        modified from https://github.com/huggingface/diffusers/issues/3064#issuecomment-1512429695
    """
    def __init__(self):
        self._records = set()

    def load_lora_weights(self, pipeline, checkpoint_path, multiplier, device, dtype):
        pipeline_id = id(pipeline)
        if (pipeline_id, checkpoint_path) in self._records:
            raise ValueError("cannot load %s for pipeline<%s> twice, lora already loaded" % (checkpoint_path, pipeline_id))

        self._records.add((pipeline_id, checkpoint_path))
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        # print("find", curr_layer)
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
        return pipeline

    def unload_lora_weight(self, pipeline, checkpoint_path, multiplier, device, dtype):
        pipeline_id = id(pipeline)
        if (pipeline_id, checkpoint_path) not in self._records:
            raise ValueError("cannot unload %s for pipeline<%s>, lora not loaded" % (checkpoint_path, pipeline_id))
        return self.load_lora_weights(pipeline, checkpoint_path, multiplier=-1, device=device, dtype=dtype)

    def load_lora_for_pipelines(self, pipes, lora_configs):
        for pipe, lora_config in zip(pipes, lora_configs):
            for k, v in lora_config.items():
                print("loading %s with weight %s" %(k, v))
                pipe = self.load_lora_weights(pipe, k, v, 'cuda', torch.float32)

    def unload_lora_for_pipelines(self, pipes, lora_configs):
        for pipe, lora_config in zip(pipes, lora_configs):
            for k, v in lora_config.items():
                print("loading %s with weight %s" %(k, v))
                pipe = self.unload_lora_weight(pipe, k, v, 'cuda', torch.float32)