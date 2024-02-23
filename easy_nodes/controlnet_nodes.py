
import folder_paths
from folder_paths import models_dir
import comfy
import comfy.model_management
import torch
import safetensors
import safetensors.torch
import os 
import torch.nn as nn
from .models.stable_cascade.controlnet import ControlNet as StableCascadeControlNet


def load_controlnet(ckpt_path, model=None):
    if ckpt_path.lower().endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    return state_dict


class ControlNet(ControlBase):
    pass 


stable_cascade_canny_config = {
    ""
}


class EasyControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "control_net_type": (["stable_cascade_canny", ])
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "⚡⚡⚡easy_nodes_hui/controlnet"

    def load_controlnet(self, control_net_name, control_net_type="stable_cascade_canny"):

        ckpt_path = os.path.join(models_dir, "controlnet", control_net_name)
        if ckpt_path.lower().endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        control_model = StableCascadeControlNet()
        missing_keys, unexpected_keys = control_model.load_state_dict(state_dict=state_dict, strict=False)
        if len(missing_keys) > 0:
            print("Missing: ", missing_keys)
        if len(unexpected_keys) > 0:
            print("Unexpected: ", unexpected_keys)

        global_average_pooling = False
        filename = os.path.splitext(ckpt_path)[0]
        if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
            global_average_pooling = True

        load_device = comfy.model_management.get_torch_device()
        unet_dtype = comfy.model_management.unet_dtype(supported_dtypes=[torch.float16, torch.bfloat16, torch.float32])
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
        control = ControlNet(
            control_model, 
            global_average_pooling=global_average_pooling,
            load_device=load_device, 
            manual_cast_dtype=manual_cast_dtype
        )

        return (control, )


class EasyControlNetApply:

    @classmethod
    def INPUT_TYPE(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "apply_controlnet"
    CATEGORY = "⚡⚡⚡easy_nodes_hui/controlnet"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        if strength == 0:
            return conditioning
        c = []
        control_hint = image.movedim(-1, 1)  # BHWC, BCHW
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]["control"])
            n[1]["control"] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )

