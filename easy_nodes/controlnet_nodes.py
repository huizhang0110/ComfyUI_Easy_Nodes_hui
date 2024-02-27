
import folder_paths
from folder_paths import models_dir
import comfy
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import torch
import safetensors
import safetensors.torch
import os 
import torch.nn as nn
from .models.stable_cascade.controlnet import ControlNet as StableCascadeControlNet


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    if current_batch_size == 1:
        return tensor 
    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]
    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)
    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor 
    else:
        return torch.cat([tensor] * batched_number, dim=0)



class ControlBase:

    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None 
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.timestep_range = None 
        self.global_average_pooling = False

        if device is None:
            device = comfy.model_management.get_torch_device()
        self.device = device 
        self.previous_controlnet = None 

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0)):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (
            percent_to_timestep_function(self.timestep_percent_range[0]),
            percent_to_timestep_function(self.timestep_percent_range[-1])
        )
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
    
    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None 
        self.timestep_range = None 

    def get_model(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out 
    
    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original 
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.global_average_pooling = self.global_average_pooling 

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0
    
    def control_merge(self, control_input, control_output, control_prev, output_dtype):
        out = {"input": [], "middle": [], "output": []}
        
        if control_input is not None:
            for i in range(len(control_input)):
                key = "input"
                x = control_input[i]
                if x is not None:
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                    out[key].insert(0, x)
        
        if control_output is not None:
            for i in range(len(control_output)):
                if i == (len(control_output) - 1):
                    key = "middle"
                    index = 0
                else:
                    key = "output"
                    index = i 
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out[key].append(x)
        
        if control_prev is not None:
            for x in ["input", "middle", "output"]:
                o = out[x]
            for i in range(len(control_prev[x])):
                prev_val = control_prev[x][i]
                if i >= len(o):
                    o.append(prev_val)
                elif prev_val is not None:
                    if o[i] is None:
                        o[i] = prev_val
                    else:
                        if o[i].shape[0] < prev_val.shape[0]:
                            o[i] = prev_val + o[i]
                        else:
                            o[i] += prev_val
        
        return out 



class ControlNet(ControlBase):

    def __init__(self, control_model, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(device)
        self.control_model = control_model 
        self.load_device = load_device 
        self.control_model_wrapped = comfy.model_patcher.ModelPatcher(
            self.control_model, load_device=load_device, 
            offload_device=comfy.model_management.unet_offload_device()
        )
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None 
        self.manual_cast_dtype = manual_cast_dtype

    def get_control(self, x_noisy, t, cond, batched_sampler):
        control_prev = None 
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_sampler)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None 
        
        dtype = self.control_model.dtype 
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype 
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 3 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None 
            self.cond_hint = comfy.utils.common_upscale(
                self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, "nearest-exact", "center").to(dtype).to(self.device)
        
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_sampler)

        context = cond.get("crossattn_controlnet", cond["c_crossattn"])

        y = cond.get("y", None)
        if y is not None:
            y = y.to(dtype)
        
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)
        
        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=context.to(dtype), y=y)
        return self.control_merge(control_input=None, control_output=control, control_prev=control_prev, output_dtype=output_dtype)

    def copy(self):
        c = ControlNet(
            self.control_model, global_average_pooling=self.global_average_pooling, 
            load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype
        )
        self.copy_to(c)
        return c 

    def get_model(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out 

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)

    def cleanup(self):
        self.model_sampling_current = None 
        super().cleanup()



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

