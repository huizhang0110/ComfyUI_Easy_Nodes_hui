from .models.rmbg.briarmbg import BriaRMBG
import folder_paths
import torch 
from PIL import Image 
import numpy as np 
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import numpy as np
from folder_paths import models_dir, recursive_search
import os 

class BgRemover_ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        files, folders_all = recursive_search(os.path.join(models_dir, "bg_remover"))
        return {
            "required": {
                "ckpt_name": (files, ),
                "bg_remover_type": (["BriaRMBG"], )
            }
        }

    RETURN_NAMES = ("model", )
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "load_model"
    CATEGORY = "⚡⚡⚡easy_nodes_hui/remove_bg"

    def load_model(self, ckpt_name, bg_remover_type):
        if bg_remover_type == "BriaRMBG":
            net = BriaRMBG()
            model_path = os.path.join(models_dir, "bg_remover", ckpt_name)
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
        return (net, )


class BgRemover:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "⚡⚡⚡easy_nodes_hui/remove_bg"

    def remove_background(self, model, image):
        output_images = []
        output_masks = []
        model_input_size = [1024, 1024]
        # To reduce memory usage, single-batch inference is used here
        for ori_img_tensor in image:
            ori_img_pil = Image.fromarray(np.clip(255. * ori_img_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            ori_img_w, ori_img_h = ori_img_pil.size 
    
            input_img_pil = ori_img_pil.convert("RGB").resize(model_input_size, Image.BILINEAR)
            input_img_np = np.array(input_img_pil)
            input_img_tensor = torch.tensor(input_img_np, dtype=torch.float32).permute(2, 0, 1) # HWC 2 CHW
            input_img_tensor = input_img_tensor.unsqueeze(dim=0)
            input_img_tensor = torch.divide(input_img_tensor, 255.)
            input_img_tensor = normalize(input_img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            result = model(input_img_tensor)
            mask_tensor = F.interpolate(result[0][0], size=(ori_img_h, ori_img_w), mode='bilinear')
            mask_tensor = (mask_tensor - torch.min(mask_tensor)) / (torch.max(mask_tensor) - torch.min(mask_tensor))
            mask_np = (mask_tensor * 255).cpu().data.numpy().astype(np.uint8)
            mask_pil = Image.fromarray(np.squeeze(mask_np))
            fg_im_pil = Image.new("RGBA", mask_pil.size, (0, 0, 0, 0))
            fg_im_pil.paste(ori_img_pil, mask=mask_pil)

            output_images.append(torch.from_numpy(np.array(fg_im_pil).astype(np.float32) / 255.0).unsqueeze(0))
            output_masks.append(torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        return (output_image, output_mask)  # torch tensor

