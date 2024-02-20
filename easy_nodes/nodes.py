import torch

res_list = [
    {
        "width": 704,
        "height": 1408
    },
    {
        "width": 704,
        "height": 1344
    },
    {
        "width": 768,
        "height": 1344
    },
    {
        "width": 768,
        "height": 1280
    },
    {
        "width": 832,
        "height": 1216
    },
    {
        "width": 832,
        "height": 1152
    },
    {
        "width": 896,
        "height": 1152
    },
    {
        "width": 896,
        "height": 1088
    },
    {
        "width": 960,
        "height": 1088
    },
    {
        "width": 960,
        "height": 1024
    },
    {
        "width": 1024,
        "height": 1024
    },
    {
        "width": 1024,
        "height": 960
    },
    {
        "width": 1088,
        "height": 960
    },
    {
        "width": 1088,
        "height": 896
    },
    {
        "width": 1152,
        "height": 896
    },
    {
        "width": 1152,
        "height": 832
    },
    {
        "width": 1216,
        "height": 832
    },
    {
        "width": 1280,
        "height": 768
    },
    {
        "width": 1344,
        "height": 768
    },
    {
        "width": 1344,
        "height": 704
    },
    {
        "width": 1408,
        "height": 704
    },
    {
        "width": 1472,
        "height": 704
    },
    {
        "width": 1536,
        "height": 640
    },
    {
        "width": 1600,
        "height": 640
    },
    {
        "width": 1664,
        "height": 576
    },
    {
        "width": 1728,
        "height": 576
    }
]


MAX_RESOLUTION = 8192


class EasyEmptyLatentImage:

    resolution_dictionaly = None

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        if s.resolution_dictionaly is None:
            s.resolution_dictionaly = {}
            for item in res_list:
                width = item["width"]
                height = item["height"]
                aspect_ratio = "{:.2f}".format(round(width / height, 2))
                key = f"{width} x {height} ({aspect_ratio})"
                s.resolution_dictionaly[key] = item
        res_keys = list(s.resolution_dictionaly.keys())
        return {
            "required": {
                "resolution": (res_keys, ),
                "compression": ("INT", {"default": 42, "min": 32, "max": 64, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            },
            "optional": {
                "width": ("INT", {"min": 256, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"min": 256, "max": MAX_RESOLUTION, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b/f4")
    FUNCTION = "generate"
    CATEGORY = "⚡⚡⚡easy_nodes_hui/latent"
 
    def generate(self, resolution, compression, batch_size=1, width=None, height=None):
        height = height if height else self.resolution_dictionaly[resolution]["height"]
        width = width if width else self.resolution_dictionaly[resolution]["width"]
        c_latent = torch.zeros([batch_size, 16, height // compression, width // compression])
        b_latent = torch.zeros([batch_size, 4, height // 4, width // 4])
        return ({"samples": c_latent}, {"samples": b_latent})


class LatentToCondition:
    COND_KEYS = ["stable_cascade_prior"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": (s.COND_KEYS, ),
                "latent": ("LATENT", )
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "set_prior"
    CATEGORY = "⚡⚡⚡easy_nodes_hui/latent"

    def set_prior(self, key, latent):
        # Redundant data is retained to match the comfyui original data checker
        # TODO: redefined KSampler 
        data = [
            torch.zeros((1, 77, 1280)),
            {
                key: latent['samples'],
                "pooled_output": torch.zeros(1, 1280)
            }
        ]
        return ([data], )  



# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "EasyEmptyLatentImage": EasyEmptyLatentImage,
    "LatentToCondition": LatentToCondition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyEmptyLatentImage": "empty latent",
    "LatentToCondition": "latent to conditionaling"
}
