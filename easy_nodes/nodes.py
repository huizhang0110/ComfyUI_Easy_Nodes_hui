from .latent_nodes import EmptyLatentImage, LatentToCondition
from .image_nodes import LoadImage
from .bg_remover_nodes import BgRemover, BgRemover_ModelLoader
from .controlnet_nodes import EasyControlNetLoader


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # image
    "EasyLoadImage": LoadImage,
    # latent
    "EasyEmptyLatentImage": EmptyLatentImage,
    "EasyLatentToCondition": LatentToCondition,
    # bg_remover
    "EasyBgRemover": BgRemover,
    "EasyBgRemover_ModelLoader": BgRemover_ModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # image
    "EasyLoadImage": "Loading Image",
    # latent
    "EasyEmptyLatentImage": "empty latent",
    "EasyLatentToCondition": "latent to conditionaling",
    # bg_remover
    "EasyBgRemover": "Remove Background",
    "EasyBgRemover_ModelLoader": "Loading BgRemover model"
}
