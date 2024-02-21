# ComfyUI Custom Nodes

Custom nodes that extend the capabilities of [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Installing 

To manually install, simply clone this repo into the custom_nodes directory with this command:

```
cd ComfyUI/custom_nodes
git clone https://github.com/huizhang0110/ComfyUI_Easy_Nodes_hui
```

and install the requirements using 

```
..\..\..\python_embeded\python.exe -s -m pip install -r .\requirements.txt
```

## ✴️ The Nodes

| Name | Description | ComfyUI category |
|:--------:|:-----------:|:-------:|
| EmptyLatentImage | The preset resolution plays a critical role in SD models, and this tool assists in identifying resolutions with various ratios from the SDXL resolution set. Moreover, it seamlessly integrates with the latest **stable cascade**. | ⚡⚡⚡easy_nodes_hui/latent  |
| LoadImage | As original comfyui LoadImage, for learning use | ⚡⚡⚡easy_nodes_hui/image  |
| LatentToCondition | Facilitates stable cascade stage-c latent conversion into stage-b condition | ⚡⚡⚡easy_nodes_hui/latent  |
| BgRemover | Isolating the foreground goals that we care about, e.g., people, objects, animals, etc.  | ⚡⚡⚡easy_nodes_hui/remove_bg  |
| BgRemover_ModelLoader | Loading model for `BgRemover` | ⚡⚡⚡easy_nodes_hui/remove_bg  |

## Examples 



