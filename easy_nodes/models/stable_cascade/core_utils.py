import os
import torch
import json
import safetensors

def load_or_fail(path):
    accepted_extensions = [".pt", ".ckpt", ".json", ".safetensors"]
    try:
        assert any(
            [path.endswith(ext) for ext in accepted_extensions]
        ), f"Automatic loading not supported for this extension: {path}"
        if not os.path.exists(path):
            checkpoint = None
        elif path.endswith(".pt") or path.endswith(".ckpt"):
            checkpoint = torch.load(path, map_location="cpu")
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        elif path.endswith(".safetensors"):
            checkpoint = {}
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
        return checkpoint
    except Exception as e:
        raise e