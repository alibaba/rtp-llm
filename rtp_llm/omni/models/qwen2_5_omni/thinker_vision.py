from typing import Dict

import torch


class OmniVisionProcessor:
    def __init__(self, vision_config_dict: Dict):
        self.config = vision_config_dict
        self.vit = None

    @staticmethod
    def validate_config(vision_config: Dict) -> None:
        required = ["depth", "embed_dim", "num_heads", "patch_size"]
        for key in required:
            if key not in vision_config:
                raise ValueError(f"Missing required vision config key: {key}")

    @torch.inference_mode()
    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.vit is None:
            raise RuntimeError("Vision model not initialized. Load weights first.")
        return self.vit(pixel_values, grid_thw=grid_thw)
