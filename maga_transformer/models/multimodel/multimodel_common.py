from typing import Any, List, Union
import torch

from torchvision import transforms


class ImageTransform:

    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def encode(self, images: List[Any], device: Union[str, torch.device], dtype: torch.dtype) -> torch.Tensor:
        tensor_images = torch.stack(
            [self.image_transform(image) for image in images], dim=0
        ).to(device=device).to(dtype=dtype)
        return tensor_images