from typing import Dict, List, Any
import os
import re

import torch
import torch.nn as nn

from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from maga_transformer.models.multimodal_mixin import BaseImageEmbedding

class LlavaImageEmbedding(BaseImageEmbedding):
    def __init__(self, config: Dict[str, Any]):
        self.vision_tower = self.build_vision_tower(config).to(device='cuda:0')
        self.mm_projector = self.build_vision_projector(config)
        self.config = config
    
    def image_embedding(self, images: List[Image.Image], device):
        processed_images = process_images(images, 
                                          self.config.get('image_aspect_ratio', None), 
                                          self.vision_tower.image_processor, 
                                          device)

        image_features = self.encode_images(processed_images)

        return image_features
    
    def encode_images(self, images):
        if images.shape[0] == 0:
            return images
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features
    
    def build_vision_tower(self, vision_tower_cfg: Dict[str, Any], **kwargs: Any):
        vision_tower = os.environ.get('LOCAL_EXTRA_DATA_PATH', None)
        if vision_tower is None:
            vision_tower = vision_tower_cfg['vit_tower_path']
        is_absolute_path_exists = os.path.exists(vision_tower)
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
    def build_vision_projector(self, config, delay_load=False, **kwargs):
        projector_type = config.get('mm_projector_type', 'linear')

        if projector_type == 'linear':
            return torch.nn.Linear(config['mm_hidden_size'], config['hidden_size'], device='cuda:0')

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(config['mm_hidden_size'], config['hidden_size'], device='cuda:0')]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(torch.nn.Linear(config['hidden_size'], config['hidden_size'], device='cuda:0'))
            return torch.nn.Sequential(*modules)

        if projector_type == 'identity':
            return IdentityMap()

        raise ValueError(f'Unknown projector type: {projector_type}')

# ViT
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args['mm_vision_select_layer']
        self.select_feature = args.get('mm_vision_select_feature', 'patch')
        self.args = args

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

# Projector
class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

# image preprocess
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def process_batch_images(batch_images, image_aspect_ratio, image_processor, device):
    batch_new_images = []
    for images in batch_images:
        batch_new_images.append(process_images(images, image_aspect_ratio, image_processor, device))
    return batch_new_images

def process_images(images, image_aspect_ratio, image_processor, device):
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        if len(new_images) == 0:
            new_images = torch.tensor([])
        else:
            new_images = torch.stack(new_images, dim=0)

    if type(new_images) is list:
        new_images = [image.to(device, dtype=torch.float16) for image in new_images]
    else:
        new_images = new_images.to(device, dtype=torch.float16)

    return new_images