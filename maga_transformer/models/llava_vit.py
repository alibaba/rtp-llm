from typing import Dict, List, Any
import os
import re

import torch
import torch.nn as nn

from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from maga_transformer.models.multimodal_mixin import BaseImageEmbedding
from maga_transformer.models.llava_utils import expand2square, process_anyres_image, unpad_image, get_anyres_image_grid_shape

class LlavaImageEmbedding(BaseImageEmbedding):
    def __init__(self, config: Dict[str, Any]):
        if config.get("vision_config", None) != None:
            raise Exception("llava-hf style config is not implemented yet")
            # self.vision_tower = CLIPVisionModel(config["vision_config"]).cuda().half()
        else:
            self.vision_tower = self.build_vision_tower(config).cuda().half()
        self.mm_projector = self.build_vision_projector(config).cuda().half()
        if "unpad" in config.get("mm_patch_merge_type", "flat"):
            self.image_newline = nn.Parameter(
                torch.empty(config["hidden_size"]).cuda().half()
            )
        self.config = config
    
    @torch.no_grad()
    def image_embedding(self, images: List[Image.Image], device):
        image_aspect_ratio = self.config["image_aspect_ratio"]
        mm_patch_merge_type = self.config.get("mm_patch_merge_type", "flat")

        processed_images = process_images(images, 
                                          image_aspect_ratio, 
                                          self.vision_tower.image_processor, 
                                          device,
                                          image_grid_pinpoints = self.config.get("image_grid_pinpoints", []))
        
        processed_images = [image.unsqueeze(0) if image.ndim == 3 else image for image in processed_images]
        split_sizes = [processed_image.shape[0] for processed_image in processed_images]
        processed_images = torch.cat(processed_images)
        image_features = self.encode_images(processed_images)
        image_features = list(torch.split(image_features, split_sizes, dim=0))

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            image_sizes = [image.size for image in images]
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.vision_tower.num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == 'anyres':
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config["image_grid_pinpoints"], self.vision_tower.config.image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        raise NotImplementedError
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((
                            image_feature,
                            self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                        ), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            self.image_newline[None].to(image_feature.device)
                        ), dim=0)
                new_image_features.append(image_feature)
            image_features = new_image_features

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
            return CLIPVisionTower(vision_tower,
                                   select_layer=vision_tower_cfg.get("mm_vision_select_layer", -2),
                                   select_feature=vision_tower_cfg.get("mm_vision_select_feature", "patch"),
                                   **kwargs)
        
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
    def __init__(self, vision_tower, select_layer=-2, select_feature="patch", delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = select_layer
        self.select_feature = select_feature

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
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

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

def process_batch_images(batch_images, image_aspect_ratio, image_processor, device, **kwargs):
    batch_new_images = []
    for images in batch_images:
        batch_new_images.append(process_images(images, image_aspect_ratio, image_processor, device, **kwargs))
    return batch_new_images

def process_images(images, image_aspect_ratio, image_processor, device, **kwargs):
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, kwargs.get('image_grid_pinpoints', []))
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']

    if type(new_images) is list:
        new_images = [image.to(device, dtype=torch.float16) for image in new_images]
    else:
        new_images = new_images.to(device, dtype=torch.float16)

    return new_images