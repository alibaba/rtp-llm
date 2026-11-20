# -*- coding: utf-8 -*-
# @Time    : 2024/10/23 15:54
# @Author  : xianqi
# @File    : configuration_mbert.py
# @Desc    :
from transformers import BertConfig, SwinConfig


class MBertRelConfig(BertConfig):

    model_type = "mbert"

    def __init__(
        self,
        num_labels=2,
        vision_token_window=2,
        vision_hidden_size=None,
        vision_encoder_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_labels = num_labels
        self.vision_token_window = vision_token_window
        self.vision_hidden_size = vision_hidden_size
        if vision_encoder_config:
            self.vision_encoder_config = SwinConfig(**vision_encoder_config)
        else:
            self.vision_encoder_config = SwinConfig()

    def to_dict(self):
        output = super().to_dict()
        output["vision_encoder_config"] = self.vision_encoder_config.to_dict()
        return output
