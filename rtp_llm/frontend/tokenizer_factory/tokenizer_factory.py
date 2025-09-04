import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Type, Union

from transformers import AutoTokenizer

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    _tokenizer_factory,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.utils.fuser import fetch_remote_file_to_local
from rtp_llm.utils.util import check_with_info


class TokenizerFactory:
    @staticmethod
    def create_from_env():
        ckpt_path = StaticConfig.model_config.checkpoint_path
        tokenizer_path = StaticConfig.model_config.tokenizer_path
        model_type = StaticConfig.model_config.model_type

        tokenizer_path = fetch_remote_file_to_local(tokenizer_path)
        ckpt_path = fetch_remote_file_to_local(ckpt_path)

        return TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)

    @staticmethod
    def create_from_config(config):
        ckpt_path = config.ckpt_path
        tokenizer_path = config.tokenizer_path
        model_type = StaticConfig.model_config.model_type
        return TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)

    @staticmethod
    def create(ckpt_path: str, tokenizer_path: str, model_type: str):
        global _tokenizer_factory
        config_json = {}
        config_json_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as reader:
                text = reader.read()
                config_json = json.loads(text)

        if _tokenizer_factory.get(model_type) is None:
            logging.info(
                f"not register special tokenizer, use transformers.AutoTokenizer for model {model_type}"
            )
            return BaseTokenizer(tokenizer_path, config_json)
        else:
            return _tokenizer_factory[model_type](tokenizer_path, config_json)
