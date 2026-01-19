import json
import logging
import os

from rtp_llm.tokenizer_factory.tokenizer_factory_register import _tokenizer_factory
from rtp_llm.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class TokenizerFactory:
    @staticmethod
    def create(ckpt_path: str, tokenizer_path: str, model_type: str):
        """
        Create a tokenizer from the given parameters.

        Args:
            ckpt_path: Path to the checkpoint directory
            tokenizer_path: Path to the tokenizer directory or file
            model_type: Type of the model (e.g., "gpt", "llama", etc.)

        Returns:
            A tokenizer instance
        """
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
