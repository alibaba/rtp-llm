from typing import Any, Dict, Union

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer


def create_generate_config(
    generate_config: Union[GenerateConfig, Dict[str, Any]],
    vocab_size: int,
    special_tokens: Any,
    tokenizer: BaseTokenizer,
    generate_env_config,
    **kwargs: Any,
) -> GenerateConfig:
    """Create or enrich a GenerateConfig with special tokens and stop words."""
    if isinstance(generate_config, dict):
        config = GenerateConfig.create_generate_config(generate_config, **kwargs)
    else:
        config = generate_config
    config.add_special_tokens(special_tokens)
    config.convert_select_tokens(vocab_size, tokenizer)
    config.add_thinking_params(tokenizer, generate_env_config)
    config.add_stop_ids_from_str(tokenizer)
    return config
