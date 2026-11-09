from typing import Any, Dict

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class HyV4Tokenizer(BaseTokenizer):
    """Load Hy4's generic fast-tokenizer files without parsing model config.

    ``PreTrainedTokenizerFast`` reads the serialized backend directly in
    Transformers 4 and aliases ``TokenizersBackend`` in Transformers 5.
    This preserves tokenizer metadata without requiring AutoConfig to
    recognize HY4's custom architecture.
    """

    def init_tokenizer(
        self, tokenizer_path: str, config_json: Dict[str, Any] = {}
    ) -> None:
        del config_json
        from transformers import PreTrainedTokenizerFast

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            verbose=False,
        )


register_tokenizer(["hy_v4", "hy_v4_mtp"], HyV4Tokenizer)
