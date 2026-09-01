from typing import Any, Dict

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class HyV4Tokenizer(BaseTokenizer):
    """Load Hy4's generic fast-tokenizer files without parsing model config.

    Transformers 5 resolves ``AutoTokenizer`` through ``AutoConfig`` first.
    Hy4 is not an upstream Transformers architecture yet, so that route rejects
    its custom ``layer_types`` before reading the checkpoint's explicit
    ``TokenizersBackend`` declaration.  Loading the declared backend directly
    mirrors vLLM's custom-model tokenizer path and keeps the workaround scoped
    to Hy4.
    """

    def init_tokenizer(
        self, tokenizer_path: str, config_json: Dict[str, Any] = {}
    ) -> None:
        del config_json
        from transformers import TokenizersBackend

        self.tokenizer = TokenizersBackend.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            verbose=False,
        )


register_tokenizer(["hy_v4", "hy_v4_mtp"], HyV4Tokenizer)
