from typing import Any, Dict, Optional

from rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer import (
    QWenV2Tokenizer,
)


class JevTokenizer(QWenV2Tokenizer):
    """Load the published JEV tokenizer with official AutoTokenizer semantics.

    JEV's reference service calls AutoTokenizer directly. Passing the generic
    RTP tokenizer_object workaround would instead restore the raw JSON backend;
    for Qwen2 this changes token boundaries around some Unicode combining marks.
    Keep this exception local to checkpoints carrying JEV task metadata.
    """

    def init_tokenizer(
        self, tokenizer_path: str, config_json: Optional[Dict[str, Any]] = None
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.tokenizer.im_start_id = self.tokenizer.encode(
            "<|im_start|>", add_special_tokens=False
        )[0]
        self.tokenizer.im_end_id = self.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )[0]
