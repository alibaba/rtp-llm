from typing import Any, Dict, Optional

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class BertTokenizer(BaseTokenizer):
    def _additional_kwargs(self, tokenizer_config: Dict[str, Any]) -> Dict[str, Any]:
        do_lower_case = self._infer_do_lower_case(tokenizer_config)
        if do_lower_case is not None:
            return {"do_lower_case": do_lower_case}
        return {}

    @staticmethod
    def _infer_do_lower_case(tokenizer_config: Dict[str, Any]) -> Optional[bool]:
        """Workaround for transformers==5.2.0 do_lower_case default change.

        In 5.2.0, BertTokenizer.__init__ defaults do_lower_case=False, whereas
        4.x's BertTokenizerFast defaulted to True. For models whose
        tokenizer_config.json lacks an explicit do_lower_case field but whose
        name_or_path contains "uncased" (e.g. bert-base-uncased, toxic-bert),
        the correct behavior is lowercase.
        """
        if "do_lower_case" in tokenizer_config:
            return None
        if "uncased" in tokenizer_config.get("name_or_path", "").lower():
            return True
        # None = fall back to transformers' default.
        # Uncased models lacking the "uncased" hint should set do_lower_case in config.
        return None

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_token_id


register_tokenizer(["bert", "roberta", "vision_bert"], BertTokenizer)
