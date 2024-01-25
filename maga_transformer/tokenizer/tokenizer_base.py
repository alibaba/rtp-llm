import os
from typing import Any, Dict, List, Optional, Union

class TokenizerBase(object):
    def encode(self, inputs: Union[str, List[Dict[str, str]]]) -> List[int]:
        raise NotImplementedError()

    def decode(self, outputs: List[int]) -> str:
        raise NotImplementedError()

    @property
    def chat_template(self) -> Optional[str]:
        return None

    @property
    def default_chat_template(self) -> Optional[str]:
        return None

    @property
    def additional_special_tokens(self) -> Optional[List[str]]:
        return None