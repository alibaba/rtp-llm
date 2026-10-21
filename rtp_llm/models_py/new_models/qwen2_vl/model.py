from typing import Callable

from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM


class Qwen2VLForCausalLM(Qwen2ForCausalLM):
    """Qwen2-VL language runtime with vision tensors filtered before loading."""

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return lambda name: not name.startswith("visual.")
