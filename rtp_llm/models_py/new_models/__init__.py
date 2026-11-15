"""Explicit production registrations; importing loader core stays backend-free."""

from rtp_llm.models_py.registry import register_lazy_model

for _name, _package, _class in (
    ("qwen_2", "qwen2", "Qwen2ForCausalLM"),
    ("qwen_3", "qwen3", "Qwen3ForCausalLM"),
    ("llama", "llama", "LlamaForCausalLM"),
):
    register_lazy_model(
        _name, f"rtp_llm.models_py.new_models.{_package}.language", _class
    )
