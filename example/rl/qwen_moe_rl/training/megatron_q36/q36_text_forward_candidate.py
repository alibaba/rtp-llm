"""Constrain verl's Qwen3.5-VL config to its Q36 text-only GPT forward.

Private candidate. Install before constructing the verl Megatron engine.
The original selector is untouched for all other model configs.
"""
from functools import wraps

_ARCH = ('Qwen3_5MoeForConditionalGeneration',)
_MODEL = 'qwen3_5_moe'
_TEXT_MODEL = 'qwen3_5_moe_text'


def install():
    import verl.models.mcore as mcore
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.utils import unwrap_model
    original = mcore.get_mcore_engine_forward_fn
    if getattr(original, '_q36_text_only_forward', False):
        return

    @wraps(original)
    def choose(hf_config):
        selected = original(hf_config)
        if (getattr(hf_config, 'model_type', None) != _MODEL
            or tuple(getattr(hf_config, 'architectures', ()) or ()) != _ARCH):
            return selected
        text = getattr(hf_config, 'text_config', None)
        if getattr(text, 'model_type', None) != _TEXT_MODEL:
            raise RuntimeError('Q36 conditional config has unexpected text_config')
        if selected is not gptmodel_forward_model_engine:
            raise RuntimeError('Q36 unexpected verl forward selector')
        @wraps(selected)
        def text_forward(model, input_ids, multi_modal_inputs, *args, **kwargs):
            real_model = unwrap_model(model)
            if type(real_model) is not GPTModel:
                raise RuntimeError('Q36 text forward requires the actual GPTModel provider')
            if multi_modal_inputs:
                raise ValueError('Q36 text-only provider rejects multimodal input')
            if args:
                raise RuntimeError('Q36 forward requires explicit keyword options')
            if kwargs.get('data_format', 'thd') != 'thd':
                raise RuntimeError('Q36 text-only candidate requires packed THD')
            kwargs['vision_model'] = False
            return selected(model, input_ids, multi_modal_inputs, **kwargs)
        return text_forward
    choose._q36_text_only_forward = True
    mcore.get_mcore_engine_forward_fn = choose
