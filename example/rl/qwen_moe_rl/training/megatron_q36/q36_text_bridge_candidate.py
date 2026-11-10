"""Opt-in Qwen3.6-35B text-only AutoBridge for verl's Megatron engine.

Private candidate: this is not a complete training/serving qualification.
It preserves official HF source names and packed expert representation.  It
does not install hooks globally unless `install_for_verl()` is called.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge

CONFIG_SHA = '93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99'
TEXT_NAMES_SHA = '150cceb3506d4c99db89e0aac7da7abe09d17a345f9210564624b7cebd437b5a'

def validate_checkpoint(path):
    root = Path(path)
    data = (root / 'config.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIG_SHA:
        raise ValueError('Q36 text bridge requires the pinned official 35B config')
    config = json.loads(data)
    index = json.loads((root / 'model.safetensors.index.json').read_bytes())['weight_map']
    names = sorted(n for n in index if n == 'lm_head.weight' or n.startswith('model.language_model.'))
    if len(names) != 693 or hashlib.sha256('\n'.join(names).encode()).hexdigest() != TEXT_NAMES_SHA:
        raise ValueError('Q36 text bridge requires the pinned 693 official source names')
    for layer in range(40):
        prefix = f'model.language_model.layers.{layer}.mlp.experts.'
        if prefix+'gate_up_proj' not in index or prefix+'down_proj' not in index:
            raise ValueError(f'missing packed experts for layer {layer}')
    return config

class Q36TextModelBridge(Qwen35MoEBridge):
    def provider_bridge(self, hf_pretrained):
        config = hf_pretrained.config
        text = getattr(config, 'text_config', None)
        if text is None:
            raise ValueError('Q36 requires a conditional-generation config with text_config')
        return super().provider_bridge(SimpleNamespace(config=text))

    def mapping_registry(self):
        # Explicitly map the text-only GPTModel parameters to the full official
        # VL checkpoint text names; no vision or MTP mappings are admitted.
        return MegatronMappingRegistry(*self._get_moe_lm_mappings(
            hf_prefix='model.language_model.', megatron_prefix='', experts_packed=True))

class Q36TextAutoBridge(AutoBridge):
    @classmethod
    def from_hf_pretrained(cls, path, **kwargs):
        validate_checkpoint(path)
        return super().from_hf_pretrained(path, **kwargs)

    @property
    def _model_bridge(self):
        bridge = Q36TextModelBridge()
        bridge.export_weight_dtype = self.export_weight_dtype
        return bridge

def install_for_verl():
    """Install on this process only, before verl builds its Megatron engine."""
    import verl.models.mcore.bridge as verl_bridge
    if verl_bridge.AutoBridge not in (AutoBridge, Q36TextAutoBridge):
        raise RuntimeError('another AutoBridge override is already installed')
    verl_bridge.AutoBridge = Q36TextAutoBridge
