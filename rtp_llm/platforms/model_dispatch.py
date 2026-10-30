"""Select an implementation only when constructing a target or draft model.

Registry/config/weight-class queries keep their original classes. This module
does not register or instantiate models; non-PPU and non-Flash calls return
the supplied class without importing PPU implementations.
"""


def resolve_model_class(name, default, config=None):
    if name not in ("deepseek_v4", "deepseek_v4_mtp") or config is None:
        return default
    from rtp_llm.device.device_type import is_ppu

    if not is_ppu():
        return default
    if config.hidden_size != 4096 or config.num_layers != (1 if name == "deepseek_v4_mtp" else 43):
        return default
    from .ppu.models.dsv4.model import PpuDeepSeekV4, PpuDeepSeekV4Mtp

    return {"deepseek_v4": PpuDeepSeekV4, "deepseek_v4_mtp": PpuDeepSeekV4Mtp}[name]
