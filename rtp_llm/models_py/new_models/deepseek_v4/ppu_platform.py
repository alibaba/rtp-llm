"""Lightweight platform qualification shared by V4 service and parameter owners."""


def is_ppu_flash(config):
    name = getattr(config, "model_type", None)
    if name not in ("deepseek_v4", "deepseek_v4_mtp"):
        return False
    from rtp_llm.device.device_type import is_ppu

    return (
        is_ppu()
        and config.hidden_size == 4096
        and config.num_layers == (1 if name == "deepseek_v4_mtp" else 43)
    )
