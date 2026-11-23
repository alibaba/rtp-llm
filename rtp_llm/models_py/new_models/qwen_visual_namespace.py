"""Qwen3.5 exporters may nest the same visual tower below different wrappers."""


def is_visual_checkpoint_name(name: str) -> bool:
    return name.startswith("visual.") or ".visual." in name
