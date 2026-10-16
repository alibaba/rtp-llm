"""In-process registry pairing the DiffusionGemma main model with its proposer.

ModelFactory creates the target model first and the speculative proposer
second (same process). The proposer's weight info is empty (it borrows the
target's backbone modules directly), so it resolves the live main model here
instead of loading a second copy of the 26B checkpoint.
"""

import threading
from typing import Dict, Optional, Tuple

_MAIN_MODELS: Dict[Tuple[str, int], object] = {}
_LOCK = threading.Lock()


def register_main_model(model, ckpt_path: str) -> None:
    key = (str(ckpt_path), int(getattr(model, "layer_num", 0)))
    with _LOCK:
        _MAIN_MODELS[key] = model


def get_main_model(ckpt_path: str, layer_num: int) -> Optional[object]:
    key = (str(ckpt_path), int(layer_num))
    with _LOCK:
        return _MAIN_MODELS.get(key)


def reset() -> None:
    with _LOCK:
        _MAIN_MODELS.clear()
