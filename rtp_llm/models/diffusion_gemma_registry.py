"""In-process registry pairing the DiffusionGemma main model with its proposer.

ModelFactory creates the target model first and the speculative proposer
second (same process). The proposer's weight info is empty (it borrows the
target's backbone modules directly), so it resolves the live main model here
instead of loading a second copy of the 26B checkpoint.
"""

import threading
from typing import Dict, Optional

_MAIN_MODELS: Dict[str, object] = {}
_LOCK = threading.Lock()


def register_main_model(model, ckpt_path: str) -> None:
    # Keyed by checkpoint path only: the proposer's own layer count (1)
    # differs from the main model's (30), so a (path, layers) key can never
    # match across the pair.
    with _LOCK:
        _MAIN_MODELS[str(ckpt_path)] = model


def get_main_model(ckpt_path: str, layer_num: int = 0) -> Optional[object]:
    del layer_num  # retained for call-site compatibility
    with _LOCK:
        return _MAIN_MODELS.get(str(ckpt_path))


def reset() -> None:
    with _LOCK:
        _MAIN_MODELS.clear()
