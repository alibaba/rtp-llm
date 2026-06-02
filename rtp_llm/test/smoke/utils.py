import json
import logging
import os
import re
import shutil
import tempfile
from typing import Any

from smoke.common_def import REL_PATH


def create_temporary_copy(rel_path: str):
    match = re.match(r"^data:[^,]*;base64,", rel_path)
    if match:
        return rel_path
    if rel_path.startswith("http"):
        return rel_path
    path = os.path.abspath(os.path.join(REL_PATH, rel_path))
    tmp = tempfile.NamedTemporaryFile(delete=False)
    shutil.copy2(path, tmp.name)
    return tmp.name


def save_logits() -> bool:
    return os.environ.get("SAVE_LOGITS", "False") == "True"


def save_hidden_states() -> bool:
    return os.environ.get("SAVE_HIDDEN_STATES", "False") == "True"


def save_response() -> bool:
    return os.environ.get("SAVE_RESPONSE", "False") == "True"


def no_compare() -> bool:
    return save_hidden_states() or save_logits() or save_response()


_PROMPT_CACHE = None

def _load_prompt_candidates():
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE
    candidates_path = os.path.join(REL_PATH, "data", "prompt_candidates.json")
    if not os.path.exists(candidates_path):
        _PROMPT_CACHE = {}
        return _PROMPT_CACHE
    with open(candidates_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _PROMPT_CACHE = {k: v["content"] for k, v in data.get("prompts", {}).items()}
    logging.info("Loaded %d prompt candidates from %s", len(_PROMPT_CACHE), candidates_path)
    return _PROMPT_CACHE


def resolve_prompt_refs(obj: Any) -> Any:
    """Recursively replace '$prompt:xxx' references with actual prompt content."""
    if isinstance(obj, str) and obj.startswith("$prompt:"):
        pid = obj[len("$prompt:"):]
        prompts = _load_prompt_candidates()
        if pid not in prompts:
            raise ValueError(f"Unknown prompt candidate ID: '{pid}'. Available: {list(prompts.keys())}")
        return prompts[pid]
    elif isinstance(obj, dict):
        return {k: (v if k == "_prompt_source" else resolve_prompt_refs(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_prompt_refs(v) for v in obj]
    return obj
