from typing import Any, Dict, List, Optional, Union

import torch


def _to_list(val: Union[torch.Tensor, List]) -> List:
    return val.tolist() if isinstance(val, torch.Tensor) else val


def build_prompt_logits_dict(
    pl_raw: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if pl_raw is None:
        return None
    topk_logprobs = pl_raw["topk_logprobs"]
    if isinstance(topk_logprobs, torch.Tensor):
        actual_top_k = topk_logprobs.shape[1]
    elif topk_logprobs:
        actual_top_k = len(topk_logprobs[0])
    else:
        actual_top_k = 0
    result = {
        "start_pos": pl_raw.get("start_pos", 0),
        "end_pos": pl_raw.get("end_pos", 0),
        "top_k": actual_top_k,
        "topk_logprobs": _to_list(pl_raw["topk_logprobs"]),
        "topk_token_ids": _to_list(pl_raw["topk_token_ids"]),
    }
    if pl_raw.get("target_logprobs") is not None:
        result["target_logprobs"] = _to_list(pl_raw["target_logprobs"])
    return result
