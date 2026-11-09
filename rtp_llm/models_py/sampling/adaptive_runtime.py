import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.models_py.sampling.adaptive_controller import AdaptiveDraftController


def create_controller(
    max_num_requests: int, num_speculative_steps: int, device: str
) -> "AdaptiveDraftController | None":
    value = os.environ.get("RTP_LLM_ADAPTIVE_VERIFICATION", "0")
    if value not in ("0", "1"):
        raise ValueError("RTP_LLM_ADAPTIVE_VERIFICATION must be 0 or 1")
    if value == "0":
        return None

    import torch
    from rtp_llm.models_py.sampling.adaptive_controller import AdaptiveDraftController

    return AdaptiveDraftController(
        max_num_requests, num_speculative_steps, torch.device(device)
    )
