
import sys
import os

import torch

from typing import Optional, Tuple

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights


def set_trace_on_tty():
    """
    启动一个连接到当前终端的 PDB 会话。
    在 Unix-like 系统上工作。
    """
    try:
        import pdb
        tty_r = open('/dev/tty', 'r')
        tty_w = open('/dev/tty', 'w')
        pdb.Pdb(stdin=tty_r, stdout=tty_w).set_trace()
    except OSError as e:
        print(f"Warning: Could not open /dev/tty: {e}. Skipping pdb.")
        import traceback
        traceback.print_exc()
        pass

class GptModelBase:
    def __init__(self, params: GptInitModelParameters, weight: ModelWeights) -> None:
        self.params = params
        self.weight = weight
        print("model base initialized")

    def forward(self,
                combo_tokens: torch.Tensor,
                input_lengths: torch.Tensor,
                sequence_lengths: torch.Tensor,
                attention_mask: torch.Tensor,
                kv_cache_block_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("model base forward called")
        return (
            torch.zeros_like(combo_tokens, dtype=torch.float32),
            torch.zeros_like(combo_tokens, dtype=torch.float32),
        )
