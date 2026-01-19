from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.embedding.render.common_input_generator import CommonInputGenerator
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


def lengths_to_slices(input_lengths: Sequence[int]) -> List[slice]:
    offset = 0
    input_slices: List[slice] = []
    for input_length in input_lengths:
        input_slices.append(slice(offset, offset + input_length))
        offset += input_length
    return input_slices


def combo_to_batch_input_ids(
    input_ids: torch.Tensor, input_slices: List[slice]
) -> torch.Tensor:
    return pad_sequence(
        [torch.IntTensor(input_ids[s]) for s in input_slices], batch_first=True
    )


def combo_to_batch_hidden_states(
    hidden_states: torch.Tensor, input_slices: List[slice]
) -> torch.Tensor:
    return pad_sequence([hidden_states[s] for s in input_slices], batch_first=True)


def combo_to_batch_moe_gating(
    moe_gating: List[Optional[torch.Tensor]], input_slices: List[slice]
) -> List[Optional[torch.Tensor]]:
    return [
        (
            pad_sequence([g[s] for s in input_slices], batch_first=True)
            if g is not None
            else None
        )
        for g in moe_gating
    ]


def generate_attention_mask(input_lengths: Sequence[int]) -> torch.Tensor:
    max_input_length: int = max(input_lengths)
    batch_size = len(input_lengths)
    device = torch.device(f"cuda:{g_parallel_info.local_rank}")
    batched_attention_mask = torch.ones(
        (batch_size, max_input_length), dtype=torch.bool, device=device
    )
    for b, input_length in enumerate(input_lengths):
        batched_attention_mask[b, input_length:] = 0
    return batched_attention_mask


def combo_to_batch_data(
    input_lengths: torch.Tensor, combo_data: Dict[str, Any]
) -> Dict[str, Any]:
    input_lengths_list = input_lengths.tolist()
    input_slices = lengths_to_slices(input_lengths_list)

    batch_data: Dict[str, Any] = {**combo_data}

    if "input_ids" in batch_data:
        batch_data["input_ids"] = combo_to_batch_input_ids(
            batch_data.pop("input_ids"), input_slices
        )

    if "hidden_states" in batch_data:
        batch_data["hidden_states"] = combo_to_batch_hidden_states(
            batch_data.pop("hidden_states"), input_slices
        )

    if "moe_gating" in batch_data:
        batch_data["moe_gating"] = combo_to_batch_moe_gating(
            batch_data.pop("moe_gating"), input_slices
        )

    if "attention_mask" in batch_data:
        batch_data["attention_mask"] = generate_attention_mask(input_lengths_list)

    return batch_data


def combo_to_batch(
    hidden_states: torch.Tensor, input_ids: torch.Tensor, input_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_lengths_list = input_lengths.tolist()
    input_slices = lengths_to_slices(input_lengths_list)
    batched_input_ids = combo_to_batch_input_ids(input_ids, input_slices)
    batched_hidden_states = combo_to_batch_hidden_states(hidden_states, input_slices)
    batched_attention_mask = generate_attention_mask(input_lengths_list)
    return batched_input_ids, batched_hidden_states, batched_attention_mask


def hidden_combo_to_batch(
    hidde_states: torch.Tensor, input_lengths: torch.Tensor
) -> torch.Tensor:
    sliced_hidden_states: List[torch.Tensor] = []
    hidden_bias = 0
    for input_length in input_lengths:
        sliced_hidden_states.append(
            hidde_states[hidden_bias : hidden_bias + input_length]
        )
        hidden_bias += input_length
    return pad_sequence(sliced_hidden_states, batch_first=True)


def combo_to_list(
    tensor: torch.Tensor, input_length: torch.Tensor
) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    bias = 0
    for length in input_length:
        result.append(tensor[bias : bias + length])
        bias += length
    return result
