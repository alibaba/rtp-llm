import torch

from rtp_llm.ops.compute_ops import PyModelInputs

_ZERO_LAYER_HIDDEN_SIZE = 4
_VOCAB_SIZE = 8


def _build_word_embedding(
    *,
    dtype: torch.dtype = torch.float16,
    hidden_size: int = _ZERO_LAYER_HIDDEN_SIZE,
) -> torch.Tensor:
    # Keep every row distinguishable after BF16 conversion. A flat low-precision
    # arange loses unit increments at large hidden sizes and can let a wrong-row
    # lookup satisfy the oracle.
    rows = torch.arange(_VOCAB_SIZE, dtype=torch.float32, device="cuda").unsqueeze(1)
    columns = (
        torch.arange(hidden_size, dtype=torch.float32, device="cuda") % 16
    ).unsqueeze(0)
    return (rows * 0.5 + columns / 64).to(dtype)


def _build_inputs(
    *,
    with_text_tokens_mask: bool = True,
    dtype: torch.dtype = torch.float16,
    hidden_size: int = _ZERO_LAYER_HIDDEN_SIZE,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
        [0, 1, 2, 3], dtype=torch.int32, device="cuda"
    )
    inputs.bert_embedding_inputs.position_encoding = torch.zeros(
        (4, hidden_size), dtype=dtype, device="cuda"
    )
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.zeros(
        4, dtype=torch.int32, device="cuda"
    )
    inputs.bert_embedding_inputs.token_type_embedding = torch.zeros(
        (1, hidden_size), dtype=dtype, device="cuda"
    )
    if with_text_tokens_mask:
        inputs.embedding_inputs.text_tokens_mask = torch.ones(
            4, dtype=torch.int32, device="cuda"
        )
    return inputs
