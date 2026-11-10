import torch

from rtp_llm.models_py.modules.dsv41.rocm_sparse_mla import _ragged_indices


def test_ragged_indices_uses_fixed_width_stride() -> None:
    indices = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.int32)
    flat, indptr = _ragged_indices(indices, None, 2, 3)
    assert flat.tolist() == [3, 4, 5, 6, 7, 8]
    assert indptr.tolist() == [0, 3, 6]


def test_ragged_indices_masks_beyond_topk_length() -> None:
    indices = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.int32)
    lengths = torch.tensor([2, 1], dtype=torch.int32)
    flat, indptr = _ragged_indices(indices, lengths, 2, 3)
    assert flat.tolist() == [3, 4, -1, 6, -1, -1]
    assert indptr.tolist() == [0, 3, 6]


def test_ragged_indices_1m_rows_does_not_overflow_int32() -> None:
    rows = 1_048_576
    indices = torch.zeros((rows, 2), dtype=torch.int32)
    _, indptr = _ragged_indices(indices, None, rows, 2)
    assert int(indptr[-1]) == rows * 2
    assert indptr.dtype == torch.int32