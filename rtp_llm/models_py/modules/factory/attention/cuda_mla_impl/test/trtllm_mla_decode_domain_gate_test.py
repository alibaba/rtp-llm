"""UT for the trtllm-gen MLA decode domain gate.

trtllm_mla_decode_domain_admitted decides whether the decode forward may
take the trtllm-gen branch: it must admit plain decode and MTP verify
(uniform q rows per request) and reject the ragged absorb-prefill batch —
including the uneven-but-divisible case that a naive tokens % qo_bs check
admits and that silently cuts request boundaries (the M-A9 regression
class) — plus the one-beat kv_bs lag and empty batches.
"""

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    trtllm_mla_decode_domain_admitted,
)


def _indptr(lens):
    t = torch.tensor([0] + list(lens), dtype=torch.int32)
    return t.cumsum(0).to(torch.int32)


def test_plain_decode_admitted():
    # 4 requests x 1 q row each, kv tensor records 4 requests
    assert trtllm_mla_decode_domain_admitted(_indptr([1, 1, 1, 1]), 4, 4, 4)


def test_mtp_verify_admitted():
    # 4 requests x (1 + 3 gen) q rows, kv tensor records 4 requests
    assert trtllm_mla_decode_domain_admitted(_indptr([4, 4, 4, 4]), 4, 4, 16)


def test_uneven_but_divisible_ragged_rejected():
    # The M-A9 class: per-request lens differ, tokens == qo_bs * q_len only
    # by coincidence of the first length; must NOT be admitted.
    assert not trtllm_mla_decode_domain_admitted(_indptr([2, 2, 2, 2, 4]), 5, 5, 12)
    assert not trtllm_mla_decode_domain_admitted(_indptr([8, 4, 8, 4]), 4, 4, 24)


def test_kv_bs_one_beat_lag_rejected():
    # kv length tensor still records the previous request count
    assert not trtllm_mla_decode_domain_admitted(_indptr([1, 1, 1, 1]), 4, 5, 4)
    assert not trtllm_mla_decode_domain_admitted(_indptr([1, 1, 1, 1]), 4, 3, 4)


def test_token_count_mismatch_rejected():
    # tokens != qo_bs * q_len (e.g. a stale batch size)
    assert not trtllm_mla_decode_domain_admitted(_indptr([1, 1, 1, 1]), 4, 4, 5)
    assert not trtllm_mla_decode_domain_admitted(_indptr([4, 4, 4, 4]), 4, 4, 15)


def test_empty_batch_rejected():
    assert not trtllm_mla_decode_domain_admitted(_indptr([0]), 0, 0, 0)
    assert not trtllm_mla_decode_domain_admitted(_indptr([]), 0, 0, 0)


def test_single_request_decode_admitted():
    assert trtllm_mla_decode_domain_admitted(_indptr([1]), 1, 1, 1)
    assert trtllm_mla_decode_domain_admitted(_indptr([4]), 1, 1, 4)


if __name__ == "__main__":
    test_plain_decode_admitted()
    test_mtp_verify_admitted()
    test_uneven_but_divisible_ragged_rejected()
    test_kv_bs_one_beat_lag_rejected()
    test_token_count_mismatch_rejected()
    test_empty_batch_rejected()
    test_single_request_decode_admitted()
    print("ALL DOMAIN GATE TESTS PASSED")
