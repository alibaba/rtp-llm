"""Phase D: lock down the CP B=1 fresh-prefill alignment between
``_get_window_topk_idxs_varlen`` and the all-gathered ``kv_full[seq_len_full]``.

Loads the function via importlib (avoids the heavyweight rtp_llm package
init / .so binding) so the test stays a pure CPU unit test.

The contract under verification:
  Q is rank-local (zigzag, T_local = chunk_length tokens, each with
  GLOBAL position in ``position_ids``). KV after ``cp_all_gather_full``
  is the full sequence ``[seq_len_full]`` in GLOBAL request order. The
  varlen topk index for Q token t (global pos g) under B=1 fresh
  (prefix=0, cu_seqlens=[0, T_local]) must be:
      [max(0, g-win+1), g]   right-padded to ``win`` with -1
  i.e. exactly the global flat indices into ``kv_full``.
"""

import importlib.util
import sys
import types

import torch


def _load_attention_helper():
    # Stub the package chain so attention.py's relative imports don't drag
    # in the .so-bound rtp_llm root. We only need `_get_window_topk_idxs_varlen`
    # which depends on torch + torch.nn.functional alone.
    pkg_root = "rtp_llm"
    if pkg_root not in sys.modules:
        for n in [
            "rtp_llm",
            "rtp_llm.models_py",
            "rtp_llm.models_py.modules",
            "rtp_llm.models_py.modules.dsv4",
        ]:
            sys.modules[n] = types.ModuleType(n)
    # Inline lift just the function from the source file (avoids importing
    # attention.py's body which pulls in compute_ops etc.).
    src_path = "rtp_llm/models_py/modules/dsv4/attention.py"
    with open(src_path) as f:
        src = f.read()
    start = src.index("def _get_window_topk_idxs_varlen(")
    # Find the next top-level "def "/"class " to bound the function
    end = src.index("\nclass SwaPrefillMeta", start)
    snippet = "import torch\n" "import torch.nn.functional as F\n" + src[start:end]
    mod = types.ModuleType("varlen_topk_helper")
    exec(compile(snippet, "<varlen_topk_helper>", "exec"), mod.__dict__)
    return mod._get_window_topk_idxs_varlen


_get_window_topk_idxs_varlen = _load_attention_helper()


def _zigzag_global_positions(
    seq_len_full: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Reproduce the rank-local global position list ZigZagProcessor produces."""
    pair = seq_len_full // (cp_size * 2)
    first = torch.arange(cp_rank * pair, (cp_rank + 1) * pair, dtype=torch.int32)
    second = torch.arange(
        seq_len_full - (cp_rank + 1) * pair,
        seq_len_full - cp_rank * pair,
        dtype=torch.int32,
    )
    return torch.cat([first, second])


def _expected_window_indices_for_global_pos(g: int, win: int) -> list:
    """Right-padded list of global KV positions covered by Q at global pos g."""
    start = max(0, g - win + 1)
    indices = list(range(start, g + 1))
    return indices + [-1] * (win - len(indices))


def test_cp_b1_fresh_topk_indices_are_global_flat() -> None:
    """CP=2, B=1, fresh prefill, win=4, seq_len_full=16: each rank-local
    Q token's flat indices == ``[max(0, g-win+1)..g]`` right-padded — i.e.
    the GLOBAL flat indices into ``kv_full[seq_len_full]``."""
    seq_len_full = 16
    cp_size = 2
    win = 4
    chunk_length = seq_len_full // cp_size  # 8
    prefix_lengths = torch.tensor([0], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, chunk_length], dtype=torch.int32)

    for r in range(cp_size):
        position_ids = _zigzag_global_positions(seq_len_full, cp_size, r)
        req_id_per_token = torch.zeros(chunk_length, dtype=torch.int32)
        topk = _get_window_topk_idxs_varlen(
            win, cu_seqlens, position_ids, prefix_lengths, req_id_per_token
        )
        assert topk.shape == (chunk_length, win)
        for t in range(chunk_length):
            g = int(position_ids[t].item())
            expected = _expected_window_indices_for_global_pos(g, win)
            actual = topk[t].tolist()
            assert (
                actual == expected
            ), f"rank={r} t={t} g={g}: expected {expected}, got {actual}"
            for v in actual:
                assert (
                    v < seq_len_full
                ), f"flat index {v} out of bounds for kv_full[{seq_len_full}]"


def test_cp_b1_fresh_topk_matches_no_cp_when_collapsed() -> None:
    """CP=1 with the same fresh-prefill config produces identical indices
    (per row) to CP=2 — by construction, since the formula is positional."""
    seq_len_full = 16
    win = 4
    prefix_lengths = torch.tensor([0], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, seq_len_full], dtype=torch.int32)
    position_ids = torch.arange(seq_len_full, dtype=torch.int32)
    req_id_per_token = torch.zeros(seq_len_full, dtype=torch.int32)
    topk_no_cp = _get_window_topk_idxs_varlen(
        win, cu_seqlens, position_ids, prefix_lengths, req_id_per_token
    )

    # Gather both ranks of CP=2 by global position and compare row-by-row.
    cp_size = 2
    chunk = seq_len_full // cp_size
    cu_seqlens_local = torch.tensor([0, chunk], dtype=torch.int32)
    rebuilt = torch.full((seq_len_full, win), -2, dtype=topk_no_cp.dtype)
    for r in range(cp_size):
        pos = _zigzag_global_positions(seq_len_full, cp_size, r)
        req_local = torch.zeros(chunk, dtype=torch.int32)
        topk_r = _get_window_topk_idxs_varlen(
            win, cu_seqlens_local, pos, prefix_lengths, req_local
        )
        for t in range(chunk):
            g = int(pos[t].item())
            rebuilt[g] = topk_r[t]
    assert torch.equal(
        rebuilt, topk_no_cp
    ), "CP=2 reassembled topk_idxs differs from CP=1 reference"


def test_cp_topk_indices_never_exceed_seq_len_full() -> None:
    """Defensive bound: regardless of cp_size / rank, no flat index should
    point past ``kv_full[seq_len_full] - 1``."""
    win = 8
    for cp_size in (2, 4):
        seq_len_full = 32
        chunk = seq_len_full // cp_size
        prefix_lengths = torch.tensor([0], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, chunk], dtype=torch.int32)
        for r in range(cp_size):
            pos = _zigzag_global_positions(seq_len_full, cp_size, r)
            req = torch.zeros(chunk, dtype=torch.int32)
            topk = _get_window_topk_idxs_varlen(
                win, cu_seqlens, pos, prefix_lengths, req
            )
            real = topk[topk >= 0]
            assert int(real.max().item()) < seq_len_full


if __name__ == "__main__":
    test_cp_b1_fresh_topk_indices_are_global_flat()
    test_cp_b1_fresh_topk_matches_no_cp_when_collapsed()
    test_cp_topk_indices_never_exceed_seq_len_full()
    print("OK")
