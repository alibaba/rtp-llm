"""Request row lengths for the eager PPU Prefill projections."""

from contextlib import contextmanager
from contextvars import ContextVar

_lengths = ContextVar("ppu_prefill_request_lengths", default=())
_hc_pair = ContextVar("ppu_prefill_hc_pair", default=None)


@contextmanager
def prefill_request_rows(boundaries, total_rows, *, enabled, hc_pair=None):
    lengths = ()
    if (
        enabled
        and boundaries is not None
        and not boundaries.is_cuda
        and boundaries.numel() > 2
    ):
        offsets = boundaries.tolist()
        if offsets[0] == 0 and offsets[-1] == total_rows:
            lengths = tuple(end - start for start, end in zip(offsets, offsets[1:]))
            if any(length <= 0 for length in lengths):
                lengths = ()
    token = _lengths.set(lengths)
    pair_token = _hc_pair.set(hc_pair)
    try:
        yield
    finally:
        _hc_pair.reset(pair_token)
        _lengths.reset(token)


def request_row_lengths(rows):
    lengths = _lengths.get()
    return lengths if lengths and sum(lengths) == rows else ()


def prefill_hc_pair():
    return _hc_pair.get()
