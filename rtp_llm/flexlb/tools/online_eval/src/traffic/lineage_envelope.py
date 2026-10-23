"""Bounded XZ decoding shared by lineage detection and both persisted formats."""

import lzma

LIMIT = 128 * 1024 * 1024


def decompress(raw, error):
    decoder = lzma.LZMADecompressor(memlimit=LIMIT)
    data = decoder.decompress(raw, max_length=LIMIT)
    if not decoder.eof or decoder.unused_data:
        raise ValueError(error)
    return data
