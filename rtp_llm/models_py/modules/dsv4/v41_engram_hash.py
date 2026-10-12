# SPDX-License-Identifier: Apache-2.0
# Hashing adapted from vLLM (Copyright contributors to the vLLM project)
# and the DeepSeek V4.1 checkpoint inference reference.
"""Explicit-window Engram hashing, independent of host-table ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from sympy import nextprime


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys, lookup = {}, []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else normalizer.normalize_str(text) or text
        )
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return lookup, len(keys)


@dataclass(frozen=True)
class V41EngramLayout:
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[int, ...], ...]
    n_heads: int
    head_dim: int
    compressed_vocab_size: int
    pad_token_id: int
    max_ngram_size: int = 4

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, text: Mapping, pad_token_id: int) -> V41EngramLayout:
        layers = tuple(text["engram_layer_ids"])
        counts = tuple(text["engram_num_embeddings"])
        for key in (
            "engram_n_heads",
            "engram_head_dim",
            "engram_compressed_vocab_size",
            "engram_vocab_size",
        ):
            if type(text[key]) is not int or text[key] <= 0:
                raise ValueError(f"V4.1 {key} must be a positive integer")
        if (
            not layers
            or any(type(i) is not int or i < 0 for i in layers)
            or tuple(sorted(set(layers))) != layers
            or len(counts) != len(layers)
            or any(type(n) is not int or n <= 0 for n in counts)
            or text["engram_max_ngram_size"] != 4
            or type(pad_token_id) is not int
            or pad_token_id < 0
        ):
            raise ValueError(
                "V4.1 invalid Engram layers, tables, pad ID or window width"
            )
        seen, primes = set(), []
        for count in counts:
            row = []
            for _ in range(3):
                current = text["engram_vocab_size"] - 1
                for _ in range(text["engram_n_heads"]):
                    current = int(nextprime(current))
                    while current in seen:
                        current = int(nextprime(current))
                    seen.add(current)
                    row.append(current)
            if sum(row) != count:
                raise ValueError(
                    "V4.1 Engram prime buckets do not match checkpoint table rows"
                )
            primes.append(tuple(row))
        return cls(
            layers,
            counts,
            tuple(primes),
            text["engram_n_heads"],
            text["engram_head_dim"],
            text["engram_compressed_vocab_size"],
            pad_token_id,
        )


@dataclass(frozen=True)
class V41EngramHashes:
    ids: torch.Tensor  # int64 [tokens, layers, 3 * heads], including pad-derived hashes
    valid_rows: torch.Tensor  # bool [tokens]; required by lookup/injection


class V41EngramHasher(torch.nn.Module):
    """No request-local history. Move constants to the device before execution."""

    def __init__(self, layout: V41EngramLayout, token_map: Sequence[int]):
        super().__init__()
        mapping = tuple(token_map)
        if (
            not mapping
            or any(
                type(i) is not int or not 0 <= i < layout.compressed_vocab_size
                for i in mapping
            )
            or set(mapping) != set(range(layout.compressed_vocab_size))
            or layout.pad_token_id >= len(mapping)
        ):
            raise ValueError("V4.1 Engram compressed vocabulary or pad ID mismatch")
        self.layout = layout
        self.pad_id = mapping[layout.pad_token_id]
        bound = max(1, (np.iinfo(np.int64).max // layout.compressed_vocab_size) // 2)
        multipliers = np.stack(
            [
                np.random.default_rng(10007 * layer).integers(
                    0, bound, size=4, dtype=np.int64
                )
                * 2
                + 1
                for layer in layout.layer_ids
            ]
        )
        offsets = np.array(
            [np.cumsum([0, *row[:-1]]) for row in layout.primes], dtype=np.int64
        )
        for name, tensor in (
            ("token_map", torch.tensor(mapping, dtype=torch.int64)),
            ("multipliers", torch.from_numpy(multipliers)),
            ("primes", torch.tensor(layout.primes, dtype=torch.int64)),
            ("offsets", torch.from_numpy(offsets)),
        ):
            self.register_buffer(name, tensor, persistent=False)

    def forward(
        self, windows: torch.Tensor, *, backend: str = "triton"
    ) -> V41EngramHashes:
        if (
            windows.ndim != 2
            or windows.shape[1] != 4
            or windows.dtype not in (torch.int32, torch.int64)
            or not windows.is_contiguous()
            or windows.device != self.token_map.device
        ):
            raise ValueError(
                "V4.1 Engram requires contiguous integer [tokens, 4] on the constants' device"
            )
        valid_rows = (windows[:, 0] >= 0) & (windows[:, 0] < self.token_map.numel())
        if backend == "eager":
            # Stop every longer n-gram at the first absent/dead token.
            valid = (windows >= 0) & (windows < self.token_map.numel())
            alive = valid.to(torch.int32).cumprod(dim=1).bool()
            mapped = self.token_map[windows.long().clamp(0, self.token_map.numel() - 1)]
            mapped = torch.where(alive, mapped, self.pad_id)
            rolling = mapped[:, None, 0] * self.multipliers[None, :, 0]
            hashes = []
            for lag in range(1, 4):
                rolling = rolling ^ (
                    mapped[:, None, lag] * self.multipliers[None, :, lag]
                )
                start = (lag - 1) * self.layout.n_heads
                primes = self.primes[:, start : start + self.layout.n_heads]
                hashes.append(rolling[..., None] % primes)
            ids = torch.cat(hashes, dim=-1) + self.offsets
        elif backend == "triton":
            from rtp_llm.models_py.modules.dsv4._v41_engram_hash_triton import (
                hash_windows,
            )

            ids = hash_windows(windows, self)
        else:
            raise ValueError("V4.1 invalid Engram hashing backend")
        return V41EngramHashes(ids, valid_rows)
