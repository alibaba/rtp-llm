import hashlib
from collections.abc import Iterable, Sequence
from types import SimpleNamespace

_UINT64_MASK = (1 << 64) - 1


def _encode_type_and_value(major_type: int, value: int) -> bytes:
    if value < 0:
        raise ValueError("CBOR value must be non-negative")
    prefix = major_type << 5
    if value < 24:
        return bytes((prefix | value,))
    if value <= 0xFF:
        return bytes((prefix | 24, value))
    if value <= 0xFFFF:
        return bytes((prefix | 25,)) + value.to_bytes(2, "big")
    if value <= 0xFFFFFFFF:
        return bytes((prefix | 26,)) + value.to_bytes(4, "big")
    if value <= 0xFFFFFFFFFFFFFFFF:
        return bytes((prefix | 27,)) + value.to_bytes(8, "big")
    raise ValueError("CBOR integer exceeds 64 bits")


def _encode_integer(value: int) -> bytes:
    if value >= 0:
        return _encode_type_and_value(0, value)
    return _encode_type_and_value(1, -1 - value)


def _encode_bytes(value: bytes) -> bytes:
    return _encode_type_and_value(2, len(value)) + value


def _encode_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _encode_type_and_value(3, len(encoded)) + encoded


def _encode_integer_array(values: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in values)
    return _encode_type_and_value(4, len(values)) + b"".join(
        _encode_integer(value) for value in values
    )


def _hash_block(parent_hash: bytes, token_ids: Sequence[int]) -> bytes:
    encoded = b"".join(
        (
            _encode_type_and_value(4, 3),
            _encode_bytes(parent_hash),
            _encode_integer_array(token_ids),
            b"\xf6",
        )
    )
    return hashlib.sha256(encoded).digest()


def block_hash_to_uint64(block_hash: bytes | int) -> int:
    if isinstance(block_hash, int):
        return block_hash & _UINT64_MASK
    return int.from_bytes(block_hash, byteorder="big") & _UINT64_MASK


def stable_request_id(request_id: str) -> int:
    value = int.from_bytes(
        hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
    )
    return value or 1


def calculate_vllm_block_hashes(
    prompt_token_ids: Sequence[int],
    block_size: int,
    *,
    request=None,
) -> list[int]:
    """Return vLLM sha256_cbor block hashes as uint64 wire values."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return _hash_normalized_token_ids(
        tuple(map(int, prompt_token_ids)), block_size, request=request
    )


def _hash_normalized_token_ids(
    token_ids: tuple[int, ...],
    block_size: int,
    *,
    request=None,
) -> list[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    full_block_count = len(token_ids) // block_size
    parent_hash = hashlib.sha256(_encode_text("0")).digest()
    hash_request = None
    if request is not None:
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import (
            generate_block_hash_extra_keys,
            hash_block_tokens,
        )

        hash_request = SimpleNamespace(
            mm_features=getattr(request, "mm_features", None) or [],
            lora_request=getattr(request, "lora_request", None),
            cache_salt=getattr(request, "cache_salt", None),
            prompt_embeds=getattr(request, "prompt_embeds", None),
            _prompt_embeds_per_block_hashes={},
        )
    result = []
    mm_index = 0
    for block_index in range(full_block_count):
        begin = block_index * block_size
        end = begin + block_size
        if hash_request is None:
            parent_hash = _hash_block(parent_hash, token_ids[begin:end])
        else:
            extra_keys, mm_index = generate_block_hash_extra_keys(
                hash_request, begin, end, mm_index
            )
            parent_hash = hash_block_tokens(
                sha256_cbor, parent_hash, token_ids[begin:end], extra_keys
            )
        result.append(block_hash_to_uint64(parent_hash))
    return result
