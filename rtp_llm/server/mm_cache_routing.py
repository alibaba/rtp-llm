from typing import Any, Dict, List, Optional, Tuple


def multimodal_routing_tokens(
    token_ids: List[int],
    separators: List[List[int]],
    include_separators: bool,
    keys: List[str],
    metadata: Optional[Dict[str, Any]],
    max_seq_len: int,
) -> Tuple[List[int], Optional[int]]:
    """Return a verified routing prefix and the full length only on a complete hit."""
    from libth_transformer_config import get_multimodal_token_spans

    spans = get_multimodal_token_spans(token_ids, separators, include_separators)
    if not spans:
        return [], None
    safe_prefix = token_ids[: spans[0][0]]
    if not keys or not metadata or metadata.get("feature_hash_version") != 1:
        return safe_prefix, None
    entries = {entry["key"]: entry for entry in metadata["entries"]}
    output, cursor, segment = [], 0, 0
    for key in keys:
        entry = entries.get(key)
        if not entry or not entry.get("hit"):
            if segment >= len(spans):
                return safe_prefix, None
            output.extend(token_ids[cursor : spans[segment][0]])
            return output, None
        sizes, hashes = entry["split_size"], entry["feature_hashes"]
        if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError("invalid multimodal segment lengths")
        if sum(sizes) != len(hashes) or any(
            type(h) is not int or not -(1 << 31) <= h < (1 << 31) for h in hashes
        ):
            raise ValueError("invalid multimodal row hashes")
        offset = 0
        for size in sizes:
            if segment >= len(spans):
                raise ValueError("multimodal metadata exceeds tag count")
            begin, end = spans[segment]
            if len(output) + begin - cursor + size >= max_seq_len:
                return safe_prefix, None
            output.extend(token_ids[cursor:begin])
            output.extend(hashes[offset : offset + size])
            offset += size
            cursor, segment = end, segment + 1
    if segment != len(spans):
        raise ValueError("multimodal metadata does not match tag count")
    output.extend(token_ids[cursor:])
    if len(output) >= max_seq_len:
        return safe_prefix, None
    return output, len(output)
