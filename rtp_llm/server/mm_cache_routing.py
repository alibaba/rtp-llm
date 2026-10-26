from array import array


def multimodal_length_tokens(
    token_ids, separators, include_separators, lengths, max_seq_len
):
    """Expand complete media spans without assigning any KV-cache identity."""
    from libth_transformer_config import get_multimodal_token_spans

    spans = get_multimodal_token_spans(token_ids, separators, include_separators)
    if len(spans) != len(lengths) or not spans:
        raise ValueError("one metadata length is required per media placeholder")
    if any(type(n) is not int or not 0 < n < (1 << 31) for n in lengths):
        raise ValueError("invalid multimodal token length")
    total = len(token_ids) + sum(lengths) - sum(end - begin for begin, end in spans)
    if total >= max_seq_len:
        raise ValueError("expanded multimodal input exceeds max_seq_len")
    output, expanded, cursor = array("i"), [], 0
    for (begin, end), length in zip(spans, lengths):
        output.extend(token_ids[cursor:begin])
        expanded.append((len(output), length))
        output.extend(array("i", [0]) * length)
        cursor = end
    output.extend(token_ids[cursor:])
    return output, expanded
