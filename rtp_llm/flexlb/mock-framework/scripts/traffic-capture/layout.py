def media_spans(media):
    """Convert text offsets to positions in the expanded sequence."""
    shift = 0
    for key, offset, count in media:
        start = offset + shift
        yield start, start + count, key
        shift += count


def text_at(media, expanded):
    return expanded - sum(
        max(0, min(end, expanded) - start) for start, end, _ in media_spans(media)
    )
