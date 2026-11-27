"""Checkpoint ownership for a DeepSeek V3-family score (non-MTP) model."""

from functools import lru_cache


def score_name_filter(active_depth, score_depth, appended_depth, *, resolve_appended=None):
    for value, label in ((active_depth, "active"), (score_depth, "checkpoint"), (appended_depth, "appended")):
        if type(value) is not int or value < 0 or (label != "appended" and value == 0):
            raise ValueError(f"invalid {label} layer count: {value!r}")
    if active_depth > score_depth:
        raise ValueError(f"requested {active_depth} score layers but checkpoint declares {score_depth}")

    @lru_cache(maxsize=1)
    def appended_range():
        if resolve_appended is not None and appended_depth <= 1:
            # A combined GLM checkpoint may omit the NextN count or retain
            # the original draft index after cropping. Only its EH projection
            # proves that an otherwise undeclared layer belongs to the draft.
            marker = resolve_appended()
            if marker is not None:
                if marker < score_depth:
                    raise ValueError(f"draft layer {marker} overlaps score depth {score_depth}")
                return range(marker, marker + 1)
        return range(score_depth, score_depth + appended_depth)

    def belongs(name):
        if name.startswith("model.layers."):
            layer, sep, _ = name[len("model.layers."):].partition(".")
            if sep and layer.isascii() and layer.isdigit():
                # Profiling may intentionally shorten the score model; appended
                # NextN layers have a different owner. A layer beyond both is
                # malformed, not another reason to silently drop a tensor.
                index = int(layer)
                if index < score_depth:
                    return index < active_depth
                return index not in appended_range()
        # Preserve unrecognized names: normal NewLoader dispatch must reject
        # them (multimodal parents apply their own namespace filter first).
        return True

    return belongs
