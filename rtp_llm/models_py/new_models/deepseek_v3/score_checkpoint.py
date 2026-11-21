"""Checkpoint ownership for a DeepSeek V3-family score (non-MTP) model."""


def score_name_filter(active_depth, score_depth, appended_depth):
    for value, label in ((active_depth, "active"), (score_depth, "checkpoint"), (appended_depth, "appended")):
        if type(value) is not int or value < 0 or (label != "appended" and value == 0):
            raise ValueError(f"invalid {label} layer count: {value!r}")
    if active_depth > score_depth:
        raise ValueError(f"requested {active_depth} score layers but checkpoint declares {score_depth}")

    def belongs(name):
        if name.startswith("model.layers."):
            layer, sep, _ = name[len("model.layers."):].partition(".")
            if sep and layer.isascii() and layer.isdigit():
                # Profiling may intentionally shorten the score model; appended
                # NextN layers have a different owner. A layer beyond both is
                # malformed, not another reason to silently drop a tensor.
                return not active_depth <= int(layer) < score_depth + appended_depth
        # Preserve unrecognized names: normal NewLoader dispatch must reject
        # them (multimodal parents apply their own namespace filter first).
        return True

    return belongs
