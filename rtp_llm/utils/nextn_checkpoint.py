"""Identify a physical NextN draft layer without loading its tensors."""

import re


_EH_WEIGHT = re.compile(r"model\.layers\.(0|[1-9][0-9]*)\.eh_proj\.weight")


def nextn_layer_from_names(names):
    layers = {int(match[1]) for name in names if (match := _EH_WEIGHT.fullmatch(name))}
    if len(layers) > 1:
        raise ValueError(
            f"ambiguous NextN checkpoint: found at least two EH projections in layers {sorted(layers)}"
        )
    return next(iter(layers), None)
