from typing import Sequence


def apply_mrope_section(
    rope_config,
    mrope_section: Sequence[int],
    *,
    model_name: str,
    interleaved: bool,
) -> None:
    """Apply the common three-axis T/H/W MRoPE configuration contract."""
    if len(mrope_section) != 3:
        raise ValueError(
            f"{model_name} mrope_section must contain exactly 3 T/H/W sections, "
            f"got {len(mrope_section)}"
        )
    rope_config.index_factor = 3
    rope_config.mrope_dim1 = mrope_section[0]
    rope_config.mrope_dim2 = mrope_section[1]
    rope_config.mrope_dim3 = mrope_section[2]
    rope_config.mrope_interleaved = interleaved
