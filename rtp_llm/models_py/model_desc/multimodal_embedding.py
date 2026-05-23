from typing import Any


def embed_with_multimodal_features(embed_tokens: Any, inputs: Any) -> Any:
    input_ids = inputs.input_ids
    multimodal_features = getattr(inputs, "multimodal_features", None)
    text_tokens_mask = getattr(inputs, "text_tokens_mask", None)
    mm_features_locs = getattr(inputs, "mm_features_locs", None)

    if (
        multimodal_features is None
        or len(multimodal_features) == 0
        or text_tokens_mask is None
        or mm_features_locs is None
        or text_tokens_mask.numel() == 0
        or mm_features_locs.numel() == 0
    ):
        return embed_tokens(input_ids)

    if text_tokens_mask.numel() != input_ids.numel():
        raise ValueError(
            "text_tokens_mask size must match input_ids size for multimodal embedding overlay"
        )
    if mm_features_locs.numel() != len(multimodal_features):
        raise ValueError("mm_features_locs size must match multimodal feature count")

    safe_input_ids = input_ids.clone()
    text_tokens_mask = text_tokens_mask.to(device=input_ids.device)
    safe_input_ids[text_tokens_mask == 0] = 0
    hidden_states = embed_tokens(safe_input_ids)

    feature_locs = mm_features_locs.to(device="cpu").contiguous().tolist()
    for feature, loc in zip(multimodal_features, feature_locs):
        loc = int(loc)
        end = loc + feature.shape[0]
        if loc < 0 or end > hidden_states.shape[0]:
            raise ValueError(
                f"multimodal feature span [{loc}, {end}) is out of hidden states range {hidden_states.shape[0]}"
            )
        if feature.shape[1:] != hidden_states.shape[1:]:
            raise ValueError(
                f"multimodal feature shape {tuple(feature.shape)} does not match hidden shape {tuple(hidden_states.shape)}"
            )
        hidden_states[loc:end] = feature.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

    return hidden_states
