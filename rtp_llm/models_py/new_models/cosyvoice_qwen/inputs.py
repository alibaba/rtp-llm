"""Inject external speech features only at the model's embedding stage."""

import torch


def embed_external_features(embedding, injector, inputs):
    ids = inputs.input_ids
    mask = getattr(getattr(inputs, "embedding_inputs", None), "text_tokens_mask", None)
    if mask is not None and mask.numel():
        if mask.shape != ids.shape:
            raise ValueError("CosyVoice text_tokens_mask must match input_ids")
        mask = mask.to(device=ids.device, dtype=torch.bool)
        # Speech placeholders (-200) are not entries in either vocabulary.
        hidden = embedding(torch.where(mask, ids, 0)) * mask.unsqueeze(-1)
    else:
        hidden = embedding(ids)
    multimodal = getattr(inputs, "multimodal_inputs", None)
    features = getattr(multimodal, "multimodal_features", None)
    if not features:
        return hidden
    return injector(hidden, features, multimodal.mm_features_locs)
