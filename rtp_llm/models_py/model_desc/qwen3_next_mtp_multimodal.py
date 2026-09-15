"""Inject multimodal embeddings already aligned by the MTP input processor."""


def mtp_word_embedding(embed_tokens, injector, inputs):
    """Use the global MTP shift followed by the optional CP token remap.

    MtpBatchStreamProcessor moves the mask, feature rows and locations together
    with token IDs before TP broadcast. The CP processor subsequently slices all
    of them with the same index map; moving anything here would shift it twice.
    """
    multimodal_features = inputs.multimodal_inputs.multimodal_features
    if not multimodal_features:
        return embed_tokens(inputs.input_ids)

    text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
    if text_tokens_mask is None or text_tokens_mask.numel() == 0:
        raise ValueError("Qwen3.5 MTP multimodal prefill requires text_tokens_mask")
    if text_tokens_mask.numel() != inputs.input_ids.numel():
        raise ValueError("MTP multimodal mask length does not match token IDs")
    inputs_embeds = embed_tokens(inputs.input_ids, text_tokens_mask=text_tokens_mask)
    return injector(
        inputs_embeds,
        multimodal_features,
        inputs.multimodal_inputs.mm_features_locs,
    )
