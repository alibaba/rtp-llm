"""DSpark geometry for the inherited PPU weight consumer."""

def draft_config(config):
    """Use real draft geometry rather than the target's 384-expert layers."""
    result = dict(config)
    layers = int(config["num_nextn_predict_layers"])
    if layers != 3:
        raise ValueError("V4.1 DSpark requires exactly three real draft layers")
    result.update(
        num_hidden_layers=layers,
        num_layers=layers,
        compress_ratios=[0] * layers,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        engram_layer_ids=[],
        engram_num_embeddings=[],
        n_routed_experts=int(config["dspark_n_routed_experts"]),
        num_experts_per_tok=int(config["dspark_num_experts_per_tok"]),
        original_seq_len=0,
        ppu_model_identity="dspark-draft",
    )
    return result
