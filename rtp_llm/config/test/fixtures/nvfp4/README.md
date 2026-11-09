# NVFP4 fixture provenance

Source: https://huggingface.co/RedHatAI/Qwen3-8B-NVFP4
Revision: `e391349c110709b87bfc2ad2fde3f50dc5839fd8` (Apache-2.0).
Fetched 2026-11-09. These fixtures are derived from that model, not generated weights.

- `config.json`: original checkpoint configuration, SHA256
  `5996fa7a47b6fff7563e320024d005a76054d3b36725a3068390570559dcbc6a`.
- `q_proj_tile.json`: first 128 output rows and first 128 input channels of
  `model.layers.0.self_attn.q_proj`, extracted with bounded HTTP Range reads
  from shard 1. Packed shape `[128,64]`, local scale shape `[128,8]`.
  Contains original packed codes, E4M3 scale bytes, and both original FP32
  global divisors. Each byte payload has its own SHA256 in the JSON.
  This is a sliced matrix, not a complete layer or a runnable model.

The complete checkpoint header and global-scale audit covered 36 layers,
504 global scalars, and 144 fused-global equality checks (two scale kinds
for QKV and gate/up at each layer). All equality checks passed. This does
not establish numerical serving accuracy or kernel performance.
