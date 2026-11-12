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

The 2026-11-12 re-audit is committed as `checkpoint_audit.json`, with all
504 global divisor values, scalar-region source bytes/hashes, header hashes,
common scalar metadata F32 `[1]`, and layer-0 local-scale metadata. Reproduce
with `scripts/audit_ct_nvfp4_checkpoint.py`; no whole shard is downloaded.

Format contract verified against the exact upstream PyPI wheel
`compressed-tensors==0.11.1a20250917`, SHA256
`b0edd61c7d2beb8ec956250ddd7525d3de2f5016105c355029543e64ed11c9a1`:
`nvfp4_quantized.py::pack_fp4_to_uint8` puts even K in the low nibble and odd
K in the high nibble; `decompress_weight` reads the scale tensor without an
unswizzle, and `lifecycle/forward.py` applies it to consecutive last-dimension
groups of 16. Scale storage is ordinary row-major `[N,K/16]`; backend-specific
swizzling is deferred to the RTP device hook. The source-function audit is
recorded in `docs/validation/ct_nvfp4_upstream_format_result.json` and can be
reproduced with `scripts/audit_ct_nvfp4_format.py`. This evidence is limited
to the pinned serializer version and does not establish native kernel accuracy.
