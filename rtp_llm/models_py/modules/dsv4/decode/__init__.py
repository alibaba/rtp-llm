"""DeepSeek-V4 decode-only ops & metadata.

This subpackage hosts the decode-path pieces — FP8 FlashMLA attention
kernels, compute primitives, FP8 KV write, indexer op, and the per-step
attention metadata builder. Strictly **decode-only**: nothing in here
touches prefill code, so PD-disagg can later split prefill/decode
services on a clean cleavage line.

FP8-only as of the rebase branch (see
:meth:`rtp_llm.models_py.modules.dsv4.attention.Attention.forward_decode`
assertion). BF16 KV-cache decode was removed together with
``sparse_attn_decode_op`` and ``paged_topk_translator.gather_dual_pool_kv_packed``.

Sub-modules:
- ``compute_qkv`` — per-request Q/KV projection + partial RoPE
  (:func:`.compute_qkv.decode_compute_qkv`).
- ``output_proj`` — fused inv-RoPE + FP8 quant + wo_a einsum + wo_b
  (:func:`.output_proj.decode_output_proj`).
- ``write_swa`` — FP8 SWA pool write wrapper
  (:func:`.write_swa.decode_write_swa_fp8`).
- ``attention_kernels`` — FlashMLA FP8 dispatch helpers
  (``attn_fp8_swa_paged`` / ``attn_fp8_dual_paged``).
- ``fp8_kv_quant_decode_op`` — CUDA ``concat_and_cache_mla("fp8_model1_mla", ...)``
  dispatcher + reference implementations.
- ``fp8_sparse_attn_decode_op`` — ``SparseAttnV4DecodeFp8Op`` wrapper
  around ``flash_mla_with_kvcache`` (sparse FP8 MLA).
- ``indexer_decode_op`` — paged FP8 MQA logits + topk via DeepGEMM,
  matching ``base/cuda/indexer_op.py``'s V3.2 path.
- ``kv_write_decode_op`` — per-token append into non-SWA pools via
  ``slot_mapping`` (still used by compressor / indexer / prefill paged
  write paths; decode SWA write uses ``write_swa`` instead).
- ``paged_topk_translator`` — local-slot → global-slot translator
  (``translate_local_to_global_slots`` + ``build_req_id_per_token``).
- ``pool_slot_mapping`` — per-token pool slot-mapping computation used
  by the metadata builder.
- ``decode_attn_metadata`` — per-step builder for slot mappings,
  block tables, indexer topk buffer.
- ``decode_fmha_impl`` — persistent metadata owner for CUDA graph capture.
- ``forward`` — orchestration layer called by the transformer model.
"""
