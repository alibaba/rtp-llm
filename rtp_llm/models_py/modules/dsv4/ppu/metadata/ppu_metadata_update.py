"""PPU graph metadata update with a replaceable state-slot kernel.

Based on mainline update_decode_metadata_in_place_fp8 at 1d20e29a428f.
Keep the orchestration local so the optional PPU kernel replaces, rather than
repeats, state-slot work without changing the shared GPU updater.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    DSv4DecodeAttnMetadataFP8,
    _build_start_pos_from_attention_inputs,
    _build_position_ids_2d,
    _update_topk_lengths_in_place,
    _resolve_paged_pool_tokens_per_block,
    _update_compressor_state_slot_mappings,
    _compressed_domain_tokens_per_block,
)


def update_decode_metadata_in_place_ppu(
    meta: "DSv4DecodeAttnMetadataFP8",
    attention_inputs: Any,
    *,
    forbid_realloc: bool = False,
    paged_block_tables: Optional[Dict[str, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[str, int]] = None,
    paged_pool_tokens_per_block: Optional[Dict[str, int]] = None,
    capture_full_width_lengths: bool = False,
    compressor_state_slot_updater=None,
) -> None:
    """Recompute every metadata buffer IN PLACE for new attention inputs.

    Contract:
      * Every output tensor reuses its prior storage (``data_ptr()``
        unchanged); ``forbid_realloc=True`` makes any accidental realloc
        an immediate error rather than silent-correctness-bug.
      * Writes the prefix ``[:bs]`` (or ``[:bs * q_len]`` for slot
        mappings); leaves the tail at sentinel (-1) from
        ``allocate_decode_metadata_fp8``. The captured CUDA graph at that
        BS only reads the prefix, so the tail never matters.

    Mirrors :func:`build_decode_metadata_fp8` arithmetic. Kept as a separate
    function (not method) so it's callable from ``DSv4DecodeFmhaImplFP8.prepare``
    and from unit tests without instantiating the impl.

    Args:
        meta: Pre-allocated metadata (from ``allocate_decode_metadata_fp8``).
        attention_inputs: Framework attention inputs. Normal decode derives
            the first-token position from ``sequence_lengths``; target verify
            derives it from ``prefix_lengths`` because C++ clears
            ``sequence_lengths`` for that path. A tensor is accepted for
            focused metadata tests.
        forbid_realloc: If True, asserts every write reuses the existing
            tensor storage (sanity check for the captured-graph path).
    """
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    device = meta.start_pos.device

    start_pos = _build_start_pos_from_attention_inputs(
        attention_inputs,
        device,
        meta.max_seq_len,
        q_len,
    )
    bs = int(start_pos.shape[0])
    position_ids_2d = _build_position_ids_2d(start_pos, q_len, device)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()
    if paged_pool_entries_per_block:
        if paged_pool_tokens_per_block is None:
            raise ValueError("paged_pool_tokens_per_block is required for paged pools")

    # snapshot pointers for the realloc-forbidden mode
    if forbid_realloc:
        ptr_snap = {
            "start_pos": meta.start_pos.data_ptr(),
            "position_ids": meta.position_ids.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        }
        if meta.position_ids_long is not None:
            ptr_snap["position_ids_long"] = meta.position_ids_long.data_ptr()
        if meta.decode_seq_start_per_req is not None:
            ptr_snap["decode_seq_start_per_req"] = (
                meta.decode_seq_start_per_req.data_ptr()
            )
        for r, t in meta.slot_mapping_compressed.items():
            ptr_snap[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            ptr_snap[f"compressed_lens[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens_per_token.items():
            ptr_snap[f"compressed_lens_per_token[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            ptr_snap[f"topk_total_by_ratio[{r}]"] = t.data_ptr()
        for at, t in meta.compressor_state_slot_mappings.items():
            ptr_snap[f"compressor_state_slot_mappings[{at}]"] = t.data_ptr()
        if meta.swa_global_slots is not None:
            ptr_snap["swa_global_slots"] = meta.swa_global_slots.data_ptr()
        if meta.hca_cmp_global_slots is not None:
            ptr_snap["hca_cmp_global_slots"] = meta.hca_cmp_global_slots.data_ptr()
        if meta.swa_topk_length is not None:
            ptr_snap["swa_topk_length"] = meta.swa_topk_length.data_ptr()
        for r, t in meta.compressed_topk_length_by_ratio.items():
            ptr_snap[f"compressed_topk_length_by_ratio[{r}]"] = t.data_ptr()

    meta.position_ids[: bs * q_len].copy_(position_ids_flat)
    if meta.position_ids_long is not None:
        meta.position_ids_long[: bs * q_len].copy_(position_ids_flat.to(torch.long))
    if meta.decode_seq_start_per_req is not None:
        meta.decode_seq_start_per_req[:bs].copy_(start_pos)
    meta.topk_buffer_compressed[:bs].fill_(-1)

    from rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton import (
        fused_update_decode_meta_pure,
    )

    fused_update_decode_meta_pure(meta, start_pos, meta.max_seq_len)

    # opt_flash_mla: derive FlashMLA topk_length / extra_topk_length from the
    # just-filled compressed_lens + start_pos, in place (graph-stable buffers).
    # During CUDA-graph capture/warmup (capture_full_width_lengths=True) write
    # the FULL width so FlashMLA's frozen tile schedule is max-provisioned and
    # safe for any real replay length; real per-request lengths are written
    # before each replay by prepare_cuda_graph(). See _update_topk_lengths_in_place.
    _update_topk_lengths_in_place(
        meta, start_pos, bs, full_width=capture_full_width_lengths
    )

    # ------------------------------------------------------------------
    # Phase 2: paged write slot mappings (per cache tag).
    # SWA-K writes EVERY decode token; CSA-K / HCA-K / INDEXER-K write
    # ONLY on their respective compression boundaries (sentinel ``-1``
    # otherwise — the write op honors that via ``mask_negative=True``).
    # ------------------------------------------------------------------
    if paged_block_tables and paged_pool_entries_per_block:
        paged_pool_tokens_per_block = _resolve_paged_pool_tokens_per_block(
            paged_pool_entries_per_block,
            paged_pool_tokens_per_block or meta.paged_pool_tokens_per_block,
        )
        meta.paged_pool_tokens_per_block = dict(paged_pool_tokens_per_block)
        # Snapshot block_table content into the metadata's stable buffer
        # (forbid_realloc-friendly: just `.copy_` the prefix). Skip pools
        # without a metadata buffer (e.g. SWA-only layers don't carry CSA).
        for at, src_bt in paged_block_tables.items():
            dst_bt = meta.pool_block_tables.get(at)
            if dst_bt is None:
                continue
            n_rows = min(src_bt.shape[0], dst_bt.shape[0])
            n_cols = min(src_bt.shape[1], dst_bt.shape[1])
            # Zero stale rows beyond current bs to avoid carrying old block
            # ids (defensive — graph reads only [:bs] anyway).
            dst_bt[bs:].zero_()
            dst_bt[:n_rows, :n_cols].copy_(src_bt[:n_rows, :n_cols])
            if n_cols < dst_bt.shape[1]:
                dst_bt[:n_rows, n_cols:].zero_()

        from rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton import (
            fused_phase2b_pool_slot_mapping,
        )

        fused_phase2b_pool_slot_mapping(
            meta,
            start_pos,
            bs,
            paged_pool_entries_per_block,
            paged_pool_tokens_per_block,
        )

        (compressor_state_slot_updater or _update_compressor_state_slot_mappings)(
            meta,
            bs,
            paged_pool_entries_per_block,
        )

    # Iter3.3: precompute translate_swa / translate_hca once per step.
    # Shared across all 43 attention layers. Graph-safe: ``.copy_()`` into
    # pre-allocated buffers on meta (allocated at max_bs); never reassigns
    # the attribute, so graph replays read from fixed addresses.
    # ``req_id_per_token`` is deterministic (``arange(B)``) so it was filled
    # at allocate time and stays stable.
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            translate_local_to_global_slots,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import HCA_KV, SWA_KV

        T = bs * q_len
        req_id_bs = (
            meta.req_id_per_token[:T] if meta.req_id_per_token is not None else None
        )

        if (
            req_id_bs is not None
            and meta.swa_global_slots is not None
            and SWA_KV in meta.pool_block_tables
            and meta.swa_abs_idx is not None
        ):
            swa_eb = paged_pool_entries_per_block[SWA_KV]
            swa_tokens_per_block = paged_pool_tokens_per_block[SWA_KV]
            swa_local = meta.swa_abs_idx[:bs].reshape(T, window_size)
            swa_global_new = translate_local_to_global_slots(
                req_id_bs,
                meta.pool_block_tables[SWA_KV][:bs],
                swa_local,
                entries_per_block=swa_eb,
                tokens_per_block_for_block_table=swa_tokens_per_block,
            )
            meta.swa_global_slots[:T].copy_(swa_global_new)

        # HCA layers all share the dense-idx-masked cmp_local_raw (already
        # materialised as topk_total[r=128][:, :, win:]); translate it once.
        if (
            req_id_bs is not None
            and meta.hca_cmp_global_slots is not None
            and HCA_KV in meta.pool_block_tables
            and 128 in meta.topk_total_by_ratio
        ):
            hca_eb = paged_pool_entries_per_block[HCA_KV]
            hca_tokens_per_block = _compressed_domain_tokens_per_block(
                HCA_KV, paged_pool_tokens_per_block[HCA_KV]
            )
            hca_tt = meta.topk_total_by_ratio[128]
            K_h = hca_tt.shape[-1] - window_size
            hca_cmp_local = hca_tt[:bs, :, window_size:].reshape(T, K_h).contiguous()
            hca_global_new = translate_local_to_global_slots(
                req_id_bs,
                meta.pool_block_tables[HCA_KV][:bs],
                hca_cmp_local,
                entries_per_block=hca_eb,
                tokens_per_block_for_block_table=hca_tokens_per_block,
            )
            meta.hca_cmp_global_slots[:T].copy_(hca_global_new)

    # Update Python-scalar geometry (cheap — these are not captured into the graph)
    meta.batch_size = bs
    meta.total_tokens = bs * q_len

    if forbid_realloc:
        # Verify every storage pointer is unchanged.
        cur = {
            "start_pos": meta.start_pos.data_ptr(),
            "position_ids": meta.position_ids.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        }
        if meta.position_ids_long is not None:
            cur["position_ids_long"] = meta.position_ids_long.data_ptr()
        if meta.decode_seq_start_per_req is not None:
            cur["decode_seq_start_per_req"] = meta.decode_seq_start_per_req.data_ptr()
        for r, t in meta.slot_mapping_compressed.items():
            cur[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            cur[f"compressed_lens[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens_per_token.items():
            cur[f"compressed_lens_per_token[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            cur[f"topk_total_by_ratio[{r}]"] = t.data_ptr()
        for at, t in meta.compressor_state_slot_mappings.items():
            cur[f"compressor_state_slot_mappings[{at}]"] = t.data_ptr()
        if meta.swa_global_slots is not None:
            cur["swa_global_slots"] = meta.swa_global_slots.data_ptr()
        if meta.hca_cmp_global_slots is not None:
            cur["hca_cmp_global_slots"] = meta.hca_cmp_global_slots.data_ptr()
        if meta.swa_topk_length is not None:
            cur["swa_topk_length"] = meta.swa_topk_length.data_ptr()
        for r, t in meta.compressed_topk_length_by_ratio.items():
            cur[f"compressed_topk_length_by_ratio[{r}]"] = t.data_ptr()
        for k, p_before in ptr_snap.items():
            assert cur[k] == p_before, (
                f"update_decode_metadata_in_place_fp8(forbid_realloc=True) "
                f"reallocated buffer '{k}': before={hex(p_before)} after={hex(cur[k])}"
            )
