"""Frozen model-validated FI CP guard and fallback call; source-test fixture only."""

def _same_layer_overlap_enabled(self):
    return _CP_PACKED_KV_OVERLAP and (not _CP_PREFIX_PREFETCH) and self.nvfp4_kv_cache and self._kv_sharded and (self._cp_size == 4) and (self.kv_head_num == 4) and (self.head_dim == 128) and (self.idx_head_dim == 128) and (self.page_size == 128) and (not _should_use_cp_compact_prefill(_CP_COMPACT_PREFILL, self.nvfp4_kv_cache))


def _write_cp_suffix_to_nvfp4_working_pages(self):
    nvfp4_quantize_cp_main_index_rows_to_planes(packed, unpad_rows, write_slots[:token_count].contiguous(), main[0], main_scales[0], main[1], main_scales[1], idx_packed, idx_scales, persistent_slots=slot_mapping[:token_count], persistent_planes=(views.main_k_fp4, views.main_k_scale, views.main_v_fp4, views.main_v_scale, views.idx_k_fp4, views.idx_k_scale), rows_per_cta=8, fi_working_layout=self._flashinfer_nvfp4_prefill)
