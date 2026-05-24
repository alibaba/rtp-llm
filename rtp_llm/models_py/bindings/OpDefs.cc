#include "OpDefs.h"

namespace torch_ext {

void registerPyOpDefs(pybind11::module& m) {
    pybind11::enum_<rtp_llm::CacheGroupType>(m, "CacheGroupType")
        .value("LINEAR", rtp_llm::CacheGroupType::LINEAR)
        .value("FULL", rtp_llm::CacheGroupType::FULL)
        .value("SWA", rtp_llm::CacheGroupType::SWA)
        .export_values();

    pybind11::enum_<rtp_llm::KVCacheRegionName>(m, "KVCacheRegionName")
        .value("DEFAULT", rtp_llm::KVCacheRegionName::DEFAULT)
        .value("CSA_KV", rtp_llm::KVCacheRegionName::CSA_KV)
        .value("HCA_KV", rtp_llm::KVCacheRegionName::HCA_KV)
        .value("INDEXER_KV", rtp_llm::KVCacheRegionName::INDEXER_KV)
        .value("INDEXER_STATE", rtp_llm::KVCacheRegionName::INDEXER_STATE)
        .value("CSA_STATE", rtp_llm::KVCacheRegionName::CSA_STATE)
        .value("HCA_STATE", rtp_llm::KVCacheRegionName::HCA_STATE)
        .value("SWA_KV", rtp_llm::KVCacheRegionName::SWA_KV)
        .export_values();

    pybind11::class_<LayerKVCache>(m, "LayerKVCache")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_base", &LayerKVCache::kv_cache_base, "Key/value cache tensor (per-layer view)")
        .def_readwrite("kv_scale_base", &LayerKVCache::kv_scale_base, "Key/value cache scale tensor")
        .def_readonly("seq_size_per_block", &LayerKVCache::seq_size_per_block, "Sequence size per block")
        .def_readonly("layer_id", &LayerKVCache::layer_id, "Global layer id")
        .def_readonly("group_id", &LayerKVCache::group_id, "KV cache group id")
        .def_readonly("region_name", &LayerKVCache::region_name, "KV cache attention type");

    pybind11::class_<KVCache>(m, "KVCache")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_base_by_layer", &KVCache::kv_cache_base_by_layer, "Per-layer KV cache tensors")
        .def_readwrite("kv_scale_base_by_layer", &KVCache::kv_scale_base_by_layer, "Per-layer KV scale tensors")
        .def_readwrite("seq_size_per_block", &KVCache::seq_size_per_block, "Physical (logical) block size in tokens")
        .def_readwrite("kernel_seq_size_per_block",
                       &KVCache::kernel_seq_size_per_block,
                       "Kernel block size (0 = same as seq_size_per_block)")
        .def_readwrite("num_kv_heads", &KVCache::num_kv_heads, "Number of KV heads per TP rank")
        .def_readwrite("head_dim", &KVCache::head_dim, "Head dimension")
        .def_readwrite("use_mla", &KVCache::use_mla, "Whether MLA cache layout is used")
        .def_readwrite("kv_lora_rank", &KVCache::kv_lora_rank, "MLA KV LoRA rank")
        .def_readwrite("rope_head_dim", &KVCache::rope_head_dim, "MLA RoPE head dimension")
        .def_readwrite("layer_group_types",
                       &KVCache::layer_group_types,
                       "Per-layer attention type (CacheGroupType::FULL or LINEAR). "
                       "Empty = all layers treated as FULL (backward compatibility).")
        .def_readwrite("group_region_names", &KVCache::group_region_names, "Per-group KV cache attention types")
        .def_readwrite("layer_region_to_group_id",
                       &KVCache::layer_region_to_group_id,
                       "Dense mapping from layer id and KVCacheRegionName to group id")
        .def_readwrite("kv_cache_base_by_layer_region",
                       &KVCache::kv_cache_base_by_layer_region,
                       "Per-layer and per-attention-type KV cache tensors")
        .def_readwrite("kv_cache_base_by_layer_region_flat",
                       &KVCache::kv_cache_base_by_layer_region_flat,
                       "Flat version of by_layer_region: [layer*8+region_name] = tensor")
        .def_readwrite("kv_scale_base_by_layer_region",
                       &KVCache::kv_scale_base_by_layer_region,
                       "Per-layer and per-attention-type KV scale tensors")
        .def("get_layer_cache",
             pybind11::overload_cast<int>(&KVCache::getLayerCache),
             "Return the legacy/default per-layer LayerKVCache for the given global layer id")
        .def("get_layer_cache",
             pybind11::overload_cast<int, rtp_llm::KVCacheRegionName>(&KVCache::getLayerCache),
             "Return a raw per-layer LayerKVCache for the given global layer id and KV cache attention type")
        .def("get_layer_caches",
             &KVCache::getLayerCaches,
             "Return every per-region LayerKVCache owned by the given global layer id")
        .def("get_raw_pool_tensor",
             &KVCache::getRawPoolTensor,
             "Return raw [total_blocks, stride_bytes] tensor for a specific layer and region name, no reshape (DSV4)");

    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readonly("kv_cache", &PyModelInitResources::kv_cache, "KV cache for all layers")
        .def_readonly(
            "is_speculative", &PyModelInitResources::is_speculative, "True when speculative decoding is active")
        .def_readonly("is_decode_role",
                      &PyModelInitResources::is_decode_role,
                      "True when this model instance runs in decode role")
        .def_readonly("max_context_batch_size",
                      &PyModelInitResources::max_context_batch_size,
                      "Max concurrent context (prefill) batches from FIFO scheduler");

    pybind11::class_<caffe2::TypeMeta>(m, "TypeMeta").def(pybind11::init<>());

    m.def(
        "get_typemeta",
        [](const torch::Tensor& tensor) { return torch::scalarTypeToTypeMeta(tensor.scalar_type()); },
        "Convert tensor dtype to TypeMeta");

    m.def(
        "get_scalar_type",
        [](caffe2::TypeMeta dtype) { return dtype.toScalarType(); },
        "Convert TypeMeta to scalar type");

    pybind11::class_<PyCacheStoreInputs>(m, "PyCacheStoreInputs")
        .def(pybind11::init<>())
        .def_readwrite("input_lengths_host", &PyCacheStoreInputs::input_lengths_host)
        .def_readwrite("prefix_lengths_host", &PyCacheStoreInputs::prefix_lengths_host);

    pybind11::class_<rtp_llm::ParamsBase, std::shared_ptr<rtp_llm::ParamsBase>>(m, "ParamsBase")
        .def(pybind11::init<>())
        .def(
            "fill_params",
            [](rtp_llm::ParamsBase& self,
               torch::Tensor        sequence_lengths,
               torch::Tensor        input_lengths,
               torch::Tensor        kv_cache_block_id_host,
               int                  batch_size,
               int                  seq_size_per_block) {
                self.fillParams(
                    sequence_lengths, input_lengths, kv_cache_block_id_host, batch_size, seq_size_per_block);
            },
            pybind11::arg("sequence_lengths"),
            pybind11::arg("input_lengths"),
            pybind11::arg("kv_cache_block_id_host"),
            pybind11::arg("batch_size"),
            pybind11::arg("seq_size_per_block"),
            "Fill parameters for CUDA graph execution");

    pybind11::class_<PyPrefillCudaGaphCopyParams>(m, "PyPrefillCudaGaphCopyParams")
        .def(pybind11::init<>())
        .def_readwrite("cuda_graph_prefill_batch_size", &PyPrefillCudaGaphCopyParams::cuda_graph_prefill_batch_size)
        .def_readwrite("max_seq_len", &PyPrefillCudaGaphCopyParams::max_seq_len)
        .def_readwrite("max_batch_size", &PyPrefillCudaGaphCopyParams::max_batch_size);

    pybind11::class_<PyContextParallelParams>(m, "PyContextParallelParams")
        .def(pybind11::init<>())
        .def_readwrite("prefill_cp_padding_lengths", &PyContextParallelParams::prefill_cp_padding_lengths)
        .def_readwrite("prefill_cp_chunk_lengths", &PyContextParallelParams::prefill_cp_chunk_lengths)
        .def_readwrite("prefill_shuffle_indices", &PyContextParallelParams::prefill_shuffle_indices)
        .def_readwrite("prefill_qkv_restore_indice", &PyContextParallelParams::prefill_qkv_restore_indice)
        .def_readwrite("prefill_qkv_padding_mask", &PyContextParallelParams::prefill_qkv_padding_mask)
        .def_readwrite("prefill_actual_input_lengths_cpu", &PyContextParallelParams::prefill_actual_input_lengths_cpu);

    // IMPORTANT (M05 §3.1 / §4.1 P0 — pybind completeness): every field declared on
    // PyKVCacheRegionDesc MUST appear here as `def_readonly`. Consumers in M06
    // (`handle.bps`, INDEXER_KV asserts, `region_max_cols` clamp) and M09
    // (`max_state_blocks` per-`at_state` dict, INDEXER contiguity / head_dim
    // asserts) silently break under partial bindings. Keep this list canonical.
    pybind11::class_<PyKVCacheRegionDesc>(m, "PyKVCacheRegionDesc")
        .def(pybind11::init<>())
        // ---- region identity ------------------------------------------------------
        .def_readonly("region_name",
                      &PyKVCacheRegionDesc::region_name,
                      "KVCacheRegionName as int (0..7); 8 region slots incl DEFAULT.")
        .def_readonly("group_id",
                      &PyKVCacheRegionDesc::group_id,
                      "Group id in CacheConfig::group_region_names order; "
                      "natural key of region_descs (Panel B §3.6).")
        // ---- per-block / per-entry shape ------------------------------------------
        .def_readonly("entries_per_block",
                      &PyKVCacheRegionDesc::entries_per_block,
                      "Tokens per kernel block (KV) or 1 (state). Source of truth; "
                      "no longer derived from base.shape[1] * element_size(). "
                      "M09 alias: `extra_page_block_size` (FlashMLA sched_meta).")
        .def_readonly("kernel_blocks_per_kv_block",
                      &PyKVCacheRegionDesc::kernel_blocks_per_kv_block,
                      "bps factor; ≡1 under F02 bps≡1 (B07-18-4). "
                      "Consumed by M06 `handle.bps` and the M09 §2.2 future "
                      "`bps[at] > 1` audit.")
        .def_readonly("num_blocks",
                      &PyKVCacheRegionDesc::num_blocks,
                      "Per-region kernel-block count for this region; "
                      "carried for descriptor-level introspection / T01 invariants.")
        .def_readonly("region_max_cols",
                      &PyKVCacheRegionDesc::region_max_cols,
                      "Per-region effective_max_blocks (clamp width). Consumed by "
                      "compute_kv_pool_slot_mapping_unified (M06) to clamp the "
                      "unified [B, max_blocks] table against per-region effective "
                      "columns. Tail columns over this carry null_block_value "
                      "(B07-18-4 / 46-5).")
        // ---- per-block / per-scale byte strides -----------------------------------
        .def_readonly("kv_block_stride_bytes",
                      &PyKVCacheRegionDesc::kv_block_stride_bytes,
                      "Per-block byte stride INCLUDING any TMA padding.")
        .def_readonly("kv_scale_stride_bytes",
                      &PyKVCacheRegionDesc::kv_scale_stride_bytes,
                      "Per-block byte stride of the scale tensor (0 if no scale).")
        // ---- raw base pointers ----------------------------------------------------
        .def_readonly("kv_pool_base_ptr",
                      &PyKVCacheRegionDesc::kv_pool_base_ptr,
                      "Raw uint8 base of region pool (int64). Pointer-stable for "
                      "the model lifetime under one-shot pool allocation. "
                      "T01 invariant: equals "
                      "kv_cache_base_by_layer_region[L][region].data_ptr() for any L.")
        .def_readonly("kv_scale_base_ptr",
                      &PyKVCacheRegionDesc::kv_scale_base_ptr,
                      "Raw uint8 base of scale pool (int64); 0 if no scale tensor.")
        // ---- per-slot byte size + null sentinel -----------------------------------
        .def_readonly("bytes_per_entry",
                      &PyKVCacheRegionDesc::bytes_per_entry,
                      "Per-slot byte count: 584 (CSA/HCA/SWA KV), 132 (INDEXER KV), "
                      "fp32-sized state. Distinct from kv_block_stride_bytes.")
        .def_readonly("null_block_value",
                      &PyKVCacheRegionDesc::null_block_value,
                      "0 for KV pools (NULL_BLOCK_IDX warmup-shared 0); "
                      "-1 for state pools. SOLE source-of-truth — kernels MUST "
                      "NOT inline 0/-1 (B07 46-1).")
        // ---- attribute predicates -------------------------------------------------
        .def_readonly("has_tma_padding",
                      &PyKVCacheRegionDesc::has_tma_padding,
                      "True iff this region's kernel blocks carry TMA pad bytes.")
        .def_readonly("is_state_pool",
                      &PyKVCacheRegionDesc::is_state_pool,
                      "STATE pools route through _compute_state_pool_slot_mapping; "
                      "KV pools through compute_kv_pool_slot_mapping_unified.")
        .def_readonly("is_contiguous_paged_blocks",
                      &PyKVCacheRegionDesc::is_contiguous_paged_blocks,
                      "INDEXER pool MUST be contiguous (no TMA pad). M06 post-init "
                      "INDEXER_KV assertion reads this.")
        // ---- INDEXER-only invariants ---------------------------------------------
        .def_readonly("head_dim",
                      &PyKVCacheRegionDesc::head_dim,
                      "INDEXER score-path head dim; routed via single "
                      "_indexer_consts.py (D07 36-10).")
        .def_readonly("entry_bytes",
                      &PyKVCacheRegionDesc::entry_bytes,
                      "Alias for bytes_per_entry surfaced to indexer import-time "
                      "asserts (D07 36-10).")
        // ---- CP layout flag (STATIC; topology lives in CPShardConfig) ------------
        .def_readonly("cp_sharded",
                      &PyKVCacheRegionDesc::cp_sharded,
                      "STATIC layout flag: does this pool's storage live in "
                      "CP-replicated or CP-sharded space? Decision-only — runtime "
                      "(cp_size, cp_rank) topology comes EXCLUSIVELY from "
                      "_cp_slot_mapping.CPShardConfig (M06 §3.4). Never query "
                      "the descriptor for CP topology.")
        // ---- STATE-pool offset constexpr -----------------------------------------
        .def_readonly("stride_elems",
                      &PyKVCacheRegionDesc::stride_elems,
                      "Per-block element stride: 2 * coff * head_dim with "
                      "coff_csa=2, coff_hca=1, coff_idx=2. STATE-pool offset math.")
        // ---- STATE-pool ring depth (Panel B §3.5 / M09 R3 P0) --------------------
        .def_readonly("max_state_blocks",
                      &PyKVCacheRegionDesc::max_state_blocks,
                      "STATE-pool cyclic ring depth = CacheConfig.state_block_num "
                      "for STATE regions; 0 for non-STATE. Consumed by M09 "
                      "DSv4DecodeAttnMetadataFP8.max_state_blocks[at_state] (Fix "
                      "150). Closes the Panel B §3.5 broken plumbing chain.");

    // Module-level constants — Python imports these instead of hard-coding `8`
    // in attn_type.py (E05 24-4 / B07 32-16 / M05 Fix 53).
    m.attr("kRegionCount")        = static_cast<int>(rtp_llm::KVCacheRegionName::REGION_COUNT);
    m.attr("kDsv4RegionsPerLayer") = 8;

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
        .def_readwrite("is_cuda_graph", &PyAttentionInputs::is_cuda_graph)
        .def_readwrite("is_target_verify", &PyAttentionInputs::is_target_verify)
        .def_readwrite("prefix_lengths", &PyAttentionInputs::prefix_lengths)
        .def_readwrite("sequence_lengths", &PyAttentionInputs::sequence_lengths)
        .def_readwrite("input_lengths", &PyAttentionInputs::input_lengths)
        .def_readwrite("kv_cache_kernel_block_id_host", &PyAttentionInputs::kv_cache_kernel_block_id_host)
        .def_readwrite("kv_cache_kernel_block_id_device", &PyAttentionInputs::kv_cache_kernel_block_id_device)
        .def_readwrite("kv_cache_block_id_host", &PyAttentionInputs::kv_cache_block_id_host)
        .def_readwrite("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readwrite("kv_cache_kernel_block_id_device_by_group",
                       &PyAttentionInputs::kv_cache_kernel_block_id_device_by_group)
        .def_readwrite("kv_cache_unified_block_id_device",
                       &PyAttentionInputs::kv_cache_unified_block_id_device,
                       "Unified [B, max_kernel_blocks] CUDA block-id tensor. "
                       "Phase 1 alias of kv_cache_kernel_block_id_device_by_group[0]. "
                       "Phase 3 sole carrier of kernel block ids.")
        .def_readwrite("kv_cache_unified_block_id_host",
                       &PyAttentionInputs::kv_cache_unified_block_id_host,
                       "Pinned host mirror, materialised when use_mla.")
        .def_readwrite("kv_cache_unified_phys_block_id_host",
                       &PyAttentionInputs::kv_cache_unified_phys_block_id_host,
                       "Pinned host physical block ids for cache-store / PD.")
        .def_readwrite("region_descs",
                       &PyAttentionInputs::region_descs,
                       "Per-region stride/pointer descriptors; index = group_id.")
        .def_readwrite("kv_cache_layer_region_descs",
                       &PyAttentionInputs::kv_cache_layer_region_descs,
                       "Typed pybind binding REPLACING the legacy "
                       "`kv_cache_layer_to_group_dpsk_v4` Python attribute set by "
                       "NormalModelInputGatherer (B07 04-8 / 18-2). Shape "
                       "[layer_num, kRegionCount] int32. -1 = unowned.")
        .def_readwrite("kv_cache_layer_to_group", &PyAttentionInputs::kv_cache_layer_to_group)
        .def_readwrite("dtype", &PyAttentionInputs::dtype)
        .def_readwrite("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readwrite("cu_seqlens_host", &PyAttentionInputs::cu_seqlens_host)
        .def_readwrite("cu_kv_seqlens", &PyAttentionInputs::cu_kv_seqlens)
        .def_readwrite("context_total_kv_length", &PyAttentionInputs::context_total_kv_length)
        .def_readwrite("total_tokens", &PyAttentionInputs::total_tokens)
        .def_readwrite("padding_offset", &PyAttentionInputs::padding_offset)
        .def_readwrite("is_s_padded", &PyAttentionInputs::is_s_padded)
        .def_readwrite("sequence_lengths_plus_1_d", &PyAttentionInputs::sequence_lengths_plus_1_d)
        .def_readwrite("decode_cu_seqlens_d", &PyAttentionInputs::decode_cu_seqlens_d)
        .def_readonly("decode_cu_seqlens_host", &PyAttentionInputs::decode_cu_seqlens_host)
        .def_readwrite("position_ids", &PyAttentionInputs::position_ids)
        .def_readwrite("cache_store_inputs", &PyAttentionInputs::cache_store_inputs)
        .def_readwrite("context_parallel_info", &PyAttentionInputs::context_parallel_info)
        .def("__repr__", [](const PyAttentionInputs& self) { return "PyAttentionInputs"; })
        .def_readwrite("prefill_cuda_graph_copy_params", &PyAttentionInputs::prefill_cuda_graph_copy_params)
        .def_readwrite("headwise_config", &PyAttentionInputs::headwise_config)
        .def("__copy__", [](const PyAttentionInputs& self) { return PyAttentionInputs(self); });

    pybind11::class_<BertEmbeddingInputs>(m, "BertEmbeddingInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float>(),
             pybind11::arg("combo_position_ids")     = torch::empty(0),
             pybind11::arg("position_encoding")      = torch::empty(0),
             pybind11::arg("combo_tokens_type_ids")  = torch::empty(0),
             pybind11::arg("token_type_embedding")   = torch::empty(0),
             pybind11::arg("input_embedding_scalar") = 1.0f)
        .def_readwrite("combo_position_ids", &BertEmbeddingInputs::combo_position_ids, "Combined position IDs tensor")
        .def_readwrite("position_encoding", &BertEmbeddingInputs::position_encoding, "Position encoding tensor")
        .def_readwrite(
            "combo_tokens_type_ids", &BertEmbeddingInputs::combo_tokens_type_ids, "Combined token type IDs tensor")
        .def_readwrite(
            "token_type_embedding", &BertEmbeddingInputs::token_type_embedding, "Token type embedding tensor")
        .def_readwrite(
            "input_embedding_scalar", &BertEmbeddingInputs::input_embedding_scalar, "Input embedding scalar value")
        .def("__repr__", [](const BertEmbeddingInputs& self) { return "BertEmbeddingInputs"; });

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, torch::Tensor, PyAttentionInputs, BertEmbeddingInputs>(),
             pybind11::arg("input_ids")             = torch::empty(0),
             pybind11::arg("input_hiddens")         = torch::empty(0),
             pybind11::arg("attention_inputs")      = PyAttentionInputs(),
             pybind11::arg("bert_embedding_inputs") = BertEmbeddingInputs())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("input_hiddens", &PyModelInputs::input_hiddens, "Input hidden states tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs, "Attention inputs structure")
        .def_readwrite(
            "bert_embedding_inputs", &PyModelInputs::bert_embedding_inputs, "BERT embedding inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<>(), "Default constructor")
        .def(pybind11::init<torch::Tensor>(),
             pybind11::arg("hidden_states"),
             "Initialize with hidden states tensor only (params_ptr defaults to nullptr)")
        .def(pybind11::init([](torch::Tensor hidden_states, pybind11::object params_obj) {
                 // Try to cast to shared_ptr, return nullptr if conversion fails
                 std::shared_ptr<rtp_llm::ParamsBase> params_ptr     = nullptr;
                 py::object                           py_attn_params = py::none();
                 try {
                     params_ptr = pybind11::cast<std::shared_ptr<rtp_llm::ParamsBase>>(params_obj);
                 } catch (const pybind11::cast_error& e) {
                     // Conversion failed, params_ptr remains nullptr
                     //  RTP_LLM_LOG_INFO("Failed to cast params_obj to shared_ptr<ParamsBase>: %s", e.what());
                     py_attn_params = params_obj;
                 }
                 return PyModelOutputs(hidden_states, params_ptr, py_attn_params);
             }),
             pybind11::arg("hidden_states"),
             pybind11::arg("params_ptr"),
             "Initialize with hidden states tensor and params pointer")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor")
        .def_readwrite("params_ptr", &PyModelOutputs::params_ptr, "Parameters pointer");
}

}  // namespace torch_ext
