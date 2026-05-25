#include "rtp_llm/cpp/cache/DSV4CacheConfigHelper.h"

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// Kernel block size in tokens. DSV4 attention/compressor kernels and the
// FlashMLA SWA path are compiled assuming 256-token kernel blocks; physical
// blocks (config.seq_size_per_block) may be larger multiples of this, with the
// FULL paged pools auto-expanding via the framework's bpk machinery.
constexpr uint32_t kDsv4KernelTokensPerBlock = 256;
// BF16 pool: head_dim=512 × 2B = 1024B per KV slot, 128 × 2B = 256B per
// indexer slot. FP8 pool packs the same logical KV into a smaller slot
// (canonical fp8_model1_mla layout: 448B fp8 NoPE + 64B bf16 RoPE + 8B
// UE8M0 scales = 584B; indexer is 128B fp8 + 4B fp32 scale = 132B).
// Selected at runtime from ``attn_config.kv_cache_dtype`` — see
// ``buildDSV4PoolDescs``.
constexpr uint32_t kDsv4KvEntryBytesBf16      = 1024;
constexpr uint32_t kDsv4IndexerEntryBytesBf16 = 256;
constexpr uint32_t kDsv4KvEntryBytesFp8       = 584;
constexpr uint32_t kDsv4IndexerEntryBytesFp8  = 132;
constexpr size_t   kDsv4PoolNum               = 7;

struct DSV4LayerSets {
    std::vector<int> csa_layers;
    std::vector<int> hca_layers;
    std::vector<int> swa_only_layers;
    std::vector<int> all_layers;
};

struct DSV4PoolDesc {
    KVCacheRegionName       region_name;
    const std::vector<int>* layer_ids;
    uint32_t                entry_elems;
    uint32_t                entries_per_block;
    DataType                store_dtype;
    bool                    is_paged;
};

DSV4LayerSets classifyDSV4Layers(const std::vector<int>& compress_ratios) {
    DSV4LayerSets sets;
    // ``compress_ratios`` must describe exactly the layers covered by this
    // cache config. The main DSV4 descriptor strips the trailing MTP tail,
    // while the MTP propose descriptor uses ``[0]`` for its SWA-only draft
    // layer. Do not strip a trailing zero here; that would erase the draft.
    const size_t num_layers = compress_ratios.size();

    for (size_t i = 0; i < num_layers; ++i) {
        const int layer_id = static_cast<int>(i);
        const int ratio    = compress_ratios[i];
        sets.all_layers.push_back(layer_id);
        if (ratio == 4) {
            sets.csa_layers.push_back(layer_id);
        } else if (ratio == 128) {
            sets.hca_layers.push_back(layer_id);
        } else if (ratio == 0) {
            sets.swa_only_layers.push_back(layer_id);
        } else {
            // FIX-B HIGH-7 (DEFEND-3 #4): strict reject unknown compress_ratio.
            // DSV4 spec only allows {0, 4, 128}; silently treating a stray
            // value as HCA produces wrong pool layout + wrong block bytes
            // and the only user-visible signal is a hash-mismatch on the
            // first PD-pair query (very far from the bad config).  Fail-loud
            // at startup instead.
            RTP_LLM_CHECK_WITH_INFO(false,
                                    "DSV4 unsupported compress_ratio=%d at layer %zu "
                                    "(allowed: {0, 4, 128})",
                                    ratio,
                                    i);
        }
    }

    RTP_LLM_LOG_INFO("DSV4 layer classification: %zu total, %zu CSA, %zu HCA, %zu SWA-only",
                     sets.all_layers.size(),
                     sets.csa_layers.size(),
                     sets.hca_layers.size(),
                     sets.swa_only_layers.size());
    return sets;
}

std::vector<DSV4PoolDesc>
buildDSV4PoolDescs(const DSV4LayerSets& sets, const ModelConfig& model_config, uint32_t physical_tokens_per_block) {
    const auto& attn         = model_config.attn_config;
    const auto  head_dim     = static_cast<uint32_t>(attn.size_per_head);
    const auto  idx_head_dim = static_cast<uint32_t>(attn.indexer_head_dim);

    const uint32_t idx_state_dim = 2 * idx_head_dim;
    const uint32_t csa_state_dim = 2 * head_dim;
    const uint32_t hca_state_dim = head_dim;

    // Pick KV / indexer slot byte size from the model's kv_cache_dtype.
    // FP8 paged pools use 584B / 132B (canonical fp8_model1_mla layout
    // shared with the Python writer in
    // dsv4/fp8/_compressor_vllm_triton.py); BF16 stays at 1024B / 256B.
    const bool     fp8_kv              = (attn.kv_cache_dtype == KvCacheDataType::FP8);
    const uint32_t kv_entry_bytes      = fp8_kv ? kDsv4KvEntryBytesFp8 : kDsv4KvEntryBytesBf16;
    const uint32_t indexer_entry_bytes = fp8_kv ? kDsv4IndexerEntryBytesFp8 : kDsv4IndexerEntryBytesBf16;

    // entries_per_block stays at the kernel-block size (256/compress_ratio for
    // paged compressed entries; 256 for state/SWA per-token slots) for every
    // pool. The framework's bpk machinery uniformly expands each physical
    // block into bpk = N/256 contiguous kernel sub-blocks (FULL paged + SWA
    // fixed alike), so kernels always see 256-token blocks and block_table
    // lengths are consistent across regions for the same token range.
    (void)physical_tokens_per_block;  // used for spec.seq_size_per_blk via makeDSV4Spec
    return {
        {KVCacheRegionName::CSA_KV,
         &sets.csa_layers,
         kv_entry_bytes,
         kDsv4KernelTokensPerBlock / 4,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::HCA_KV,
         &sets.hca_layers,
         kv_entry_bytes,
         kDsv4KernelTokensPerBlock / 128,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::INDEXER_KV,
         &sets.csa_layers,
         indexer_entry_bytes,
         kDsv4KernelTokensPerBlock / 4,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::INDEXER_STATE,
         &sets.csa_layers,
         idx_state_dim * 2,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::CSA_STATE,
         &sets.csa_layers,
         csa_state_dim * 2,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::HCA_STATE,
         &sets.hca_layers,
         hca_state_dim * 2,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::SWA_KV,
         &sets.all_layers,
         kv_entry_bytes,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_UINT8,
         false},
    };
}

KVCacheSpecPtr makeDSV4Spec(const DSV4PoolDesc& pool, uint32_t physical_tokens_per_block) {
    const auto layer_count = static_cast<uint32_t>(pool.layer_ids->size());
    // All pools use the same physical seq_size so cache_keys stay aligned across
    // groups (HybridKVCacheAllocator::reuseCache iterates a single shared keys
    // array). FULL paged pools split each physical block into bpk kernel blocks
    // via the framework's bpk machinery; SWA/state pools have bpk = 1 and
    // entries_per_block scaled to physical_tokens_per_block.
    if (pool.is_paged) {
        return std::make_shared<DSV4KVSpec>(pool.region_name,
                                            layer_count,
                                            pool.entry_elems,
                                            pool.entries_per_block,
                                            pool.store_dtype,
                                            physical_tokens_per_block);
    }
    return std::make_shared<DSV4StateSpec>(pool.region_name,
                                           layer_count,
                                           pool.entry_elems,
                                           pool.entries_per_block,
                                           pool.store_dtype,
                                           physical_tokens_per_block);
}

}  // namespace

void DSV4CacheConfigHelper::applyConfig(CacheConfig&         config,
                                        const ModelConfig&   model_config,
                                        const KVCacheConfig& kv_cache_config) {
    RTP_LLM_LOG_INFO("Creating DSV4 typed hybrid-pool cache config with %zu compress_ratios",
                     model_config.attn_config.layer_compress_ratios.size());

    // Honor user-supplied --seq_size_per_block when it's a positive multiple of
    // the kernel block size; otherwise fall back to the kernel block size. Paged
    // FULL groups split each physical block into integer-many 256-token kernel
    // sub-blocks via the framework's bpk machinery; non-multiples would break it.
    const auto user_seq_size = kv_cache_config.seq_size_per_block;
    uint32_t   physical_tokens_per_block;
    if (user_seq_size > 0 && user_seq_size % kDsv4KernelTokensPerBlock == 0) {
        physical_tokens_per_block = static_cast<uint32_t>(user_seq_size);
    } else {
        if (user_seq_size > 0) {
            RTP_LLM_LOG_WARNING("DSV4 ignoring seq_size_per_block=%d (not a positive multiple of %u); "
                                "using kernel block size %u as physical block size",
                                user_seq_size,
                                kDsv4KernelTokensPerBlock,
                                kDsv4KernelTokensPerBlock);
        }
        physical_tokens_per_block = kDsv4KernelTokensPerBlock;
    }
    RTP_LLM_LOG_INFO("DSV4 physical block = %u tokens, kernel block = %u tokens (bpk = %u)",
                     physical_tokens_per_block,
                     kDsv4KernelTokensPerBlock,
                     physical_tokens_per_block / kDsv4KernelTokensPerBlock);

    const auto sets = classifyDSV4Layers(model_config.attn_config.layer_compress_ratios);
    auto       pools = buildDSV4PoolDescs(sets, model_config, physical_tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(pools.size() == kDsv4PoolNum, "DSV4 must produce %zu pools", kDsv4PoolNum);

    // R6 DEV-ε / DEFEND-3 HIGH-3: structural invariant CHECKs hoisted OUT
    // of the K_state>0 branch so legacy (OFF) deployments also exercise
    // them.  Pool ordering [CSA_KV, HCA_KV, INDEXER_KV, INDEXER_STATE,
    // CSA_STATE, HCA_STATE, SWA_KV] is baked into CacheConfigCreator
    // (pool-id 3..5 ↔ STATE pinned-CPU branch), the K_state hook below
    // (``kStatePoolIndices = {3,4,5}``), the hash-salt producer, and the
    // connector init.  Previously only the K_state hook re-asserted
    // ``!is_paged`` for idx 3..5; OFF-path deployments ran with ZERO
    // structural guard, so a refactor that re-ordered ``buildDSV4PoolDescs``
    // would silently miscategorise HBM-vs-CPU residency and only manifest
    // as a PD hash mismatch on the first peer query.  Fail-loud here.
    constexpr size_t kPagedFullIndices[] = {0, 1, 2};
    constexpr size_t kStatePoolIndices_inv[] = {3, 4, 5};
    constexpr size_t kSwaPoolIndex            = 6;
    for (size_t i : kPagedFullIndices) {
        RTP_LLM_CHECK_WITH_INFO(pools[i].is_paged,
                                "DSV4 pool idx %zu must be paged FULL (got is_paged=false); "
                                "buildDSV4PoolDescs ordering was refactored without updating "
                                "the downstream {kStatePoolIndices, CacheConfigCreator pinned-CPU} consumers",
                                i);
    }
    for (size_t i : kStatePoolIndices_inv) {
        RTP_LLM_CHECK_WITH_INFO(!pools[i].is_paged,
                                "DSV4 pool idx %zu must be a non-paged STATE pool (got is_paged=true)",
                                i);
        RTP_LLM_CHECK_WITH_INFO(isStateRegion(pools[i].region_name),
                                "DSV4 pool idx %zu must be a STATE region (got region=%d)",
                                i,
                                static_cast<int>(pools[i].region_name));
    }
    RTP_LLM_CHECK_WITH_INFO(!pools[kSwaPoolIndex].is_paged,
                            "DSV4 pool idx %zu must be the non-paged SWA pool (got is_paged=true)",
                            kSwaPoolIndex);
    RTP_LLM_CHECK_WITH_INFO(pools[kSwaPoolIndex].region_name == KVCacheRegionName::SWA_KV,
                            "DSV4 pool idx %zu must have region=SWA_KV (got region=%d)",
                            kSwaPoolIndex,
                            static_cast<int>(pools[kSwaPoolIndex].region_name));

    // ---- F01 PR-1: K_state phase-2 hook (default OFF; byte-identical) ----
    // When `dsv4_state_entries_per_block > 0`, collapse the 3 STATE pools
    // (INDEXER_STATE @ idx 3, CSA_STATE @ idx 4, HCA_STATE @ idx 5 — see
    // ``buildDSV4PoolDescs`` ordering) from kernel-block size
    // (kDsv4KernelTokensPerBlock = 256) entries_per_block down to K_state,
    // shrinking each per-block bytes by 256/K_state. PR-1 only flips the
    // sizing surface; kernel-side compressor / decode_attn_metadata land
    // in F01-PR2 (M08) and the HBM-savings smoke in F01-PR3 (M02 §3.3).
    // makeDSV4Spec runs *after* this override so the resulting spec's
    // block_size_bytes reflects the reduced K_state.
    const int k_state = kv_cache_config.dsv4_state_entries_per_block;
    // F01-PR2-followup validate-on-load: reject negative and oversized
    // values up-front so a misconfigured deployment crashes at startup
    // rather than silently producing a degenerate cache layout.
    RTP_LLM_CHECK_WITH_INFO(k_state >= 0 && k_state <= static_cast<int>(kDsv4KernelTokensPerBlock),
                            "dsv4_state_entries_per_block (%d) out of range [0, %u]",
                            k_state,
                            kDsv4KernelTokensPerBlock);
    // FIX-B HIGH-3 (DEFEND-3 #2): strict divisibility CHECK — non-divisors of
    // kDsv4KernelTokensPerBlock=256 produce non-integer per-block byte math
    // downstream (e.g. K_state=3 ⇒ 256/3=85 truncated, layout silently
    // corrupts).  This is a HARD kernel invariant; the power-of-2 WARN
    // below catches OFF-fast-path values, but the WARN-only path lets
    // K_state=3/5/7/9 silently produce a degenerate cache layout.  Reject
    // up-front so a misconfigured deployment crashes at startup rather
    // than silently shipping bad cache_keys to the kernel.
    RTP_LLM_CHECK_WITH_INFO(k_state == 0 || (kDsv4KernelTokensPerBlock % static_cast<uint32_t>(k_state) == 0),
                            "dsv4_state_entries_per_block (%d) must divide kernel block size %u evenly "
                            "(non-divisors produce non-integer per-block byte math)",
                            k_state,
                            kDsv4KernelTokensPerBlock);
    // F01-PR2-followup: warn on non-power-of-2 K_state. The compressor /
    // decode_attn_metadata kernels in F01-PR2 expect K_state ∈ {1,2,4,8,
    // 16,32,64,128,256}; other values land but trigger a slow scalar tail.
    if (k_state > 0 && (k_state & (k_state - 1)) != 0) {
        // Find the closest power-of-2 divisor of 256 for operator guidance.
        int suggested = 1;
        while (suggested < k_state && suggested < static_cast<int>(kDsv4KernelTokensPerBlock)) {
            suggested <<= 1;
        }
        if (suggested - k_state > k_state - (suggested >> 1)) {
            suggested >>= 1;
        }
        RTP_LLM_LOG_WARNING("dsv4_state_entries_per_block=%d is not a power-of-2 divisor of %u; "
                            "kernel fast paths only cover {1,2,4,8,16,32,64,128,256} — closest "
                            "supported value is %d",
                            k_state,
                            kDsv4KernelTokensPerBlock,
                            suggested);
    }
    // F01-PR2-followup task 2: K_state == 256 is the kernel-block identity
    // (no shrink). Normalize it to OFF so salt bit3 stays dark and legacy
    // hash bytes are preserved byte-for-byte. The override loop below would
    // be a no-op at 256 anyway, but writing the mirror to 256 would flip
    // the salt bit3 / handshake bitmap for a config that is operationally
    // identical to OFF — a needless mixed-mode PD-pair fail-loud.
    if (k_state > 0 && k_state < static_cast<int>(kDsv4KernelTokensPerBlock)) {
        constexpr size_t kStatePoolIndices[]   = {3, 4, 5};  // INDEXER_STATE, CSA_STATE, HCA_STATE
        for (size_t state_idx : kStatePoolIndices) {
            auto& pool = pools[state_idx];
            RTP_LLM_CHECK_WITH_INFO(!pool.is_paged, "DSV4 state pool idx %zu unexpectedly paged", state_idx);
            pool.entries_per_block = static_cast<uint32_t>(k_state);
        }
        config.state_entries_per_block_constant = k_state;
        RTP_LLM_LOG_INFO("F01 PR-1: K_state=%d active; 3 DSV4 STATE pools collapsed from "
                         "%u to %d entries/block (per-block bytes shrunk by %ux)",
                         k_state,
                         kDsv4KernelTokensPerBlock,
                         k_state,
                         kDsv4KernelTokensPerBlock / static_cast<uint32_t>(k_state));
    } else {
        if (k_state == static_cast<int>(kDsv4KernelTokensPerBlock)) {
            RTP_LLM_LOG_INFO("F01 PR-1: K_state=%d == kernel block size %u (identity), normalizing "
                             "to OFF; salt/hash bytes remain byte-identical to legacy.",
                             k_state,
                             kDsv4KernelTokensPerBlock);
        }
        config.state_entries_per_block_constant = 0;
    }

    config.layer_num                                = static_cast<uint32_t>(sets.all_layers.size());
    config.layer_all_num                            = config.layer_num;
    config.use_mla                                  = false;
    config.is_sparse                                = true;
    config.seq_size_per_block                       = physical_tokens_per_block;
    config.kernel_seq_size_per_block                = kDsv4KernelTokensPerBlock;
    config.use_typed_cache_regions                  = true;
    config.use_opaque_kv_cache_store                = true;
    config.disable_decode_first_malloc_device_reuse = true;

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();
    // All groups share the same physical seq_size — required so the global
    // cache_keys array (initCacheKeys uses config.seq_size_per_block) aligns
    // with every group's match() / insertIntoCache() granularity.
    config.group_seq_size_per_block.assign(pools.size(), physical_tokens_per_block);
    config.cache_specs.reserve(pools.size());
    config.global_layer_ids.reserve(pools.size());
    config.layer_ids.reserve(pools.size());
    config.group_types.reserve(pools.size());
    config.group_region_names.reserve(pools.size());
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        const auto& pool = pools[gid];
        auto        spec = makeDSV4Spec(pool, physical_tokens_per_block);

        config.cache_specs.push_back(spec);
        config.global_layer_ids.push_back(*pool.layer_ids);
        config.layer_ids.push_back(*pool.layer_ids);
        config.group_types.push_back(pool.is_paged ? CacheGroupType::FULL : CacheGroupType::SWA);
        config.group_region_names.push_back(pool.region_name);
    }

    // ---- F02 super-block layout (M02-PR1: default OFF, no behaviour change) ----
    // bps[p] == 1 for every DSV4 pool today. `enabled` only flips when the
    // user explicitly opts in via DSV4_UNIFIED_BLOCKS=1 (tri-state default -1
    // resolves to legacy OFF until canary §1.0 11-item checklist clears and
    // the default is re-flipped to 1). num_super_blocks stays 0 in PR-1;
    // HybridPoolConfigCreator populates it in M02-PR2.
    config.super_block_layout.bps.assign(pools.size(), 1u);
    config.super_block_layout.enabled = (kv_cache_config.dsv4_unified_block_count == 1);
}

}  // namespace rtp_llm
