#include "rtp_llm/cpp/cache/KVCacheManager.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/metrics/KVCacheCanaryMetrics.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

size_t expectedCPShardedLocalBlocks(const CPSlotMapper& mapper, int seq_len, int reserve_step) {
    const int effective_seq_len = mapper.effectiveSeqLenForAlloc(std::max(seq_len, 0));
    const int block_size        = mapper.blockSize();
    const int total_len         = effective_seq_len + std::max(reserve_step, 0);
    return static_cast<size_t>((total_len + block_size - 1) / block_size);
}

}  // namespace

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const KVCacheConfig&               kv_cache_config,
                               const ParallelismConfig&           parallelism_config,
                               const RuntimeConfig&               runtime_config,
                               const SpeculativeExecutionConfig&  sp_config,
                               const PDSepConfig&                 pd_sep_config,
                               const CacheStoreConfig&            cache_store_config):
    config_(buildFinalConfig(config, warmup, parallelism_config, runtime_config)),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config),
    sp_config_(sp_config),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config) {
    // R6 DEFEND-2 HIGH-8: KVCacheConfig field-range validation at the
    // earliest possible point in construction.  Catches argparse / unit-
    // conversion bugs that today silently cap or wrap (e.g.
    // ``--reserve_block_ratio 500`` would reserve 5x available_blocks then
    // underflow into a stealth perf cliff).  All ranges below are documented
    // in ConfigModules.h; rejecting at boundary surfaces misconfig before
    // any pool allocation.
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.reserve_block_ratio >= 0
                                && kv_cache_config_.reserve_block_ratio <= 100,
                            "KVCacheConfig.reserve_block_ratio=%ld must be in [0,100] (percent)",
                            static_cast<long>(kv_cache_config_.reserve_block_ratio));
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.dsv4_unified_block_count >= -1
                                && kv_cache_config_.dsv4_unified_block_count <= 1,
                            "KVCacheConfig.dsv4_unified_block_count=%d must be in {-1,0,1}",
                            kv_cache_config_.dsv4_unified_block_count);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.dsv4_state_entries_per_block >= 0
                                && kv_cache_config_.dsv4_state_entries_per_block <= 256,
                            "KVCacheConfig.dsv4_state_entries_per_block=%d must be in [0,256]",
                            kv_cache_config_.dsv4_state_entries_per_block);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.kv_cache_mem_mb != 0,
                            "KVCacheConfig.kv_cache_mem_mb=0 is meaningless (use -1 for auto, "
                            "or a positive explicit budget)");
    if (config_.state_pool_uses_pinned_cpu && kv_cache_config_.state_pool_memory_mb == 0) {
        // ``state_pool_uses_pinned_cpu`` is set by CacheConfigCreator either
        // because an explicit env budget (state_pool_memory_mb > 0) is
        // present OR because role_type==PREFILL forces STATE off-HBM.  The
        // second branch yields the (flag-true, budget-zero) combination
        // documented as "auto-size from HBM-derived block_num" — warn once
        // so operators know the budget knob did NOT participate.
        RTP_LLM_LOG_WARNING("state_pool_uses_pinned_cpu=true but state_pool_memory_mb=0 — STATE "
                            "pool will be auto-sized to HBM-derived block_num via DSV4 PD-prefill "
                            "override");
    }

    // R6 DEFEND-2 HIGH-7: CacheConfig divisor guards.  ``seq_size_per_block``
    // is the primary divisor for every per-block / per-token conversion
    // (availableTokensNum, scheduler capacity probes, CP slot mapper) — a
    // hand-tuned task_info with this set to 0 today silently makes
    // ``available_blocks * 0 == 0`` so the scheduler infers no capacity.
    // Likewise ``kernel_seq_size_per_block``, when nonzero, MUST divide
    // ``seq_size_per_block`` exactly — otherwise the kernel <-> KV block
    // ratio is fractional and ``kernelBlocksPerKvBlock`` rounds, dropping
    // blocks.  Both surfaces fire once at construction with a clear
    // message instead of as silent zero-throughput later.
    RTP_LLM_CHECK_WITH_INFO(config_.seq_size_per_block > 0,
                            "CacheConfig.seq_size_per_block must be > 0 (got %zu)",
                            config_.seq_size_per_block);
    RTP_LLM_CHECK_WITH_INFO(
        config_.kernel_seq_size_per_block == 0
            || config_.seq_size_per_block % config_.kernel_seq_size_per_block == 0,
        "CacheConfig.kernel_seq_size_per_block=%zu must divide seq_size_per_block=%zu (or be 0)",
        config_.kernel_seq_size_per_block,
        config_.seq_size_per_block);
    for (size_t gid = 0; gid < config_.group_seq_size_per_block.size(); ++gid) {
        // Zero is a sentinel meaning "fall back to seq_size_per_block";
        // negative is impossible (size_t).  Reject only the explicit zero
        // when the surrounding caller treats it as an out-of-band divisor
        // — the fallback path at availableTokensNum (lines 569-571) already
        // guards against 0 by replacing it; here we surface configs where
        // a non-fallback CHK was intended.
        if (config_.group_seq_size_per_block[gid] == 0) {
            RTP_LLM_LOG_WARNING("CacheConfig.group_seq_size_per_block[%zu]=0 — caller fallback "
                                "to seq_size_per_block=%zu will apply",
                                gid,
                                config_.seq_size_per_block);
        }
    }

    // R6 DEFEND-2 HIGH-4: post-buildFinalConfig invariant CHK.  Catches a
    // cross-rank ``min_element`` that returned 0 (a peer rank evaluated zero
    // blocks → silent under-provision) AND a ``finalizeBlockNums`` skip
    // (block_num unchanged so the per-group sizing was never re-derived
    // against the new global).  Without this CHK the post-warmup engine
    // would have an empty pool and every malloc would return success=false
    // forever, hanging the scheduler.  Locks down the XR4-9 contract that
    // every ``config_`` read outside the ctor sees a fully finalised
    // snapshot.  Skipped on warmup path (block_num is intentionally 1
    // there and group_block_nums is not yet populated).
    if (!warmup) {
        RTP_LLM_CHECK_WITH_INFO(config_.block_num > 0,
                                "post-buildFinalConfig invariant: block_num=%d must be > 0 "
                                "(cross-rank min collapsed to 0 — peer under-provision?)",
                                config_.block_num);
        RTP_LLM_CHECK_WITH_INFO(!config_.use_independent_block_pools
                                    || config_.group_block_nums.size() == config_.cache_specs.size(),
                                "post-buildFinalConfig invariant: use_independent_block_pools=true "
                                "but group_block_nums.size=%zu != cache_specs.size=%zu "
                                "(finalizeBlockNums was skipped — block_num didn't change after sync?)",
                                config_.group_block_nums.size(),
                                config_.cache_specs.size());
    }

    // F01-PR2-followup task 4: WARN when the user opted into the K_state
    // hook via --dsv4_state_entries_per_block but the DSV4 helper never
    // ran (the model is not DSV4) — otherwise the override is silently
    // ignored and the operator gets no signal that their flag was a
    // no-op. ``state_entries_per_block_constant`` is the mirror written
    // only by DSV4CacheConfigHelper::applyConfig (DSV4-typed hybrid
    // configs); a non-DSV4 model leaves it at 0. K_state==256 is the
    // identity case that the helper normalizes to mirror=0 (task 2);
    // do not WARN there because the helper did fire.
    const int requested_k_state = kv_cache_config_.dsv4_state_entries_per_block;
    if (requested_k_state > 0 && requested_k_state != 256
        && config_.state_entries_per_block_constant == 0) {
        RTP_LLM_LOG_WARNING("--dsv4_state_entries_per_block=%d was set but the active model is "
                            "not a DSV4 hybrid config (layer_compress_ratios empty); the "
                            "K_state hook is silently ignored. Either deploy on DSV4 or unset "
                            "the flag (default 0).",
                            requested_k_state);
    }

    // Page-level RR sharding context: one CPSlotMapper for the lifetime of the
    // manager, broadcast into every malloc/insert via auto-injection in
    // malloc()/insertIntoCache().  When kv_cache_sharded=false (or tp_size==1),
    // cp_slot_mapper_ stays nullptr and every call site stays bit-equal to the
    // pre-RR behaviour.
    const auto& cp_cfg = parallelism_config_.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && parallelism_config_.tp_size > 1) {
        cp_slot_mapper_ = std::make_shared<CPSlotMapper>(static_cast<int>(parallelism_config_.tp_rank),
                                                         static_cast<int>(parallelism_config_.tp_size),
                                                         static_cast<int>(config_.seq_size_per_block));
        RTP_LLM_LOG_INFO("CP sharded KV cache enabled: cp_rank=%d, cp_size=%d, block_size=%zu, "
                         "virtual_block_size=%d",
                         (int)parallelism_config_.tp_rank,
                         (int)parallelism_config_.tp_size,
                         config_.seq_size_per_block,
                         cp_slot_mapper_->virtualBlockSize());
    }

    RTP_LLM_LOG_INFO("cache config: layer_num=%d, block_num=%d, block_size=%dB, seq_size_per_block=%zu",
                     config_.layer_num,
                     config_.block_num,
                     config_.block_size_bytes,
                     config_.seq_size_per_block);
}

KVCacheManager::~KVCacheManager() {
    stop_.store(true, std::memory_order_relaxed);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    allocator_.reset();
    coordinator_.reset();
}

// 初始化和配置相关

bool KVCacheManager::init() {
    // FIX-B HIGH-6 (DEFEND-2 HIGH-3): reject double-init.  Without this guard
    // a second call would spawn an extra metrics_reporter_thread_ (orphaning
    // the first), overwrite allocator_ mid-flight, and re-bind the connector
    // socket inside coordinator_->init().  Cheap caller-bug catcher that
    // closes the XR4-9 config_-drift hazard without requiring the full
    // const-config refactor.
    RTP_LLM_CHECK_WITH_INFO(!initialized_, "KVCacheManager::init called twice");
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "cache specs must not be empty");

    auto shared_cache = std::make_shared<SharedBlockCache>();

    const bool is_hybrid = config_.groupNums() > 1;
    if (config_.use_independent_block_pools) {
        allocator_ = std::make_shared<rtp_llm::HybridPoolKVCacheAllocator>(
            config_,
            AllocationType::DEVICE,
            metrics_reporter_,
            kv_cache_config_.reserve_block_ratio,
            pd_sep_config_.role_type);
    } else if (is_hybrid) {
        allocator_ = std::make_shared<rtp_llm::HybridTypeKVCacheAllocator>(
            config_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
    } else {
        allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(
            config_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
    }

    allocator_->setSharedBlockCache(shared_cache);
    RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "KVCacheAllocator init failed");

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    initConnectorCoordinator();
    initialized_ = true;
    return true;
}

const CacheConfig& KVCacheManager::cacheConfig() const {
    return config_;
}

const CacheConfig& KVCacheManager::getMTPModuleCacheConfig(int mtp_module_id) const {
    return *config_.mtp_sub_configs[mtp_module_id];
}

// 显存管理和缓存分配

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids);

    // Auto-inject cp_slot_mapper when CP sharding is active and the caller
    // didn't supply one. Fast path: if cp_slot_mapper_ is null (sharding off)
    // we use the original const-ref directly with no copy.
    const MallocInfo* effective = &malloc_info;
    MallocInfo        patched;
    if (cp_slot_mapper_ && !malloc_info.cp_slot_mapper) {
        patched                = malloc_info;
        patched.cp_slot_mapper = cp_slot_mapper_;
        effective              = &patched;
    }

    // Cache-key computation is identical for CP and non-CP — we always have
    // the full sequence's token ids; rolling hash is at block_size granularity.
    //
    // F01-PR2 part A: the salt is constructed from CacheConfig via the
    // shared ``makeCacheKeySalt`` producer (also used by the PD-handshake
    // wiring in KVCacheConnectorCoordinator::init).  Default config
    // (``state_entries_per_block_constant == 0``) → all-zero salt → XOR
    // with zero in ``applySalt`` → byte-identical legacy cache_keys.
    // When K_state > 0, salt.K_state mirrors the override so two PD peers
    // running different ``--dsv4_state_entries_per_block`` values produce
    // distinct cache_keys (Risk 9.6: silent reuse-miss rather than silent
    // corruption from cross-K_state cache_key collision).
    const int          seq_size_per_block = config_.seq_size_per_block;
    const CacheKeySalt salt               = makeCacheKeySalt(config_);
    if (!effective->batch_kv_cache_resource->curBlocksNum()) {
        initCacheKeys(effective->batch_kv_cache_resource, effective->complete_token_ids, seq_size_per_block, salt);
    } else {
        updateCacheKeys(effective->batch_kv_cache_resource, effective->complete_token_ids, seq_size_per_block, salt);
    }

    auto result = allocator_->malloc(*effective);

    // CP invariant: blocks holds this rank's local share of logical KV pages.
    // cacheKeys can be shorter than logical blocks because an in-flight partial
    // tail is not always cacheable, so derive the expected count from seq_len.
    if (result.success && effective->cp_slot_mapper && effective->cp_slot_mapper->isSharded()) {
        const auto& res        = effective->batch_kv_cache_resource->cacheResource(0);
        size_t      num_blocks = res.blocks().size();
        size_t      expected   = expectedCPShardedLocalBlocks(*effective->cp_slot_mapper,
                                                       effective->complete_token_ids->seqLength(),
                                                       effective->complete_token_ids->getReserveStep());
        RTP_LLM_CHECK_WITH_INFO(num_blocks == expected,
                                "CP invariant violated: blocks=%zu != expected_local_blocks=%zu "
                                "(seq_len=%d, reserve_step=%d, cp_size=%d, block_size=%d, cacheKeys=%zu)",
                                num_blocks,
                                expected,
                                effective->complete_token_ids->seqLength(),
                                effective->complete_token_ids->getReserveStep(),
                                effective->cp_slot_mapper->cpSize(),
                                effective->cp_slot_mapper->blockSize(),
                                res.cacheKeys().size());
    }

    return result;
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    allocator_->free(free_info);
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    RTP_LLM_PROFILE_FUNCTION();
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    if (cp_slot_mapper_ && !insert_info.cp_slot_mapper) {
        InsertInfo patched     = insert_info;
        patched.cp_slot_mapper = cp_slot_mapper_;
        allocator_->insertIntoCache(patched);
        return;
    }
    allocator_->insertIntoCache(insert_info);
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    return allocator_->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

// 块操作相关

void KVCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    return allocator_->blockCopy(src_block_index, dest_block_index);
}

void KVCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const torch::Tensor& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) {
    return allocator_->blockBatchCopy(copy_mapping_begin, copy_mapping_end);
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<BlockIdPair>&      block_update_mapping) {
    RTP_LLM_PROFILE_FUNCTION();
    return allocator_->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
}

// Write one KV block (optionally per-layer) from host/device tensors for test
bool KVCacheManager::setKVBlockValue(int                  block_index,
                                     int                  layer_id,
                                     const torch::Tensor& k_buffer,
                                     const torch::Tensor& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    auto&  spec             = config_.cache_specs[0];
    size_t expected_k_bytes = spec->k_block_size_bytes();
    size_t expected_v_bytes = spec->v_block_size_bytes();
    size_t src_k_bytes      = k_buffer.nbytes();
    size_t src_v_bytes      = v_buffer.nbytes();
    if (src_k_bytes < expected_k_bytes || src_v_bytes < expected_v_bytes) {
        RTP_LLM_LOG_ERROR("setKVBlockValue src bytes too small: k[%zu]<[%zu] or v[%zu]<[%zu]",
                          src_k_bytes,
                          expected_k_bytes,
                          src_v_bytes,
                          expected_v_bytes);
        return false;
    }

    auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);
    RTP_LLM_CHECK_WITH_INFO(
        !dst.empty(), "convertIndexToBuffer returned empty for layer %d, block %d", layer_id, block_index);
    if (!dst[0].addr) {
        RTP_LLM_LOG_ERROR("convertIndexToBuffer returned null for layer %d, block %d", layer_id, block_index);
        return false;
    }

    auto copyFunc = [&](const torch::Tensor& src_tensor, const BlockInfo& dst_block, size_t dst_byte_offset) -> bool {
        const size_t dst_bytes = dst_block.size_bytes;
        const size_t src_bytes = src_tensor.nbytes();
        if (dst_bytes < dst_byte_offset + src_bytes) {
            RTP_LLM_LOG_ERROR("dst block bytes[%zu] < dst_offset[%zu] + src bytes[%zu] in setKVBlockValue(layer=%d)",
                              dst_bytes,
                              dst_byte_offset,
                              src_bytes,
                              layer_id);
            return false;
        }

        auto* dst_ptr    = static_cast<char*>(dst_block.addr) + dst_byte_offset;
        auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
        auto  src_device = src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
        auto  dst_t      = torch::from_blob(
            dst_ptr, {(int64_t)src_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(dst_device));
        auto src_t = torch::from_blob(src_tensor.data_ptr(),
                                      {(int64_t)src_bytes},
                                      torch::TensorOptions().dtype(torch::kUInt8).device(src_device));
        dst_t.copy_(src_t);
        return true;
    };

    if (!copyFunc(k_buffer, dst[0], 0)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst[0], expected_k_bytes)) {
        return false;
    }

    cudaSyncAndCheck();
    return true;
}

bool KVCacheManager::setKVBlockValue(int block_index, const torch::Tensor& k_buffer, const torch::Tensor& v_buffer) {
    if (block_index < 0 || block_index >= config_.block_num) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_num);
        return false;
    }

    bool all_success = true;
    for (int layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        all_success = setKVBlockValue(block_index, layer_id, k_buffer, v_buffer) && all_success;
    }
    return all_success;
}

// 地址转换和缓冲区访问

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return allocator_->convertIndexToAddr(layer_id, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(int block_index, int layer_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index, partition_count, partition_id);
}

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id, KVCacheRegionName region_name) const {
    return allocator_->convertIndexToAddr(layer_id, region_name, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, KVCacheRegionName region_name) const {
    return allocator_->convertIndexToBuffer(layer_id, region_name, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(
    int block_index, int layer_id, KVCacheRegionName region_name, int partition_count, int partition_id) const {
    return allocator_->convertIndexToBuffer(layer_id, region_name, block_index, partition_count, partition_id);
}

CacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return allocator_->allLayerCacheBase();
}

CacheLayerLayout KVCacheManager::getMainModelCacheLayerLayout() const {
    CacheLayerLayout layout;

    auto  all_layout        = allocator_->allLayerCacheBase();
    auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
    auto& all_scale_tensors = all_layout.layers_to_scale_buffer_ptrs;

    layout.layer_to_groups.resize(config_.layer_num);
    layout.layer_to_group_ids.resize(config_.layer_num);
    layout.layer_region_to_group_id.resize(config_.layer_num);
    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_num);
    if (!all_scale_tensors.empty()) {
        layout.layers_to_scale_buffer_ptrs.resize(config_.layer_num);
    }

    layout.group_types        = config_.group_types;
    layout.group_region_names = config_.group_region_names;
    layout.layer_group_types.resize(config_.layer_num, CacheGroupType::FULL);
    layout.layers_to_kv_buffer_ptrs_by_attn.resize(config_.layer_num);
    if (!all_layout.layers_to_scale_buffer_ptrs_by_attn.empty()) {
        layout.layers_to_scale_buffer_ptrs_by_attn.resize(config_.layer_num);
    }

    RTP_LLM_CHECK_WITH_INFO(config_.layer_num <= all_layer_tensors.size(),
                            "config_.layer_num[%d] > all_layer_tensors.size()[%ld]",
                            config_.layer_num,
                            all_layer_tensors.size());

    for (int layer_id = 0; layer_id < static_cast<int>(config_.layer_num); ++layer_id) {
        if (static_cast<size_t>(layer_id) < all_layer_tensors.size()) {
            layout.layer_to_groups[layer_id]          = all_layout.layer_to_groups[layer_id];
            layout.layers_to_kv_buffer_ptrs[layer_id] = all_layer_tensors[layer_id];
        } else {
            RTP_LLM_CHECK(false);
        }

        if (!all_scale_tensors.empty()) {
            if (static_cast<size_t>(layer_id) < all_scale_tensors.size()) {
                layout.layers_to_scale_buffer_ptrs[layer_id] = all_scale_tensors[layer_id];
            } else {
                RTP_LLM_CHECK(false);
            }
        }
        if (static_cast<size_t>(layer_id) < config_.layer_group_types.size()) {
            layout.layer_group_types[layer_id] = config_.layer_group_types[static_cast<size_t>(layer_id)];
        }
        if (static_cast<size_t>(layer_id) < config_.layer_to_group_ids.size()) {
            layout.layer_to_group_ids[layer_id] = config_.layer_to_group_ids[static_cast<size_t>(layer_id)];
        }
        if (static_cast<size_t>(layer_id) < config_.layer_region_to_group_id.size()) {
            layout.layer_region_to_group_id[layer_id] = config_.layer_region_to_group_id[static_cast<size_t>(layer_id)];
        }
        if (static_cast<size_t>(layer_id) < all_layout.layers_to_kv_buffer_ptrs_by_attn.size()) {
            layout.layers_to_kv_buffer_ptrs_by_attn[layer_id] =
                all_layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)];
        }
        if (static_cast<size_t>(layer_id) < all_layout.layers_to_scale_buffer_ptrs_by_attn.size()) {
            layout.layers_to_scale_buffer_ptrs_by_attn[layer_id] =
                all_layout.layers_to_scale_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)];
        }
    }

    return layout;
}

CacheLayerLayout KVCacheManager::getMTPModuleCacheLayerLayout(int mtp_module_id) const {
    CacheLayerLayout layout;

    RTP_LLM_CHECK_WITH_INFO(mtp_module_id >= 0 && static_cast<size_t>(mtp_module_id) < config_.mtp_sub_configs.size(),
                            "Invalid mtp_module_id: %d, must be in range [0, %zu)",
                            mtp_module_id,
                            config_.mtp_sub_configs.size());

    const auto& mtp_sub_config = config_.mtp_sub_configs[mtp_module_id];
    RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%d] is null", mtp_module_id);
    RTP_LLM_CHECK_WITH_INFO(
        !mtp_sub_config->global_layer_ids.empty(), "mtp_sub_configs[%d]->global_layer_ids is empty", mtp_module_id);

    // Flatten across all groups: SWA-only DSV4 propose configs put the
    // single MTP layer in the SWA group (gid=6), not FULL[0], so reading
    // group 0 alone would be empty.  Walk every group and collect any
    // global_layer_ids it contributed so this layout is independent of
    // which pool the propose layer lives in.
    std::vector<int> mtp_global_layer_ids;
    for (const auto& group_ids : mtp_sub_config->global_layer_ids) {
        for (int lid : group_ids) {
            mtp_global_layer_ids.push_back(lid);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(
        !mtp_global_layer_ids.empty(), "mtp_sub_configs[%d] has no layers across any group", mtp_module_id);
    const uint32_t mtp_layer_num = mtp_sub_config->layer_num;

    auto  all_layout        = allocator_->allLayerCacheBase();
    auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
    auto& all_scale_tensors = all_layout.layers_to_scale_buffer_ptrs;

    layout.layer_to_groups.resize(mtp_layer_num);
    layout.layers_to_kv_buffer_ptrs.resize(mtp_layer_num);
    if (!all_scale_tensors.empty()) {
        layout.layers_to_scale_buffer_ptrs.resize(mtp_layer_num);
    }
    layout.layer_group_types.resize(mtp_layer_num, CacheGroupType::FULL);
    // Propagate the propose's group identity / typed-pool views so the
    // Python decode path can build per-attn-type paged metadata.
    // Mirrors what ``getCacheLayerLayout()`` does for the main model;
    // without these, ``build_metadata_eager`` finds an empty
    // ``group_region_names`` and emits zero ``paged_block_tables``,
    // which trips Attention.forward_decode's "no paged metadata" gate.
    layout.group_region_names = mtp_sub_config->group_region_names;
    layout.group_types        = mtp_sub_config->group_types;
    // Typed-pool views are indexed by LOCAL layer id from the MTP model's
    // attention modules (self.layer_id ∈ [0, mtp_layer_num)).  The full
    // layout's by_attn arrays are indexed by GLOBAL layer id (main + MTP
    // appended), so we MUST remap from global → local — copying the full
    // arrays verbatim makes local index 0 return main layer 0's typed
    // buffers, which causes the draft to write into the main model's KV
    // pool and corrupts target verify (0% acceptance regression).
    const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    layout.layers_to_kv_buffer_ptrs_by_attn.assign(mtp_layer_num, std::vector<torch::Tensor>(region_name_count));
    layout.layers_to_scale_buffer_ptrs_by_attn.assign(mtp_layer_num, std::vector<torch::Tensor>(region_name_count));
    layout.layer_region_to_group_id.assign(mtp_layer_num, std::vector<int>(region_name_count, -1));

    for (uint32_t local_layer_id = 0; local_layer_id < mtp_layer_num; ++local_layer_id) {
        if (local_layer_id < mtp_global_layer_ids.size()) {
            const int global_layer_id = mtp_global_layer_ids[local_layer_id];

            if (global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < all_layer_tensors.size()) {
                layout.layer_to_groups[local_layer_id]          = all_layout.layer_to_groups[global_layer_id];
                layout.layers_to_kv_buffer_ptrs[local_layer_id] = all_layer_tensors[global_layer_id];
            } else {
                RTP_LLM_CHECK(false);
            }

            if (!all_scale_tensors.empty()) {
                if (global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < all_scale_tensors.size()) {
                    layout.layers_to_scale_buffer_ptrs[local_layer_id] = all_scale_tensors[global_layer_id];
                } else {
                    RTP_LLM_CHECK(false);
                }
            }
            if (local_layer_id < mtp_sub_config->layer_group_types.size()) {
                layout.layer_group_types[local_layer_id] = mtp_sub_config->layer_group_types[local_layer_id];
            }

            // Remap typed-pool buffers from GLOBAL layer id row to the MTP
            // model's LOCAL layer id row.  Each region slot is copied as-is
            // (it points at the per-group BlockPool's per-layer slice that
            // KVCacheGroup::init bound for global_layer_id), so the draft's
            // SWA write for local id 0 lands in the SWA pool slot that was
            // created for the appended MTP global layer — NOT main layer 0.
            const size_t gid = static_cast<size_t>(global_layer_id);
            if (gid < all_layout.layers_to_kv_buffer_ptrs_by_attn.size()) {
                const auto& src_kv = all_layout.layers_to_kv_buffer_ptrs_by_attn[gid];
                for (size_t a = 0; a < region_name_count && a < src_kv.size(); ++a) {
                    layout.layers_to_kv_buffer_ptrs_by_attn[local_layer_id][a] = src_kv[a];
                }
            }
            if (gid < all_layout.layers_to_scale_buffer_ptrs_by_attn.size()) {
                const auto& src_scale = all_layout.layers_to_scale_buffer_ptrs_by_attn[gid];
                for (size_t a = 0; a < region_name_count && a < src_scale.size(); ++a) {
                    layout.layers_to_scale_buffer_ptrs_by_attn[local_layer_id][a] = src_scale[a];
                }
            }
            // Remap the typed region→group sentinel so OpDefs.h getLayerCache
            // ownership check (`layer_region_to_group_id[layer][attn] < 0`)
            // sees the propose-config's per-attn ownership for THIS local
            // layer, not main layer 0's CSA/HCA/INDEXER ownership.
            if (gid < all_layout.layer_region_to_group_id.size()) {
                const auto& src_region = all_layout.layer_region_to_group_id[gid];
                for (size_t a = 0; a < region_name_count && a < src_region.size(); ++a) {
                    layout.layer_region_to_group_id[local_layer_id][a] = src_region[a];
                }
            }
        } else {
            RTP_LLM_CHECK(false);
        }
    }

    return layout;
}

// 资源统计和信息查询

size_t KVCacheManager::freeBlocksNum() const {
    return allocator_->freeBlocksNum();
}

size_t KVCacheManager::availableBlocksNum() const {
    return allocator_->availableBlocksNum();
}

size_t KVCacheManager::notInUseBlocksNum() const {
    return allocator_->notInUseBlocksNum();
}

BatchKVCacheResourcePtr KVCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    return allocator_->popBlocksFromCache(min_blocks_to_free);
}

void KVCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    allocator_->blockCacheFree(batch_kv_cache_resource);
}

size_t KVCacheManager::availableTokensNum() const {
    return allocator_->availableTokensNum();
}

size_t KVCacheManager::totalBlocksNum() const {
    return allocator_->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    return allocator_->maxAvailableTokensNum();
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;

    if (!allocator_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        info.version = latest_version;
        return info;
    }

    if (need_cache_keys) {
        std::unordered_set<CacheKeyType> all_keys;
        // device cache keys
        auto shared_cache = allocator_->sharedBlockCache();
        if (shared_cache) {
            auto cache_keys = shared_cache->allCacheKeys();
            all_keys.insert(cache_keys.begin(), cache_keys.end());
            info.version = shared_cache->version();
        }
        // memory cache keys
        const auto mem_cache_keys = coordinator_->memoryCacheKeys();
        all_keys.insert(mem_cache_keys.begin(), mem_cache_keys.end());

        info.cached_keys.assign(all_keys.begin(), all_keys.end());
    }

    const size_t block_size_tokens = config_.seq_size_per_block;

    info.block_size = block_size_tokens;
    if (auto hybrid = std::dynamic_pointer_cast<rtp_llm::HybridPoolKVCacheAllocator>(allocator_)) {
        const auto& pools            = hybrid->groupBlockPools();
        size_t      total_tokens     = std::numeric_limits<size_t>::max();
        size_t      available_tokens = std::numeric_limits<size_t>::max();
        bool        has_pool         = false;
        for (size_t gid = 0; gid < pools.size(); ++gid) {
            const auto& pool = pools[gid];
            if (!pool) {
                continue;
            }
            const size_t seq_size =
                (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                    config_.group_seq_size_per_block[gid] :
                    block_size_tokens;
            total_tokens     = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
            available_tokens = std::min(available_tokens, pool->availableBlocksNum() * seq_size);
            has_pool         = true;
        }
        info.total_kv_cache     = has_pool ? total_tokens : 0;
        info.available_kv_cache = has_pool ? available_tokens : 0;
    } else {
        const size_t total_blocks     = allocator_->totalBlocksNum();
        const size_t available_blocks = allocator_->availableBlocksNum();
        info.total_kv_cache           = total_blocks * block_size_tokens;
        info.available_kv_cache       = available_blocks * block_size_tokens;
    }
    // cached_keys left empty for now; can be populated when distributed cache is wired up.

    return info;
}

// 系统资源管理

void KVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    allocator_->regUserMr(model_id, std::move(cache_store));
}

void KVCacheManager::setCacheStore(std::shared_ptr<CacheStore> cache_store) {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    cache_store_ = std::move(cache_store);
}

std::shared_ptr<CacheStore> KVCacheManager::getCacheStore() const {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    return cache_store_;
}

bool KVCacheManager::hasActiveConnectors() const {
    return coordinator_ && coordinator_->hasActiveConnectors();
}

// PD separation: increment KV cache reference count
std::shared_ptr<KVCacheResource>
KVCacheManager::incrKVCacheRef(const KVCacheResource& resource, const CacheKeysType& cache_keys, bool is_connector) {
    return allocator_->incrKVCacheRef(resource, cache_keys, is_connector);
}

bool KVCacheManager::hasP2PConnector() const {
    return coordinator_ && coordinator_->hasP2PConnector();
}

// 异步连接器操作

std::shared_ptr<AsyncContext>
KVCacheManager::asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    return coordinator_->asyncRead(connector_context);
}

std::shared_ptr<AsyncContext>
KVCacheManager::asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    return coordinator_->asyncWrite(connector_context);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    return coordinator_->executeFunction(request, response);
}

void KVCacheManager::initConnectorCoordinator() {
    RTP_LLM_LOG_INFO(
        "init connector coordinator, cache config: [%s], kv cache config: [%s], runtime config: [%s], parallelism config: [%s], sp config: [%s]",
        config_.debugString().c_str(),
        kv_cache_config_.to_string().c_str(),
        runtime_config_.to_string().c_str(),
        parallelism_config_.to_string().c_str(),
        sp_config_.to_string().c_str());
    coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(config_,
                                                                 kv_cache_config_,
                                                                 runtime_config_,
                                                                 parallelism_config_,
                                                                 sp_config_,
                                                                 allocator_,
                                                                 metrics_reporter_,
                                                                 pd_sep_config_,
                                                                 cache_store_config_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_->init(), "connector coordinator init failed");
}

CacheConfig KVCacheManager::buildFinalConfig(CacheConfig              config,
                                             bool                     warmup,
                                             const ParallelismConfig& parallelism_config,
                                             const RuntimeConfig&     runtime_config) {
    // R6 DEFEND-2 / DEV-2: this is the lifted ``allocateAndSync`` body.
    // Operating on a LOCAL ``config`` value (passed by value) and returning
    // the finalised snapshot lets the caller (KVCacheManager ctor init list)
    // bind ``config_`` as a ``const`` member — no mid-flight mutation can
    // ever drift the field after this returns.  Behaviour is byte-equal
    // to the previous in-place mutation: warmup yields block_num=1; non-
    // warmup runs the cross-rank gather + min + conditional
    // ``finalizeBlockNums`` exactly as before.
    if (warmup) {
        config.block_num = 1;
        return config;
    }

    const int original_block_num = config.block_num;
    size_t    world_size         = parallelism_config.tp_size * parallelism_config.dp_size;
    if (world_size > 1) {
        size_t local_rank =
            parallelism_config.tp_size * parallelism_config.dp_rank + parallelism_config.tp_rank;
        auto block_num_t          = torch::empty({(int64_t)world_size}, torch::kInt32).pin_memory();
        auto block_num_ptr        = block_num_t.data_ptr<int>();
        block_num_ptr[local_rank] = config.block_num;
        execAllGather({{block_num_t}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        if (parallelism_config.ffn_disaggregate_config.is_ffn_service()) {
            config.block_num = 1;
        } else {
            config.block_num = *std::min_element(block_num_ptr, block_num_ptr + world_size);
        }
    }
    // Only re-derive per-group sizing when the cross-rank sync actually shrunk
    // block_num — otherwise we would clobber test/production configs that have
    // pre-populated differentiated group_block_nums (e.g. DSV4 stress tests
    // that hand-tune SWA pool sizes vs FULL paged pools, or
    // CacheConfigCreator-built configs that already finalized once).
    if (config.use_independent_block_pools && config.block_num != original_block_num) {
        config.finalizeBlockNums(static_cast<uint32_t>(config.block_num), runtime_config);
    }
    RTP_LLM_LOG_INFO("block_num is %d after tp sync", config.block_num);
    return config;
}

void KVCacheManager::reportMetricsLoop() {
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    // Raw "kvc raw" log lines are throttled to once every 3 minutes — kmonitor
    // gauges still report every 1s so dashboards stay continuous, but the
    // diagnostic log is intended for sporadic spot-checks, not per-tick spam.
    // Initialise to "3 min ago" so the first iteration emits one line right
    // away (gives operators an immediate baseline after restart).
    constexpr auto kLogInterval  = std::chrono::minutes(3);
    auto           last_log_time = std::chrono::steady_clock::now() - kLogInterval;
    while (!stop_.load(std::memory_order_relaxed)) {
        if (!metrics_reporter_ || !allocator_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector collector;

        auto shared_cache = allocator_->sharedBlockCache();

        const auto total_blocks         = allocator_->totalBlocksNum();
        const auto available_blocks     = allocator_->availableBlocksNum();
        const auto request_ref_blocks   = allocator_->requestRefBlocksNum();
        const auto connector_ref_blocks = allocator_->connectorRefBlocksNum();

        collector.kv_cache_item_num             = shared_cache ? static_cast<int64_t>(shared_cache->size()) : 0;
        collector.kv_cache_left_seq             = static_cast<int64_t>(available_blocks * config_.seq_size_per_block);
        collector.kv_cache_available_blocks     = static_cast<int64_t>(available_blocks);
        collector.kv_cache_request_ref_blocks   = static_cast<int64_t>(request_ref_blocks);
        collector.kv_cache_connector_ref_blocks = static_cast<int64_t>(connector_ref_blocks);
        collector.kv_cache_free_blocks          = static_cast<int64_t>(allocator_->freeBlocksNum());
        collector.kv_cache_used_ratio =
            (total_blocks == 0) ?
                0.0f :
                static_cast<float>(100.0 * (total_blocks - available_blocks) / static_cast<double>(total_blocks));
        collector.mr_cost_time_ms = allocator_->getMrCostTimeMs();

        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &collector);

        // G5 canary alias (PHASE5_CANARY_PROCEDURE.md §1 item kv_cache.hbm_used_blocks).
        // Used = total - free; semantics match the "hbm_used_blocks" axis the
        // canary dashboard graphs (G5b threshold: ≤ 2 blocks vs baseline).
        // Fires every tick alongside the legacy rtp_llm_kv_cache_free_blocks /
        // _available_blocks gauges so the canary path stays bit-aligned with
        // the existing pool numbers.
        const int64_t hbm_used_blocks = static_cast<int64_t>(total_blocks) - collector.kv_cache_free_blocks;
        recordCanaryHbmUsedBlocks(metrics_reporter_, hbm_used_blocks);

        // Decide once per tick whether the throttled diagnostic log should fire.
        // Math is self-consistent within this tick by construction; the log is
        // for spot-checking when dashboards look off — once every 3 min suffices.
        const auto now        = std::chrono::steady_clock::now();
        const bool should_log = (now - last_log_time) >= kLogInterval;
        if (should_log) {
            last_log_time = now;
            RTP_LLM_LOG_INFO(
                "kvc raw global: total=%zu avail=%zu req_ref=%zu con_ref=%zu free=%zu items=%ld ratio=%.4f%%",
                total_blocks,
                available_blocks,
                request_ref_blocks,
                connector_ref_blocks,
                static_cast<size_t>(collector.kv_cache_free_blocks),
                static_cast<long>(collector.kv_cache_item_num),
                collector.kv_cache_used_ratio);
        }

        // Per-pool breakdown — only meaningful for HybridPoolKVCacheAllocator
        // (DSv4's 7-pool layout etc.). Single-pool allocators skip this.
        // kmonitor samples are emitted every tick (dashboards stay continuous);
        // the diagnostic "kvc raw pool[gid]" lines piggyback on should_log so
        // they fire alongside "kvc raw global" once every 3 min.
        if (auto hybrid = std::dynamic_pointer_cast<rtp_llm::HybridPoolKVCacheAllocator>(allocator_)) {
            const auto& pools = hybrid->groupBlockPools();
            for (size_t gid = 0; gid < pools.size(); ++gid) {
                const auto& pool = pools[gid];
                if (!pool) {
                    continue;
                }
                const size_t pool_total     = pool->totalBlocksNum();
                const size_t pool_available = pool->availableBlocksNum();
                const size_t pool_free      = pool->freeBlocksNum();
                const size_t pool_req_ref   = pool->requestRefBlocksNum();
                const size_t pool_con_ref   = pool->connectorRefBlocksNum();
                const float  pool_used_ratio =
                    (pool_total == 0) ?
                         0.0f :
                         static_cast<float>(100.0 * (pool_total - pool_available) / static_cast<double>(pool_total));

                if (should_log) {
                    RTP_LLM_LOG_INFO(
                        "kvc raw pool[%zu]: total=%zu avail=%zu req_ref=%zu con_ref=%zu free=%zu ratio=%.4f%%",
                        gid,
                        pool_total,
                        pool_available,
                        pool_req_ref,
                        pool_con_ref,
                        pool_free,
                        pool_used_ratio);
                }

                RtpLLMCachePoolMetricsCollector pool_collector;
                pool_collector.free_blocks          = static_cast<int64_t>(pool_free);
                pool_collector.available_blocks     = static_cast<int64_t>(pool_available);
                pool_collector.request_ref_blocks   = static_cast<int64_t>(pool_req_ref);
                pool_collector.connector_ref_blocks = static_cast<int64_t>(pool_con_ref);
                pool_collector.total_blocks         = static_cast<int64_t>(pool_total);
                pool_collector.used_ratio           = pool_used_ratio;

                kmonitor::MetricsTags pool_tags("pool", std::to_string(gid));
                metrics_reporter_->report<RtpLLMCachePoolMetrics, RtpLLMCachePoolMetricsCollector>(&pool_tags,
                                                                                                   &pool_collector);
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

void KVCacheManager::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                P2PConnectorStartLoadResponsePB&      response,
                                std::function<bool()>                 is_cancelled) {
    if (coordinator_) {
        coordinator_->handleRead(request, response, is_cancelled);
    }
}

}  // namespace rtp_llm
