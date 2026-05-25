#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#ifdef USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#endif

namespace rtp_llm {
namespace {

CacheGroupType groupTypeForConnector(const CacheConfig& cache_config, int group_id) {
    if (group_id >= 0 && static_cast<size_t>(group_id) < cache_config.group_types.size()) {
        return cache_config.group_types[static_cast<size_t>(group_id)];
    }
    return CacheGroupType::FULL;
}

bool selectedLastRankKeysAreAligned(const KVCacheResource& source, int cp_size) {
    if (source.lastBlockAligned()) {
        return true;
    }
    const auto& keys = source.cacheKeys();
    if (keys.empty() || cp_size <= 1) {
        return source.lastBlockAligned();
    }
    const int partial_key_pos = static_cast<int>(keys.size() - 1);
    const int last_rank       = cp_size - 1;
    return partial_key_pos % cp_size != last_rank;
}

KVCacheResource makeCpShardedConnectorResource(const KVCacheResource& source,
                                               const CacheConfig&     cache_config,
                                               const CacheKeysType&   selected_keys,
                                               int                    cp_size) {
    std::vector<CacheGroupType> group_types;
    group_types.reserve(static_cast<size_t>(source.groupNums()));
    for (int gid = 0; gid < source.groupNums(); ++gid) {
        group_types.push_back(groupTypeForConnector(cache_config, gid));
    }

    KVCacheResource selected = source;
    selected.initGroups(source.groupNums(),
                        static_cast<int>(cache_config.layer_all_num),
                        cache_config.layer_to_group_id,
                        cache_config.kernelBlocksPerKvBlock(),
                        group_types,
                        cache_config.layer_region_to_group_id);
    selected.cacheKeys()        = selected_keys;
    const bool selected_aligned = selectedLastRankKeysAreAligned(source, cp_size);
    selected.setLastBlockAligned(selected_aligned);

    // Memory connector intentionally drops the last key to avoid matching a
    // partial tail.  After CP Page-RR remap, a source partial can belong to a
    // non-last rank, making the selected last-rank key complete.  Append the
    // original partial key as a connector-only dummy tail so the drop-last
    // contract discards the dummy, not the usable selected key.
    if (!source.lastBlockAligned() && selected_aligned && !source.cacheKeys().empty()) {
        selected.cacheKeys().push_back(source.cacheKeys().back());
        selected.setLastBlockAligned(false);
    }

    for (int gid = 0; gid < source.groupNums(); ++gid) {
        const auto&      src_blocks = source.blocks(gid);
        BlockIndicesType dst_blocks;
        dst_blocks.reserve(selected_keys.size());

        if (group_types[static_cast<size_t>(gid)] == CacheGroupType::FULL) {
            // FULL groups are physically RR-sharded: blocks are rank-local and
            // compact, keyed by the canonical last-rank key sequence.
            for (size_t i = 0; i < selected_keys.size(); ++i) {
                dst_blocks.push_back(i < src_blocks.size() ? src_blocks[i] : NULL_BLOCK_IDX);
            }
        } else {
            // SWA/state groups keep the non-sharded logical coordinate system.
            // Select the block at the original logical key position instead of
            // reinterpreting the group as rank-local compact storage.
            for (size_t logical_pos = static_cast<size_t>(cp_size - 1); dst_blocks.size() < selected_keys.size();
                 logical_pos += static_cast<size_t>(cp_size)) {
                dst_blocks.push_back(logical_pos < src_blocks.size() ? src_blocks[logical_pos] : NULL_BLOCK_IDX);
            }
        }

        selected.mutableBlockIds(gid).assign(std::move(dst_blocks));
    }

    return selected;
}

// M04-PR2: per-pool resource builder driven by the unified block-major planner.
//
// Gated by ``cache_config.super_block_layout.enabled`` at the call site.
// Default behaviour (gate off) is completely unchanged — the legacy
// ``makeCpShardedConnectorResource`` above is still reachable and bit-identical
// to today.
//
// Semantics:
//   * Pool descriptors are derived from ``cache_config.group_{types,region_names}``.
//   * The unified planner is invoked with ``shard_owner_rank = cp_size - 1``
//     (canonical last-rank namespace, matching the legacy
//     ``localCacheKeys(cp_size-1, cp_size)`` projection).
//   * Per-pool block lists are populated from each plan item's
//     ``offset_index`` — FULL pools get the rank-local compact offset
//     (``B / cp_size``); non-FULL pools are replicated to rank 0 only per the
//     planner's ownership rule (semantic improvement vs legacy, which shipped
//     non-FULL groups from every rank).
//   * Cache-keys are populated via the M01-sanctioned controlled mutators
//     ``clearCacheKeys`` / ``appendCacheKey`` / ``setLastBlockAlignedAll``
//     (Panel-A item 5 / C3 closure; Fix 19 / Fix 76 revised) — NO direct
//     ``selected.cacheKeys() =`` vector assignment, NO direct
//     ``setLastBlockAligned`` (legacy spelling) at this site.
//   * The dummy-tail trick is retained for byte-equivalence with legacy
//     under the unified path; PR-4 will atomically delete it and wire
//     per-pool ``PoolDescriptor.last_partial`` instead (Risk 9.2).
//   * ``last_partial`` on every pool descriptor is left at ``false`` —
//     M01 surface (per-pool alignment fan-out) lands in PR-4.
//
// Caller contract: invoked only when ``cp_size > 1`` (mirrors legacy).
KVCacheResource makeUnifiedConnectorResource(const KVCacheResource& source,
                                             const CacheConfig&     cache_config,
                                             const CacheKeysType&   selected_keys,
                                             int                    cp_size) {
    const int group_num = source.groupNums();

    std::vector<PoolDescriptor> pools;
    std::vector<size_t>         pool_block_counts;
    std::vector<CacheGroupType> group_types;
    pools.reserve(static_cast<size_t>(group_num));
    pool_block_counts.reserve(static_cast<size_t>(group_num));
    group_types.reserve(static_cast<size_t>(group_num));

    // M04-PR4 atomic cutover (REQ-D3):
    //   * FULL pools: last_partial = !source.lastBlockAligned() — the
    //     planner Step 2 drops B+1 == total_cache_keys for these pools,
    //     replacing the legacy dummy-tail append (deleted below).
    //   * SWA / LINEAR pools: last_partial = false unconditionally — SWA
    //     pools still emit the last 2 blocks regardless (M04 §3.3 Step 2
    //     comment), LINEAR emits exactly one tail block.  Setting
    //     last_partial=true on SWA would silently truncate the kvcache
    //     window by 1 vs legacy peers and break cross-version pairing.
    const bool source_last_partial = !source.lastBlockAligned();
    for (int gid = 0; gid < group_num; ++gid) {
        PoolDescriptor pd{};
        pd.pool_id      = gid;
        pd.layer_id     = -1;
        pd.region_name  = (static_cast<size_t>(gid) < cache_config.group_region_names.size())
                              ? cache_config.group_region_names[static_cast<size_t>(gid)]
                              : KVCacheRegionName::DEFAULT;
        pd.group_type   = groupTypeForConnector(cache_config, gid);
        pd.stride_bytes = 0;
        pd.last_partial = source_last_partial && pd.group_type == CacheGroupType::FULL;
        pools.push_back(pd);
        pool_block_counts.push_back(static_cast<size_t>(source.blocks(gid).size()));
        group_types.push_back(pd.group_type);
    }

    const size_t total_logical_blocks = source.cacheKeys().size();
    const auto   plan                  = buildUnifiedTransferPlan(/*model_id=*/0,
                                                                  total_logical_blocks,
                                                                  /*total_cache_keys=*/total_logical_blocks,
                                                                  /*reuse_block_size=*/0u,
                                                                  pools,
                                                                  pool_block_counts,
                                                                  /*shard_owner_rank=*/cp_size - 1,
                                                                  cp_size,
                                                                  /*use_hybrid=*/true,
                                                                  /*hybrid_full_from_begin=*/true,
                                                                  PlannerRole::kPrefillWrite);

    KVCacheResource selected = source;
    selected.initGroups(group_num,
                        static_cast<int>(cache_config.layer_all_num),
                        cache_config.layer_to_group_id,
                        cache_config.kernelBlocksPerKvBlock(),
                        group_types,
                        cache_config.layer_region_to_group_id);

    // -- Cache keys via controlled mutators (M01 §3.6 Fix 19 / M03 §4 Fix 76) --
    selected.clearCacheKeys();
    for (const auto& k : selected_keys) {
        selected.appendCacheKey(k);
    }
    // M04-PR4 atomic cutover (Risk 9.2): legacy ``makeCpShardedConnectorResource``
    // appended a dummy-tail cache_key here so the memory connector's
    // unconditional drop-last contract discarded the dummy rather than the
    // usable selected-last key.  Under the unified path the planner Step 2
    // performs the drop *per-pool* — FULL pools whose source.lastBlockAligned()
    // == false skip B+1 == total_cache_keys, SWA pools keep both tail blocks
    // (REQ-D3) — so the wire-level dummy-tail trick is no longer required.
    // ``lastBlockAlignedAll`` carries the source signal verbatim for receivers
    // that still walk the request-level bit (mixed-mode pairs during rollout).
    const bool selected_aligned = selectedLastRankKeysAreAligned(source, cp_size);
    selected.setLastBlockAlignedAll(selected_aligned);

    // -- Per-pool block lists from plan items (vLLM SupportsHMA per-pool view). --
    std::vector<BlockIndicesType> per_pool_dst(static_cast<size_t>(group_num));
    for (size_t p = 0; p < per_pool_dst.size(); ++p) {
        per_pool_dst[p].reserve(selected_keys.size());
    }
    for (const auto& it : plan) {
        const auto&        src_blocks = source.blocks(it.pool_id);
        const BlockIdxType v          = (it.offset_index >= 0 && static_cast<size_t>(it.offset_index) < src_blocks.size())
                                            ? src_blocks[static_cast<size_t>(it.offset_index)]
                                            : NULL_BLOCK_IDX;
        per_pool_dst[static_cast<size_t>(it.pool_id)].push_back(v);
    }
    for (int gid = 0; gid < group_num; ++gid) {
        selected.mutableBlockIds(gid).assign(std::move(per_pool_dst[static_cast<size_t>(gid)]));
    }

    return selected;
}

}  // namespace

KVCacheConnectorCoordinator::KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                                         const KVCacheConfig&                     kv_cache_config,
                                                         const RuntimeConfig&                     runtime_config,
                                                         const ParallelismConfig&                 parallelism_config,
                                                         const SpeculativeExecutionConfig&        sp_config,
                                                         const std::shared_ptr<KVCacheAllocator>& allocator,
                                                         const kmonitor::MetricsReporterPtr&      metrics_reporter,
                                                         const PDSepConfig&                       pd_sep_config,
                                                         const CacheStoreConfig&                  cache_store_config):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    runtime_config_(runtime_config),
    parallelism_config_(parallelism_config),
    sp_config_(sp_config),
    allocator_(allocator),
    metrics_reporter_(metrics_reporter),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config) {}

KVCacheConnectorCoordinator::~KVCacheConnectorCoordinator() {
    stop_.store(true);
    // release all connectors to make sure all async context done
    memory_connector_.reset();
    connectors_.clear();
    // connectors already released, all async context should be done
    autil::ScopedTime2 timer;
    while (true) {
        if (timer.done_ms() > update_interval_ms_ * 2) {
            RTP_LLM_LOG_WARNING(
                "coordinator destructor timeout, read or write list not empty, timeout: %d ms, read list size: %zu, write list size: %zu",
                timer.done_ms(),
                fused_async_read_context_list_.size(),
                fused_async_write_context_list_.size());
            break;
        }
        {
            std::lock_guard<std::mutex> lock(update_mutex_);
            if (fused_async_read_context_list_.empty() && fused_async_write_context_list_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (update_thread_) {
        update_thread_->stop();
        update_thread_.reset();
    }
}

bool KVCacheConnectorCoordinator::hasActiveConnectors() const {
    return !connectors_.empty();
}

bool KVCacheConnectorCoordinator::hasP2PConnector() const {
    return p2p_connector_ != nullptr;
}

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());

    // M04 PR-3: pre-compute the local handshake info from the cache config.
    // Legacy path keeps all fields at 0; unified path
    // (super_block_layout.enabled == true) populates the pinned-input
    // pool_descriptor_hash and the M03 salt schema.
    //
    // F01-PR2 part A: the salt is now built from
    // ``CacheConfig::state_entries_per_block_constant`` via
    // ``makeCacheKeySalt`` (single source of truth shared with the
    // cache_key producer in KVCacheManager).  When K_state == 0 the salt
    // is all-zero → bitmap=0 → hash_salt_version stays 0 → handshake is
    // byte-identical to today.  When K_state > 0 the bitmap acquires
    // bit3 and the schema version flips to ``kCacheKeySaltSchemaVersion``
    // so peers with different K_state values trip
    // ``validateHandshake``'s REQ-D1 check.
    //
    // The validator ``validatePeerHandshake`` is invoked by per-connector
    // peer-pairing code once the peer's HandshakeInfo arrives over the
    // wire.  That call site is still TODO (see F01-PR2-followup note
    // below) — but the actual cache_key XOR is wired through to
    // KVCacheManager in this PR so mixed-mode peers always produce
    // divergent cache_keys (Risk 9.6: silent-reuse-miss instead of
    // silent corruption) even if the handshake validation hook hasn't
    // landed yet.
    const CacheKeySalt local_salt    = makeCacheKeySalt(cache_config_);
    const uint32_t     salt_bitmap   = nonzeroFieldBitmap(local_salt);
    const uint32_t     salt_version  = (salt_bitmap != 0) ? kCacheKeySaltSchemaVersion : 0u;
    local_handshake_info_ = computeLocalHandshakeInfo(cache_config_, salt_version, salt_bitmap);
    RTP_LLM_LOG_INFO("PD pair local handshake info: %s (K_state=%d, salt.K_state=%u)",
                     local_handshake_info_.toString().c_str(),
                     cache_config_.state_entries_per_block_constant,
                     local_salt.K_state);

    // F01-PR2-followup wired (R4-24): the validator API is no longer a
    // dead-letter.  Connectors with a salt-aware wire (cache_store_service.
    // proto fields 103/104/105) should invoke
    // ``acceptPeerHandshakeFields(magic, version, bitmap)`` from their
    // per-request accept path — see CacheStoreServiceImpl::load for the
    // first call site.  Connectors whose wire does NOT yet carry the
    // salt triple fall back to the cache_key XOR safety net (divergent
    // salts ⇒ reuse-miss instead of silent corruption, Risk 9.6) and
    // never observe a mismatch.  The validator logs WARN + increments
    // ``pd.cache.salt_mismatch_skipped`` on mismatch — it does NOT
    // RTP_LLM_FAIL, so a mixed-mode misconfiguration is observable in
    // kmonitor but does not crash the engine.

    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
        memory_connector_ = initMemoryConnector();
        connectors_.emplace_back(memory_connector_);
    }
#ifdef USE_REMOTE_KV_CACHE
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_remote_cache) {
        remote_connector_ = initRemoteConnector();
        connectors_.emplace_back(remote_connector_);
    }
#endif
    if (!initP2PConnectorInternal()) {
        RTP_LLM_LOG_WARNING("init P2P connector failed, P2P path disabled — engine continues without it");
    }
    initUpdateThread();
    return true;
}

void KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ = autil::LoopThread::createLoopThread(
        [this]() { updateOnce(); }, update_interval_ms_ * 1000, "CoordinatorUpdateThread");
    RTP_LLM_CHECK_WITH_INFO(update_thread_ != nullptr, "init update thread failed");
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async read failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    // empty cache keys will not handled by coordinator.
    if (kvcache_resource.cacheKeys().empty()) {
        return nullptr;
    }

    const int       cp_size      = cpSize();
    CacheKeysType   ref_keys     = kvcache_resource.cacheKeys();
    KVCacheResource ref_resource = kvcache_resource;
    if (cp_size > 1) {
        ref_keys = kvcache_resource.localCacheKeys(cp_size - 1, cp_size);
        // Short requests (< cp_size logical blocks) have no complete virtual
        // block, so the canonical last-rank-key namespace is empty by design.
        // Skip silently — connector activity for these is a no-op anyway.
        if (ref_keys.empty()) {
            return nullptr;
        }
        // M04-PR2 gate: under the unified super-block layout, build the
        // per-pool resource via the block-major planner. Default (env=0,
        // super_block_layout.enabled=false) preserves legacy path verbatim.
        ref_resource = cache_config_.super_block_layout.enabled
                           ? makeUnifiedConnectorResource(kvcache_resource, cache_config_, ref_keys, cp_size)
                           : makeCpShardedConnectorResource(kvcache_resource, cache_config_, ref_keys, cp_size);
        ref_keys     = ref_resource.cacheKeys();
    }
    auto resource = allocator_->incrKVCacheRef(ref_resource, ref_keys, true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("async read failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> match_contexts(connectors_.size());
    for (int i = 0; i < connectors_.size(); i++) {
        match_contexts.at(i) = connectors_.at(i)->asyncMatch(resource, connector_context->meta());
    }

    auto fused_match_context = std::make_shared<FusedAsyncContext>(std::move(match_contexts));
    auto fused_read_context =
        std::make_shared<FusedAsyncReadContext>(fused_match_context, resource, connector_context->meta());
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_read_context_list_.push_back(fused_read_context);
    }
    return fused_read_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async write failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_DEBUG("async write failed, kvcache resource cache keys is empty, resource: [%s]",
                          kvcache_resource.debugString().c_str());
        return nullptr;
    }

    const int       cp_size      = cpSize();
    CacheKeysType   ref_keys     = kvcache_resource.cacheKeys();
    KVCacheResource ref_resource = kvcache_resource;
    if (cp_size > 1) {
        ref_keys = kvcache_resource.localCacheKeys(cp_size - 1, cp_size);
        if (ref_keys.empty()) {
            return nullptr;  // request shorter than one virtual block — nothing to write
        }
        // M04-PR2 gate: see asyncRead for rationale. Default path unchanged.
        ref_resource = cache_config_.super_block_layout.enabled
                           ? makeUnifiedConnectorResource(kvcache_resource, cache_config_, ref_keys, cp_size)
                           : makeCpShardedConnectorResource(kvcache_resource, cache_config_, ref_keys, cp_size);
        ref_keys     = ref_resource.cacheKeys();
    }
    auto resource = allocator_->incrKVCacheRef(ref_resource, ref_keys, true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("async write failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> write_contexts(connectors_.size());
    for (int i = 0; i < connectors_.size(); i++) {
        write_contexts.at(i) = connectors_.at(i)->asyncWrite(resource, connector_context->meta());
    }

    auto fused_write_context = std::make_shared<FusedAsyncContext>(std::move(write_contexts));
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_write_context_list_.push_back(fused_write_context);
    }
    return fused_write_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWriteByLayer(int                                                  layer_id,
                                               const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    if (!p2p_connector_) {
        return nullptr;
    }
    if (!layer_context) {
        RTP_LLM_LOG_WARNING("asyncWriteByLayer: layer_context is null, skip P2P write for layer %d", layer_id);
        return nullptr;
    }
    if (layer_id == 0) {
        RTP_LLM_LOG_INFO("asyncWriteByLayer [P2P]: dispatching layer_id=%d, request_id=%ld to P2PConnector",
                         layer_id,
                         layer_context->requestId());
    }
    return p2p_connector_->asyncWriteByLayer(layer_id, layer_context);
}

std::shared_ptr<KVCacheMemoryConnector> KVCacheConnectorCoordinator::initMemoryConnector() {
    auto memory_connector = std::make_shared<KVCacheMemoryConnector>(cache_config_,
                                                                     kv_cache_config_,
                                                                     parallelism_config_,
                                                                     allocator_,
                                                                     runtime_config_.worker_grpc_addrs,
                                                                     metrics_reporter_);
    RTP_LLM_CHECK_WITH_INFO(memory_connector->init(), "memory connector init failed");
    return memory_connector;
}

std::shared_ptr<RemoteConnector> KVCacheConnectorCoordinator::initRemoteConnector() {
#ifdef USE_REMOTE_KV_CACHE
    // TODO : get lora info map
    auto remote_connector_ = std::make_shared<RemoteConnector>(cache_config_,
                                                               kv_cache_config_,
                                                               runtime_config_,
                                                               parallelism_config_,
                                                               sp_config_,
                                                               allocator_->getBlockPool()->getBaseAddress(),
                                                               allocator_->getBlockPool()->getTotalSizeBytes(),
                                                               allocator_,
                                                               metrics_reporter_);

    RTP_LLM_CHECK_WITH_INFO(remote_connector_->init(), "remote connector init failed");
    return remote_connector_;
#else
    RTP_LLM_LOG_ERROR("not RemoteConnector");
    return nullptr;
#endif
}

int KVCacheConnectorCoordinator::cpSize() const {
    const auto& cp_cfg = parallelism_config_.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && parallelism_config_.tp_size > 1) {
        return static_cast<int>(parallelism_config_.tp_size);
    }
    return 1;
}

void KVCacheConnectorCoordinator::updateOnce() {
    RTP_LLM_PROFILE_FUNCTION();
    processReadContexts();
    processWriteContexts();
}

void KVCacheConnectorCoordinator::processReadContexts() {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_read_context_list_.begin(); it != fused_async_read_context_list_.end();) {
        auto fused_read_context = *it;
        if (fused_read_context->done()) {
            fused_read_context->notifyDone();
            it = fused_async_read_context_list_.erase(it);
            continue;
        }
        // 没有 done 但是有 read context, 或者 match context 还没 done, 或者 match 失败 (下一轮调度处理), 继续等待
        auto read_context  = fused_read_context->fusedReadContext();
        auto match_context = fused_read_context->fusedMatchContext();
        if (read_context || !match_context->done() || !match_context->success()) {
            it = std::next(it);
            continue;
        }
        // match success, start read
        asyncReadAfterMatch(fused_read_context);
        it = std::next(it);
    }
}

void KVCacheConnectorCoordinator::processWriteContexts() {
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_write_context_list_.begin(); it != fused_async_write_context_list_.end();) {
        auto fused_write_context = *it;
        if (fused_write_context->done()) {
            it = fused_async_write_context_list_.erase(it);
            continue;
        }
        it = std::next(it);
    }
}

// this function is called under lock
void KVCacheConnectorCoordinator::asyncReadAfterMatch(std::shared_ptr<FusedAsyncReadContext> fused_read_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto match_contexts = fused_read_context->fusedMatchContext()->contexts();
    RTP_LLM_CHECK_WITH_INFO(
        match_contexts.size() == connectors_.size(),
        "match contexts size is not equal to connectors size, match contexts size: [%d], connectors size: [%d]",
        match_contexts.size(),
        connectors_.size());

    int                                        already_reuse_num = fused_read_context->resource()->reuseBlockNum();
    std::vector<std::shared_ptr<AsyncContext>> connector_read_contexts;
    for (int i = 0; i < match_contexts.size(); i++) {
        auto match_context = std::dynamic_pointer_cast<AsyncMatchContext>(match_contexts.at(i));
        if (!match_context) {
            continue;
        }
        const auto matched_num = match_context->matchedBlockCount();
        if (matched_num <= already_reuse_num) {
            continue;
        }
        auto connector_read_context = connectors_.at(i)->asyncRead(fused_read_context->resource(),
                                                                   fused_read_context->meta(),
                                                                   match_context,
                                                                   already_reuse_num,
                                                                   matched_num - already_reuse_num);
        if (connector_read_context) {
            connector_read_contexts.emplace_back(connector_read_context);
            already_reuse_num = matched_num;
        }
    }
    fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
}

void KVCacheConnectorCoordinator::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                             P2PConnectorStartLoadResponsePB&      response,
                                             std::function<bool()>                 is_cancelled) {
    if (!p2p_connector_) {
        RTP_LLM_LOG_WARNING("handleRead called but P2P connector not initialized");
        return;
    }
    p2p_connector_->handleRead(request, response, std::move(is_cancelled));
}

bool KVCacheConnectorCoordinator::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_mem_request()) {
        RTP_LLM_CHECK(memory_connector_ != nullptr);
        return memory_connector_->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else if (request.has_remote_request()) {
#ifdef USE_REMOTE_KV_CACHE
        RTP_LLM_CHECK(remote_connector_ != nullptr);
        return remote_connector_->copyCache(request.remote_request(), *(response.mutable_remote_response()));
#endif
        RTP_LLM_CHECK(false);
        return false;
    } else if (request.has_p2p_request()) {
        if (!p2p_connector_) {
            RTP_LLM_LOG_WARNING("executeFunction: p2p_request received but P2P connector not initialized");
            return false;
        }
        return p2p_connector_->executeFunction(request, response);
    } else {
        RTP_LLM_LOG_WARNING("execute function failed, request is invalid, request: [%s]",
                            request.DebugString().c_str());
        return false;
    }
}

bool KVCacheConnectorCoordinator::isPdInvertMode() const {
    return (pd_sep_config_.role_type == RoleType::PREFILL || pd_sep_config_.role_type == RoleType::DECODE)
           && pd_sep_config_.decode_entrance;
}

bool KVCacheConnectorCoordinator::initP2PConnectorInternal() {
    // TODO: P2P connector initialization is disabled until the next PR enables
    // scheduler async load cache support. Change to `#if 1` to activate.
#if 0
    if (!isPdInvertMode()) {
        return true;
    }
    const uint32_t layer_all_num         = static_cast<uint32_t>(cache_config_.layer_all_num);
    auto           layer_block_converter = std::make_shared<LayerBlockConverterImpl>(allocator_);

    auto p2p_config = P2PConnectorConfig::create(
        runtime_config_, cache_store_config_, parallelism_config_, pd_sep_config_, layer_all_num);
    auto p2p = std::make_shared<P2PConnector>(std::move(p2p_config), layer_block_converter, metrics_reporter_);
    if (!p2p->init()) {
        RTP_LLM_LOG_ERROR("P2PConnector init failed");
        p2p.reset();  // 显式释放，避免半初始化状态的 P2PConnector 意外使用
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        p2p_connector_ = std::move(p2p);
        connectors_.emplace_back(p2p_connector_);
    }
    RTP_LLM_LOG_INFO("P2PConnector initialized successfully, total connectors: %zu", connectors_.size());
#endif
    return true;
}

bool KVCacheConnectorCoordinator::validatePeerHandshake(const HandshakeInfo& peer_info) const {
    // F01-PR2-followup: returns bool (was void+FAIL).  A mixed-mode peer
    // is a misconfiguration, not a crash-worthy invariant violation —
    // RTP_LLM_FAIL would have brought down the engine on a single
    // misbehaving wire-peer.  Instead we WARN, bump
    // ``pd.cache.salt_mismatch_skipped``, and return false so the caller
    // can downgrade the pairing to legacy-only OR drop the peer
    // depending on connector semantics.  The cache_key salt XOR
    // (KVCacheHashUtil) already prevents silent corruption at the
    // request layer (Risk 9.6); this counter surfaces the misconfig.
    std::string err;
    if (validateHandshake(local_handshake_info_, peer_info, &err)) {
        return true;
    }
    RTP_LLM_LOG_WARNING("PD handshake mismatch — pairing falls back to legacy-only: %s", err.c_str());
    recordPdSaltMismatchSkipped(metrics_reporter_);
    return false;
}

bool KVCacheConnectorCoordinator::acceptPeerHandshakeFields(uint32_t peer_salt_magic,
                                                            uint32_t peer_hash_salt_version,
                                                            uint32_t peer_hash_salt_nonzero_bitmap) const {
    // FIX-B HIGH-5 (DEFEND-4 #5, production-fatal): the wire's ``salt_magic``
    // proto field (tag 103) is a fixed sentinel ({0,100}) used only for
    // forward-compat envelope detection.  It is NOT the same semantic as
    // ``HandshakeInfo::protocol_magic`` ({0,1} = legacy / unified-aware).
    // Pre-fix, the proto field defaulted to 100 and a naive route into
    // HandshakeInfo::protocol_magic produced ``HandshakeInfo{magic=100,...}``
    // which falls through ``validateHandshake``'s unified↔unified branch and
    // REFUSES every legacy peer (zero hash/version mismatch) → fleet-wide
    // REFUSE storm at every PD pairing.  The proto rename + default=0 below
    // means a legacy peer is now wire-identifiable as ``salt_magic == 0``;
    // the real "is the peer's cache_key payload salted" gate uses
    // ``hash_salt_version > 0``, NOT salt_magic.
    //
    // CHECK invariant (caller-bug catcher): if any non-salt-aware caller
    // wires a stray byte into salt_magic, it must still be in the
    // forward-compat sentinel set {0, 100}.  We don't FAIL (mixed-mode is
    // a misconfiguration, not a fatal crash — same policy as
    // validatePeerHandshake), but we log + downgrade to legacy semantics
    // so we never accidentally inject the sentinel into the legacy
    // protocol_magic field.  Note: bare ``DCHECK`` (release-stripped) is
    // not enough — use a warn-once so prod sees it.
    if (peer_salt_magic != 0 && peer_salt_magic != 100) {
        RTP_LLM_LOG_WARNING(
            "PD handshake: peer advertised unexpected salt_magic=%u (expected {0,100}); "
            "downgrading peer to legacy semantics for envelope detection",
            peer_salt_magic);
    }

    // Map the proto sentinel onto the {0,1} HandshakeInfo::protocol_magic
    // domain.  REQ: salt_magic == 0 (proto default = field NOT sent by
    // legacy peer) MUST be treated as legacy-OK, NOT mixed-mode REFUSE.
    HandshakeInfo peer{};
    peer.pool_descriptor_hash     = 0;  // not yet on the wire (deferred to PR adding field 106)
    peer.hash_salt_version        = peer_hash_salt_version;
    peer.hash_salt_nonzero_bitmap = peer_hash_salt_nonzero_bitmap;

    // FIX-B HIGH-5: the actual "peer is salt-aware (unified-mode hash domain)"
    // gate is ``hash_salt_version > 0`` — only then advertise unified-aware
    // protocol_magic=1.  A legacy peer (no field 103, defaults to 0) AND a
    // salt-aware peer that happens to send {salt_magic=100, version=0}
    // (e.g. a unified-aware sender on a fresh boot with K_state=0) both
    // route to the legacy↔legacy branch, which is byte-identical to
    // today's behaviour.
    peer.protocol_magic = (peer_hash_salt_version > 0) ? 1u : 0u;
    return validatePeerHandshake(peer);
}

std::vector<CacheKeyType> KVCacheConnectorCoordinator::memoryCacheKeys() const {
    if (!memory_connector_) {
        return {};
    }
    return memory_connector_->cacheKeys();
}

}  // namespace rtp_llm
