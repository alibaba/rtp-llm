#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool isCanonicalEvictionTarget(Tier source_tier, Tier target_tier) {
    switch (source_tier) {
        case Tier::DEVICE:
            return target_tier == Tier::HOST || target_tier == Tier::NONE;
        case Tier::HOST:
            return target_tier == Tier::DISK || target_tier == Tier::NONE;
        case Tier::DISK:
            return target_tier == Tier::NONE;
        default:
            return false;
    }
}

}  // namespace

EvictionTaskRunner::EvictionTaskRunner(ExecuteTransferFn               execute_transfer,
                                       const std::vector<GroupSetPtr>& group_sets,
                                       const BlockTransferDispatcher*  transfer_dispatcher,
                                       BlockTreeTaskPool*              task_pool,
                                       BlockTreeCacheMetricsReporter&  metrics_reporter,
                                       std::mutex&                     mutex,
                                       int                             memory_timeout_ms,
                                       int                             disk_timeout_ms,
                                       IsTierEnabledFn                 is_tier_enabled,
                                       SettledFn                       settled,
                                       RemoteWriteFn                   remote_write):
    execute_transfer_(std::move(execute_transfer)),
    group_sets_(group_sets),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(&metrics_reporter),
    mutex_(&mutex),
    memory_timeout_ms_(memory_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    is_tier_enabled_(std::move(is_tier_enabled)),
    settled_(std::move(settled)),
    remote_write_(std::move(remote_write)) {}

bool EvictionTaskRunner::submitLocked(BlockTreeEvictor& evictor, TransferDescriptor& eviction_desc) {
    if (!isCanonicalEvictionTarget(eviction_desc.source_tier, eviction_desc.target_tier)) {
        RTP_LLM_LOG_WARNING("rejecting eviction move with non-canonical target: source=%s target=%s group_set=%zu",
                            tierName(eviction_desc.source_tier),
                            tierName(eviction_desc.target_tier),
                            eviction_desc.group_set_id);
        return false;
    }
    if (eviction_desc.target_tier != Tier::NONE) {
        eviction_desc.target_tier = normalizeTargetTier(eviction_desc.source_tier);
    }

    auto plan = evictor.buildPlan(eviction_desc);
    if (!plan.has_value()) {
        return false;
    }

    if (!plan->needsCopy()) {
        BlockTreeEvictor::CopyResultSet results;
        results.primary_success = true;
        results.cascade_success.assign(plan->cascade_descs.size(), true);
        evictor.complete(*plan, results);
        metrics_reporter_->reportEvictionFinished(*plan, results, group_sets_);
        settled_(true, false);
        return true;
    }

    auto plan_ptr = std::make_shared<BlockTreeEvictor::EvictionPlan>(std::move(*plan));
    evictor.reservePendingReleases(*plan_ptr);
    const bool submitted = task_pool_->submit([this, &evictor, plan_ptr]() { runTask(evictor, *plan_ptr); });
    if (!submitted) {
        evictor.settlePendingReleases(*plan_ptr);
        evictor.rollbackPreparedPlan(*plan_ptr);
        return false;
    }
    return true;
}

void EvictionTaskRunner::runTask(BlockTreeEvictor& evictor, const BlockTreeEvictor::EvictionPlan& plan) {
    const Tier    source_tier          = plan.primary_desc.source_tier;
    const Tier    target_tier          = plan.primary_desc.target_tier;
    const size_t  transfer_block_count = plan.cascade_descs.size() + 1;
    const int64_t transfer_begin_time_us =
        metrics_reporter_->reportTransferStarted(CacheTransferOperation::EVICT, source_tier, target_tier);
    BlockTreeEvictor::CopyResultSet copy_results;
    copy_results.primary_success = false;
    copy_results.cascade_success.assign(plan.cascade_descs.size(), false);

    auto finalization_action = [this,
                                &evictor,
                                &plan,
                                &copy_results,
                                source_tier,
                                target_tier,
                                transfer_block_count,
                                transfer_begin_time_us]() noexcept {
        const bool transfer_success = copy_results.primary_success
                                      && std::all_of(copy_results.cascade_success.begin(),
                                                     copy_results.cascade_success.end(),
                                                     [](bool success) { return success; });
        BlockTreeTransferBytes transfer_bytes;
        if (copy_results.primary_success) {
            metrics_reporter_->accumulateTransferBytes(
                plan.primary_desc, group_sets_[plan.primary_desc.group_set_id], transfer_bytes);
        }
        for (size_t desc_index = 0; desc_index < plan.cascade_descs.size(); ++desc_index) {
            if (copy_results.cascade_success[desc_index]) {
                const TransferDescriptor& desc = plan.cascade_descs[desc_index];
                metrics_reporter_->accumulateTransferBytes(desc, group_sets_[desc.group_set_id], transfer_bytes);
            }
        }
        metrics_reporter_->reportTransferFinished(CacheTransferOperation::EVICT,
                                                  source_tier,
                                                  target_tier,
                                                  transfer_block_count,
                                                  transfer_begin_time_us,
                                                  transfer_success,
                                                  transfer_bytes);

        auto pending_release_settlement_action = [&evictor, &plan]() noexcept {
            evictor.settlePendingReleases(plan);
        };
        block_tree_cache_detail::ScopeRollback<decltype(pending_release_settlement_action)>
            pending_release_settlement_guard(std::move(pending_release_settlement_action));

        bool       completion_succeeded = false;
        bool       plan_terminalized    = false;
        bool       plan_succeeded       = false;
        const bool copy_ok              = copy_results.primary_success;

        CacheKeyType          remote_cache_key = 0;
        std::optional<size_t> remote_group_set_id;
        if (copy_ok && plan.primary_desc.node != nullptr) {
            remote_cache_key    = plan.primary_desc.node->cache_key;
            remote_group_set_id = plan.primary_desc.group_set_id;
        }

        try {
            std::lock_guard<std::mutex> lock(*mutex_);
            try {
                evictor.complete(plan, copy_results);
                completion_succeeded = true;
                plan_terminalized    = true;
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("eviction completion failed; rolling back accepted plan: %s", error.what());
                evictor.rollbackPreparedPlan(plan);
                plan_terminalized = true;
            } catch (...) {
                RTP_LLM_LOG_ERROR("eviction completion failed with unknown exception; rolling back "
                                  "accepted plan");
                evictor.rollbackPreparedPlan(plan);
                plan_terminalized = true;
            }

            pending_release_settlement_guard.run();

            const bool mutated = plan_terminalized && completion_succeeded
                                 && (copy_results.primary_success
                                     || std::any_of(copy_results.cascade_success.begin(),
                                                    copy_results.cascade_success.end(),
                                                    [](bool success) { return success; }));
            plan_succeeded = plan_terminalized && completion_succeeded && copy_results.primary_success
                             && copy_results.cascade_success.size() == plan.cascade_descs.size()
                             && std::all_of(copy_results.cascade_success.begin(),
                                            copy_results.cascade_success.end(),
                                            [](bool success) { return success; });
            settled_(mutated, plan_succeeded);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("eviction terminalization lock/follow-up failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("eviction terminalization lock/follow-up failed with unknown exception");
        }
        metrics_reporter_->reportEvictionFinished(plan, copy_results, group_sets_);

        if (plan_terminalized && completion_succeeded && copy_ok && remote_group_set_id.has_value()) {
            try {
                remote_write_(remote_cache_key, *remote_group_set_id);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("remote eviction write-through failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("remote eviction write-through failed with unknown exception");
            }
        }
    };
    block_tree_cache_detail::ScopeRollback<decltype(finalization_action)> finalization_guard(
        std::move(finalization_action));

    try {
        copy_results = runTransfer(plan);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("eviction copy failed with unknown exception");
    }
}

BlockTreeEvictor::CopyResultSet EvictionTaskRunner::performCopy(const BlockTreeEvictor::EvictionPlan& plan) const {
    BlockTreeEvictor::CopyResultSet results;
    results.primary_success = true;

    if (plan.primary_desc.target_tier != Tier::NONE) {
        results.primary_success = executeTierCopy(plan.primary_desc);
        if (!results.primary_success) {
            RTP_LLM_LOG_WARNING("primary copy FAILED "
                                "group_set[%zu] node_key=%ld %s->%s",
                                plan.primary_desc.group_set_id,
                                plan.primary_desc.node ? plan.primary_desc.node->cache_key : 0,
                                tierName(plan.primary_desc.source_tier),
                                tierName(plan.primary_desc.target_tier));
            results.cascade_success.assign(plan.cascade_descs.size(), false);
            return results;
        }
        RTP_LLM_LOG_DEBUG("primary copy OK "
                          "group_set[%zu] node_key=%ld %s->%s",
                          plan.primary_desc.group_set_id,
                          plan.primary_desc.node ? plan.primary_desc.node->cache_key : 0,
                          tierName(plan.primary_desc.source_tier),
                          tierName(plan.primary_desc.target_tier));
    }

    results.cascade_success.reserve(plan.cascade_descs.size());
    for (const auto& cascade_desc : plan.cascade_descs) {
        bool copy_ok = true;
        if (cascade_desc.target_tier != Tier::NONE) {
            copy_ok = executeTierCopy(cascade_desc);
        }
        results.cascade_success.push_back(copy_ok);

        if (!copy_ok) {
            RTP_LLM_LOG_WARNING("cascade copy FAILED "
                                "group_set[%zu] node_key=%ld %s->%s",
                                cascade_desc.group_set_id,
                                cascade_desc.node ? cascade_desc.node->cache_key : 0,
                                tierName(cascade_desc.source_tier),
                                tierName(cascade_desc.target_tier));
        } else if (cascade_desc.target_tier != Tier::NONE) {
            RTP_LLM_LOG_DEBUG("cascade copy OK "
                              "group_set[%zu] node_key=%ld %s->%s",
                              cascade_desc.group_set_id,
                              cascade_desc.node ? cascade_desc.node->cache_key : 0,
                              tierName(cascade_desc.source_tier),
                              tierName(cascade_desc.target_tier));
        }
    }
    return results;
}

BlockTreeEvictor::CopyResultSet EvictionTaskRunner::runTransfer(const BlockTreeEvictor::EvictionPlan& plan) const {
    if (!transfer_dispatcher_->hasMultiRankEngine()) {
        return performCopy(plan);
    }

    BlockTreeEvictor::CopyResultSet results;
    std::vector<TransferDescriptor> descriptors;
    const bool                      batch_ready      = buildTransferBatch(plan, descriptors);
    const bool                      transfer_success = batch_ready
                                  && transfer_dispatcher_->executeMultiRank(
                                      descriptors, transferTimeoutMs(plan, memory_timeout_ms_, disk_timeout_ms_));
    results.primary_success = transfer_success;
    results.cascade_success.assign(plan.cascade_descs.size(), transfer_success);
    return results;
}

bool EvictionTaskRunner::executeTierCopy(const TransferDescriptor& eviction_desc) const {
    if (!execute_transfer_) {
        return false;
    }

    if (!eviction_desc.isExecutable()) {
        return false;
    }
    return execute_transfer_(eviction_desc);
}

Tier EvictionTaskRunner::normalizeTargetTier(Tier source_tier) const {
    switch (source_tier) {
        case Tier::DEVICE:
            if (is_tier_enabled_(Tier::HOST)) {
                return Tier::HOST;
            }
            if (is_tier_enabled_(Tier::DISK)) {
                return Tier::DISK;
            }
            return Tier::NONE;
        case Tier::HOST:
            if (is_tier_enabled_(Tier::DISK)) {
                return Tier::DISK;
            }
            return Tier::NONE;
        default:
            return Tier::NONE;
    }
}

bool EvictionTaskRunner::buildTransferBatch(const BlockTreeEvictor::EvictionPlan& plan,
                                            std::vector<TransferDescriptor>&      descriptors) {
    descriptors.clear();
    descriptors.reserve(1 + plan.cascade_descs.size());

    if (!plan.primary_desc.isExecutable()) {
        return false;
    }
    descriptors.push_back(plan.primary_desc);

    for (const TransferDescriptor& cascade_desc : plan.cascade_descs) {
        if (!cascade_desc.isExecutable()) {
            descriptors.clear();
            return false;
        }
        descriptors.push_back(cascade_desc);
    }
    return true;
}

int EvictionTaskRunner::transferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan,
                                          int                                   memory_timeout_ms,
                                          int                                   disk_timeout_ms) {
    bool uses_disk = plan.primary_desc.source_tier == Tier::DISK || plan.primary_desc.target_tier == Tier::DISK;
    for (const TransferDescriptor& cascade_desc : plan.cascade_descs) {
        if (cascade_desc.source_tier == Tier::DISK || cascade_desc.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    return uses_disk ? disk_timeout_ms : memory_timeout_ms;
}

}  // namespace rtp_llm
