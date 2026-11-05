#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <algorithm>
#include <chrono>
#include <cassert>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

std::string jsonEscape(const std::string& value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(ch) << std::dec;
                } else {
                    escaped << static_cast<char>(ch);
                }
        }
    }
    escaped << '"';
    return escaped.str();
}

void logPrefixPlanFormed(const std::string&                       instance,
                         const std::vector<PrefixCachePlanPoint>& points,
                         Tier                                      target_tier,
                         std::optional<int64_t> observation_request_id) {
    std::ostringstream payload;
    payload << "{\"kind\":\"plan\",\"schema_version\":1,\"instance\":" << jsonEscape(instance)
            << ",\"state\":\"formed\","
               "\"target_tier\":\""
            << tierName(target_tier) << "\",\"formed_time_us\":" << currentTimeUs() << ",\"points\":[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i != 0) {
            payload << ',';
        }
        payload << "{\"key\":\"" << points[i].key << "\",\"pos\":" << points[i].pos << '}';
    }
    payload << "],\"request_id\":" << (observation_request_id ? std::to_string(*observation_request_id) : "null") << '}';
    RTP_LLM_LOG_INFO("event=prefix_cache_observation payload=%s", payload.str().c_str());
}

void logPrefixPlanUnfulfilled(const std::string&                       instance,
                              const std::vector<PrefixCachePlanPoint>& points,
                              const char*                              reason,
                              const ErrorInfo&                        error,
                              std::optional<int64_t> observation_request_id) {
    for (const PrefixCachePlanPoint& point : points) {
        std::ostringstream payload;
        payload << "{\"kind\":\"plan_unfulfilled\",\"schema_version\":1,\"instance\":"
                << jsonEscape(instance) << ",\"key\":\"" << point.key
                << "\",\"pos\":" << point.pos << ",\"reason\":\"" << reason << "\",\"error_code\":"
                << static_cast<int>(error.code()) << ",\"error_message\":" << jsonEscape(error.ToString())
                << ",\"request_id\":" << (observation_request_id ? std::to_string(*observation_request_id) : "null") << '}';
        RTP_LLM_LOG_WARNING("event=prefix_cache_observation payload=%s", payload.str().c_str());
    }
}

// A settlement event reports tree mutation facts, never request token reuse.
void logPrefixPublication(const std::string& instance,
                          std::optional<int64_t> request_id,
                          Tier tier,
                          const BlockTreeInsertResult* result,
                          const char* reason,
                          size_t unfulfilled_count,
                          std::optional<size_t> internal_fork_pos = std::nullopt,
                          std::optional<bool> internal_fork_ready = std::nullopt) {
    std::ostringstream payload;
    payload << "{\"kind\":\"publication\",\"schema_version\":1,\"instance\":" << jsonEscape(instance)
            << ",\"request_id\":" << (request_id ? std::to_string(*request_id) : "null")
            << ",\"settled_time_us\":" << currentTimeUs() << ",\"target_tier\":" << jsonEscape(tierName(tier))
            << ",\"insert_attempted\":" << (result ? "true" : "false")
            << ",\"inserted_nodes\":" << (result ? result->inserted_nodes.size() : 0)
            << ",\"adopted_nodes\":" << (result ? result->adopted_nodes.size() : 0)
            << ",\"accepted_resources\":" << (result ? result->accepted_resource_count : 0)
            << ",\"unfulfilled_plan_points\":" << unfulfilled_count
            << ",\"internal_fork\":" << (internal_fork_pos ? "true" : "false")
            << ",\"internal_fork_pos\":";
    if (internal_fork_pos) {
        payload << *internal_fork_pos;
    } else {
        payload << "null";
    }
    payload << ",\"internal_fork_ready\":";
    if (internal_fork_ready) {
        payload << (*internal_fork_ready ? "true" : "false");
    } else {
        payload << "null";
    }
    payload << ",\"reason\":";
    if (reason) {
        payload << jsonEscape(reason);
    } else {
        payload << "null";
    }
    payload << '}';
    RTP_LLM_LOG_INFO("event=prefix_cache_observation payload=%s", payload.str().c_str());
}

}  // namespace

BlockTreeStorer::BlockTreeStorer(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 BlockTransferDispatcher*        transfer_dispatcher,
                                 BlockTreeTaskPool*              task_pool,
                                 BlockTreeCacheMetricsReporter&  metrics_reporter,
                                 std::mutex&                     mutex,
                                 int                             host_timeout_ms,
                                 int                             disk_timeout_ms,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled):
    tree_(tree),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    store_task_runner_(tree_->groupSets()),
    mutex_(mutex),
    host_timeout_ms_(host_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)) {}

void BlockTreeStorer::observePublicationSkipped(std::optional<int64_t> request_id,
                                                Tier tier,
                                                const char* reason,
                                                std::optional<size_t> internal_fork_pos) {
    if (tree_->observabilityEnabled()) {
        logPrefixPublication(tree_->observabilityInstance(), request_id, tier, nullptr, reason, 0, internal_fork_pos);
    }
}

void BlockTreeStorer::stopAdmissionLocked() {
    stopping_.store(true);
}

StorageWriteTask BlockTreeStorer::storeLocked(const CacheKeysType&                              cache_keys,
                                              const std::vector<std::vector<GroupSetResource>>& resources,
                                              Tier                                              target_tier,
                                              bool                                              is_resident,
                                              size_t& resident_prefix_length,
                                              const std::vector<std::vector<int>>* token_segments,
                                              PrefixCachePlanState* prefix_plan_state,
                                              const std::vector<size_t>* token_ends,
                                              std::optional<int64_t> observation_request_id,
                                              bool internal_fork,
                                              bool* ready_after_insert) {
    resident_prefix_length = 0;
    assert(!is_resident || target_tier == Tier::DEVICE);
    RTP_LLM_CHECK_WITH_INFO(target_tier == Tier::DEVICE || target_tier == Tier::HOST || target_tier == Tier::DISK,
                            "unsupported store target tier: %s",
                            tierName(target_tier));
    if (target_tier == Tier::DEVICE) {
        publishDeviceLocked(cache_keys,
                            resources,
                            is_resident,
                            resident_prefix_length,
                            token_segments,
                            prefix_plan_state,
                            token_ends,
                            observation_request_id,
                            internal_fork,
                            ready_after_insert);
    } else {
        submitLowerTierLocked(cache_keys, resources, target_tier, token_segments, prefix_plan_state, observation_request_id);
        return {};
    }
    return storage_backend_ && !internal_fork ?
               storage_backend_->prepareWrite(makeStorageRequest(cache_keys, resources, token_ends)) :
                              StorageWriteTask{};
}

void BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                          const std::vector<std::vector<GroupSetResource>>& resources,
                                          bool                                              is_resident,
                                          size_t&                                           resident_prefix_length,
                                          const std::vector<std::vector<int>>*              token_segments,
                                          PrefixCachePlanState*                             prefix_plan_state,
                                          const std::vector<size_t>*                         token_ends,
                                          std::optional<int64_t> observation_request_id,
                                          bool internal_fork,
                                          bool* ready_after_insert) {
    const bool plan_reached = tree_->observabilityEnabled() && !cache_keys.empty()
                              && resources.size() == cache_keys.size();
    std::vector<PrefixCachePlanPoint> plan_points;
    if (plan_reached) {
        plan_points = tree_->planRecoveryPoints(cache_keys, resources, token_segments, Tier::DEVICE, token_ends);
        if (prefix_plan_state != nullptr) {
            *prefix_plan_state = PrefixCachePlanState::FORMED;
        }
    }
    if (tree_->observabilityEnabled() && plan_reached) {
        logPrefixPlanFormed(tree_->observabilityInstance(), plan_points, Tier::DEVICE, observation_request_id);
    }
    const BlockTreeInsertResult insert_result =
        tree_->insertNode(cache_keys, resources, false, is_resident, token_segments, token_ends);
    std::optional<bool> fork_ready;
    if (internal_fork) {
        const auto path = tree_->findNode(cache_keys);
        fork_ready = token_ends != nullptr && path.size() == cache_keys.size()
                     && path.back()->token_end == token_ends->back()
                     && std::all_of(tree_->groupSets().begin(), tree_->groupSets().end(), [&](const auto& group) {
                            return group->hasDeviceCoverage(path, token_ends->back());
                        });
        if (ready_after_insert != nullptr) {
            *ready_after_insert = *fork_ready;
        }
    }
    const auto unfulfilled = tree_->unfulfilledPlanPoints(plan_points, cache_keys);
    logPrefixPlanUnfulfilled(tree_->observabilityInstance(),
                             unfulfilled,
                             stopping_.load() ? "store_stopped" : "insert_not_fulfilled",
                             ErrorInfo::OkStatus(), observation_request_id);
    if (tree_->observabilityEnabled()) {
        const char* publication_reason = unfulfilled.empty() ? nullptr : "insert_not_fulfilled";
        if (internal_fork && fork_ready.has_value() && !*fork_ready) {
            publication_reason = "internal_fork_insert_incomplete";
        }
        logPrefixPublication(tree_->observabilityInstance(), observation_request_id, Tier::DEVICE,
                             &insert_result, publication_reason, unfulfilled.size(),
                             internal_fork ? std::optional<size_t>(token_ends->back()) : std::nullopt, fork_ready);
    }
    const bool tree_data_mutated = !insert_result.inserted_nodes.empty() || !insert_result.adopted_nodes.empty();
    if (tree_data_mutated || !insert_result.newly_resident_nodes.empty()) {
        evictor_.onInserted(insert_result);
    }
    if (tree_data_mutated) {
        settled_(true);
    }
    resident_prefix_length = insert_result.resident_prefix_length;
}

StorageRequest BlockTreeStorer::makeStorageRequest(const CacheKeysType&                              cache_keys,
                                                   const std::vector<std::vector<GroupSetResource>>& resources,
                                                   const std::vector<size_t>* token_ends) const {
    auto keys = std::make_shared<CacheKeysType>();
    StorageRequest request{keys, {}};
    for (size_t key_index = 0; key_index < resources.size(); ++key_index) {
        // The existing backend protocol only names complete cumulative keys.
        if (token_ends != nullptr && (*token_ends)[key_index] % tree_->reuseUnitTokens() != 0) {
            continue;
        }
        keys->push_back(cache_keys[key_index]);
        auto& key_handles = request.handles.emplace_back();
        for (size_t group_set = 0; group_set < tree_->groupSets().size(); ++group_set) {
            if (tree_->groupSets()[group_set]->localOnly()) {
                continue;
            }
            const auto& resource = resources[key_index][group_set];
            if (!resource.hasCompleteDeviceValue()) {
                continue;
            }
            const auto& group = *tree_->groupSets()[group_set];
            for (size_t member = 0; member < group.groupTags().size(); ++member) {
                key_handles.push_back({group.groupTags()[member], resource.device_blocks[member]});
            }
        }
    }
    return request;
}

void BlockTreeStorer::submitLowerTierLocked(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            Tier                                              target_tier,
                                            const std::vector<std::vector<int>>*              token_segments,
                                            PrefixCachePlanState*                             prefix_plan_state,
                                            std::optional<int64_t> observation_request_id) {
    const bool plan_reached = tree_->observabilityEnabled() && !cache_keys.empty()
                              && resources.size() == cache_keys.size();
    std::vector<PrefixCachePlanPoint> plan_points;
    if (plan_reached) {
        plan_points = tree_->planRecoveryPoints(cache_keys, resources, token_segments, target_tier);
        if (prefix_plan_state != nullptr) {
            *prefix_plan_state = PrefixCachePlanState::FORMED;
        }
    }
    if (tree_->observabilityEnabled() && plan_reached) {
        logPrefixPlanFormed(tree_->observabilityInstance(), plan_points, target_tier, observation_request_id);
    }
    if (stopping_.load()) {
        logPrefixPlanUnfulfilled(tree_->observabilityInstance(), plan_points,
                                 "store_stopped",
                                 ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store stopped before admission"), observation_request_id);
        if (tree_->observabilityEnabled()) {
            logPrefixPublication(tree_->observabilityInstance(), observation_request_id, target_tier,
                                 nullptr, "store_stopped", plan_points.size());
        }
        return;
    }

    StoreTaskPtr task = std::make_shared<StoreTask>(
        target_tier,
        cache_keys,
        std::chrono::milliseconds(target_tier == Tier::DISK ? disk_timeout_ms_ : host_timeout_ms_),
        token_segments ? *token_segments : std::vector<std::vector<int>>{},
        plan_points);
    task->observation_request_id = observation_request_id;

    const char* failure_reason = "prepare_failed";
    block_tree_cache_detail::ScopeRollback prepare_guard(
        [this, &task, &failure_reason]() { settleLocked(*task, /*publish=*/false, failure_reason); });

    if (!store_task_runner_.prepareTask(*task, resources)) {
        task->error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store task preparation failed");
        return;
    }
    const int64_t queue_begin = currentTimeUs();
    auto          on_timeout  = [this, task, queue_begin]() {
        metrics_reporter_.reportQueueWaitMetric(false,
                                                "business",
                                                cacheTransferOperationName(CacheTransferOperation::STORE),
                                                Tier::NONE,
                                                Tier::NONE,
                                                currentTimeUs() - queue_begin);
        RTP_LLM_LOG_WARNING("store expired in business queue, target=%s blocks=%zu",
                            tierName(task->target_tier),
                            task->descriptors().size());
        scheduleStoreSettlement(task,
                                ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "store business queue timeout"),
                                "store_timeout");
    };
    const auto deadline = task->transfer_task.deadline();
    failure_reason = "queue_rejected";
    if (!task_pool_->submit(
            BlockTreeTaskClass::BACKGROUND,
            [this, task, queue_begin]() {
                metrics_reporter_.reportQueueWaitMetric(false,
                                                        "business",
                                                        cacheTransferOperationName(CacheTransferOperation::STORE),
                                                        Tier::NONE,
                                                        Tier::NONE,
                                                        currentTimeUs() - queue_begin);
                runStoreTask(task);
            },
            deadline,
            std::move(on_timeout))) {
        RTP_LLM_LOG_WARNING("store aborted: business task submission rejected, target=%s blocks=%zu",
                            tierName(target_tier),
                            task->descriptors().size());
        task->error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store business task submission rejected");
        return;
    }
    prepare_guard.dismiss();
}

void BlockTreeStorer::runStoreTask(const StoreTaskPtr& task) {
    if (stopping_.load()) {
        scheduleStoreSettlement(task,
                                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store stopped before transfer"),
                                "store_stopped");
        return;
    }

    try {
        store_task_runner_.runTransfer(task, *transfer_dispatcher_, metrics_reporter_, [this, task](ErrorInfo error) {
            scheduleStoreSettlement(task, std::move(error), "copy_failed");
        });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("store copy threw: %s", error.what());
        scheduleStoreSettlement(task, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()), "copy_failed");
    } catch (...) {
        RTP_LLM_LOG_ERROR("store copy threw an unknown exception");
        scheduleStoreSettlement(task,
                                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store copy exception"),
                                "copy_failed");
    }
}

void BlockTreeStorer::scheduleStoreSettlement(const StoreTaskPtr& task,
                                             ErrorInfo            error,
                                             const char*          failure_reason) {
    if (!task->markFinished()) {
        return;
    }
    task->error = error;
    auto        settle      = [this, task, error = std::move(error), failure_reason]() mutable {
        settleTask(*task, error.ok(), failure_reason);
    };
    const int64_t queue_begin = currentTimeUs();
    if (!task_pool_->submitCompletion([this, settle, queue_begin]() mutable {
            metrics_reporter_.reportQueueWaitMetric(true,
                                                    "business",
                                                    cacheTransferOperationName(CacheTransferOperation::STORE),
                                                    Tier::NONE,
                                                    Tier::NONE,
                                                    currentTimeUs() - queue_begin);
            settle();
        })) {
        RTP_LLM_LOG_WARNING("store completion queue is closed; settling inline");
        settle();
    }
}

void BlockTreeStorer::settleTask(StoreTask& task, bool copy_success, const char* failure_reason) {
    bool   stopping = false;
    size_t accepted = 0;
    if (stopping_.load() && task.error.ok()) {
        task.error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store stopped before settlement");
    }
    if (!copy_success && task.error.ok()) {
        task.error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "store copy failed");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping = stopping_.load();
        accepted = settleLocked(task, copy_success && !stopping, stopping ? "store_stopped" : failure_reason);
    }

    if (stopping) {
        RTP_LLM_LOG_INFO("store rolled back during shutdown, target=%s blocks=%zu",
                         tierName(task.target_tier),
                         task.descriptors().size());
    } else if (!copy_success) {
        RTP_LLM_LOG_WARNING("store copy failed, target=%s blocks=%zu; tree unchanged",
                            tierName(task.target_tier),
                            task.descriptors().size());
    } else {
        metrics_reporter_.reportStorePublish(task.target_tier, accepted, task.descriptors().size() - accepted);
        RTP_LLM_LOG_DEBUG("store published target=%s accepted=%zu duplicate=%zu",
                          tierName(task.target_tier),
                          accepted,
                          task.descriptors().size() - accepted);
    }
}

size_t BlockTreeStorer::settleLocked(StoreTask& task, bool publish, const char* failure_reason) {
    BlockTreeInsertResult insert_result;
    std::vector<PrefixCachePlanPoint> unfulfilled;
    if (publish) {
        std::vector<std::vector<GroupSetResource>> resources(task.cache_keys.size(),
                                                             std::vector<GroupSetResource>(tree_->groupSets().size()));
        for (const TransferDescriptor& descriptor : task.descriptors()) {
            resources[descriptor.path_index][descriptor.group_set_id].setBlocks(
                task.target_tier, {descriptor.singleBlockAt(task.target_tier)});
        }
        insert_result = tree_->insertNode(task.cache_keys, resources, true, false, &task.token_segments);
        unfulfilled   = tree_->unfulfilledPlanPoints(task.plan_points, task.cache_keys);
        logPrefixPlanUnfulfilled(tree_->observabilityInstance(),
                                 unfulfilled,
                                 "insert_not_fulfilled",
                                 ErrorInfo::OkStatus(), task.observation_request_id);
    } else {
        unfulfilled = std::move(task.plan_points);
        logPrefixPlanUnfulfilled(tree_->observabilityInstance(), unfulfilled, failure_reason, task.error, task.observation_request_id);
    }
    if (tree_->observabilityEnabled()) {
        logPrefixPublication(tree_->observabilityInstance(), task.observation_request_id, task.target_tier,
                             publish ? &insert_result : nullptr,
                             publish ? (unfulfilled.empty() ? nullptr : "insert_not_fulfilled") : failure_reason,
                             unfulfilled.size());
    }
    task.plan_points.clear();

    store_task_runner_.releaseTaskResources(task);

    if (insert_result.accepted_resource_count > 0) {
        evictor_.onInserted(insert_result);
    }
    if (!task.descriptors().empty()) {
        settled_(true);
    }
    return insert_result.accepted_resource_count;
}

}  // namespace rtp_llm
