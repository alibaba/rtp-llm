#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/connector/IKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

class ConnectorLayerWriteContext final: public KVCacheConnectorLayerContext {
public:
    ConnectorLayerWriteContext(KVCacheResourcePtr          resource,
                               int64_t                     request_id,
                               std::shared_ptr<c10::Event> event,
                               int64_t                     deadline_ms):
        resource_(std::move(resource)),
        request_id_(request_id),
        event_(std::move(event)),
        deadline_ms_(deadline_ms) {}

    const KVCacheResource& kvCacheResource() const override {
        return *resource_;
    }
    KVCacheResourcePtr heldKVCacheResource() const override {
        return resource_;
    }
    int64_t requestId() const override {
        return request_id_;
    }
    std::shared_ptr<c10::Event> attentionEvent() const override {
        return event_;
    }
    int64_t deadlineMs() const override {
        return deadline_ms_;
    }

private:
    KVCacheResourcePtr          resource_;
    int64_t                     request_id_;
    std::shared_ptr<c10::Event> event_;
    int64_t                     deadline_ms_;
};

void writeCacheToP2PConnector(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                              const torch_ext::LayerKVCache&       layer_kv,
                              const std::shared_ptr<KVCacheManager>& cache_manager,
                              size_t                               cache_model_id,
                              int                                  cp_rank,
                              int                                  cp_size,
                              const std::shared_ptr<c10::Event>&   event) {
    if (!cache_manager || !cache_manager->hasP2PConnector()) {
        return;
    }
    auto coordinator = cache_manager->connectorCoordinator();
    if (!coordinator) {
        return;
    }

    const auto& param = cache_store_inputs;
    if (!param.request_id.defined() || param.request_id.numel() == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(param.input_lengths_host.device().is_cpu()
                                && param.prefix_lengths_host.device().is_cpu()
                                && param.host_kv_cache_offset.device().is_cpu()
                                && param.request_id.device().is_cpu()
                                && param.request_pd_separation.device().is_cpu()
                                && param.cache_keys.device().is_cpu(),
                            "P2P cache-write metadata must be host-resident");
    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.dim() == 2 && param.cache_keys.dim() == 2,
                            "P2P cache-write expects tag-local 2-D block table and cache keys");

    const auto global_layer_id = coordinator->convertToGlobalLayerId(cache_model_id, layer_kv.layer_id);
    if (global_layer_id == std::numeric_limits<uint32_t>::max()) {
        RTP_LLM_LOG_ERROR("P2P cache-write cannot map model=%zu local_layer=%d", cache_model_id, layer_kv.layer_id);
        coordinator->reportP2PCacheWriteFailure();
        return;
    }

    const auto& global_cache_config = cache_manager->cacheConfig();
    const auto& group = global_cache_config.groupForLayer(static_cast<int>(global_layer_id), layer_kv.tag);
    const auto  group_id = global_cache_config.groupIdForTag(layer_kv.tag);
    const auto  context_batch_size = static_cast<size_t>(param.request_id.numel());
    const auto  total_batch_size   = static_cast<size_t>(param.input_lengths_host.numel());
    RTP_LLM_CHECK_WITH_INFO(total_batch_size >= context_batch_size,
                            "P2P cache-write total batch=%zu smaller than context batch=%zu",
                            total_batch_size,
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.numel() == static_cast<int64_t>(context_batch_size)
                                && param.request_pd_separation.numel() == static_cast<int64_t>(context_batch_size)
                                && param.cache_keys.size(0) == static_cast<int64_t>(context_batch_size)
                                && param.host_kv_cache_offset.size(0) == static_cast<int64_t>(total_batch_size),
                            "P2P cache-write metadata batch dimensions disagree");

    const size_t decoder_batch_size = total_batch_size - context_batch_size;
    const size_t key_width           = static_cast<size_t>(param.cache_keys.size(1));
    const size_t offset_width        = static_cast<size_t>(param.host_kv_cache_offset.size(1));
    const auto   input_lengths       = param.input_lengths_host.accessor<int32_t, 1>();
    const auto   prefix_lengths      = param.prefix_lengths_host.accessor<int32_t, 1>();
    const auto   block_offsets       = param.host_kv_cache_offset.accessor<int32_t, 2>();
    const auto   request_ids         = param.request_id.accessor<int64_t, 1>();
    const auto   pd_requests         = param.request_pd_separation.accessor<bool, 1>();
    const auto   cache_keys          = param.cache_keys.accessor<int64_t, 2>();
    const bool   has_deadlines       = param.request_deadline_ms.defined()
                                     && param.request_deadline_ms.device().is_cpu()
                                     && param.request_deadline_ms.numel() == static_cast<int64_t>(context_batch_size);
    const int64_t* deadlines = has_deadlines ? param.request_deadline_ms.data_ptr<int64_t>() : nullptr;
    const bool use_hybrid = global_cache_config.topology().groups().size() > 1;
    const size_t physical_tokens_per_block = group.seq_size_per_block;

    for (size_t batch_id = 0; batch_id < context_batch_size; ++batch_id) {
        if (!pd_requests[static_cast<int64_t>(batch_id)]) {
            continue;
        }
        const bool uses_cp_keys = cp_size > 1 && group.policy.cp_mapping != CpBlockMappingMode::NONE
                                  && physical_tokens_per_block % static_cast<size_t>(cp_size) == 0;
        const size_t canonical_tokens_per_block =
            uses_cp_keys ? physical_tokens_per_block / static_cast<size_t>(cp_size) : physical_tokens_per_block;
        const int prefix_length = prefix_lengths[static_cast<int64_t>(batch_id)];
        RTP_LLM_CHECK_WITH_INFO(prefix_length % static_cast<int>(canonical_tokens_per_block) == 0,
                                "P2P cache-write tag=%s prefix=%d is not aligned to canonical block=%zu",
                                layer_kv.tag.c_str(),
                                prefix_length,
                                canonical_tokens_per_block);
        const int input_length = input_lengths[static_cast<int64_t>(decoder_batch_size + batch_id)];
        const int canonical_total_blocks =
            prefix_length / static_cast<int>(canonical_tokens_per_block)
            + (input_length + static_cast<int>(canonical_tokens_per_block) - 1)
                  / static_cast<int>(canonical_tokens_per_block);
        const auto plan = buildCacheStorePlan(group.policy,
                                              static_cast<size_t>(std::min<int>(canonical_total_blocks, key_width)),
                                              0,
                                              use_hybrid,
                                              cp_rank,
                                              cp_size);
        if (plan.empty()) {
            continue;
        }

        auto resource = std::make_shared<KVCacheResource>();
        resource->initGroups(global_cache_config.topologyPtr());
        auto& resource_blocks = resource->mutableBlockIds(group_id);
        CacheKeysType selected_keys;
        BlockIndicesType selected_blocks;
        selected_keys.reserve(plan.size());
        selected_blocks.reserve(plan.size());
        for (const auto& pair : plan) {
            RTP_LLM_CHECK_WITH_INFO(pair.key_index >= 0 && static_cast<size_t>(pair.key_index) < key_width
                                        && pair.offset_index >= 0
                                        && static_cast<size_t>(pair.offset_index) < offset_width,
                                    "P2P cache-write plan out of range: key=%d/%zu offset=%d/%zu",
                                    pair.key_index,
                                    key_width,
                                    pair.offset_index,
                                    offset_width);
            const auto block_id = block_offsets[static_cast<int64_t>(decoder_batch_size + batch_id)]
                                               [static_cast<int64_t>(pair.offset_index)];
            if (isNullBlockIdx(block_id)) {
                continue;
            }
            selected_keys.push_back(
                cache_keys[static_cast<int64_t>(batch_id)][static_cast<int64_t>(pair.key_index)]);
            selected_blocks.push_back(block_id);
        }
        if (selected_keys.empty()) {
            continue;
        }
        resource->setCacheKeys(std::move(selected_keys));
        resource_blocks.assign(std::move(selected_blocks));
        resource->setCacheKeysAreCpCanonical(uses_cp_keys);

        auto held = coordinator->holdKVCacheResourceForConnector(*resource, static_cast<int>(global_layer_id));
        if (!held) {
            coordinator->reportP2PCacheWriteFailure();
            continue;
        }
        const int64_t deadline_ms = deadlines ? deadlines[batch_id] : std::numeric_limits<int64_t>::max();
        auto context = std::make_shared<ConnectorLayerWriteContext>(
            std::move(held), request_ids[static_cast<int64_t>(batch_id)], event, deadline_ms);
        if (!coordinator->asyncWriteByLayer(static_cast<int>(global_layer_id), context)) {
            coordinator->reportP2PCacheWriteFailure();
        }
    }
}

}  // namespace

CacheStoreAsyncWriter::PendingTaskGuard::PendingTaskGuard(CacheStoreAsyncWriter& writer): writer_(writer) {}

CacheStoreAsyncWriter::PendingTaskGuard::~PendingTaskGuard() {
    writer_.completePendingTask();
}

CacheStoreAsyncWriter::CacheStoreAsyncWriter(int                             device_id,
                                             std::shared_ptr<KVCacheManager> cache_manager,
                                             size_t                          cache_model_id,
                                             std::optional<int>              mtp_cache_config_index):
    device_id_(device_id), cache_manager_(std::move(cache_manager)), cache_model_id_(cache_model_id) {
    if (cache_manager_) {
        const CacheConfig* selected_config = &cache_manager_->cacheConfig();
        if (mtp_cache_config_index.has_value()) {
            selected_config = &cache_manager_->getMTPModuleCacheConfig(*mtp_cache_config_index);
        }
        cache_config_ = std::shared_ptr<const CacheConfig>(cache_manager_, selected_config);

        if (const auto cp_slot_mapper = cache_manager_->cpSlotMapper()) {
            cp_rank_ = cp_slot_mapper->cpRank();
            cp_size_ = cp_slot_mapper->cpSize();
        }
    }

    constexpr size_t kThreadCount = 3;
    constexpr size_t kQueueSize   = 10000;
    auto pool = std::make_shared<autil::LockFreeThreadPool>(kThreadCount, kQueueSize, nullptr, "CacheStoreAsync");
    RTP_LLM_CHECK_WITH_INFO(pool->start(), "CacheStoreAsyncWriter: failed to start thread pool");
    thread_pool_ = std::move(pool);
}

void CacheStoreAsyncWriter::completePendingTask() {
    if (pending_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

void CacheStoreAsyncWriter::storeCurrentException() {
    std::lock_guard<std::mutex> ex_lock(exception_mutex_);
    if (!stored_exception_) {
        stored_exception_ = std::current_exception();
    }
}

CacheStoreAsyncWriter::~CacheStoreAsyncWriter() {
    if (state_ == State::RUNNING) {
        RTP_LLM_LOG_WARNING("CacheStoreAsyncWriter destroyed while RUNNING - "
                            "caller should call waitAllDone() before destruction");
    }
    if (thread_pool_) {
        thread_pool_->stop();
    }
}

// IDLE -> RUNNING. Resets bookkeeping for a new forward-pass cycle.
void CacheStoreAsyncWriter::init() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::IDLE,
                            "CacheStoreAsyncWriter::init() called while already RUNNING. "
                            "Must call waitAllDone() before re-initializing.");
    pending_count_.store(0, std::memory_order_relaxed);
    stored_exception_   = nullptr;
    active_cache_store_ = cache_manager_ ? cache_manager_->getCacheStore() : nullptr;
    state_              = State::RUNNING;
}

// Enqueue a task to the background thread pool. Must be in RUNNING state.
void CacheStoreAsyncWriter::submit(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                            "CacheStoreAsyncWriter::submit() called when not RUNNING. "
                            "Call init() first.");

    pending_count_.fetch_add(1, std::memory_order_acq_rel);
    auto wrapped = [this, task = std::move(task)]() {
        PendingTaskGuard pending_task_guard(*this);
        try {
            setCurrentThreadDeviceIfNeeded(device_id_);
            task();
        } catch (...) {
            storeCurrentException();
            RTP_LLM_LOG_ERROR("CacheStoreAsyncWriter: background task threw an exception");
        }
    };

    const auto rc = thread_pool_->pushTask(std::move(wrapped));
    if (rc != autil::ThreadPoolBase::ERROR_NONE) {
        completePendingTask();
        RTP_LLM_CHECK_WITH_INFO(false,
                                "CacheStoreAsyncWriter: pushTask failed (rc=%d). "
                                "Queue full or thread pool in bad state.",
                                static_cast<int>(rc));
    }
}

// Block until all submitted tasks complete, then RUNNING -> IDLE.
// Re-throws the first stored exception after state transition.
void CacheStoreAsyncWriter::waitAllDone() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        RTP_LLM_CHECK_WITH_INFO(state_ == State::RUNNING,
                                "CacheStoreAsyncWriter::waitAllDone() called when not RUNNING. "
                                "Call init() first.");
    }

    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait(lock, [this]() { return pending_count_.load(std::memory_order_acquire) == 0; });
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        active_cache_store_.reset();
        state_ = State::IDLE;
    }

    if (stored_exception_) {
        auto ex           = stored_exception_;
        stored_exception_ = nullptr;
        std::rethrow_exception(ex);
    }
}

void CacheStoreAsyncWriter::write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                                  const torch_ext::LayerKVCache&       layer_kv) {
    if (!cache_config_) {
        return;
    }

    // Capture tensors by value so their underlying storage stays alive in the background thread.
    // A torch::Tensor copy only increments the reference count.
    auto captured_cache_store_inputs = cache_store_inputs;
    auto captured_layer_kv           = layer_kv;
    auto cache_config                = cache_config_;
    // Create the event on the main thread to avoid event-record contention in worker threads.
    auto event = runtimeCreateEvent();
    writeCacheToP2PConnector(
        captured_cache_store_inputs, captured_layer_kv, cache_manager_, cache_model_id_, cp_rank_, cp_size_, event);
    auto cache_store = active_cache_store_;
    if (!cache_store) {
        return;
    }
    auto run   = [captured_cache_store_inputs,
                captured_layer_kv,
                cache_config,
                cache_store,
                cache_model_id = cache_model_id_,
                cp_rank        = cp_rank_,
                cp_size        = cp_size_,
                event          = std::move(event)]() mutable {
        runtimeWriteCacheStore(captured_cache_store_inputs,
                               captured_layer_kv,
                               *cache_config,
                               cache_store,
                               cache_model_id,
                               cp_rank,
                               cp_size,
                               std::move(event));
    };
    submit(std::move(run));
}

}  // namespace rtp_llm
