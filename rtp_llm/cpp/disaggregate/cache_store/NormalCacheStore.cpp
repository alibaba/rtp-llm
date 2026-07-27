#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include "autil/LockFreeThreadPool.h"

#include <cstring>

namespace rtp_llm {

NormalCacheStore::~NormalCacheStore() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    lifecycle_state_ = LifecycleState::Destroyed;
    stopThreadPoolLocked();
    if (messager_) {
        messager_->beginCheckpointDrain();
        messager_->teardownForCheckpoint();
        messager_.reset();
    }
    if (request_block_buffer_store_) {
        request_block_buffer_store_->stop();
        request_block_buffer_store_.reset();
    }
    if (owns_memory_util_) {
        memory_util_.reset();
    }
    RTP_LLM_LOG_INFO("destory cache store done");
}

std::shared_ptr<NormalCacheStore> NormalCacheStore::createNormalCacheStore(const CacheStoreInitParams& params) {
    std::shared_ptr<NormalCacheStore> normal_cache_store(new NormalCacheStore);
    if (normal_cache_store && normal_cache_store->init(params)) {
        return normal_cache_store;
    }
    return nullptr;
}

bool NormalCacheStore::init(const CacheStoreInitParams& params) {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    params_           = params;
    device_id_        = params.device_id;
    owns_memory_util_ = params_.memory_util == nullptr;

    if (params_.memory_util != nullptr) {
        memory_util_ = params.memory_util;
    } else {
        memory_util_ = createMemoryUtilImpl(params_.rdma_mode);
    }

    // always has metric
    metrics_reporter_ = params.metrics_reporter;

    if (!memory_util_ || !memory_util_->isAvailable() || !initRuntimeLocked(false)) {
        lifecycle_state_ = LifecycleState::Failed;
        return false;
    }
    restored_paused_ = false;
    lifecycle_state_ = LifecycleState::Running;
    RTP_LLM_LOG_INFO("normal cache store init done, thread pool thread count is %d", params.thread_count);
    return true;
}

bool NormalCacheStore::initRuntimeLocked(bool start_paused) {
    request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_);

    messager_ = createMessager(memory_util_, request_block_buffer_store_, metrics_reporter_);
    MessagerInitParams messager_init_params;
    messager_init_params.server_port                  = params_.listen_port;
    messager_init_params.rdma_server_port             = params_.rdma_listen_port;
    messager_init_params.rdma_connect_timeout_ms      = params_.rdma_connect_timeout_ms;
    messager_init_params.rdma_qp_count_per_connection = params_.rdma_qp_count_per_connection;
    messager_init_params.rdma_io_thread_count         = params_.rdma_io_thread_count;
    messager_init_params.rdma_worker_thread_count     = params_.rdma_worker_thread_count;
    messager_init_params.io_thread_count              = params_.messager_io_thread_count;
    messager_init_params.worker_thread_count          = params_.messager_worker_thread_count;
    messager_init_params.worker_queue_size            = params_.queue_size;
    messager_init_params.device_id                    = params_.device_id;
    messager_init_params.start_paused                 = start_paused;

    if (!messager_->init(messager_init_params)) {
        RTP_LLM_LOG_ERROR("normal cache store init failed : init messager failed");
        return false;
    }

    return startThreadPoolLocked();
}

bool NormalCacheStore::startThreadPoolLocked() {
    thread_pool_close_.store(false, std::memory_order_release);
    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        params_.thread_count, params_.queue_size, nullptr, "NormalCacheStoreTask");
    if (!thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("normal cache store init failed : init thread pool failed");
        thread_pool_.reset();
        thread_pool_close_.store(true, std::memory_order_release);
        return false;
    }

    auto check_task_readiness = [this]() {
        pinThreadToDeviceOnce(this->device_id_);
        while (!thread_pool_close_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::unique_lock<std::shared_mutex> lock(store_tasks_mutex_);
            for (auto it = this->store_tasks_.begin(); it != this->store_tasks_.end();) {
                auto& [buffer, item]   = *it;
                auto& [callback, task] = item;
                auto event             = buffer->getEvent();
                if ((event && event->query()) || event == nullptr) {
                    if (this->thread_pool_->pushTask(task) != autil::ThreadPoolBase::ERROR_NONE) {
                        RTP_LLM_LOG_WARNING("normal cache store push store task to thread pool failed");
                        callback(false, CacheStoreErrorCode::PushWorkerItemFailed);
                    }

                    it = store_tasks_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    };

    if (thread_pool_->pushTask(check_task_readiness) != autil::ThreadPoolBase::ERROR_NONE) {
        RTP_LLM_LOG_WARNING("normal cache store push check task to thread pool failed");
        return false;
    }

    return true;
}

void NormalCacheStore::stopThreadPoolLocked() {
    thread_pool_close_.store(true, std::memory_order_release);
    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_.reset();
    }
}

void NormalCacheStore::store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                             CacheStoreStoreDoneCallback                callback) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running) {
        lifecycle_lock.unlock();
        callback(false, CacheStoreErrorCode::PushWorkerItemFailed);
        return;
    }
    if (request_block_buffer == nullptr || !request_block_buffer->isValid()) {
        RTP_LLM_LOG_WARNING("normal cache store call store failed, request block is invalid");
        lifecycle_lock.unlock();
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        lifecycle_lock.unlock();
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    auto collector = std::make_shared<CacheStoreStoreMetricsCollector>(
        metrics_reporter_, request_block_buffer->getBlocksCount(), request_block_buffer->getBlocksSize());
    auto counted_callback = countTransfer(std::move(callback));
    // task 只在threadpool中运行, threadpool退出前会清理所有running task, 用this是安全的
    auto task = [this, request_block_buffer, counted_callback, collector]() {
        pinThreadToDeviceOnce(this->device_id_);
        try {
            this->runStoreTask(request_block_buffer, counted_callback, collector);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("normal cache store run store task exception, request id is %s, error is %s",
                              request_block_buffer->getRequestId().c_str(),
                              e.what());
            counted_callback(false, CacheStoreErrorCode::StoreFailed);
        } catch (...) {
            RTP_LLM_LOG_ERROR("normal cache store run store task unknown exception, request id is %s",
                              request_block_buffer->getRequestId().c_str());
            counted_callback(false, CacheStoreErrorCode::StoreFailed);
        }
    };

    std::unique_lock<std::shared_mutex> lock(store_tasks_mutex_);
    store_tasks_[request_block_buffer] = {counted_callback, task};
}

std::shared_ptr<StoreContext>
NormalCacheStore::storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                               int64_t                                                 timeout_ms) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running) {
        return nullptr;
    }
    if (request_block_buffers.empty()) {
        return nullptr;
    }
    lifecycle_lock.unlock();
    auto store_context = std::make_shared<StoreContext>(shared_from_this());
    store_context->store(request_block_buffers, timeout_ms);
    return store_context;
}

void NormalCacheStore::debugInfo() {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ == LifecycleState::Running && request_block_buffer_store_) {
        request_block_buffer_store_->debugInfo();
    }
}

void NormalCacheStore::runStoreTask(const std::shared_ptr<RequestBlockBuffer>&              request_block_buffer,
                                    CacheStoreStoreDoneCallback                             callback,
                                    const std::shared_ptr<CacheStoreStoreMetricsCollector>& collector) {
    // store to local
    collector->markTaskRun();

    auto ret = request_block_buffer_store_->setRequestBlockBuffer(request_block_buffer);
    collector->markEnd(ret);

    if (!ret) {
        RTP_LLM_LOG_WARNING("normal cache store run store task failed, request id is %s",
                            request_block_buffer->getRequestId().c_str());
        callback(false, CacheStoreErrorCode::StoreFailed);
        return;
    }
    callback(true, CacheStoreErrorCode::None);
}

void NormalCacheStore::load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                            CacheStoreLoadDoneCallback                 callback,
                            const std::string&                         ip,
                            uint32_t                                   port,
                            uint32_t                                   rdma_port,
                            uint32_t                                   timeout_ms,
                            int                                        partition_count,
                            int                                        partition_id) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running) {
        lifecycle_lock.unlock();
        callback(false, CacheStoreErrorCode::LoadConnectFailed);
        return;
    }
    if (request_block_buffer == nullptr || !request_block_buffer->isValid() || ip.empty()) {
        RTP_LLM_LOG_WARNING("normal cache store run load failed, invalid params");
        lifecycle_lock.unlock();
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (port == 0 || (memory_util_->isRdmaMode() && rdma_port == 0)) {
        RTP_LLM_LOG_WARNING("normal cache store run load failed, port is 0");
        lifecycle_lock.unlock();
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        lifecycle_lock.unlock();
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(
        metrics_reporter_, request_block_buffer->getBlocksCount(), request_block_buffer->getBlocksSize());

    auto counted_callback = countTransfer(std::move(callback));
    auto task             = [this,
                 request_block_buffer,
                 counted_callback,
                 ip,
                 port,
                 rdma_port,
                 timeout_ms,
                 collector,
                 partition_count,
                 partition_id]() {
        pinThreadToDeviceOnce(this->device_id_);
        try {
            this->runLoadTask(request_block_buffer,
                              counted_callback,
                              ip,
                              port,
                              rdma_port,
                              timeout_ms,
                              collector,
                              partition_count,
                              partition_id);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("normal cache store run load task exception, request id is %s, error is %s",
                              request_block_buffer->getRequestId().c_str(),
                              e.what());
            counted_callback(false, CacheStoreErrorCode::LoadErrorUnknown);
        } catch (...) {
            RTP_LLM_LOG_ERROR("normal cache store run load task unknown exception, request id is %s",
                              request_block_buffer->getRequestId().c_str());
            counted_callback(false, CacheStoreErrorCode::LoadErrorUnknown);
        }
    };

    if (thread_pool_->pushTask(task) != autil::ThreadPoolBase::ERROR_NONE) {
        RTP_LLM_LOG_WARNING("normal cache store push load task for request id [%s] to thread pool failed",
                            request_block_buffer->getRequestId().c_str());
        collector->markEnd(false);
        lifecycle_lock.unlock();
        counted_callback(false, CacheStoreErrorCode::PushWorkerItemFailed);
        return;
    }
}

void NormalCacheStore::runLoadTask(const std::shared_ptr<RequestBlockBuffer>&                   request_block_buffer,
                                   CacheStoreLoadDoneCallback                                   callback,
                                   const std::string&                                           ip,
                                   uint32_t                                                     port,
                                   uint32_t                                                     rdma_port,
                                   uint32_t                                                     timeout_ms,
                                   const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                                   int                                                          partition_count,
                                   int                                                          partition_id) {
    collector->markTaskRun();
    auto load_request = std::make_shared<LoadRequest>(
        ip, port, rdma_port, request_block_buffer, callback, timeout_ms, partition_count, partition_id);
    messager_->load(load_request, collector);
}

std::shared_ptr<LoadContext>
NormalCacheStore::loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                              const std::string&                                      ip,
                              uint32_t                                                port,
                              uint32_t                                                rdma_port,
                              int64_t                                                 timeout_ms,
                              LoadContext::CheckCancelFunc                            check_cancel_func,
                              int                                                     partition_count,
                              int                                                     partition_id) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running) {
        return nullptr;
    }
    if (request_block_buffers.empty() || ip.empty()) {
        return nullptr;
    }

    auto memory_util = memory_util_;
    lifecycle_lock.unlock();
    auto load_context = std::make_shared<LoadContext>(shared_from_this(), memory_util->isRdmaMode());
    load_context->load(
        request_block_buffers, ip, port, rdma_port, timeout_ms, check_cancel_func, partition_count, partition_id);
    return load_context;
}

std::shared_ptr<RemoteStoreTask>
NormalCacheStore::submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&                    request,
                                        const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                        RemoteStoreTask::CheckCancelFunc check_cancel_func) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running) {
        return nullptr;
    }
    auto task = std::make_shared<RemoteStoreTaskImpl>(request, collector, check_cancel_func);
    std::unique_lock<std::shared_mutex> lock(remote_store_tasks_mutex_);
    auto&                               tasks = remote_store_tasks_[request->request_id];
    tasks.push_back(task);

    RTP_LLM_LOG_DEBUG("normal cache store submit remote store task, request id is %s, request is %s",
                      request->request_id.c_str(),
                      request->toString().c_str());

    auto                               request_id = request->request_id;
    std::weak_ptr<RemoteStoreTaskImpl> weak_task  = task;
    RequestBlockBuffer::WatchFunc      watchFunc =
        [this, request_id, weak_task](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            auto task = weak_task.lock();
            if (!task) {
                RTP_LLM_LOG_DEBUG("task has been released, request id is %s", request_id.c_str());
                return;
            }
            if (!ok) {
                RTP_LLM_LOG_WARNING("normal cache store run store task watch func failed, request id is %s",
                                    request_id.c_str());
                task->notifyRequestDone({}, false);
                return;
            }

            auto transfer_request = task->makeAvailableRequest(blocks);

            if (transfer_request == nullptr) {
                RTP_LLM_LOG_WARNING("normal cache store make available request failed, request id is %s",
                                    request_id.c_str());
                return;
            }

            std::shared_ptr<Messager> messager;
            {
                std::shared_lock<std::shared_mutex> lifecycle_lock(this->lifecycle_mutex_);
                if ((this->lifecycle_state_ != LifecycleState::Running
                     && this->lifecycle_state_ != LifecycleState::Draining)
                    || !this->messager_) {
                    transfer_request->callback(
                        false, CacheStoreErrorCode::LoadConnectFailed, transfer_request->buffer_pairs);
                    return;
                }
                messager = this->messager_;
                this->trackPhysicalTransfer(transfer_request);
            }
            messager->transfer(transfer_request);
        };

    this->request_block_buffer_store_->setRequestBlockBufferWatchFunc(request_id, std::move(watchFunc));
    return std::dynamic_pointer_cast<RemoteStoreTask>(task);
}

void NormalCacheStore::releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if ((lifecycle_state_ != LifecycleState::Running && lifecycle_state_ != LifecycleState::Draining) || !task) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(remote_store_tasks_mutex_);
    auto                                iter = remote_store_tasks_.find(task->getRequestId());
    if (iter == remote_store_tasks_.end()) {
        return;
    }
    auto& tasks = iter->second;
    tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
    if (tasks.empty()) {
        remote_store_tasks_.erase(iter);
    }
}

void NormalCacheStore::markRequestEnd(const std::string& requestid) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running && lifecycle_state_ != LifecycleState::Draining) {
        return;
    }
    std::vector<CacheStoreStoreDoneCallback> pending_store_callbacks;
    {
        std::unique_lock<std::shared_mutex> lock(store_tasks_mutex_);
        for (auto it = store_tasks_.begin(); it != store_tasks_.end();) {
            auto& buffer = it->first;
            if (buffer && buffer->getRequestId() == requestid) {
                pending_store_callbacks.emplace_back(std::move(it->second.first));
                it = store_tasks_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto& callback : pending_store_callbacks) {
        if (callback) {
            callback(true, CacheStoreErrorCode::None);
        }
    }
    request_block_buffer_store_->delRequestBlockBuffer(requestid);
}

bool NormalCacheStore::regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    const bool                          may_register = lifecycle_state_ == LifecycleState::Running
                              || (lifecycle_state_ == LifecycleState::Draining && restored_paused_);
    if (!may_register || !request_block_buffer_store_) {
        return false;
    }
    return request_block_buffer_store_->regUserBuffers(buffers);
}

std::shared_ptr<BlockBuffer> NormalCacheStore::findUserBuffer(const std::string& buffer_key) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Running || !request_block_buffer_store_) {
        return nullptr;
    }
    return request_block_buffer_store_->findUserBuffer(buffer_key);
}

const std::shared_ptr<MemoryUtil>& NormalCacheStore::getMemoryUtil() const {
    return memory_util_;
}

size_t NormalCacheStore::activeTransferCount() const {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    return activeTransferCountLocked();
}

size_t NormalCacheStore::activeTransferCountLocked() const {
    size_t count = active_transfer_count_->load(std::memory_order_acquire)
                   + active_physical_transfer_count_->load(std::memory_order_acquire);
    std::shared_lock<std::shared_mutex> lock(remote_store_tasks_mutex_);
    for (const auto& [request_id, tasks] : remote_store_tasks_) {
        for (const auto& task : tasks) {
            if (task && !task->done()) {
                count++;
            }
        }
    }
    if (messager_) {
        count += messager_->activeRequestCount();
    }
    return count;
}

void NormalCacheStore::trackPhysicalTransfer(const std::shared_ptr<TransferRequest>& request) {
    if (!request || !request->callback) {
        return;
    }
    auto active_count = active_physical_transfer_count_;
    active_count->fetch_add(1, std::memory_order_acq_rel);
    auto released = std::make_shared<std::atomic<bool>>(false);
    auto callback = std::move(request->callback);
    request->callback =
        [active_count, released, callback = std::move(callback)](
            bool success, CacheStoreErrorCode error_code, const std::map<std::string, std::string>& block_keys) {
            auto release = [&]() {
                if (!released->exchange(true, std::memory_order_acq_rel)) {
                    active_count->fetch_sub(1, std::memory_order_acq_rel);
                }
            };
            try {
                callback(success, error_code, block_keys);
            } catch (...) {
                release();
                throw;
            }
            release();
        };
}

size_t NormalCacheStore::activePhysicalTransferCount() const {
    return active_physical_transfer_count_->load(std::memory_order_acquire);
}

bool NormalCacheStore::beginCheckpointDrain() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ == LifecycleState::Draining) {
        return true;
    }
    if (lifecycle_state_ != LifecycleState::Running || !messager_) {
        return false;
    }
    if (!messager_->beginCheckpointDrain()) {
        lifecycle_state_ = LifecycleState::Failed;
        return false;
    }
    lifecycle_state_ = LifecycleState::Draining;
    return true;
}

bool NormalCacheStore::resumeAfterCheckpoint() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ == LifecycleState::Running) {
        return true;
    }
    if (lifecycle_state_ != LifecycleState::Draining || !messager_ || memory_owner_suspended_ || !memory_util_
        || !memory_util_->isAvailable()) {
        return false;
    }
    if (!messager_->resumeAfterCheckpoint()) {
        lifecycle_state_ = LifecycleState::Failed;
        return false;
    }
    restored_paused_ = false;
    lifecycle_state_ = LifecycleState::Running;
    return true;
}

bool NormalCacheStore::teardownForCheckpoint() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ == LifecycleState::Suspended) {
        return true;
    }
    if (lifecycle_state_ != LifecycleState::Draining) {
        return false;
    }
    if (activeTransferCountLocked() != 0) {
        RTP_LLM_LOG_WARNING("cache store checkpoint teardown rejected with active transfers");
        return false;
    }

    bool ok = true;
    stopThreadPoolLocked();
    {
        std::unique_lock<std::shared_mutex> lock(store_tasks_mutex_);
        ok = store_tasks_.empty() && ok;
        store_tasks_.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(remote_store_tasks_mutex_);
        remote_store_tasks_.clear();
    }
    if (messager_) {
        ok = messager_->teardownForCheckpoint() && ok;
        messager_.reset();
    }
    if (request_block_buffer_store_) {
        request_block_buffer_store_->stop();
        request_block_buffer_store_.reset();
    }
    // Keep MemoryUtil alive and running. The Level-3 controller deregisters KV
    // MRs only after this transport-only teardown has completed.
    restored_paused_ = false;
    lifecycle_state_ = ok ? LifecycleState::Suspended : LifecycleState::Failed;
    return ok;
}

bool NormalCacheStore::teardownMemoryOwnerAfterMrDereg() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (memory_owner_suspended_) {
        return true;
    }
    if (lifecycle_state_ != LifecycleState::Suspended || !memory_util_ || activeTransferCountLocked() != 0) {
        return false;
    }
    if (!memory_util_->teardownForCheckpoint()) {
        lifecycle_state_ = LifecycleState::Failed;
        return false;
    }
    memory_owner_suspended_ = true;
    return true;
}

bool NormalCacheStore::rebuildMemoryOwnerBeforeMrReg() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ != LifecycleState::Suspended || !memory_util_) {
        return false;
    }
    if (!memory_owner_suspended_) {
        return memory_util_->isAvailable();
    }
    if (!memory_util_->rebuildAfterRestore()) {
        lifecycle_state_ = LifecycleState::Failed;
        return false;
    }
    memory_owner_suspended_ = false;
    return true;
}

bool NormalCacheStore::rebuildAfterRestore() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (lifecycle_state_ == LifecycleState::Running) {
        return true;
    }
    if (lifecycle_state_ == LifecycleState::Draining && restored_paused_) {
        return true;
    }
    if (lifecycle_state_ != LifecycleState::Suspended || !memory_util_ || memory_owner_suspended_
        || !memory_util_->isAvailable()) {
        return false;
    }
    lifecycle_state_ = LifecycleState::Rebuilding;

    auto fail_rebuild = [this]() {
        stopThreadPoolLocked();
        if (messager_) {
            messager_->beginCheckpointDrain();
            messager_->teardownForCheckpoint();
            messager_.reset();
        }
        if (request_block_buffer_store_) {
            request_block_buffer_store_->stop();
            request_block_buffer_store_.reset();
        }
        if (memory_util_->teardownForCheckpoint()) {
            memory_owner_suspended_ = true;
        }
        lifecycle_state_ = LifecycleState::Failed;
        return false;
    };

    if (!initRuntimeLocked(true)) {
        return fail_rebuild();
    }
    // Keep all admission gates closed. The lifecycle coordinator re-registers
    // MRs and runs health checks before calling resumeAfterCheckpoint().
    restored_paused_ = true;
    lifecycle_state_ = LifecycleState::Draining;
    return true;
}

bool NormalCacheStore::isAvailable() const {
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    return lifecycle_state_ == LifecycleState::Running;
}

const std::shared_ptr<RequestBlockBufferStore>& NormalCacheStore::getRequestBlockBufferStore() const {
    return request_block_buffer_store_;
}

}  // namespace rtp_llm
