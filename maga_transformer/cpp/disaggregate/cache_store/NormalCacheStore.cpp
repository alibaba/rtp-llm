#include "maga_transformer/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/Interface.h"
#include "maga_transformer/cpp/utils/Logger.h"

#include "autil/LockFreeThreadPool.h"

#include <cstring>

namespace rtp_llm {

NormalCacheStore::~NormalCacheStore() {
    if (metrics_reporter_) {
        metrics_reporter_->stop();
        metrics_reporter_.reset();
    }

    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_.reset();
    }

    messager_client_.reset();
    messager_server_.reset();
    request_block_buffer_store_.reset();
    FT_LOG_INFO("destory cache store done");
}

std::shared_ptr<NormalCacheStore> NormalCacheStore::createNormalCacheStore(const CacheStoreInitParams& params) {
    std::shared_ptr<NormalCacheStore> normal_cache_store(new NormalCacheStore);
    if (normal_cache_store && normal_cache_store->init(params)) {
        return normal_cache_store;
    }
    return nullptr;
}

bool NormalCacheStore::init(const CacheStoreInitParams& params) {
    params_ = params;

    if (params_.memory_util != nullptr) {
        memory_util_ = params.memory_util;
    } else {
        memory_util_ = createMemoryUtilImpl(params_.rdma_mode);
    }

    request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_, params.device);

    metrics_reporter_ = std::make_shared<CacheStoreMetricsReporter>();
    if (params.enable_metric) {
        if (!metrics_reporter_->init()) {
            FT_LOG_WARNING("normal cache store init metrics reporter failed");
        }
    }
    timer_manager_ = std::make_shared<arpc::TimerManager>();
    if (!timer_manager_) {
        FT_LOG_INFO("normal cache store init failed, timer init failed");
        return false;
    }
    messager_client_ = std::move(createMessagerClient(memory_util_));
    if (!messager_client_ || !messager_client_->init(params.enable_metric)) {
        FT_LOG_ERROR("normal cache store init failed : init messager client failed");
        return false;
    }

    messager_server_ =
        std::move(createMessagerServer(memory_util_, request_block_buffer_store_, metrics_reporter_, timer_manager_));
    if (!messager_server_
        || !messager_server_->init(params.listen_port, params.rdma_listen_port, params.enable_metric)) {
        FT_LOG_ERROR("normal cache store init failed : init messager server failed");
        return false;
    }

    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        params.thread_count, params.queue_size, nullptr, "NormalCacheStoreTask");
    if (!thread_pool_ || !thread_pool_->start()) {
        FT_LOG_ERROR("normal cache store init failed : init thread pool failed");
        return false;
    }

    FT_LOG_INFO("normal cache store init done");
    return true;
}

void NormalCacheStore::store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                             CacheStoreStoreDoneCallback                callback) {
    if (request_block_buffer == nullptr || !request_block_buffer->isValid()) {
        FT_LOG_WARNING("normal cache store call store failed, request block is invalid");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    auto collector = metrics_reporter_->makeClientStoreMetricsCollector(request_block_buffer->getBlocksCount());
    // task 只在threadpool中运行, threadpool退出前会清理所有running task, 用this是安全的
    auto task = [this, request_block_buffer, callback, collector]() {
        this->runStoreTask(request_block_buffer, callback, collector);
    };

    if (thread_pool_->pushTask(task) != autil::ThreadPoolBase::ERROR_NONE) {
        FT_LOG_WARNING("normal cache store push store task to thread pool failed");
        callback(false, CacheStoreErrorCode::PushWorkerItemFailed);
        return;
    }
}

std::shared_ptr<StoreContext>
NormalCacheStore::storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                               int64_t                                                 timeout_ms) {
    if (request_block_buffers.empty()) {
        return nullptr;
    }
    auto store_context = std::make_shared<StoreContext>(shared_from_this());
    store_context->store(request_block_buffers, timeout_ms);
    return store_context;
}

void NormalCacheStore::debugInfo() {
    request_block_buffer_store_->debugInfo();
}

void NormalCacheStore::runStoreTask(const std::shared_ptr<RequestBlockBuffer>&                    request_block_buffer,
                                    CacheStoreStoreDoneCallback                                   callback,
                                    const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector) {
    // store to local
    CacheStoreClientStoreMetricsCollector::markStoreLocalBegin(collector);
    auto ret = request_block_buffer_store_->setRequestBlockBuffer(request_block_buffer);
    CacheStoreClientStoreMetricsCollector::markStoreLocalEnd(collector);

    CacheStoreClientStoreMetricsCollector::markEnd(collector, ret);
    if (!ret) {
        FT_LOG_WARNING("normal cache store run store task failed, request id is %s",
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
    if (request_block_buffer == nullptr || !request_block_buffer->isValid() || ip.empty()) {
        FT_LOG_WARNING("normal cache store run load failed, invalid params");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (port == 0 || (memory_util_->isRdmaMode() && rdma_port == 0)) {
        FT_LOG_WARNING("normal cache store run load failed, port is 0");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    auto collector = metrics_reporter_->makeClientLoadMetricsCollector(request_block_buffer->getBlocksCount());
    auto task      = [this,
                 request_block_buffer,
                 callback,
                 ip,
                 port,
                 rdma_port,
                 timeout_ms,
                 collector,
                 partition_count,
                 partition_id]() {
        this->runLoadTask(
            request_block_buffer, callback, ip, port, rdma_port, timeout_ms, collector, partition_count, partition_id);
    };

    if (thread_pool_->pushTask(task) != autil::ThreadPoolBase::ERROR_NONE) {
        FT_LOG_WARNING("normal cache store push load task for request id [%s] to thread pool failed",
                       request_block_buffer->getRequestId().c_str());
        callback(false, CacheStoreErrorCode::PushWorkerItemFailed);
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

    CacheStoreClientLoadMetricsCollector::markLoadRequestBegin(collector, request_block_buffer->getBlocksCount());
    messager_client_->load(
        ip, port, rdma_port, request_block_buffer, callback, timeout_ms, collector, partition_count, partition_id);
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
    if (request_block_buffers.empty() || ip.empty()) {
        return nullptr;
    }

    auto load_context = std::make_shared<LoadContext>(shared_from_this(), memory_util_->isRdmaMode());
    load_context->load(
        request_block_buffers, ip, port, rdma_port, timeout_ms, check_cancel_func, partition_count, partition_id);
    return load_context;
}

void NormalCacheStore::markRequestEnd(const std::string& requestid) {
    request_block_buffer_store_->delRequestBlockBuffer(requestid);
}

const std::shared_ptr<MemoryUtil>& NormalCacheStore::getMemoryUtil() const {
    return memory_util_;
}

const std::shared_ptr<RequestBlockBufferStore>& NormalCacheStore::getRequestBlockBufferStore() const {
    return request_block_buffer_store_;
}

}  // namespace rtp_llm
