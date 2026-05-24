#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include "autil/LockFreeThreadPool.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>

namespace rtp_llm {

namespace {

bool pdDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_PD_DEBUG");
    return env != nullptr && std::string(env) == "1";
}

std::string summarizeBlocks(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, size_t limit = 3) {
    if (request_block_buffer == nullptr) {
        return "null";
    }
    std::ostringstream oss;
    oss << "request_id=" << request_block_buffer->getRequestId()
        << " request_key=" << request_block_buffer->getRequestKey()
        << " blocks=" << request_block_buffer->getBlocksCount() << " bytes=" << request_block_buffer->getBlocksSize();
    auto   blocks = request_block_buffer->getBlocks();
    size_t idx    = 0;
    oss << " sample_keys=[";
    for (const auto& [key, block] : blocks) {
        if (idx++ >= limit) {
            oss << "...";
            break;
        }
        if (idx > 1) {
            oss << ",";
        }
        oss << key << ":" << (block == nullptr ? 0 : block->len)
            << (block != nullptr && block->gpu_mem ? ":gpu" : ":cpu");
    }
    oss << "]";
    return oss.str();
}

std::vector<std::shared_ptr<RequestBlockBuffer>>
chunkTcpLoadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers) {
    constexpr size_t kMaxChunkBytes  = 48ULL * 1024ULL * 1024ULL;
    constexpr size_t kMaxChunkLayers = 4;

    std::vector<std::shared_ptr<RequestBlockBuffer>> chunked;
    chunked.reserve(request_block_buffers.size());

    std::vector<std::shared_ptr<BlockBuffer>> pending_blocks;
    std::string                               pending_request_id;
    std::string                               first_request_key;
    std::string                               last_request_key;
    size_t                                    pending_bytes  = 0;
    size_t                                    pending_layers = 0;

    auto flush = [&]() {
        if (pending_blocks.empty()) {
            return;
        }
        std::string chunk_key = first_request_key;
        if (last_request_key != first_request_key) {
            chunk_key += "..";
            chunk_key += last_request_key;
        }
        auto combined = std::make_shared<RequestBlockBuffer>(pending_request_id, chunk_key);
        combined->addBlocks(pending_blocks);
        chunked.push_back(std::move(combined));
        pending_blocks.clear();
        pending_request_id.clear();
        first_request_key.clear();
        last_request_key.clear();
        pending_bytes  = 0;
        pending_layers = 0;
    };

    for (const auto& request_block_buffer : request_block_buffers) {
        if (request_block_buffer == nullptr || request_block_buffer->getBlocksCount() == 0) {
            flush();
            if (request_block_buffer != nullptr) {
                chunked.push_back(request_block_buffer);
            }
            continue;
        }

        const auto& request_id  = request_block_buffer->getRequestId();
        const auto& request_key = request_block_buffer->getRequestKey();
        const auto  block_bytes = request_block_buffer->getBlocksSize();
        if (!pending_blocks.empty()
            && (request_id != pending_request_id || pending_layers >= kMaxChunkLayers
                || pending_bytes + block_bytes > kMaxChunkBytes)) {
            flush();
        }

        if (pending_blocks.empty()) {
            pending_request_id = request_id;
            first_request_key  = request_key;
        }
        last_request_key = request_key;

        auto blocks = request_block_buffer->getBlocks();
        pending_blocks.reserve(pending_blocks.size() + blocks.size());
        for (auto& [_, block] : blocks) {
            pending_blocks.push_back(block);
        }
        pending_bytes += block_bytes;
        ++pending_layers;
    }
    flush();

    return chunked;
}

size_t tcpLoadMaxInflightChunks() {
    // This limit is per load context (one decode request loading from one
    // prefill peer). In PD with decode DP x prefill TP, total server-side TCP
    // load fanout is dp_size * request_concurrency * this value per prefill
    // rank. Keep the default at one rolling chunk per context so large
    // 20-40MiB TCP responses do not overrun the prefill RPC worker pool.
    constexpr size_t kDefaultMaxInflightChunks = 1;
    const char*      env                       = std::getenv("CACHE_STORE_TCP_LOAD_MAX_INFLIGHT_CHUNKS");
    if (env == nullptr || std::strlen(env) == 0) {
        return kDefaultMaxInflightChunks;
    }
    char* end = nullptr;
    auto  val = std::strtoull(env, &end, 10);
    if (end == env || val == 0) {
        return kDefaultMaxInflightChunks;
    }
    return static_cast<size_t>(val);
}

}  // namespace

NormalCacheStore::~NormalCacheStore() {
    if (thread_pool_) {
        thread_pool_close_ = true;
        thread_pool_->stop();
        thread_pool_.reset();
    }

    request_block_buffer_store_->stop();
    messager_.reset();
    request_block_buffer_store_.reset();
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
    params_ = params;

    if (params_.memory_util != nullptr) {
        memory_util_ = params.memory_util;
    } else {
        memory_util_ = createMemoryUtilImpl(params_.rdma_mode);
    }

    request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_);

    // always has metric
    metrics_reporter_ = params.metrics_reporter;

    messager_ = createMessager(memory_util_, request_block_buffer_store_, metrics_reporter_);
    MessagerInitParams messager_init_params;
    messager_init_params.server_port                  = params.listen_port;
    messager_init_params.rdma_server_port             = params.rdma_listen_port;
    messager_init_params.rdma_connect_timeout_ms      = params.rdma_connect_timeout_ms;
    messager_init_params.rdma_qp_count_per_connection = params.rdma_qp_count_per_connection;
    messager_init_params.rdma_io_thread_count         = params.rdma_io_thread_count;
    messager_init_params.rdma_worker_thread_count     = params.rdma_worker_thread_count;
    messager_init_params.io_thread_count              = params.messager_io_thread_count;
    messager_init_params.worker_thread_count          = params.messager_worker_thread_count;
    messager_init_params.worker_queue_size            = params.queue_size;

    if (!messager_->init(messager_init_params)) {
        RTP_LLM_LOG_ERROR("normal cache store init failed : init messager failed");
        return false;
    }

    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        params.thread_count, params.queue_size, nullptr, "NormalCacheStoreTask");
    if (!thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("normal cache store init failed : init thread pool failed");
        return false;
    }

    auto check_task_readiness = [this]() {
        while (!thread_pool_close_) {
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

    RTP_LLM_LOG_INFO("normal cache store init done, thread pool thread count is %d", params.thread_count);
    return true;
}

void NormalCacheStore::store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                             CacheStoreStoreDoneCallback                callback) {
    if (request_block_buffer == nullptr || !request_block_buffer->isValid()) {
        RTP_LLM_LOG_WARNING("normal cache store call store failed, request block is invalid");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][NORMAL_CACHE_STORE_STORE_ENQUEUE] %s",
                         summarizeBlocks(request_block_buffer).c_str());
    }

    auto collector = std::make_shared<CacheStoreStoreMetricsCollector>(
        metrics_reporter_, request_block_buffer->getBlocksCount(), request_block_buffer->getBlocksSize());
    // task 只在threadpool中运行, threadpool退出前会清理所有running task, 用this是安全的
    auto task = [this, request_block_buffer, callback, collector]() {
        this->runStoreTask(request_block_buffer, callback, collector);
    };

    std::unique_lock<std::shared_mutex> lock(store_tasks_mutex_);
    store_tasks_[request_block_buffer] = {callback, task};
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

void NormalCacheStore::runStoreTask(const std::shared_ptr<RequestBlockBuffer>&              request_block_buffer,
                                    CacheStoreStoreDoneCallback                             callback,
                                    const std::shared_ptr<CacheStoreStoreMetricsCollector>& collector) {
    // store to local
    collector->markTaskRun();

    // In local cache-store mode, RequestBlockBufferStore may copy GPU blocks
    // into pinned host buffers. The blocks are produced on the model stream, so
    // wait for the event recorded by WriteCacheStoreOp before the copy starts.
    auto    event         = request_block_buffer->getEvent();
    int64_t event_wait_ms = 0;
    if (event) {
        auto wait_begin = std::chrono::steady_clock::now();
#if USE_PPU
        while (!event->query()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
#else
        event->synchronize();
#endif
        event_wait_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wait_begin)
                .count();
    }

    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][NORMAL_CACHE_STORE_RUN_STORE_BEGIN] event=%d event_wait_ms=%ld %s",
                         static_cast<int>(event != nullptr),
                         event_wait_ms,
                         summarizeBlocks(request_block_buffer).c_str());
    }
    auto ret = request_block_buffer_store_->setRequestBlockBuffer(request_block_buffer);
    collector->markEnd(ret);

    if (!ret) {
        RTP_LLM_LOG_WARNING("normal cache store run store task failed, request id is %s",
                            request_block_buffer->getRequestId().c_str());
        if (pdDebugEnabled()) {
            RTP_LLM_LOG_WARNING("[PD_DEBUG][NORMAL_CACHE_STORE_RUN_STORE_END] ret=0 %s",
                                summarizeBlocks(request_block_buffer).c_str());
        }
        callback(false, CacheStoreErrorCode::StoreFailed);
        return;
    }
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][NORMAL_CACHE_STORE_RUN_STORE_END] ret=1 %s",
                         summarizeBlocks(request_block_buffer).c_str());
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
        RTP_LLM_LOG_WARNING("normal cache store run load failed, invalid params");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (port == 0 || (memory_util_->isRdmaMode() && rdma_port == 0)) {
        RTP_LLM_LOG_WARNING("normal cache store run load failed, port is 0");
        callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }

    if (request_block_buffer->getBlocksCount() == 0) {
        callback(true, CacheStoreErrorCode::None);
        return;
    }

    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][NORMAL_CACHE_STORE_LOAD_ENQUEUE] peer=%s:%u rdma_port=%u timeout_ms=%u "
                         "partition_count=%d partition_id=%d %s",
                         ip.c_str(),
                         port,
                         rdma_port,
                         timeout_ms,
                         partition_count,
                         partition_id,
                         summarizeBlocks(request_block_buffer).c_str());
    }

    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(
        metrics_reporter_, request_block_buffer->getBlocksCount(), request_block_buffer->getBlocksSize());

    auto task = [this,
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
        RTP_LLM_LOG_WARNING("normal cache store push load task for request id [%s] to thread pool failed",
                            request_block_buffer->getRequestId().c_str());
        collector->markEnd(false);
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
    collector->markTaskRun();
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][NORMAL_CACHE_STORE_RUN_LOAD] peer=%s:%u rdma_port=%u timeout_ms=%u "
                         "partition_count=%d partition_id=%d %s",
                         ip.c_str(),
                         port,
                         rdma_port,
                         timeout_ms,
                         partition_count,
                         partition_id,
                         summarizeBlocks(request_block_buffer).c_str());
    }
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
    if (request_block_buffers.empty() || ip.empty()) {
        return nullptr;
    }

    std::vector<std::shared_ptr<RequestBlockBuffer>> tcp_chunked_buffers;
    const auto*                                      load_buffers = &request_block_buffers;
    if (!memory_util_->isRdmaMode()) {
        tcp_chunked_buffers = chunkTcpLoadBuffers(request_block_buffers);
        if (!tcp_chunked_buffers.empty()) {
            load_buffers = &tcp_chunked_buffers;
        }
        if (pdDebugEnabled() && load_buffers->size() < request_block_buffers.size()) {
            RTP_LLM_LOG_INFO("normal cache store tcp load chunked request buffers from %zu to %zu",
                             request_block_buffers.size(),
                             load_buffers->size());
        }
    }

    auto load_context = std::make_shared<LoadContext>(shared_from_this(), memory_util_->isRdmaMode());
    if (!memory_util_->isRdmaMode()) {
        const auto max_inflight_chunks = tcpLoadMaxInflightChunks();
        if (max_inflight_chunks > 0 && load_buffers->size() > max_inflight_chunks) {
            load_context->setMaxInflightRequestCount(max_inflight_chunks);
            if (pdDebugEnabled()) {
                RTP_LLM_LOG_INFO("normal cache store tcp load max inflight chunks per context set to %zu",
                                 max_inflight_chunks);
            }
        }
    }
    load_context->load(
        *load_buffers, ip, port, rdma_port, timeout_ms, check_cancel_func, partition_count, partition_id);
    return load_context;
}

std::shared_ptr<RemoteStoreTask>
NormalCacheStore::submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&                    request,
                                        const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                        RemoteStoreTask::CheckCancelFunc check_cancel_func) {
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
            if (!ok) {
                RTP_LLM_LOG_WARNING("normal cache store run store task watch func failed, request id is %s",
                                    request_id.c_str());
                return;
            }

            auto task = weak_task.lock();
            if (!task) {
                RTP_LLM_LOG_DEBUG("task has been released, request id is %s", request_id.c_str());
                return;
            }

            auto transfer_request = task->makeAvailableRequest(blocks);

            if (transfer_request == nullptr) {
                RTP_LLM_LOG_WARNING("normal cache store make available request failed, request id is %s",
                                    request_id.c_str());
                return;
            }

            this->messager_->transfer(transfer_request);
        };

    this->request_block_buffer_store_->setRequestBlockBufferWatchFunc(request_id, std::move(watchFunc));
    return std::dynamic_pointer_cast<RemoteStoreTask>(task);
}

void NormalCacheStore::releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task) {
    std::unique_lock<std::shared_mutex> lock(remote_store_tasks_mutex_);
    auto&                               tasks = remote_store_tasks_[task->getRequestId()];
    tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
}

void NormalCacheStore::markRequestEnd(const std::string& requestid) {
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][NORMAL_CACHE_STORE_MARK_REQUEST_END] request_id=%s current=%s",
                         requestid.c_str(),
                         request_block_buffer_store_->debugInfoOnRequest(requestid).c_str());
    }
    request_block_buffer_store_->delRequestBlockBuffer(requestid);
}

bool NormalCacheStore::regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) {
    return request_block_buffer_store_->regUserBuffers(buffers);
}

std::shared_ptr<BlockBuffer> NormalCacheStore::findUserBuffer(const std::string& buffer_key) {
    return request_block_buffer_store_->findUserBuffer(buffer_key);
}

const std::shared_ptr<MemoryUtil>& NormalCacheStore::getMemoryUtil() const {
    return memory_util_;
}

const std::shared_ptr<RequestBlockBufferStore>& NormalCacheStore::getRequestBlockBufferStore() const {
    return request_block_buffer_store_;
}

}  // namespace rtp_llm
