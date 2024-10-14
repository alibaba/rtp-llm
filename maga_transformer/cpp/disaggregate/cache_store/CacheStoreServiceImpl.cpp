#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"
#include "autil/TimeUtility.h"
#include <unistd.h>

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, CacheStoreServiceImpl);
AUTIL_LOG_SETUP(rtp_llm, TcpCacheStoreServiceImpl);

CacheStoreServiceImpl::CacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&              memory_util,
                                             const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                             const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter):
    memory_util_(memory_util),
    request_block_buffer_store_(request_block_buffer_store),
    metrics_reporter_(metrics_reporter) {}

void CacheStoreServiceImpl::load(::google::protobuf::RpcController* controller,
                                 const ::CacheLoadRequest*          request,
                                 ::CacheLoadResponse*               response,
                                 ::google::protobuf::Closure*       done) {
    if (request_block_buffer_store_ == nullptr) {
        AUTIL_LOG(WARN,
                  "cache store service has no block cache store, request failed, request from [%s:%u], request id [%s]",
                  request->client_ip().c_str(),
                  request->client_port(),
                  request->requestid().c_str());
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }
    loadImpl(controller, request, response, done);
}

TcpCacheStoreServiceImpl::TcpCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&              memory_util,
                                             const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                                             const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter): CacheStoreServiceImpl(memory_util, request_block_buffer_store, metrics_reporter) {}

void TcpCacheStoreServiceImpl::loadImpl(::google::protobuf::RpcController* controller,
                                          const ::CacheLoadRequest*          request,
                                          ::CacheLoadResponse*               response,
                                          ::google::protobuf::Closure*       done) {
    int64_t start_time_us        = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t request_send_cost_us = start_time_us - request->request_send_start_time_us();
    auto    collector            = metrics_reporter_->makeServerLoadMetricsCollector(
        request->blocks_size(), request->blocks_size() ? request->blocks(0).len() : 0, request_send_cost_us);

    auto retcode = loadTcpBlocks(request, response, collector);
    if (retcode != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        AUTIL_LOG(WARN,
                  "cache store service load failed, request %s from [%s:%u]",
                  request->requestid().c_str(),
                  request->client_ip().c_str(),
                  request->client_port());
        CacheStoreServerLoadMetricsCollector::markEnd(collector, false);
        response->clear_blocks();
        response->set_error_code(retcode);
        done->Run();
        return;
    }

    CacheStoreServerLoadMetricsCollector::markEnd(collector, true);
    response->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
    response->set_response_send_start_time_us(autil::TimeUtility::currentTimeInMicroSeconds());
    response->set_direct_write_response(false);

    done->Run();
}

KvCacheStoreServiceErrorCode
TcpCacheStoreServiceImpl::loadTcpBlocks(const ::CacheLoadRequest*                                    request,
                                     ::CacheLoadResponse*                                         response,
                                     const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector) {
    std::map<std::string, BlockBufferInfo> unloaded_blocks;
    for (int i = 0; i < request->blocks_size(); i++) {
        unloaded_blocks[request->blocks(i).key()] = request->blocks(i);
    }

    auto request_send_start_time = request->request_send_start_time_us();
    auto start_time_ms           = autil::TimeUtility::currentTimeInMilliSeconds();
    bool first_block             = true;
    while (!unloaded_blocks.empty()
           && autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms < request->timeout_ms()) {
        for (auto it = unloaded_blocks.begin(); it != unloaded_blocks.end();) {
            auto block = request_block_buffer_store_->getBlockBuffer(request->requestid(), it->first);
            if (block == nullptr) {
                it++;
                continue;
            }
            if (first_block) {
                CacheStoreServerLoadMetricsCollector::setFirstBlockCostUs(
                    collector, autil::TimeUtility::currentTimeInMicroSeconds() - request_send_start_time);
            }
            auto* block_info = response->add_blocks();
            block_info->set_key(it->first);
            block_info->set_len(block->len);

            auto block_content = block_info->mutable_content();
            block_content->assign(
                std::shared_ptr<const char>(block->addr, reinterpret_cast<const char*>(block->addr.get())),
                size_t(block->len));

            it = unloaded_blocks.erase(it);
        }
        usleep(100);
    }
    return unloaded_blocks.empty() ? KvCacheStoreServiceErrorCode::EC_SUCCESS :
                                     KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER;
}

}  // namespace rtp_llm
