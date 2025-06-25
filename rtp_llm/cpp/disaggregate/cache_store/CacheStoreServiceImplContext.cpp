#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include <atomic>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheStoreServiceImplContext::CacheStoreServiceImplContext(
    const CacheLoadRequest*                                      request,
    CacheLoadResponse*                                           response,
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
    ::google::protobuf::Closure*                                 done,
    const std::shared_ptr<RequestBlockBufferStore>&              request_block_buffer_store):
    request_(request),
    request_send_start_time_us_(request->request_send_start_time_us()),
    total_block_count_(request_->blocks_size()),
    request_id_(request_->requestid()),
    peer_ip_(request->client_ip()),
    partition_count_(request->partition_count() == 0 ? 1 : request->partition_count()),  // compatible with old version
    partition_id_(request->partition_id()),
    response_(response),
    collector_(collector),
    done_(done),
    request_block_buffer_store_(request_block_buffer_store),
    write_cnt_(0) {
    // init set unloaded blocks
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    for (int i = 0; i < request_->blocks_size(); i++) {
        unloaded_blocks_[request_->blocks(i).key()] = std::make_shared<BlockBufferInfo>(request_->blocks(i));
    }
}

std::shared_ptr<BlockBufferInfo> CacheStoreServiceImplContext::getAndEraseUnLoadedBlock(const std::string& block_key) {
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    auto                                it = unloaded_blocks_.find(block_key);
    if (it == unloaded_blocks_.end()) {
        return nullptr;
    }
    if (unloaded_blocks_.size() == total_block_count_) {
        collector_->markFirstBlockReady();
    }

    auto block_info = it->second;
    unloaded_blocks_.erase(it);

    if (unloaded_blocks_.empty()) {
        collector_->markAllBlocksReady();
    }
    return block_info;
}

void CacheStoreServiceImplContext::runSuccess(bool direct_write) {
    RTP_LLM_LOG_DEBUG("request [%s] run success", request_id_.c_str());
    bool expected = false;
    if (!done_run_.compare_exchange_strong(expected, true)) {
        return;
    }

    stopTimer();

    // run success, set response
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (response_ != nullptr) {
            response_->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
            response_->set_response_send_start_time_us(currentTimeUs());
            response_->set_direct_write_response(direct_write);
            response_ = nullptr;
        }
    }

    collector_->markEnd(true);
    // call callback
    if (done_) {
        done_->Run();
        done_ = nullptr;
    }
}

void CacheStoreServiceImplContext::runFailed(KvCacheStoreServiceErrorCode error_code) {
    bool expected = false;
    if (!done_run_.compare_exchange_strong(expected, true)) {
        return;
    }

    stopTimer();

    auto request_block_buffer_store = request_block_buffer_store_.lock();
    if (request_block_buffer_store) {
        RTP_LLM_LOG_WARNING(
            "cache store service load failed, request %s from [%s], error code is %d, block buffer is %s",
            request_id_.c_str(),
            peer_ip_.c_str(),
            error_code,
            request_block_buffer_store->debugInfoOnRequest(request_id_).c_str());
    } else {
        RTP_LLM_LOG_WARNING(
            "cache store service load failed, request %s from [%s], error code is %d, block buffer is null",
            request_id_.c_str(),
            peer_ip_.c_str(),
            error_code);
    }

    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (response_ != nullptr) {
            response_->clear_blocks();
            response_->set_error_code(error_code);
            response_ = nullptr;
        }
    }

    collector_->markEnd(false);
    if (done_) {
        done_->Run();
        done_ = nullptr;
    }
}

void CacheStoreServiceImplContext::stopTimer() {
    if (auto timer_shared_ptr = timer_.lock()) {
        timer_shared_ptr->stop();
        timer_shared_ptr.reset();
    }
}

}  // namespace rtp_llm