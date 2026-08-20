#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include <atomic>
#include <sstream>

#include "rtp_llm/cpp/utils/K3PdTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

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
    k3_pd_trace_(k3PdTraceMarkedRequestId(request_id_)),
    response_(response),
    collector_(collector),
    done_(done),
    request_block_buffer_store_(request_block_buffer_store),
    write_cnt_(0) {
    // init set unloaded blocks
    {
        std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
        for (int i = 0; i < request_->blocks_size(); i++) {
            unloaded_blocks_[request_->blocks(i).key()] = std::make_shared<BlockBufferInfo>(request_->blocks(i));
        }
    }
    if (k3_pd_trace_) {
        RTP_LLM_LOG_INFO("[K3_PD_TRACE] event=cache_store_load_recv requestid=%s peer=%s expected_keys=[%s]",
                         request_id_.c_str(),
                         peer_ip_.c_str(),
                         unloadedKeysSummary().c_str());
    }
}

std::string CacheStoreServiceImplContext::unloadedKeysSummary() {
    std::shared_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    std::ostringstream                  oss;
    for (const auto& [key, info] : unloaded_blocks_) {
        oss << key << " ";
    }
    return oss.str();
}

std::shared_ptr<BlockBufferInfo> CacheStoreServiceImplContext::getAndEraseUnLoadedBlock(const std::string& block_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    auto                                it = unloaded_blocks_.find(block_key);
    if (it == unloaded_blocks_.end()) {
        if (k3_pd_trace_) {
            RTP_LLM_LOG_WARNING(
                "[K3_PD_TRACE] event=cache_store_load_miss requestid=%s peer=%s block_key=%s already_loaded_or_unknown",
                request_id_.c_str(),
                peer_ip_.c_str(),
                block_key.c_str());
        }
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
    RTP_LLM_PROFILE_FUNCTION();
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

    if (k3_pd_trace_) {
        RTP_LLM_LOG_INFO(
            "[K3_PD_TRACE] event=cache_store_load_success requestid=%s peer=%s write_cnt=%d total_blocks=%u",
            request_id_.c_str(),
            peer_ip_.c_str(),
            write_cnt_.load(),
            total_block_count_);
    }

    collector_->markEnd(true);
    // call callback
    if (done_) {
        done_->Run();
        done_ = nullptr;
    }
}

void CacheStoreServiceImplContext::runFailed(KvCacheStoreServiceErrorCode error_code) {
    RTP_LLM_PROFILE_FUNCTION();
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
    if (k3_pd_trace_) {
        RTP_LLM_LOG_WARNING("[K3_PD_TRACE] event=cache_store_load_failed requestid=%s peer=%s error_code=%d "
                            "write_cnt=%d total_blocks=%u remaining_unloaded_keys=[%s]",
                            request_id_.c_str(),
                            peer_ip_.c_str(),
                            error_code,
                            write_cnt_.load(),
                            total_block_count_,
                            unloadedKeysSummary().c_str());
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