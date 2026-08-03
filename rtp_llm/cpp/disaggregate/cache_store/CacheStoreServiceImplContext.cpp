#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include <atomic>
#include <cstdlib>
#include <sstream>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

const bool kPdDebugEnabled = []() {
    const char* env = std::getenv("RTP_LLM_PD_DEBUG");
    return env != nullptr && std::string(env) == "1";
}();

bool pdDebugEnabled() {
    return kPdDebugEnabled;
}

}  // namespace

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
    if (pdDebugEnabled()) {
        std::ostringstream sample;
        int                sample_count = 0;
        for (const auto& [key, block] : unloaded_blocks_) {
            if (sample_count++ >= 5) {
                sample << ",...";
                break;
            }
            if (sample_count > 1) {
                sample << ",";
            }
            sample << key << ":" << (block == nullptr ? 0 : block->len());
        }
        RTP_LLM_LOG_INFO("[PD_DEBUG][CACHE_SERVICE_CONTEXT_INIT] request_id=%s peer=%s total_blocks=%u "
                         "partition_count=%d partition_id=%d send_start_us=%ld sample_unloaded=[%s]",
                         request_id_.c_str(),
                         peer_ip_.c_str(),
                         total_block_count_,
                         partition_count_,
                         partition_id_,
                         request_send_start_time_us_,
                         sample.str().c_str());
    }
}

bool CacheStoreServiceImplContext::computeSourceRanges(const std::shared_ptr<BlockBuffer>&     block,
                                                       const std::shared_ptr<BlockBufferInfo>& peer_block,
                                                       std::vector<SourceRange>&               ranges) const {
    const int32_t count = peer_block->partition_count() > 0 ? peer_block->partition_count() : partition_count_;
    const int32_t id    = peer_block->partition_count() > 0 ? peer_block->partition_id() : partition_id_;

    auto reject = [&](const char* reason) {
        RTP_LLM_LOG_WARNING("cache store service load block not match expect block len, key: %s, len %u vs stored %u "
                            "(partition %d/%d, kv_halves=%d): %s, peer is %s",
                            block->key.c_str(),
                            peer_block->len(),
                            block->len,
                            id,
                            count,
                            static_cast<int>(peer_block->partition_kv_halves()),
                            reason,
                            peer_ip_.c_str());
        return false;
    };

    if (count <= 0 || id < 0 || id >= count) {
        return reject("partition id out of range");
    }
    if (block->len % static_cast<uint32_t>(count) != 0) {
        return reject("stored len not divisible by partition count");
    }
    const uint32_t slice_len = block->len / static_cast<uint32_t>(count);
    if (slice_len != peer_block->len()) {
        return reject("peer len is not the requested partition of the stored block");
    }

    ranges.clear();
    if (!peer_block->partition_kv_halves() || count == 1) {
        ranges.push_back({slice_len * static_cast<uint32_t>(id), slice_len});
        return true;
    }

    // [K region][V region]: the KV-head partition of the whole block is the same
    // partition taken inside each region, so the slice is two disjoint ranges.
    if (block->len % 2 != 0) {
        return reject("kv_halves block has odd stored len");
    }
    const uint32_t half = block->len / 2;
    if (half % static_cast<uint32_t>(count) != 0) {
        return reject("kv_halves region not divisible by partition count");
    }
    const uint32_t sub = half / static_cast<uint32_t>(count);
    ranges.push_back({sub * static_cast<uint32_t>(id), sub});
    ranges.push_back({half + sub * static_cast<uint32_t>(id), sub});
    return true;
}

std::shared_ptr<BlockBufferInfo> CacheStoreServiceImplContext::getAndEraseUnLoadedBlock(const std::string& block_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    auto                                it = unloaded_blocks_.find(block_key);
    if (it == unloaded_blocks_.end()) {
        return nullptr;
    }
    if (unloaded_blocks_.size() == total_block_count_) {
        collector_->markFirstBlockReady();
        if (pdDebugEnabled()) {
            RTP_LLM_LOG_INFO("[PD_DEBUG][CACHE_SERVICE_FIRST_BLOCK_READY] request_id=%s key=%s total_blocks=%u",
                             request_id_.c_str(),
                             block_key.c_str(),
                             total_block_count_);
        }
    }

    auto block_info = it->second;
    unloaded_blocks_.erase(it);

    if (unloaded_blocks_.empty()) {
        collector_->markAllBlocksReady();
        if (pdDebugEnabled()) {
            RTP_LLM_LOG_INFO("[PD_DEBUG][CACHE_SERVICE_ALL_BLOCKS_READY] request_id=%s total_blocks=%u",
                             request_id_.c_str(),
                             total_block_count_);
        }
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

    collector_->markEnd(true);
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][CACHE_SERVICE_RUN_SUCCESS] request_id=%s peer=%s total_blocks=%u "
                         "write_cnt=%d direct_write=%d",
                         request_id_.c_str(),
                         peer_ip_.c_str(),
                         total_block_count_,
                         write_cnt_.load(),
                         static_cast<int>(direct_write));
    }
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

    if (pdDebugEnabled()) {
        std::shared_lock<std::shared_mutex> unloaded_lock(unloaded_blocks_mutex_);
        std::ostringstream                  sample;
        int                                 sample_count = 0;
        for (const auto& [key, block] : unloaded_blocks_) {
            if (sample_count++ >= 10) {
                sample << ",...";
                break;
            }
            if (sample_count > 1) {
                sample << ",";
            }
            sample << key << ":" << (block == nullptr ? 0 : block->len());
        }
        RTP_LLM_LOG_WARNING("[PD_DEBUG][CACHE_SERVICE_RUN_FAILED] request_id=%s peer=%s error_code=%d "
                            "total_blocks=%u write_cnt=%d unloaded=%zu sample_unloaded=[%s]",
                            request_id_.c_str(),
                            peer_ip_.c_str(),
                            error_code,
                            total_block_count_,
                            write_cnt_.load(),
                            unloaded_blocks_.size(),
                            sample.str().c_str());
    }

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
