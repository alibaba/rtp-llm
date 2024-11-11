#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include <atomic>

#include "src/fastertransformer/utils/logger.h"

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

CacheStoreServiceImplContext::~CacheStoreServiceImplContext() {}

void CacheStoreServiceImplContext::loadBlockOnTcp(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    if (done_run_) {
        // already done run, most likely timeout, no need load
        return;
    }

    if (!ok) {
        // request been canceled in cache store, just failed
        runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
        return;
    }

    for (auto& block : blocks) {
        auto unloaded_block_info = getAndEraseUnLoadedBlock(block->key);
        if (unloaded_block_info == nullptr) {
            // block already loaded
            continue;
        }

        if (unloaded_block_info->len() != block->len) {
            FT_LOG_WARNING(
                "cache store service load block not match exepct block len, key: %s, len %d vs %d, peer is %s",
                block->key.c_str(),
                unloaded_block_info->len(),
                block->len,
                peer_ip_.c_str());
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
            return;
        }

        if (!writeResponseBlock(block, unloaded_block_info)) {
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
            return;
        }

        if (++write_cnt_ == 1) {
            CacheStoreServerLoadMetricsCollector::setFirstBlockCostUs(
                collector_, autil::TimeUtility::currentTimeInMicroSeconds() - request_send_start_time_us_);
        }
    }

    if (write_cnt_ == total_block_count_) {
        runSuccess(false);
    }
}

std::shared_ptr<BlockBufferInfo> CacheStoreServiceImplContext::getAndEraseUnLoadedBlock(const std::string& block_key) {
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    auto                                it = unloaded_blocks_.find(block_key);
    if (it == unloaded_blocks_.end()) {
        return nullptr;
    }

    auto block_info = it->second;
    unloaded_blocks_.erase(it);
    return block_info;
}

bool CacheStoreServiceImplContext::writeResponseBlock(const std::shared_ptr<BlockBuffer>&     block,
                                                      const std::shared_ptr<BlockBufferInfo>& peer_block) {
    std::lock_guard<std::mutex> lock(response_mutex_);
    if (response_ == nullptr) {
        // try write response while already done
        return false;
    }

    auto* block_info = response_->add_blocks();
    block_info->set_key(block->key);
    block_info->set_len(block->len);
    auto block_content = block_info->mutable_content();
    block_content->assign(std::shared_ptr<const char>(block->addr, reinterpret_cast<const char*>(block->addr.get())),
                          size_t(block->len));
    return true;
}

void CacheStoreServiceImplContext::runSuccess(bool direct_write) {
    FT_LOG_DEBUG("run success");
    bool expected = false;
    if (!done_run_.compare_exchange_strong(expected, true)) {
        return;
    }

    stopTimer();

    CacheStoreServerLoadMetricsCollector::markEnd(collector_, true);

    // run success, set response
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (response_ != nullptr) {
            response_->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
            response_->set_response_send_start_time_us(autil::TimeUtility::currentTimeInMicroSeconds());
            response_->set_direct_write_response(direct_write);
            response_ = nullptr;
        }
    }

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
        FT_LOG_WARNING("cache store service load failed, request %s from [%s], error code is %d, block buffer is %s",
                       request_id_.c_str(),
                       peer_ip_.c_str(),
                       error_code,
                       request_block_buffer_store->debugInfoOnRequest(request_id_).c_str());
    } else {
        FT_LOG_WARNING("cache store service load failed, request %s from [%s], error code is %d, block buffer is null",
                       request_id_.c_str(),
                       peer_ip_.c_str(),
                       error_code);
    }

    CacheStoreServerLoadMetricsCollector::markEnd(collector_, false);

    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (response_ != nullptr) {
            response_->clear_blocks();
            response_->set_error_code(error_code);
            response_ = nullptr;
        }
    }

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