#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImplContext.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaCacheStoreServiceImplContext::RdmaCacheStoreServiceImplContext(
    const CacheLoadRequest*                                      request,
    CacheLoadResponse*                                           response,
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
    ::google::protobuf::Closure*                                 done,
    const std::shared_ptr<RequestBlockBufferStore>&              request_block_buffer_store,
    const std::shared_ptr<RdmaConnection>&                       connection):
    CacheStoreServiceImplContext(request, response, collector, done, request_block_buffer_store),
    rdma_connection_(connection) {}

void RdmaCacheStoreServiceImplContext::loadBlockOnRdma(bool                                             ok,
                                                       const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    if (done_run_) {
        return;
    }

    if (!ok) {
        // request been finished, while load not finished
        runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
        return;
    }

    auto                                                   start_write_time_us = currentTimeUs();
    std::vector<std::shared_ptr<RdmaConnectionWriteBlock>> write_blocks;
    for (auto& block : blocks) {
        auto block_info = getAndEraseUnLoadedBlock(block->key);
        if (block_info == nullptr) {
            // block already loaded
            continue;
        }

        auto block_len = block->len / partition_count_;
        if (block_info->len() != block_len) {
            RTP_LLM_LOG_WARNING(
                "cache store service load block not match exepct block len, key: %s, len %d vs %d, peer is %s",
                block->key.c_str(),
                block_info->len(),
                block_len,
                peer_ip_.c_str());
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
            return;
        }

        std::shared_ptr<BlockBuffer> write_block = std::make_shared<BlockBuffer>(
            block->key,
            std::shared_ptr<void>(block, (void*)((int64_t)(block->addr.get()) + partition_id_ * block_len)),
            block_len,
            block->gpu_mem,
            block->adopted);

        auto callback = [shared_this = shared_from_this(),
                         block_key   = block->key,
                         block_len,
                         start_write_time_us,
                         collector = collector_](bool success) {
            collector->setWriteInfo(1, block_len, currentTimeUs() - start_write_time_us);
            std::dynamic_pointer_cast<RdmaCacheStoreServiceImplContext>(shared_this)->onWriteDone(success, block_key);
        };

        write_blocks.push_back(
            std::make_shared<RdmaConnectionWriteBlock>(write_block, block_info, std::move(callback)));
    }
    if (!write_blocks.empty()) {
        rdma_connection_->writeBlocks(write_blocks);
    }
}

void RdmaCacheStoreServiceImplContext::onWriteDone(bool success, const std::string& block_key) {
    if (!success) {
        RTP_LLM_LOG_WARNING(
            "cache store service write failed, block key: %s, peer is %s", block_key.c_str(), peer_ip_.c_str());
        runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE);
        return;
    }

    if (++write_cnt_ == total_block_count_) {
        runSuccess(true);
    }
}

}  // namespace rtp_llm