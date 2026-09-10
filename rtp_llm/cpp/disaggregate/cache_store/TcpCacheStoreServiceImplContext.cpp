#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void TcpCacheStoreServiceImplContext::loadBlockOnTcp(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
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

        std::vector<SourceRange> ranges;
        if (!computeSourceRanges(block, unloaded_block_info, ranges)) {
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
            return;
        }

        if (!writeResponseBlock(block, unloaded_block_info, ranges)) {
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
            return;
        }
        ++write_cnt_;
    }

    if (write_cnt_ == total_block_count_) {
        runSuccess(false);
    }
}

bool TcpCacheStoreServiceImplContext::writeResponseBlock(const std::shared_ptr<BlockBuffer>&     block,
                                                         const std::shared_ptr<BlockBufferInfo>& peer_block,
                                                         const std::vector<SourceRange>&         ranges) {
    std::lock_guard<std::mutex> lock(response_mutex_);
    if (response_ == nullptr) {
        // try write response while already done
        return false;
    }
    if (ranges.empty()) {
        return false;
    }

    const auto  block_len   = peer_block->len();
    const char* stored_base = reinterpret_cast<const char*>(block->addr.get());

    auto* block_info = response_->add_blocks();
    block_info->set_key(block->key);
    block_info->set_len(block_len);
    auto block_content = block_info->mutable_content();

    if (ranges.size() == 1) {
        block_content->assign(std::shared_ptr<const char>(block->addr, stored_base + ranges[0].offset),
                              size_t(ranges[0].len));
        return true;
    }

    // Disjoint source ranges (KV-head slice of an [K region][V region] block) have
    // to be concatenated, so unlike the single-range case this cannot alias the
    // stored block. TCP mode keeps stored blocks in pinned host memory
    // (RequestBlockBufferStore::isValidBlock rejects device pointers), so a plain
    // memcpy is safe here.
    auto joined = std::shared_ptr<char[]>(new char[block_len]);
    size_t written = 0;
    for (const auto& range : ranges) {
        memcpy(joined.get() + written, stored_base + range.offset, range.len);
        written += range.len;
    }
    if (written != block_len) {
        return false;
    }
    block_content->assign(std::shared_ptr<const char>(joined, joined.get()), size_t(block_len));
    return true;
}

}  // namespace rtp_llm