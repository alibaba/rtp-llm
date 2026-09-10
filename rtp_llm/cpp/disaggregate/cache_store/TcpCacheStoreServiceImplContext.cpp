#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <atomic>
#include <cstdlib>
#include <string>

namespace rtp_llm {

// [KVDIAG] opt-in (RTP_LLM_KV_DIAG=1). This callback is the ONLY thing that can turn
// a local publication into a reply to the waiting peer, and it has a silent no-reply
// path: it receives only the newly added blocks, skips quietly when a key is not in
// this context's unloaded set, and replies only once write_cnt_ reaches
// total_block_count_. A context that never sees its key therefore logs nothing,
// fails nothing and never answers - the peer just times out.
static bool kvDiagEnabled() {
    static const bool value = [] {
        const char* e = ::getenv("RTP_LLM_KV_DIAG");
        return e != nullptr && *e != '\0' && std::string(e) != "0";
    }();
    return value;
}

void TcpCacheStoreServiceImplContext::loadBlockOnTcp(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    if (kvDiagEnabled()) {
        static std::atomic<int> cb_budget{16000};
        if (cb_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            RTP_LLM_LOG_WARNING(
                "[KVDIAG-SVC-CB] ctx=%p done_run=%d ok=%d triggered_blocks=%zu total_expected=%u written_so_far=%d "
                "partition_count=%d first=%s",
                (void*)this,
                (int)done_run_,
                (int)ok,
                blocks.size(),
                total_block_count_,
                (int)write_cnt_,
                (int)partition_count_,
                blocks.empty() ? "<none>" : blocks[0]->key.c_str());
        }
    }
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

        if (unloaded_block_info->len() != block->len / partition_count_) {
            RTP_LLM_LOG_WARNING(
                "cache store service load block not match expect block len, key: %s, len %d vs %d, peer is %s",
                block->key.c_str(),
                unloaded_block_info->len(),
                block->len / partition_count_,
                peer_ip_.c_str());
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
            return;
        }

        if (!writeResponseBlock(block, unloaded_block_info)) {
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
            return;
        }
        ++write_cnt_;
    }

    if (write_cnt_ == total_block_count_) {
        if (kvDiagEnabled()) {
            static std::atomic<int> reply_budget{8000};
            if (reply_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                RTP_LLM_LOG_WARNING("[KVDIAG-SVC-REPLY] ctx=%p complete: written=%d total=%u -> runSuccess",
                                    (void*)this,
                                    (int)write_cnt_,
                                    total_block_count_);
            }
        }
        runSuccess(false);
    }
}

bool TcpCacheStoreServiceImplContext::writeResponseBlock(const std::shared_ptr<BlockBuffer>&     block,
                                                         const std::shared_ptr<BlockBufferInfo>& peer_block) {
    std::lock_guard<std::mutex> lock(response_mutex_);
    if (response_ == nullptr) {
        // try write response while already done
        return false;
    }

    auto block_len = block->len / partition_count_;
    if (block_len != peer_block->len()) {
        RTP_LLM_LOG_WARNING(
            "cache store service load block not match expect block len, key: %s, len %d vs %d, peer is %s",
            block->key.c_str(),
            block_len,
            peer_block->len(),
            peer_ip_.c_str());
        return false;
    }

    auto* block_info = response_->add_blocks();
    block_info->set_key(block->key);
    block_info->set_len(block_len);
    auto block_content = block_info->mutable_content();
    block_content->assign(
        std::shared_ptr<const char>(
            block->addr, reinterpret_cast<const char*>((int64_t)(block->addr.get()) + block_len * partition_id_)),
        size_t(block_len));
    return true;
}

}  // namespace rtp_llm