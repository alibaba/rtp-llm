#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillWorkerLoadContext.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

PrefillWorkerLoadContext::PrefillWorkerLoadContext(int64_t            request_id,
                                                   const std::string& unique_key,
                                                   int64_t            deadline_ms,
                                                   int                transfer_count):
    request_id_(request_id), unique_key_(unique_key), deadline_ms_(deadline_ms), transfer_count_(transfer_count) {
    for (int i = 0; i < transfer_count_; i++) {
        need_transfer_ids_.insert(i);
    }
}

bool PrefillWorkerLoadContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transferred_ids_.size() == transfer_count_;
}

bool PrefillWorkerLoadContext::canceled() const {
    return canceled_;
}

bool PrefillWorkerLoadContext::timeout() const {
    return currentTimeMs() >= deadline_ms_;
}

bool PrefillWorkerLoadContext::isAllTransfersDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transferring_ids_.empty();
}

bool PrefillWorkerLoadContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return all_success_ && transferred_ids_.size() == transfer_count_;
}

bool PrefillWorkerLoadContext::startTransfer(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = need_transfer_ids_.find(id);
    if (iter == need_transfer_ids_.end()) {
        return false;
    }
    need_transfer_ids_.erase(iter);
    transferring_ids_.insert(id);
    RTP_LLM_LOG_DEBUG("PrefillWorkerLoadContext startTransfer success, id: %d", id);
    return true;
}

void PrefillWorkerLoadContext::notifyDone(int id, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!success) {
        all_success_ = false;
        RTP_LLM_LOG_DEBUG("PrefillWorkerLoadContext notifyDone failed, id: %d", id);
    }
    if (transferring_ids_.find(id) == transferring_ids_.end()) {
        return;
    }
    RTP_LLM_LOG_DEBUG("PrefillWorkerLoadContext notifyDone success, id: %d", id);
    transferring_ids_.erase(id);
    transferred_ids_.insert(id);
}

void PrefillWorkerLoadContext::setCanceled() {
    std::lock_guard<std::mutex> lock(mutex_);
    canceled_ = true;
}

bool PrefillWorkerLoadContext::isAllTransferStarted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return need_transfer_ids_.empty();
}

PrefillWorkerLoadContextStore::PrefillWorkerLoadContextStore() {}

PrefillWorkerLoadContextStore::~PrefillWorkerLoadContextStore() {}

std::shared_ptr<PrefillWorkerLoadContext>
PrefillWorkerLoadContextStore::addContext(int64_t                                 request_id,
                                          const std::string&                      unique_key,
                                          int64_t                                 deadline_ms,
                                          const std::vector<AsymmetricTPContext>& asymmetric_tp_contexts,
                                          int                                     num_layers) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto                        prefill_worker_load_context = std::make_shared<PrefillWorkerLoadContext>(
        request_id, unique_key, deadline_ms, static_cast<int>(asymmetric_tp_contexts.size()) * num_layers);
    contexts_[request_id] = prefill_worker_load_context;
    return prefill_worker_load_context;
}

std::shared_ptr<PrefillWorkerLoadContext> PrefillWorkerLoadContextStore::getContext(int request_id) const {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto                        iter = contexts_.find(request_id);
    if (iter == contexts_.end()) {
        return nullptr;
    }
    return iter->second;
}

void PrefillWorkerLoadContextStore::removeContext(int64_t request_id) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    contexts_.erase(request_id);
}

void PrefillWorkerLoadContextStore::checkTimeout() {
    std::unique_lock<std::mutex> lock(contexts_mutex_);
    for (auto iter = contexts_.begin(); iter != contexts_.end();) {
        if (iter->second->timeout()) {
            iter = contexts_.erase(iter);
        } else {
            ++iter;
        }
    }
}

int PrefillWorkerLoadContextStore::getContextsCount() const {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    return contexts_.size();
}
}  // namespace rtp_llm