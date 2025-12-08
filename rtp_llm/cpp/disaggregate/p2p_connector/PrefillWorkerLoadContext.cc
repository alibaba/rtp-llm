#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillWorkerLoadContext.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

PrefillWorkerLoadContext::PrefillWorkerLoadContext(int64_t                                 request_id,
                                                   const std::string&                      unique_key,
                                                   int64_t                                 deadline_ms,
                                                   const std::vector<AsymmetricTPContext>& asymmetric_tp_contexts,
                                                   int                                     num_layers):
    request_id_(request_id),
    unique_key_(unique_key),
    deadline_ms_(deadline_ms),
    asymmetric_tp_contexts_(asymmetric_tp_contexts),
    num_layers_(num_layers) {
    for (int i = 0; i < num_layers_; i++) {
        need_transfer_layer_ids_.insert(i);
    }
}

bool PrefillWorkerLoadContext::isDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transferred_ids_.size() == num_layers_ * asymmetric_tp_contexts_.size();
}

bool PrefillWorkerLoadContext::isCanceled() const {
    return canceled_;
}

bool PrefillWorkerLoadContext::isTimeout() const {
    return currentTimeMs() >= deadline_ms_;
}

bool PrefillWorkerLoadContext::isAllTransfersDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transferred_ids_.size() == (num_layers_ - need_transfer_layer_ids_.size()) * asymmetric_tp_contexts_.size();
}

bool PrefillWorkerLoadContext::isAllSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return all_success_;
}

bool PrefillWorkerLoadContext::needTransfer(int layer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return need_transfer_layer_ids_.find(layer_id) != need_transfer_layer_ids_.end();
}

void PrefillWorkerLoadContext::startTransfer(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    need_transfer_layer_ids_.erase(layer_id);
}

void PrefillWorkerLoadContext::notifyDone(int id, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!success) {
        all_success_ = false;
    }
    RTP_LLM_LOG_INFO(
        "PrefillWorkerLoadContext notifyDone, request_id: %ld, id: %d, success: %d", request_id_, id, success);
    transferred_ids_.insert(id);
}

void PrefillWorkerLoadContext::setCanceled() {
    std::lock_guard<std::mutex> lock(mutex_);
    canceled_ = true;
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
        request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);
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
        if (iter->second->isTimeout()) {
            iter = contexts_.erase(iter);
        } else {
            ++iter;
        }
    }
}
}  // namespace rtp_llm