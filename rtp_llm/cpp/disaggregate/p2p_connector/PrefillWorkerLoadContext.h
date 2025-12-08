#pragma once

#include "rtp_llm/cpp/disaggregate/p2p_connector/AsymmetricTpUtil.h"
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <mutex>

namespace rtp_llm {

class PrefillWorkerLoadContext {
public:
    PrefillWorkerLoadContext(int64_t                                 request_id,
                             const std::string&                      unique_key,
                             int64_t                                 deadline_ms,
                             const std::vector<AsymmetricTPContext>& asymmetric_tp_contexts,
                             int                                     num_layers);

public:
    bool isDone() const;
    bool isCanceled() const;
    bool isTimeout() const;
    bool isAllSuccess() const;

    bool needTransfer(int layer_id) const;
    void startTransfer(int layer_id);
    void notifyDone(int id, bool success);  // not layerid, layerid * asymmetric_tp_contexts_.size()

    bool isAllTransfersDone() const;
    void setCanceled();

    const std::vector<AsymmetricTPContext>& asymmetricTPContexts() const {
        return asymmetric_tp_contexts_;
    }
    int64_t requestId() const {
        return request_id_;
    }

private:
    int64_t                          request_id_;
    std::string                      unique_key_;
    int64_t                          deadline_ms_;
    std::vector<AsymmetricTPContext> asymmetric_tp_contexts_;
    int                              num_layers_;

    bool canceled_    = false;
    bool all_success_ = true;

    mutable std::mutex mutex_;
    std::set<int>      need_transfer_layer_ids_;
    std::set<int>      transferred_ids_;
};

class PrefillWorkerLoadContextStore {
public:
    PrefillWorkerLoadContextStore();
    ~PrefillWorkerLoadContextStore();

public:
    std::shared_ptr<PrefillWorkerLoadContext> addContext(int64_t                                 request_id,
                                                         const std::string&                      unique_key,
                                                         int64_t                                 deadline_ms,
                                                         const std::vector<AsymmetricTPContext>& asymmetric_tp_contexts,
                                                         int                                     num_layers);
    std::shared_ptr<PrefillWorkerLoadContext> getContext(int request_id) const;
    void                                      removeContext(int64_t request_id);
    void                                      checkTimeout();

private:
    mutable std::mutex                                                     contexts_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<PrefillWorkerLoadContext>> contexts_;
};

}  // namespace rtp_llm