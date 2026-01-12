#pragma once

#include "rtp_llm/cpp/disaggregate/p2p_connector/AsymmetricTpUtil.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <mutex>

namespace rtp_llm {

class PrefillWorkerLoadContext {
public:
    PrefillWorkerLoadContext(int64_t            request_id,
                             const std::string& unique_key,
                             int64_t            deadline_ms,
                             int                transfer_count);

public:
    bool done() const;
    bool canceled() const;
    bool timeout() const;
    bool success() const;

    void setCanceled();

    bool startTransfer(int id);
    void notifyDone(int id, bool success);  // not layerid, layerid * asymmetric_tp_contexts_.size()

    bool isAllTransfersDone() const;
    bool isAllTransferStarted() const;

    int64_t requestId() const {
        return request_id_;
    }

    const std::set<int>& getNeedTransferIds() const {
        return need_transfer_ids_;
    }

private:
    int64_t     request_id_;
    std::string unique_key_;
    int64_t     deadline_ms_;
    int         transfer_count_;

    bool canceled_    = false;
    bool all_success_ = true;

    mutable std::mutex mutex_;
    std::set<int>      need_transfer_ids_;
    std::set<int>      transferring_ids_;
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
    int                                       getContextsCount() const;

private:
    mutable std::mutex                                                     contexts_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<PrefillWorkerLoadContext>> contexts_;
};

}  // namespace rtp_llm