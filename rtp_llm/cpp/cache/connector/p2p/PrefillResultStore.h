#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace rtp_llm {

// Process-local results returned by StartLoad, independent of KV resource ownership.
// ResourceStore supplies deadlines and serializes publication, consumption and termination.
class PrefillResultStore {
public:
    // Published CPU tensors are owned by this payload and must remain read-only.
    struct SideChannelData {
        bool                                 has_first_token  = false;
        int64_t                              first_token_id   = 0;
        int32_t                              total_reuse_len  = 0;
        int32_t                              local_reuse_len  = 0;
        int32_t                              remote_reuse_len = 0;
        int32_t                              memory_reuse_len = 0;
        int32_t                              disk_reuse_len   = 0;
        std::vector<int>                     propose_tokens;
        torch::Tensor                        propose_probs;
        torch::Tensor                        propose_hidden;
        std::vector<int32_t>                 position_ids;
        std::map<std::string, torch::Tensor> first_token_tensors;
    };

    void                           updateDeadline(const std::string& unique_key, int64_t deadline_ms);
    std::optional<SideChannelData> publishPrefillPayload(const std::string& unique_key, SideChannelData&& data);
    bool                           waitPrefillPayloadReady(const std::string&    unique_key,
                                                           int64_t               deadline_ms,
                                                           std::function<bool()> is_cancelled = nullptr);
    std::optional<SideChannelData> takePrefillPayload(const std::string& unique_key);
    std::optional<SideChannelData> clearPrefillPayload(const std::string& unique_key);
    std::optional<SideChannelData> markTerminal(const std::string& unique_key);
    void                           eraseRequest(const std::string& unique_key);
    void                           stop();

    static grpc::Status fillStartLoadResponsePayload(const SideChannelData&           data,
                                                     P2PConnectorStartLoadResponsePB& response);

private:
    struct Entry {
        int64_t                        deadline_ms = 0;
        bool                           consumed    = false;
        bool                           terminal    = false;
        std::optional<SideChannelData> side_channel_data;
    };

    std::mutex                   mutex_;
    std::condition_variable      cv_;
    std::map<std::string, Entry> entries_;
    bool                         stopping_ = false;
};

}  // namespace rtp_llm
