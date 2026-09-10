#pragma once

#include "autil/LoopThread.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace rtp_llm {

// Process-local results returned by StartLoad, independent of KV resource ownership.
class PrefillResultStore {
public:
    struct Data {
        bool                 has_first_token  = false;
        int64_t              first_token_id   = 0;
        int32_t              total_reuse_len  = 0;
        int32_t              local_reuse_len  = 0;
        int32_t              remote_reuse_len = 0;
        int32_t              memory_reuse_len = 0;
        int32_t              disk_reuse_len   = 0;
        std::vector<int>     propose_tokens;
        TensorPB             propose_probs;
        TensorPB             propose_hidden;
        std::vector<int32_t> position_ids;
    };

    PrefillResultStore(int timeout_check_interval_ms, int64_t hold_ms, int64_t terminal_ttl_ms = 3600000);
    ~PrefillResultStore();
    bool init();

    // ResourceStore supplies already-normalized deadlines while holding its resource lock.
    bool registerRequest(const std::string& unique_key, int64_t deadline_ms, int64_t request_deadline_ms);
    bool beginTransfer(const std::string& unique_key, int64_t deadline_ms);
    void notify(const std::string& unique_key, int64_t request_deadline_ms, const Data& data);
    bool waitAndFill(const std::string&    unique_key,
                     int64_t               deadline_ms,
                     Data&                 data,
                     std::function<bool()> is_cancelled = nullptr);
    void waitAndFill(const std::string&               unique_key,
                     int64_t                          deadline_ms,
                     P2PConnectorStartLoadResponsePB& response,
                     std::function<bool()>            is_cancelled = nullptr);
    void seal(const std::string& unique_key, int64_t request_deadline_ms);

private:
    struct Entry {
        std::optional<Data> data;
        int64_t             deadline_ms          = 0;
        int64_t             terminal_deadline_ms = 0;
        bool                terminal             = false;
        // Only ResourceStore may expire a registered resource before its transfer starts.
        bool                resource_owned       = false;
    };
    void        checkTimeout();
    void        checkTimeout(int64_t now_ms);
    static void fillPayload(const Data& data, SideChannelPayloadPB& payload);

    std::mutex                   mutex_;
    std::condition_variable      cv_;
    std::map<std::string, Entry> entries_;
    const int                    timeout_check_interval_ms_;
    const int64_t                hold_ms_;
    const int64_t                terminal_ttl_ms_;
    autil::LoopThreadPtr         cleanup_thread_;
};

}  // namespace rtp_llm
