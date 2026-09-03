#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"

namespace rtp_llm {

class ConstraintTreeService {
public:
    ConstraintTreeService();
    ~ConstraintTreeService();

    void updateConstraintTree(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                              const http_server::HttpRequest&                         request);
    void constraintTreeStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                              const http_server::HttpRequest&                         request);

private:
    struct PendingUpdate {
        uint64_t    version;
        std::string body;
    };

    void updateLoop();

private:
    mutable std::mutex           mutex_;
    std::condition_variable      condition_;
    bool                         stopping_ = false;
    std::optional<PendingUpdate> pending_update_;
    uint64_t                     latest_requested_version_ = 0;
    std::string                  update_state_             = "idle";
    std::string                  update_message_           = "no runtime update has been submitted";
    std::thread                  update_thread_;
};

}  // namespace rtp_llm
