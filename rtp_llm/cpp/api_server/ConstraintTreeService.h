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

class DeviceBase;

class ConstraintTreeService {
public:
    explicit ConstraintTreeService(DeviceBase* device = nullptr, std::string mapping_json = "{}");
    ~ConstraintTreeService();

    void updateConstraintTree(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                              const http_server::HttpRequest&                         request);
    void constraintTreeStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                              const http_server::HttpRequest&                         request);
    void constraintTreeMapping(const std::unique_ptr<http_server::HttpResponseWriter>& writer, bool full);

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
    std::string                  latest_requested_content_sha256_;
    std::string                  mapping_json_;
    std::string                  mapping_status_json_;
    std::string                  mapping_fingerprint_;
    std::string                  update_state_   = "idle";
    std::string                  update_message_ = "no runtime update has been submitted";
    DeviceBase*                  device_         = nullptr;
    std::thread                  update_thread_;
};

}  // namespace rtp_llm
