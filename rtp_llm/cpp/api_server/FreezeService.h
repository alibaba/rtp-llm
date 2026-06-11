#pragma once

#include <memory>
#include <string>

#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/engine_base/EngineBase.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

// Body of POST /admin/freeze. All fields are optional (design doc M2):
//   {"mode": "graceful"|"force", "drain_timeout_ms": 30000, "reason": "..."}
class FreezeAdminRequest: public autil::legacy::Jsonizable {
public:
    ~FreezeAdminRequest() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("mode", mode, mode);
        json.Jsonize("drain_timeout_ms", drain_timeout_ms, drain_timeout_ms);
        json.Jsonize("reason", reason, reason);
    }

public:
    std::string mode             = "graceful";
    int64_t     drain_timeout_ms = 0;
    std::string reason;
};

// Response of POST /admin/freeze and POST /admin/resume on success.
class FreezeActionResponse: public autil::legacy::Jsonizable {
public:
    ~FreezeActionResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("status", status);
        json.Jsonize("state", state);
        json.Jsonize("freeze_epoch", freeze_epoch);
    }

public:
    std::string status = "ok";
    std::string state;
    int64_t     freeze_epoch = 0;
};

// Response of GET /admin/freeze_status. Mirrors proto FreezeStatusResponsePB.
class FreezeStatusHttpResponse: public autil::legacy::Jsonizable {
public:
    ~FreezeStatusHttpResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("state", state);
        json.Jsonize("freeze_epoch", freeze_epoch);
        json.Jsonize("kv_memory_state", kv_memory_state);
        json.Jsonize("device_kv_cache_valid", device_kv_cache_valid);
        json.Jsonize("active_request_count", active_request_count);
        json.Jsonize("active_cache_transfer_count", active_cache_transfer_count);
        json.Jsonize("gpu_resource_state", gpu_resource_state);
        json.Jsonize("last_error", last_error);
    }

public:
    std::string state;
    int64_t     freeze_epoch = 0;
    std::string kv_memory_state;
    bool        device_kv_cache_valid       = true;
    int64_t     active_request_count        = 0;
    int64_t     active_cache_transfer_count = 0;
    std::string gpu_resource_state;
    std::string last_error;
};

// Admin HTTP endpoints for the freeze/resume lifecycle (design doc M2).
// Delegates to EngineBase::freezeController() (M1 FreezeLifecycleController).
//   POST /admin/freeze
//   POST /admin/resume
//   GET  /admin/freeze_status
class FreezeService {
public:
    explicit FreezeService(const std::shared_ptr<EngineBase>& engine);
    ~FreezeService() = default;

public:
    void freeze(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                const http_server::HttpRequest&                         request);
    void resume(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                const http_server::HttpRequest&                         request);
    void freezeStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request);

private:
    // Returns false (after writing a 503 response) when there is no engine,
    // e.g. embedding-only server.
    bool checkEngine(const std::unique_ptr<http_server::HttpResponseWriter>& writer);

private:
    std::shared_ptr<EngineBase> engine_;
};

}  // namespace rtp_llm
