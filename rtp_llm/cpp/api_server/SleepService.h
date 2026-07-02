#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/engine_base/EngineBase.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

// Body of POST /sleep. All fields are optional:
//   {"level": 1, "mode": "wait"|"abort", "timeout_ms": 30000, "reason": "...", "tags": []}
// level=0 is defined as state-preserving sleep, but currently returns 501/UNIMPLEMENTED.
class SleepHttpRequest: public autil::legacy::Jsonizable {
public:
    ~SleepHttpRequest() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("level", level, level);
        json.Jsonize("mode", mode, mode);
        json.Jsonize("timeout_ms", timeout_ms, timeout_ms);
        json.Jsonize("reason", reason, reason);
        json.Jsonize("tags", tags, tags);
    }

public:
    int32_t     level            = 1;
    std::string mode             = "wait";
    int64_t     timeout_ms       = 0;
    std::string reason;
    std::vector<std::string> tags;
};

// Response of POST /sleep and POST /wake_up on success.
class SleepActionResponse: public autil::legacy::Jsonizable {
public:
    ~SleepActionResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("status", status);
    }

public:
    std::string status = "ok";
};

// Response of GET /sleep_status. Mirrors proto SleepStatusResponsePB.
class SleepStatusHttpResponse: public autil::legacy::Jsonizable {
public:
    ~SleepStatusHttpResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("sleep_mode_enabled", sleep_mode_enabled);
        json.Jsonize("effective", effective);
        json.Jsonize("supported_levels", supported_levels);
        json.Jsonize("supported_modes", supported_modes);
        json.Jsonize("disabled_reason", disabled_reason);
        json.Jsonize("state", state);
        json.Jsonize("sleep_epoch", sleep_epoch);
        json.Jsonize("kv_memory_state", kv_memory_state);
        json.Jsonize("device_kv_cache_valid", device_kv_cache_valid);
        json.Jsonize("active_request_count", active_request_count);
        json.Jsonize("active_cache_transfer_count", active_cache_transfer_count);
        json.Jsonize("gpu_resource_state", gpu_resource_state);
        json.Jsonize("last_error", last_error);
    }

public:
    bool        sleep_mode_enabled = false;
    bool        effective          = false;
    std::vector<int32_t> supported_levels;
    std::vector<std::string> supported_modes;
    std::string disabled_reason;
    std::string state;
    int64_t     sleep_epoch = 0;
    std::string kv_memory_state;
    bool        device_kv_cache_valid       = true;
    int64_t     active_request_count        = 0;
    int64_t     active_cache_transfer_count = 0;
    std::string gpu_resource_state;
    std::string last_error;
};

class IsSleepingHttpResponse: public autil::legacy::Jsonizable {
public:
    ~IsSleepingHttpResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("is_sleeping", is_sleeping);
        json.Jsonize("sleep_mode_enabled", sleep_mode_enabled);
        json.Jsonize("effective", effective);
        json.Jsonize("supported_levels", supported_levels);
        json.Jsonize("supported_modes", supported_modes);
        json.Jsonize("state", state);
        json.Jsonize("disabled_reason", disabled_reason);
    }

public:
    bool        is_sleeping        = false;
    bool        sleep_mode_enabled = false;
    bool        effective          = false;
    std::vector<int32_t> supported_levels;
    std::vector<std::string> supported_modes;
    std::string state;
    std::string disabled_reason;
};

// HTTP endpoints for the sleep/wake_up lifecycle.
// Delegates to EngineBase::sleepController().
//   POST /sleep
//   POST /wake_up
//   GET  /is_sleeping
//   GET  /sleep_status
class SleepService {
public:
    explicit SleepService(const std::shared_ptr<EngineBase>& engine);
    ~SleepService() = default;

public:
    void sleep(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
               const http_server::HttpRequest&                         request);
    void wakeUp(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                const http_server::HttpRequest&                         request);
    void isSleeping(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                    const http_server::HttpRequest&                         request);
    void sleepStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                     const http_server::HttpRequest&                         request);

private:
    // Returns false (after writing a 503 response) when there is no engine,
    // e.g. embedding-only server.
    bool checkEngine(const std::unique_ptr<http_server::HttpResponseWriter>& writer);

private:
    std::shared_ptr<EngineBase> engine_;
};

}  // namespace rtp_llm
