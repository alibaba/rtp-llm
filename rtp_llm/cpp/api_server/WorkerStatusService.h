#pragma once

#include <atomic>

#include "rtp_llm/cpp/engine_base/EngineBase.h"

#include "rtp_llm/cpp/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"

namespace rtp_llm {

class WorkerStatusResponse: public autil::legacy::Jsonizable {
public:
    ~WorkerStatusResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("available_concurrency", available_concurrency);
        json.Jsonize("available_kv_cache", cache_status.available_kv_cache);
        json.Jsonize("total_kv_cache", cache_status.total_kv_cache);
        json.Jsonize("alive", alive);
    }

public:
    int         available_concurrency;
    KVCacheInfo cache_status;
    bool        alive;
};

class WorkerStatusService {
public:
    WorkerStatusService(const std::shared_ptr<EngineBase>&            engine,
                        const std::shared_ptr<ConcurrencyController>& controller,
                        const int                                     load_balance = 0);
    ~WorkerStatusService() = default;
    int getLoadBalanceEnv() {
        return load_balance_env_;
    }

public:
    void workerStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request);
    void stop();

private:
    std::atomic_bool                       is_stopped_{false};
    std::shared_ptr<EngineBase>            engine_;
    std::shared_ptr<ConcurrencyController> controller_;
    int                                    load_balance_env_{0};
};

}  // namespace rtp_llm
