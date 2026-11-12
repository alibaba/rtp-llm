#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "autil/AtomicCounter.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"
#include "rtp_llm/cpp/api_server/TokenProcessor.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {

// Implements the evaluator-specific /tokenize, /awi/generate, /awi/barrier and
// /awi/observations endpoints.  All events exposed by observations are produced
// by engine/scheduler/allocator instrumentation; this class only parses the
// request and serializes the runtime state.
class AwiService {
public:
    AwiService(const std::shared_ptr<EngineBase>&           engine,
               const std::shared_ptr<TokenProcessor>&       token_processor,
               const std::shared_ptr<autil::AtomicCounter>& request_counter,
               const ModelConfig&                            model_config,
               const ParallelismConfig&                      parallelism_config,
               py::object                                    weights_loader);

    ~AwiService() = default;

    void tokenize(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                  const http_server::HttpRequest&                         request);

    void generate(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                  const http_server::HttpRequest&                         request);

    void barrier(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                 const http_server::HttpRequest&                         request);

    void observations(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request);

private:
    std::string weightsCensusJson();
    void        writeJson(const std::unique_ptr<http_server::HttpResponseWriter>& writer, const std::string& body);
    void        writeError(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                           int                                                 status,
                           const std::string&                                  error,
                           const std::string&                                  failure_kind = {});

private:
    std::shared_ptr<EngineBase>                         engine_;
    std::shared_ptr<TokenProcessor>                     token_processor_;
    std::shared_ptr<autil::AtomicCounter>               request_counter_;
    ModelConfig                                         model_config_;
    py::object                                          weights_loader_;
    std::mutex                                          weights_mutex_;
};

}  // namespace rtp_llm