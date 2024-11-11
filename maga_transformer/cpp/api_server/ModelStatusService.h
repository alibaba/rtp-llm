#pragma once

#include <atomic>
#include "maga_transformer/cpp/http_server/http_server/HttpResponseWriter.h"
#include "maga_transformer/cpp/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

class ModelStatusService {
public:
    ModelStatusService()  = default;
    ~ModelStatusService() = default;

public:
    void modelStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                     const http_server::HttpRequest&                         request);
};

}  // namespace rtp_llm