#pragma once

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

class SysCmdService {
public:
    SysCmdService()  = default;
    ~SysCmdService() = default;

public:
    void setLogLevel(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                     const http_server::HttpRequest&                         request);
};

}  // namespace rtp_llm