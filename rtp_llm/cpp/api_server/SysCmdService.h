#pragma once

#include <functional>

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

class SysCmdService {
public:
    using StartProfile = std::function<void(const std::string&, int, int, bool)>;
    void startProfile(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request,
                      const StartProfile&                                     start_profile);

    SysCmdService()  = default;
    ~SysCmdService() = default;

public:
    void setLogLevel(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                     const http_server::HttpRequest&                         request);
};

}  // namespace rtp_llm