#ifndef HTTP_SERVER_RESTFULPROCESSOR_H
#define HTTP_SERVER_RESTFULPROCESSOR_H

#include <functional>
#include <shared_mutex>
#include <optional>
#include <mutex>

#include "autil/Log.h"
#include "http_server/HttpRequest.h"

namespace http_server {

class HttpResponseWriter;

using ResponseHandler = std::function<void(std::unique_ptr<HttpResponseWriter>, const HttpRequest &)>;

class HttpRouter {
public:
    HttpRouter() = default;
    ~HttpRouter() = default;

public:
    bool RegisterRoute(const std::string &method, const std::string &endpoint, const ResponseHandler &func);
    std::optional<ResponseHandler> FindRoute(const std::string &method, const std::string &endpoint);

private:
    using HandlerMap = std::map<std::string, ResponseHandler>;
    std::map<std::string, HandlerMap> _registeredHandlers;
    std::shared_mutex _handlerMutex;

    AUTIL_LOG_DECLARE();
};

} // namespace http_server

#endif // HTTP_SERVER_RESTFULPROCESSOR_H
