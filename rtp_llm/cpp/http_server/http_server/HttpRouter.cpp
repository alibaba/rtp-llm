#include <mutex>
#include "http_server/HttpRouter.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpRouter);

bool HttpRouter::RegisterRoute(const std::string& method, const std::string& endpoint, const ResponseHandler& func) {
    if (method.empty() || endpoint.empty()) {
        AUTIL_LOG(WARN,
                  "register route failed, method or endpoint is empty, method: %s, endpoint: %s",
                  method.c_str(),
                  endpoint.c_str());
        return false;
    }
    {
        std::unique_lock<std::shared_mutex> lock(_handlerMutex);
        if (_registeredHandlers.count(method) && _registeredHandlers[method].count(endpoint)) {
            AUTIL_LOG(WARN,
                      "route has already registered, will be replaced. method: %s, endpoint: %s",
                      method.c_str(),
                      endpoint.c_str());
        }
        _registeredHandlers[method][endpoint] = func;
    }
    AUTIL_LOG(INFO, "register restful api [%s] %s success", method.c_str(), endpoint.c_str());
    return true;
}

std::optional<ResponseHandler> HttpRouter::FindRoute(const std::string& method, const std::string& endpoint) const {
    std::optional<ResponseHandler> handler;
    {
        std::shared_lock<std::shared_mutex> lock(_handlerMutex);
        if (!_registeredHandlers.count(method)) {
            AUTIL_LOG(WARN, "find route failed, no methdod registered for [%s: %s]", method.c_str(), endpoint.c_str());
            return std::nullopt;
        }
        if (!_registeredHandlers.at(method).count(endpoint)) {
            AUTIL_LOG(WARN,
                      "find route failed, no response handler registered for [%s: %s]",
                      method.c_str(),
                      endpoint.c_str());
            return std::nullopt;
        }
        handler = _registeredHandlers.at(method).at(endpoint);
    }
    return handler;
}

}  // namespace http_server
