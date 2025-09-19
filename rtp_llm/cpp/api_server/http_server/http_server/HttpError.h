#pragma once

#include <string>

namespace http_server {

struct HttpError {
public:
    static HttpError OK() {
        return {};
    }
    static HttpError BadRequest(const std::string& message) {
        return {400, message};
    }
    static HttpError NotFound(const std::string& message) {
        return {404, message};
    }
    static HttpError InternalError(const std::string& message) {
        return {500, message};
    }

    bool IsOK() const {
        return code == 0;
    }
    std::string ToString() const {
        return std::string("[code: ") + std::to_string(code) + ", msg: " + message + "]";
    }

public:
    int         code = 0;
    std::string message;
};

}  // namespace http_server
