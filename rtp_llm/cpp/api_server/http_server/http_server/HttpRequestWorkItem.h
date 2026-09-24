#pragma once

#include "autil/Log.h"
#include "autil/WorkItem.h"
#include "http_server/HttpRouter.h"
#include "http_server/HttpResponseWriter.h"

namespace anet {
class Connection;
}

namespace http_server {

class HttpRequestWorkItem: public autil::WorkItem {
public:
    HttpRequestWorkItem(const ResponseHandler&                   func,
                        const std::shared_ptr<anet::Connection>& conn,
                        const std::shared_ptr<HttpRequest>&      request,
                        std::unique_ptr<HttpResponseWriter>      writer = {}):
        _func(func), _conn(conn), _request(request), _writer(std::move(writer)) {}
    ~HttpRequestWorkItem() {}

public:
    void process() override;
    void reject(int status, const std::string& message);

private:
    ResponseHandler                   _func;
    std::shared_ptr<anet::Connection> _conn;
    std::shared_ptr<HttpRequest>      _request;

    std::unique_ptr<HttpResponseWriter> _writer;

    AUTIL_LOG_DECLARE();
};

}  // namespace http_server