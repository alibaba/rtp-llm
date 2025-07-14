#pragma once

#include "autil/Log.h"
#include "autil/WorkItem.h"
#include "http_server/HttpRouter.h"

namespace anet {
class Connection;
}

namespace http_server {

class HttpRequestWorkItem: public autil::WorkItem {
public:
    HttpRequestWorkItem(const ResponseHandler&                   func,
                        const std::shared_ptr<anet::Connection>& conn,
                        const std::shared_ptr<HttpRequest>&      request):
        _func(func), _conn(conn), _request(request) {}
    ~HttpRequestWorkItem() {}

public:
    void process() override;

private:
    ResponseHandler                   _func;
    std::shared_ptr<anet::Connection> _conn;
    std::shared_ptr<HttpRequest>      _request;

    AUTIL_LOG_DECLARE();
};

}  // namespace http_server