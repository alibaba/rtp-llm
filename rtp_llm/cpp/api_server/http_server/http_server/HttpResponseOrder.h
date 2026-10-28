#pragma once

#include "http_server/HttpResponseWriter.h"
#include <deque>
#include <memory>
#include <mutex>

namespace http_server {

// Optional HTTP/1.1 non-streaming response ordering. Reservation happens on the
// transport thread before work dispatch; completion may happen on any worker.
// The bound prevents an indefinitely slow first request from accumulating an
// unbounded number of completed pipelined responses on its connection.
class HttpResponseOrder: public std::enable_shared_from_this<HttpResponseOrder> {
public:
    HttpResponseOrder(std::shared_ptr<anet::Connection> connection, size_t limit);
    HttpResponseWriter::ResponseSender reserve();
    void                               cancel();

private:
    struct Ticket;
    bool                              complete(uint64_t sequence, const std::shared_ptr<HttpResponse>& response);
    std::shared_ptr<anet::Connection> connection_;
    const size_t                      limit_;
    std::mutex                        mutex_;
    uint64_t                          first_  = 0;
    bool                              closed_ = false;
    std::deque<std::shared_ptr<HttpResponse>> pending_;
};

}  // namespace http_server
