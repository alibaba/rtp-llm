#include "http_server/HttpResponseOrder.h"
#include "http_server/HttpResponse.h"
#include "aios/network/anet/connection.h"
#include <stdexcept>

namespace http_server {

struct HttpResponseOrder::Ticket {
    std::shared_ptr<HttpResponseOrder> owner;
    uint64_t                           sequence  = 0;
    bool                               completed = true;  // Armed only after all throwing reservation steps.
    ~Ticket() {
        if (!completed)
            owner->cancel();
    }
};

HttpResponseOrder::HttpResponseOrder(std::shared_ptr<anet::Connection> connection, size_t limit):
    connection_(std::move(connection)), limit_(limit) {}

HttpResponseWriter::ResponseSender HttpResponseOrder::reserve() {
    // Allocate the callback before locking. Unwinding an armed Ticket must never
    // re-enter the order mutex from inside that mutex's critical section.
    auto ticket                               = std::make_shared<Ticket>();
    ticket->owner                             = shared_from_this();
    HttpResponseWriter::ResponseSender sender = [ticket](const std::shared_ptr<HttpResponse>& response) {
        if (ticket->completed)
            return false;
        ticket->completed = true;
        return ticket->owner->complete(ticket->sequence, response);
    };
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!closed_ && pending_.size() < limit_) {
            ticket->sequence = first_ + pending_.size();
            pending_.emplace_back();
            ticket->completed = false;
            return sender;
        }
    }
    cancel();
    return {};
}

bool HttpResponseOrder::complete(uint64_t sequence, const std::shared_ptr<HttpResponse>& response) {
    size_t postedCount = 0;
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || sequence < first_ || sequence - first_ >= pending_.size())
                return false;
            pending_[sequence - first_] = response;
            while (!pending_.empty() && pending_.front()) {
                auto                                             deleter = [](anet::Packet* packet) { packet->free(); };
                std::unique_ptr<anet::Packet, decltype(deleter)> packet(pending_.front()->Encode(), deleter);
                if (!packet || !connection_->postPacket(packet.get()))
                    throw std::runtime_error("cannot post ordered HTTP response");
                packet.release();
                pending_.pop_front();
                ++first_;
                ++postedCount;
            }
        }
        // Never take the component mutex while holding the order mutex.
        connection_->completePostedResponses(postedCount);
        return true;
    } catch (...) {
        // Encoding and posting can fail. A missing response makes HTTP/1.1
        // correlation ambiguous, so terminate the connection and all tickets.
        cancel();
        return false;
    }
}

void HttpResponseOrder::cancel() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        pending_.clear();
    }
    // ANet close may deliver control packets; never call it under our mutex.
    connection_->close();
}

}  // namespace http_server
