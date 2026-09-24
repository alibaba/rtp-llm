#include "http_server/HttpRequest.h"
#include "http_server/HttpServer.h"

#include "autil/NetUtil.h"
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

namespace http_server {
namespace {

TEST(HttpNativeOptionsTest, BinaryBodyViewPreservesNulAndHeader) {
    auto* packet = new anet::HTTPPacket();
    packet->setURI("/api/generate");
    packet->setMethod(anet::HTTPPacket::HM_POST);
    packet->addHeader("Content-Type", "application/x-protobuf");
    const std::string bytes("\x0a\x00\xff\x05", 4);
    packet->setBody(bytes.data(), bytes.size());
    HttpRequest request;
    ASSERT_TRUE(request.Parse({packet, [](anet::HTTPPacket* p) { p->free(); }}).IsOK());
    EXPECT_EQ(request.GetBodySize(), bytes.size());
    EXPECT_EQ(std::string(request.GetBodyView()), bytes);
    EXPECT_EQ(request.GetHeader("content-type"), "application/x-protobuf");
    EXPECT_EQ(request.GetHeader("Missing"), "");
    EXPECT_EQ(request.GetBodyView().data(), packet->getBody());
}

TEST(HttpNativeOptionsTest, ReusePortAllowsTwoNativeListeners) {
    const auto address = "tcp:127.0.0.1:" + std::to_string(autil::NetUtil::randomPort());
    HttpServer first(nullptr, 1, 4);
    HttpServer second(nullptr, 1, 4);
    ASSERT_TRUE(first.Start(address, 1000, 1000, 16, true, 1024 * 1024));
    EXPECT_TRUE(second.Start(address, 1000, 1000, 16, true, 1024 * 1024));
    EXPECT_TRUE(second.Stop());
    EXPECT_TRUE(first.Stop());
}

TEST(HttpNativeOptionsTest, DefaultListenerRemainsExclusive) {
    const auto address = "tcp:127.0.0.1:" + std::to_string(autil::NetUtil::randomPort());
    HttpServer first(nullptr, 1, 4);
    HttpServer second(nullptr, 1, 4);
    ASSERT_TRUE(first.Start(address));
    EXPECT_FALSE(second.Start(address));
    EXPECT_TRUE(second.Stop());
    EXPECT_TRUE(first.Stop());
}

TEST(HttpNativeOptionsTest, PacketLimitRejectsOversizedBodyBeforeRoute) {
    const int        port    = autil::NetUtil::randomPort();
    const auto       address = "tcp:127.0.0.1:" + std::to_string(port);
    std::atomic<int> invoked{0};
    HttpServer       server(nullptr, 1, 4);
    ASSERT_TRUE(server.RegisterRoute("POST", "/api/generate", [&](auto writer, const auto&) {
        ++invoked;
        writer->SetWriteType(HttpResponseWriter::WriteType::Normal);
        writer->Write("unexpected-route");
    }));
    ASSERT_TRUE(server.Start(address, 1000, 60000, 16, false, 256));
    struct SocketOwner {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~SocketOwner() {
            if (fd >= 0)
                close(fd);
        }
    } client;
    ASSERT_GE(client.fd, 0);
    timeval timeout{2, 0};
    ASSERT_EQ(setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
    sockaddr_in peer{};
    peer.sin_family      = AF_INET;
    peer.sin_port        = htons(port);
    peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(connect(client.fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)), 0);
    const auto request = std::string("POST /api/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1024\r\n\r\n")
                         + std::string(1024, 'x');
    ASSERT_EQ(send(client.fd, request.data(), request.size(), MSG_NOSIGNAL), static_cast<ssize_t>(request.size()));
    char       response[128];
    const auto received = recv(client.fd, response, sizeof(response), 0);
    // ANet closes malformed/oversized packets. A timeout does not establish
    // rejection, and an accidentally accepted packet would invoke the route.
    EXPECT_TRUE(received == 0 || (received < 0 && errno == ECONNRESET));
    EXPECT_EQ(invoked, 0);
    EXPECT_TRUE(server.Stop());
}

TEST(HttpNativeOptionsTest, OrderedResponsesFlushInRequestOrderAfterReverseCompletion) {
    const int                                                  port = autil::NetUtil::randomPort();
    std::mutex                                                 mutex;
    std::condition_variable                                    ready;
    std::map<std::string, std::unique_ptr<HttpResponseWriter>> writers;
    HttpServer                                                 server(nullptr, 2, 8, true);
    ASSERT_TRUE(server.RegisterRoute("POST", "/ordered", [&](auto writer, const auto& request) {
        std::lock_guard<std::mutex> lock(mutex);
        writers.emplace(std::string(request.GetBodyView()), std::move(writer));
        ready.notify_all();
    }));
    ASSERT_TRUE(server.Start("tcp:127.0.0.1:" + std::to_string(port)));
    struct SocketOwner {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~SocketOwner() {
            if (fd >= 0)
                close(fd);
        }
    } client;
    ASSERT_GE(client.fd, 0);
    timeval timeout{3, 0};
    ASSERT_EQ(setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
    sockaddr_in peer{};
    peer.sin_family      = AF_INET;
    peer.sin_port        = htons(port);
    peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(connect(client.fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)), 0);
    std::string bytes;
    for (const auto* body : {"one", "two", "tri"})
        bytes += std::string("POST /ordered HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\n") + body;
    ASSERT_EQ(send(client.fd, bytes.data(), bytes.size(), MSG_NOSIGNAL), static_cast<ssize_t>(bytes.size()));
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(ready.wait_for(lock, std::chrono::seconds(3), [&] { return writers.size() == 3; }));
        // These are real completed Write calls, not merely submitted backend
        // requests. The old unordered implementation deterministically fails.
        for (const auto* body : {"tri", "two", "one"}) {
            auto& writer = writers.at(body);
            writer->SetWriteType(HttpResponseWriter::WriteType::Normal);
            ASSERT_TRUE(writer->Write(body));
            writer.reset();
        }
    }
    std::string received;
    while (received.find("\r\n\r\ntri") == std::string::npos) {
        char       buffer[2048];
        const auto count = recv(client.fd, buffer, sizeof(buffer), 0);
        ASSERT_GT(count, 0);
        received.append(buffer, count);
    }
    ASSERT_NE(received.find("\r\n\r\none"), std::string::npos);
    ASSERT_NE(received.find("\r\n\r\ntwo"), std::string::npos);
    EXPECT_LT(received.find("\r\n\r\none"), received.find("\r\n\r\ntwo"));
    EXPECT_LT(received.find("\r\n\r\ntwo"), received.find("\r\n\r\ntri"));
}

TEST(HttpNativeOptionsTest, RequestAwareIdleRequiresOrderedResponses) {
    EXPECT_THROW((HttpServer(nullptr, 1, 4, false, 100)), std::invalid_argument);
}

TEST(HttpNativeOptionsTest, RequestAwareIdleKeepsSameReadPipelineOpenAtZeroTimeout) {
    const int                           port = autil::NetUtil::randomPort();
    std::mutex                          mutex;
    std::condition_variable             ready;
    std::unique_ptr<HttpResponseWriter> second_writer;
    // PgNativeService maps timeout_keep_alive=0 to 1 ms because ANet reserves zero.
    HttpServer server(nullptr, 2, 8, true, 1);
    ASSERT_TRUE(server.RegisterRoute("POST", "/pipeline", [&](auto writer, const auto& request) {
        writer->SetWriteType(HttpResponseWriter::WriteType::Normal);
        if (request.GetBodyView() == "one") {
            writer->Write("one");
        } else {
            std::lock_guard<std::mutex> lock(mutex);
            second_writer = std::move(writer);
            ready.notify_all();
        }
    }));
    ASSERT_TRUE(server.Start("tcp:127.0.0.1:" + std::to_string(port)));
    struct SocketOwner {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~SocketOwner() {
            if (fd >= 0)
                close(fd);
        }
    } client;
    ASSERT_GE(client.fd, 0);
    timeval timeout{2, 0};
    ASSERT_EQ(setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
    sockaddr_in peer{};
    peer.sin_family      = AF_INET;
    peer.sin_port        = htons(port);
    peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(connect(client.fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)), 0);
    const std::string wire = "POST /pipeline HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\none"
                             "POST /pipeline HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\ntwo";
    ASSERT_EQ(send(client.fd, wire.data(), wire.size(), MSG_NOSIGNAL), static_cast<ssize_t>(wire.size()));
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(ready.wait_for(lock, std::chrono::seconds(2), [&] { return second_writer != nullptr; }));
    }
    std::string response;
    while (response.find("\r\n\r\none") == std::string::npos) {
        char       bytes[2048];
        const auto count = recv(client.fd, bytes, sizeof(bytes), 0);
        ASSERT_GT(count, 0);
        response.append(bytes, count);
    }
    pollfd pending{client.fd, POLLIN, 0};
    EXPECT_EQ(poll(&pending, 1, 350), 0);  // B remains active through several ANet timeout sweeps.
    {
        std::lock_guard<std::mutex> lock(mutex);
        ASSERT_TRUE(second_writer->Write("two"));
        second_writer.reset();
    }
    response.clear();
    while (response.find("\r\n\r\ntwo") == std::string::npos) {
        char       bytes[2048];
        const auto count = recv(client.fd, bytes, sizeof(bytes), 0);
        ASSERT_GT(count, 0);
        response.append(bytes, count);
    }
    pending.revents = 0;
    ASSERT_GT(poll(&pending, 1, 2000), 0);
    char trailing;
    EXPECT_EQ(recv(client.fd, &trailing, 1, 0), 0);
    EXPECT_TRUE(server.Stop());
}

TEST(HttpNativeOptionsTest, RequestAwareIdleWaitsForPostedResponseAndPartialNextRequest) {
    const int                                                  port = autil::NetUtil::randomPort();
    std::mutex                                                 mutex;
    std::condition_variable                                    ready;
    std::map<std::string, std::unique_ptr<HttpResponseWriter>> writers;
    HttpServer                                                 server(nullptr, 2, 8, true, 500);
    ASSERT_TRUE(server.RegisterRoute("POST", "/idle", [&](auto writer, const auto& request) {
        std::lock_guard<std::mutex> lock(mutex);
        writers.emplace(std::string(request.GetBodyView()), std::move(writer));
        ready.notify_all();
    }));
    ASSERT_TRUE(server.Start("tcp:127.0.0.1:" + std::to_string(port)));
    struct SocketOwner {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~SocketOwner() {
            if (fd >= 0)
                close(fd);
        }
    } client;
    ASSERT_GE(client.fd, 0);
    timeval timeout{2, 0};
    ASSERT_EQ(setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
    sockaddr_in peer{};
    peer.sin_family      = AF_INET;
    peer.sin_port        = htons(port);
    peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(connect(client.fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)), 0);
    const std::string first = "POST /idle HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\none";
    ASSERT_EQ(send(client.fd, first.data(), first.size(), MSG_NOSIGNAL), static_cast<ssize_t>(first.size()));
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(ready.wait_for(lock, std::chrono::seconds(2), [&] { return writers.count("one") == 1; }));
    }
    pollfd pending{client.fd, POLLIN, 0};
    EXPECT_EQ(poll(&pending, 1, 850), 0);  // Active work exceeds the 500 ms idle interval.
    {
        std::lock_guard<std::mutex> lock(mutex);
        writers.at("one")->SetWriteType(HttpResponseWriter::WriteType::Normal);
        ASSERT_TRUE(writers.at("one")->Write("one"));
        writers.erase("one");
    }
    std::string response;
    while (response.find("\r\n\r\none") == std::string::npos) {
        char       bytes[2048];
        const auto count = recv(client.fd, bytes, sizeof(bytes), 0);
        ASSERT_GT(count, 0);
        response.append(bytes, count);
    }
    const std::string prefix = "POST /idle HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\n";
    ASSERT_EQ(send(client.fd, prefix.data(), prefix.size(), MSG_NOSIGNAL), static_cast<ssize_t>(prefix.size()));
    pending.revents = 0;
    EXPECT_EQ(poll(&pending, 1, 850), 0);  // Partial body cancels the prior response's idle timer.
    ASSERT_EQ(send(client.fd, "two", 3, MSG_NOSIGNAL), 3);
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(ready.wait_for(lock, std::chrono::seconds(2), [&] { return writers.count("two") == 1; }));
        writers.at("two")->SetWriteType(HttpResponseWriter::WriteType::Normal);
        ASSERT_TRUE(writers.at("two")->Write("two"));
        writers.erase("two");
    }
    response.clear();
    while (response.find("\r\n\r\ntwo") == std::string::npos) {
        char       bytes[2048];
        const auto count = recv(client.fd, bytes, sizeof(bytes), 0);
        ASSERT_GT(count, 0);
        response.append(bytes, count);
    }
    pending.revents = 0;
    ASSERT_GT(poll(&pending, 1, 2000), 0);
    char trailing;
    EXPECT_EQ(recv(client.fd, &trailing, 1, 0), 0);
    EXPECT_TRUE(server.Stop());
}

}  // namespace
}  // namespace http_server
