// Publisher-specific behavior tests are owned by the cache/events subsystem.
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherFactory.h"
#include "rtp_llm/cpp/cache/events/LogPublisher.h"
#include "rtp_llm/cpp/cache/events/NullPublisher.h"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

constexpr auto kAsyncTestTimeout = std::chrono::seconds(10);

class RecordingReporter final: public KVCacheEventReporter {
public:
    struct Request {
        std::string                           route;
        std::string                           body;
        std::chrono::steady_clock::time_point recorded_at;
    };

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        bool fail_request = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            requests_.push_back({route, request, std::chrono::steady_clock::now()});
            if (fail_body_count_ > 0 && !fail_body_.empty() && request.find(fail_body_) != std::string::npos) {
                fail_request = true;
                --fail_body_count_;
                if (fail_body_count_ == 0) {
                    fail_body_.clear();
                }
            }
        }
        response = R"({"header":{"status":{"code":"OK"}}})";
        cv_.notify_all();
        return !fail_request;
    }

    bool waitForBodyCount(const std::string& text, size_t expected_count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            size_t count = 0;
            for (const auto& request : requests_) {
                count += request.body.find(text) != std::string::npos;
            }
            return count >= expected_count;
        });
    }

    void failNextBodyContaining(std::string text) {
        failNextBodiesContaining(std::move(text), 1);
    }

    void failNextBodiesContaining(std::string text, size_t count) {
        std::lock_guard<std::mutex> lock(mu_);
        fail_body_       = std::move(text);
        fail_body_count_ = count;
    }

    bool waitForBody(const std::string& text, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            for (const auto& request : requests_) {
                if (request.body.find(text) != std::string::npos) {
                    return true;
                }
            }
            return false;
        });
    }

    std::vector<Request> requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return requests_;
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::vector<Request>    requests_;
    std::string             fail_body_;
    size_t                  fail_body_count_{0};
};

class BlockingReporter final: public KVCacheEventReporter {
public:
    bool post(const std::string&, const std::string& request, std::string& response) noexcept override {
        std::unique_lock<std::mutex> lock(mu_);
        requests_.push_back(request);
        cv_.notify_all();
        if (block_next_mutation_ && request.find("EVENT_BLOCK_ADD") != std::string::npos) {
            block_next_mutation_ = false;
            mutation_blocked_    = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return release_mutation_; });
        }
        if (block_next_snapshot_ && request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            block_next_snapshot_ = false;
            snapshot_blocked_    = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return release_snapshot_; });
        }
        response = R"({"header":{"status":{"code":"OK"}}})";
        return true;
    }

    bool waitForBodyCount(const std::string& text, size_t expected_count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            size_t count = 0;
            for (const auto& request : requests_) {
                count += request.find(text) != std::string::npos;
            }
            return count >= expected_count;
        });
    }

    void blockNextMutation() {
        std::lock_guard<std::mutex> lock(mu_);
        block_next_mutation_ = true;
    }

    bool waitUntilMutationBlocked(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return mutation_blocked_; });
    }

    void releaseMutation() {
        std::lock_guard<std::mutex> lock(mu_);
        release_mutation_ = true;
        cv_.notify_all();
    }

    void blockNextSnapshot() {
        std::lock_guard<std::mutex> lock(mu_);
        block_next_snapshot_ = true;
    }

    bool waitUntilSnapshotBlocked(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return snapshot_blocked_; });
    }

    void releaseSnapshot() {
        std::lock_guard<std::mutex> lock(mu_);
        release_snapshot_ = true;
        cv_.notify_all();
    }

    std::vector<std::string> requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return requests_;
    }

private:
    mutable std::mutex       mu_;
    std::condition_variable  cv_;
    std::vector<std::string> requests_;
    bool                     block_next_mutation_ = false;
    bool                     mutation_blocked_    = false;
    bool                     release_mutation_    = false;
    bool                     block_next_snapshot_ = false;
    bool                     snapshot_blocked_    = false;
    bool                     release_snapshot_    = false;
};

class CountingReporter final: public KVCacheEventReporter {
public:
    bool post(const std::string&, const std::string& request, std::string& response) noexcept override {
        size_t count = 0;
        size_t pos   = 0;
        while ((pos = request.find("EVENT_BLOCK_ADD", pos)) != std::string::npos) {
            ++count;
            pos += 15;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            mutation_count_ += count;
        }
        response = R"({"header":{"status":{"code":"OK"}}})";
        cv_.notify_all();
        return true;
    }

    bool waitForMutationCount(size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] { return mutation_count_ >= expected; });
    }

private:
    size_t                  mutation_count_ = 0;
    std::mutex              mu_;
    std::condition_variable cv_;
};

class LocalHttpStub {
public:
    LocalHttpStub() {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            throw std::runtime_error("socket failed: " + std::string(std::strerror(errno)));
        }
        int reuse = 1;
        (void)::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in address{};
        address.sin_family      = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port        = 0;
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || ::listen(listen_fd_, 8) != 0) {
            const auto message = std::string(std::strerror(errno));
            ::close(listen_fd_);
            throw std::runtime_error("HTTP stub setup failed: " + message);
        }

        socklen_t address_size = sizeof(address);
        if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
            const auto message = std::string(std::strerror(errno));
            ::close(listen_fd_);
            throw std::runtime_error("getsockname failed: " + message);
        }
        port_   = ntohs(address.sin_port);
        worker_ = std::thread([this] { serve(); });
    }

    ~LocalHttpStub() {
        releaseSnapshot();
        stopping_.store(true, std::memory_order_release);

        const int wake_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (wake_fd >= 0) {
            sockaddr_in address{};
            address.sin_family      = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            address.sin_port        = htons(port_);
            (void)::connect(wake_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
            ::close(wake_fd);
        }
        if (worker_.joinable()) {
            worker_.join();
        }
        ::close(listen_fd_);
    }

    std::string endpoint() const {
        return "127.0.0.1:" + std::to_string(port_);
    }

    bool waitUntilSnapshotIsInFlight(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return snapshot_in_flight_; });
    }

    void releaseSnapshot() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            release_snapshot_ = true;
        }
        cv_.notify_all();
    }

    std::vector<std::string> requestBodies() const {
        std::lock_guard<std::mutex> lock(mu_);
        return request_bodies_;
    }

private:
    static bool readRequest(int fd, std::string& body) {
        std::string request;
        char        buffer[4096];
        size_t      header_end     = std::string::npos;
        size_t      content_length = 0;
        while (header_end == std::string::npos) {
            const auto bytes = ::recv(fd, buffer, sizeof(buffer), 0);
            if (bytes <= 0) {
                return false;
            }
            request.append(buffer, static_cast<size_t>(bytes));
            header_end = request.find("\r\n\r\n");
        }

        const auto content_length_pos = request.find("Content-Length:");
        if (content_length_pos != std::string::npos) {
            const auto value_start = content_length_pos + std::strlen("Content-Length:");
            const auto value_end   = request.find("\r\n", value_start);
            content_length = static_cast<size_t>(std::stoull(request.substr(value_start, value_end - value_start)));
        }

        const size_t body_start = header_end + 4;
        while (request.size() < body_start + content_length) {
            const auto bytes = ::recv(fd, buffer, sizeof(buffer), 0);
            if (bytes <= 0) {
                return false;
            }
            request.append(buffer, static_cast<size_t>(bytes));
        }
        body = request.substr(body_start, content_length);
        return true;
    }

    static void writeSuccess(int fd) {
        const std::string body     = R"({"header":{"status":{"code":"OK"}}})";
        const std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                     + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        size_t sent = 0;
        while (sent < response.size()) {
            const auto bytes = ::send(fd, response.data() + sent, response.size() - sent, 0);
            if (bytes <= 0) {
                return;
            }
            sent += static_cast<size_t>(bytes);
        }
    }

    void serve() {
        while (!stopping_.load(std::memory_order_acquire)) {
            const int client_fd = ::accept(listen_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                continue;
            }
            if (stopping_.load(std::memory_order_acquire)) {
                ::close(client_fd);
                break;
            }

            std::string body;
            if (readRequest(client_fd, body)) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    request_bodies_.push_back(body);
                }
                if (body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
                    std::unique_lock<std::mutex> lock(mu_);
                    snapshot_in_flight_ = true;
                    cv_.notify_all();
                    cv_.wait(lock, [this] { return release_snapshot_ || stopping_.load(std::memory_order_acquire); });
                } else {
                    writeSuccess(client_fd);
                }
            }
            ::close(client_fd);
        }
    }

private:
    int                      listen_fd_{-1};
    uint16_t                 port_{0};
    std::thread              worker_;
    std::atomic<bool>        stopping_{false};
    mutable std::mutex       mu_;
    std::condition_variable  cv_;
    std::vector<std::string> request_bodies_;
    bool                     snapshot_in_flight_{false};
    bool                     release_snapshot_{false};
};

size_t countOccurrences(const std::string& text, const std::string& pattern) {
    size_t count = 0;
    size_t pos   = 0;
    while ((pos = text.find(pattern, pos)) != std::string::npos) {
        ++count;
        pos += pattern.size();
    }
    return count;
}

KVCacheEventPublisherContext makeContext() {
    KVCacheEventPublisherContext context;
    context.instance_group    = "test_group";
    context.instance_id       = "test_instance";
    context.host_ip_port      = "127.0.0.1:9000";
    context.model_name        = "test_model";
    context.dtype             = "BF16";
    context.spec_name         = "rtp_llm_hbm_64";
    context.location_uri      = "rtp-llm://127.0.0.1:9000/hbm";
    context.block_size_tokens = 64;
    context.spec_size_bytes   = 4096;
    context.tp_size           = 2;
    context.dp_size           = 1;
    return context;
}

bool waitForState(KVCacheEventPublisher& publisher, PublisherState expected, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (publisher.status().state == expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return publisher.status().state == expected;
}

TEST(KVCacheEventPublisherTest, NullPublisherHasNoRuntimeResources) {
    NullPublisher publisher;
    EXPECT_TRUE(publisher.start());
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublishResult::DISABLED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 1, 0}));
    EXPECT_EQ(PublisherState::DISABLED, publisher.status().state);
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, FactorySelectsConfiguredPublisherWithoutLeakingConcreteTypesToCache) {
    KVCacheEventPublisherConfig config;
    const auto                  context = makeContext();

    config.type    = "none";
    auto publisher = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_FALSE(publisher->enabled());

    config.type = "log";
    publisher   = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_TRUE(publisher->enabled());

    config.type   = "kvcm";
    auto reporter = std::make_shared<RecordingReporter>();
    publisher     = createKVCacheEventPublisher(
        config,
        context,
        [] {
            return KVCacheSnapshot{1, {}};
        },
        reporter);
    ASSERT_NE(nullptr, publisher);
    EXPECT_TRUE(publisher->enabled());
    ASSERT_TRUE(publisher->start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    publisher->stop();

    config.type = "unsupported";
    publisher   = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_FALSE(publisher->enabled());
}

TEST(KVCacheEventPublisherTest, PublisherOwnershipRejectsPipelineParallelism) {
    EXPECT_TRUE(isKVCacheEventPublisherOwner(/*tp_rank=*/0, /*pp_size=*/1));
    EXPECT_FALSE(isKVCacheEventPublisherOwner(/*tp_rank=*/1, /*pp_size=*/1));
    EXPECT_FALSE(isKVCacheEventPublisherOwner(/*tp_rank=*/0, /*pp_size=*/2));
}

TEST(KVCacheEventPublisherTest, KVCMEndpointNormalizationIsStable) {
    EXPECT_EQ("", detail::normalizeKVCacheEventEndpoint(""));
    EXPECT_EQ("http://kvcm-meta:56020", detail::normalizeKVCacheEventEndpoint("kvcm-meta:56020///"));
    EXPECT_EQ("http://kvcm-meta:56020", detail::normalizeKVCacheEventEndpoint("http://kvcm-meta:56020/"));
    EXPECT_EQ("https://kvcm-meta.example", detail::normalizeKVCacheEventEndpoint("https://kvcm-meta.example///"));
}

TEST(KVCacheEventPublisherTest, KVCMResponseValidationCoversProtocolVariantsAndFailures) {
    const std::vector<std::pair<std::string, bool>> cases = {
        {R"({"header":{"status":{"code":"OK"}}})", true},
        {R"({"header":{"status":{"code":"1"}}})", true},
        {R"({"header":{"status":{"code":1}}})", true},
        {R"({"header":{"status":{"code":"OK"}},"item_results":[1,"1","OK"]})", true},
        {R"({"header":{"status":{"code":1}},"itemResults":["OK",1]})", true},
        {R"({"header":{"status":{"code":"FAILED"}}})", false},
        {R"({"header":{"status":{"code":0}}})", false},
        {R"({"header":{"status":{"code":"OK"}},"item_results":[1,0]})", false},
        {R"({"header":{"status":{"code":"OK"}},"itemResults":"OK"})", false},
        {R"({"header":{"status":{}}})", false},
        {R"({"header":{}})", false},
        {R"({"header":{"status":{"code":"OK"}})", false},
        {"not-json", false},
    };

    for (const auto& [response, expected] : cases) {
        EXPECT_EQ(expected, detail::kvcmResponseIsOk(response)) << response;
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRejectsIncompleteIdentity) {
    KVCacheEventPublisherConfig config;
    config.type  = "kvcm";
    auto context = makeContext();
    context.instance_id.clear();
    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(
        config, context, [] { return KVCacheSnapshot{}; }, reporter);

    EXPECT_FALSE(publisher.start());
    EXPECT_EQ(PublisherState::DEGRADED, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 1, 0}));
    EXPECT_TRUE(reporter->requests().empty());
}

TEST(KVCacheEventPublisherTest, RealCurlSnapshotRequestIsCancelledDuringStop) {
    LocalHttpStub               server;
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.manager_endpoint      = server.endpoint();
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.request_timeout_ms    = 2000;
    config.snapshot_timeout_ms   = 30000;
    config.retry_interval_ms     = 1;
    config.snapshot_interval_ms  = 60000;

    auto publisher = createKVCacheEventPublisher(config, makeContext(), [] { return KVCacheSnapshot{1, {10}}; });
    ASSERT_TRUE(publisher->enabled());
    ASSERT_TRUE(publisher->start());
    if (!server.waitUntilSnapshotIsInFlight(std::chrono::seconds(5))) {
        publisher->stop();
        FAIL() << "real curl snapshot request did not reach the local HTTP stub";
    }

    auto       stopped     = std::async(std::launch::async, [&] { publisher->stop(); });
    const auto stop_status = stopped.wait_for(std::chrono::seconds(5));
    server.releaseSnapshot();
    ASSERT_EQ(std::future_status::ready, stop_status)
        << "stop waited for the 30 second snapshot timeout instead of cancelling curl";
    stopped.get();

    const auto request_bodies = server.requestBodies();
    ASSERT_GE(request_bodies.size(), 3u);
    EXPECT_NE(std::string::npos, request_bodies[0].find("\"instance_group\""));
    EXPECT_NE(std::string::npos, request_bodies[1].find("EVENT_NODE_REGISTER"));
    EXPECT_NE(std::string::npos, request_bodies[2].find("EVENT_BLOCK_SNAPSHOT"));
}

TEST(KVCacheEventPublisherTest, LogPublisherAcceptsEventsAsynchronously) {
    KVCacheEventPublisherConfig config;
    config.type              = "log";
    config.queue_capacity    = 8;
    config.report_batch_size = 8;
    config.flush_interval_ms = 1;

    LogPublisher publisher(config, makeContext());
    ASSERT_TRUE(publisher.start());
    EXPECT_TRUE(publisher.enabled());
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
    EXPECT_EQ(1, publisher.status().accepted_count);
}

TEST(KVCacheEventPublisherTest, PublisherLifecycleIsIdempotent) {
    KVCacheEventPublisherConfig log_config;
    log_config.type              = "log";
    log_config.queue_capacity    = 8;
    log_config.report_batch_size = 8;
    log_config.flush_interval_ms = 1;

    LogPublisher log_publisher(log_config, makeContext());
    EXPECT_TRUE(log_publisher.start());
    EXPECT_TRUE(log_publisher.start());
    EXPECT_EQ(PublishResult::ACCEPTED, log_publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    log_publisher.stop();
    log_publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, log_publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, log_publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 43, 0}));
    EXPECT_FALSE(log_publisher.start());
    EXPECT_EQ(PublisherState::STOPPED, log_publisher.status().state);

    KVCacheEventPublisherConfig kvcm_config;
    kvcm_config.type                  = "kvcm";
    kvcm_config.queue_capacity        = 8;
    kvcm_config.report_batch_size     = 8;
    kvcm_config.flush_interval_ms     = 1;
    kvcm_config.heartbeat_interval_ms = 60000;
    kvcm_config.snapshot_interval_ms  = 60000;
    kvcm_config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher kvcm_publisher(
        kvcm_config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {}};
        },
        reporter);
    EXPECT_TRUE(kvcm_publisher.start());
    EXPECT_TRUE(kvcm_publisher.start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    kvcm_publisher.stop();
    kvcm_publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, kvcm_publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, kvcm_publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 44, 0}));
    EXPECT_FALSE(kvcm_publisher.start());
    EXPECT_EQ(PublisherState::STOPPED, kvcm_publisher.status().state);
}

TEST(KVCacheEventPublisherTest, ConcurrentStartAndStopAlwaysLeavesJoinedWorkers) {
    for (size_t iteration = 0; iteration < 32; ++iteration) {
        KVCacheEventPublisherConfig config;
        config.type              = "log";
        config.queue_capacity    = 8;
        config.report_batch_size = 8;
        config.flush_interval_ms = 1;

        LogPublisher      publisher(config, makeContext());
        std::atomic<bool> go{false};
        bool              started = false;
        std::thread       starter([&] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            started = publisher.start();
        });
        std::thread       stopper([&] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            publisher.stop();
        });

        go.store(true, std::memory_order_release);
        starter.join();
        stopper.join();
        publisher.stop();

        EXPECT_TRUE(started);
        EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRegistersSnapshotsAndReportsDeltas) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 32;
    config.report_batch_size     = 16;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter  = std::make_shared<RecordingReporter>();
    auto publisher = std::make_shared<KVCMPublisher>(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{7, {10, 20}};
        },
        reporter);

    ASSERT_TRUE(publisher->start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"30\"", kAsyncTestTimeout));

    reporter->failNextBodyContaining("EVENT_BLOCK_ADD");
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"31\"", kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    publisher->stop();

    const auto requests = reporter->requests();
    ASSERT_GE(requests.size(), 5);
    EXPECT_EQ("/api/registerInstance", requests.front().route);
    EXPECT_NE(std::string::npos, requests.front().body.find("rtp_llm_hbm_64"));
    EXPECT_NE(std::string::npos, requests.front().body.find("location_spec_groups"));
    EXPECT_EQ(1, countOccurrences(requests.front().body, "\"dtype\""));

    bool   saw_node_register = false;
    bool   saw_snapshot      = false;
    bool   saw_snapshot_item = false;
    bool   saw_add           = false;
    bool   saw_delete        = false;
    size_t host_down_count   = 0;
    for (const auto& request : requests) {
        saw_node_register = saw_node_register || request.body.find("EVENT_NODE_REGISTER") != std::string::npos;
        saw_snapshot      = saw_snapshot || request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        saw_snapshot_item =
            saw_snapshot_item || request.body.find("\"block_key\":\"10\",\"medium\":\"hbm\"") != std::string::npos;
        saw_add    = saw_add || request.body.find("EVENT_BLOCK_ADD") != std::string::npos;
        saw_delete = saw_delete || request.body.find("EVENT_BLOCK_DELETE") != std::string::npos;
        host_down_count += request.body.find("EVENT_HOST_DOWN") != std::string::npos;
    }
    EXPECT_TRUE(saw_node_register);
    EXPECT_TRUE(saw_snapshot);
    EXPECT_TRUE(saw_snapshot_item);
    EXPECT_TRUE(saw_add);
    EXPECT_TRUE(saw_delete);
    // HOST_DOWN is terminal. Registration and recovery use NODE_REGISTER plus
    // an authoritative snapshot rather than pretending the live engine exited.
    EXPECT_EQ(1, host_down_count);
    EXPECT_EQ(3, publisher->status().accepted_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromRegistrationFailure) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->failNextBodyContaining("\"instance_group\"");
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {10}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromSnapshotProviderFailure) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    std::atomic<size_t> snapshot_attempts{0};
    auto                reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher       publisher(
        config,
        makeContext(),
        [&snapshot_attempts] {
            if (snapshot_attempts.fetch_add(1, std::memory_order_relaxed) == 0) {
                throw std::runtime_error("injected snapshot failure");
            }
            return KVCacheSnapshot{2, {10, 20}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    EXPECT_GE(snapshot_attempts.load(std::memory_order_relaxed), 2);
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherReusesSnapshotPayloadAndExponentiallyBacksOff) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 40;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->failNextBodiesContaining("EVENT_BLOCK_SNAPSHOT", 2);
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {10, 20}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
    publisher.stop();

    std::vector<RecordingReporter::Request> snapshot_requests;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshot_requests.push_back(request);
        }
    }
    ASSERT_GE(snapshot_requests.size(), 3u);
    // Retrying an upload must reuse the already captured and serialized
    // snapshot (including its trace id) instead of recopying the cache.
    EXPECT_EQ(snapshot_requests[0].body, snapshot_requests[1].body);
    EXPECT_EQ(snapshot_requests[1].body, snapshot_requests[2].body);

    const auto first_retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        snapshot_requests[1].recorded_at - snapshot_requests[0].recorded_at);
    const auto second_retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        snapshot_requests[2].recorded_at - snapshot_requests[1].recorded_at);
    EXPECT_GE(first_retry_delay.count(), 30);
    EXPECT_GE(second_retry_delay.count(), 60);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherThrottlesContinuousDirtySnapshotsWithoutStarvingHeartbeat) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 1;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 10;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 50;

    std::atomic<int64_t> next_key{100};
    auto                 reporter      = std::make_shared<RecordingReporter>();
    KVCMPublisher*       publisher_ptr = nullptr;
    KVCMPublisher        publisher(
        config,
        makeContext(),
        [&] {
            const auto key = next_key.fetch_add(2, std::memory_order_relaxed);
            // The first event fills the queue and the second marks the
            // publisher dirty while each snapshot is being captured.
            (void)publisher_ptr->tryPublish({KVCacheEventType::BLOCK_ADD, key, 0});
            (void)publisher_ptr->tryPublish({KVCacheEventType::BLOCK_ADD, key + 1, 0});
            return KVCacheSnapshot{key, {key}};
        },
        reporter);
    publisher_ptr = &publisher;

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    publisher.stop();

    const auto requests              = reporter->requests();
    size_t     first_snapshot_index  = requests.size();
    size_t     second_snapshot_index = requests.size();
    for (size_t i = 0; i < requests.size(); ++i) {
        if (requests[i].body.find("EVENT_BLOCK_SNAPSHOT") == std::string::npos) {
            continue;
        }
        if (first_snapshot_index == requests.size()) {
            first_snapshot_index = i;
        } else {
            second_snapshot_index = i;
            break;
        }
    }
    ASSERT_LT(first_snapshot_index, requests.size());
    ASSERT_LT(second_snapshot_index, requests.size());

    const auto resync_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        requests[second_snapshot_index].recorded_at - requests[first_snapshot_index].recorded_at);
    EXPECT_GE(resync_delay.count(), 40);

    bool heartbeat_between_snapshots = false;
    for (size_t i = first_snapshot_index + 1; i < second_snapshot_index; ++i) {
        heartbeat_between_snapshots =
            heartbeat_between_snapshots || requests[i].body.find("EVENT_HEARTBEAT") != std::string::npos;
    }
    EXPECT_TRUE(heartbeat_between_snapshots);
    EXPECT_GT(publisher.status().dropped_count, 0u);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromHeartbeatFailure) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 100;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {10}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    reporter->failNextBodyContaining("EVENT_HEARTBEAT");
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherPreservesMutationsCreatedWhileSnapshotIsInFlight) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 50;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {10}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    reporter->blockNextSnapshot();
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "periodic snapshot request did not reach the blocking reporter";
    }

    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    reporter->releaseSnapshot();
    ASSERT_TRUE(reporter->waitForBodyCount("\"block_key\":\"20\"", 1, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherCoalescesEachKeyToItsLastMutation) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 32;
    config.report_batch_size     = 16;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {}};
        },
        reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 999, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }

    // KVCM applies all ADDs before all DELETEs within one request. Keeping both
    // transitions would make DELETE->ADD end deleted. The publisher must send
    // only the final state for each key in this queued batch.
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 42, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 43, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 43, 0}));
    reporter->releaseMutation();

    ASSERT_TRUE(reporter->waitForBodyCount("\"block_key\":\"42\"", 1, kAsyncTestTimeout));
    publisher.stop();

    bool saw_final_add_42    = false;
    bool saw_final_delete_43 = false;
    for (const auto& request : reporter->requests()) {
        if (request.find("\"block_key\":\"42\"") != std::string::npos) {
            saw_final_add_42 = request.find("EVENT_BLOCK_ADD") != std::string::npos;
            EXPECT_EQ(std::string::npos, request.find("\"block_delete\":{\"block_key\":\"42\""));
        }
        if (request.find("\"block_key\":\"43\"") != std::string::npos) {
            saw_final_delete_43 = request.find("EVENT_BLOCK_DELETE") != std::string::npos;
            EXPECT_EQ(std::string::npos, request.find("\"block_add\":{\"block_key\":\"43\""));
        }
    }
    EXPECT_TRUE(saw_final_add_42);
    EXPECT_TRUE(saw_final_delete_43);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherQueueDoesNotDropConcurrentProducers) {
    constexpr size_t kThreadCount     = 8;
    constexpr size_t kEventsPerThread = 1000;
    constexpr size_t kEventCount      = kThreadCount * kEventsPerThread;

    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = kEventCount + 1;
    config.report_batch_size     = 256;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<CountingReporter>();
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {}};
        },
        reporter);
    ASSERT_TRUE(publisher.start());
    const auto ready_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
    while (publisher.status().state != PublisherState::READY && std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::yield();
    }
    ASSERT_EQ(PublisherState::READY, publisher.status().state);

    std::atomic<size_t>      accepted_count{0};
    std::vector<std::thread> producers;
    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        producers.emplace_back([&, thread_id] {
            for (size_t i = 0; i < kEventsPerThread; ++i) {
                const int64_t key = static_cast<int64_t>(thread_id * kEventsPerThread + i);
                if (publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0}) == PublishResult::ACCEPTED) {
                    accepted_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }

    EXPECT_EQ(kEventCount, accepted_count.load(std::memory_order_relaxed));
    EXPECT_EQ(0, publisher.status().dropped_count);
    EXPECT_TRUE(reporter->waitForMutationCount(kEventCount, std::chrono::seconds(5)));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromQueueOverflowWithSnapshot) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 1;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    std::atomic<int64_t> snapshot_version{1};
    auto                 reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher        publisher(
        config,
        makeContext(),
        [&snapshot_version] {
            return KVCacheSnapshot{snapshot_version.load(std::memory_order_relaxed), {10, 20, 30, 31}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    EXPECT_EQ(PublishResult::QUEUE_FULL, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    EXPECT_EQ(1, publisher.status().dropped_count);

    snapshot_version.store(2, std::memory_order_relaxed);
    reporter->releaseMutation();
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
}

}  // namespace
}  // namespace rtp_llm
