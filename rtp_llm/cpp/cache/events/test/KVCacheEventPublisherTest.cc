// Publisher-specific behavior tests are owned by the cache/events subsystem.
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherFactory.h"
#include "rtp_llm/cpp/cache/events/LogPublisher.h"
#include "rtp_llm/cpp/cache/events/NullPublisher.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <netinet/in.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

constexpr auto kAsyncTestTimeout = std::chrono::seconds(10);
constexpr auto kSnapshotVersion  = "0123456789abcdef0123456789abcdef";

enum class ValidationTarget {
    ENDPOINT,
    HOST,
    IDENTITY,
};

struct ValidationCase {
    ValidationTarget target;
    bool             expected_valid;
    const char*      value;
};

#define KV_CACHE_EVENT_VALIDATION_CASE(target, expected_valid, value) {ValidationTarget::target, expected_valid, value},
const ValidationCase kValidationCases[] = {
#include "rtp_llm/config/test/kv_cache_event_validation_cases.inc"
};
#undef KV_CACHE_EVENT_VALIDATION_CASE

class RecordingReporter final: public KVCacheEventReporter {
public:
    struct Request {
        std::string                           route;
        std::string                           body;
        std::chrono::steady_clock::time_point recorded_at;
    };
    struct ScriptedResponse {
        std::string body_substring;
        std::string response;
        bool        transport_ok = true;
    };

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        bool                       fail_request = false;
        std::optional<std::string> scripted_response;
        {
            std::lock_guard<std::mutex> lock(mu_);
            requests_.push_back({route, request, std::chrono::steady_clock::now()});
            // Match the first applicable script, preserving FIFO among
            // scripts for the same request class without making unrelated
            // heartbeat and snapshot scripts depend on arrival order.
            const auto script =
                std::find_if(scripted_responses_.begin(), scripted_responses_.end(), [&request](const auto& candidate) {
                    return request.find(candidate.body_substring) != std::string::npos;
                });
            if (script != scripted_responses_.end()) {
                fail_request      = !script->transport_ok;
                scripted_response = std::move(script->response);
                scripted_responses_.erase(script);
            }
            if (fail_body_count_ > 0 && !fail_body_.empty() && request.find(fail_body_) != std::string::npos) {
                fail_request = true;
                --fail_body_count_;
                if (fail_body_count_ == 0) {
                    fail_body_.clear();
                }
            }
        }
        if (scripted_response) {
            // An explicitly scripted empty body must remain empty so tests can
            // exercise the production malformed-response path.
            response = std::move(*scripted_response);
        } else if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            response = std::string(R"({"header":{"status":{"code":"OK"}},"committed_snapshot_version":")")
                       + kSnapshotVersion + R"("})";
        } else {
            response = R"({"header":{"status":{"code":"OK"}}})";
        }
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
    void respondNextBodyContaining(std::string text, std::string response, bool transport_ok = true) {
        std::lock_guard<std::mutex> lock(mu_);
        scripted_responses_.push_back({std::move(text), std::move(response), transport_ok});
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
    mutable std::mutex            mu_;
    std::condition_variable       cv_;
    std::vector<Request>          requests_;
    std::string                   fail_body_;
    size_t                        fail_body_count_{0};
    std::vector<ScriptedResponse> scripted_responses_;
};

class BlockingReporter final: public KVCacheEventReporter {
public:
    bool post(const std::string&, const std::string& request, std::string& response) noexcept override {
        std::unique_lock<std::mutex> lock(mu_);
        requests_.push_back(request);
        cv_.notify_all();
        bool request_snapshot = false;
        if (block_next_mutation_ && request.find("EVENT_BLOCK_ADD") != std::string::npos) {
            block_next_mutation_ = false;
            mutation_blocked_    = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return release_mutation_ || cancelled_; });
        }
        if (request_snapshot_on_next_mutation_ && request.find("EVENT_BLOCK_ADD") != std::string::npos) {
            request_snapshot                   = true;
            request_snapshot_on_next_mutation_ = false;
        }
        if (request_snapshot_on_next_heartbeat_ && request.find("EVENT_HEARTBEAT") != std::string::npos) {
            request_snapshot                    = true;
            request_snapshot_on_next_heartbeat_ = false;
        }
        if (block_next_snapshot_ && request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            block_next_snapshot_ = false;
            snapshot_blocked_    = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return release_snapshot_ || cancelled_; });
        }
        if (request_snapshot) {
            response = R"({"header":{"status":{"code":"OK"}},"snapshot_required":true})";
        } else if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            response = std::string(R"({"header":{"status":{"code":"OK"}},"committed_snapshot_version":")")
                       + kSnapshotVersion + R"("})";
        } else {
            response = R"({"header":{"status":{"code":"OK"}}})";
        }
        return true;
    }

    void cancel() noexcept override {
        {
            std::unique_lock<std::mutex> lock(mu_);
            ++cancel_count_;
            cv_.notify_all();
            if (block_cancel_) {
                cv_.wait(lock, [this] { return release_cancel_; });
            }
            cancelled_ = true;
        }
        cv_.notify_all();
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

    bool waitForBody(const std::string& text, std::chrono::milliseconds timeout) {
        return waitForBodyCount(text, 1, timeout);
    }

    void blockNextMutation() {
        std::lock_guard<std::mutex> lock(mu_);
        block_next_mutation_ = true;
        mutation_blocked_    = false;
        release_mutation_    = false;
    }

    void requestSnapshotOnNextMutation() {
        std::lock_guard<std::mutex> lock(mu_);
        request_snapshot_on_next_mutation_ = true;
    }

    void requestSnapshotOnNextHeartbeat() {
        std::lock_guard<std::mutex> lock(mu_);
        request_snapshot_on_next_heartbeat_ = true;
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
        snapshot_blocked_    = false;
        release_snapshot_    = false;
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

    void blockCancel() {
        std::lock_guard<std::mutex> lock(mu_);
        block_cancel_ = true;
    }

    void releaseCancel() {
        std::lock_guard<std::mutex> lock(mu_);
        release_cancel_ = true;
        cv_.notify_all();
    }

    size_t cancelCount() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cancel_count_;
    }

    std::vector<std::string> requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return requests_;
    }

private:
    mutable std::mutex       mu_;
    std::condition_variable  cv_;
    std::vector<std::string> requests_;
    bool                     block_next_mutation_                = false;
    bool                     request_snapshot_on_next_mutation_  = false;
    bool                     request_snapshot_on_next_heartbeat_ = false;
    bool                     mutation_blocked_                   = false;
    bool                     release_mutation_                   = false;
    bool                     block_next_snapshot_                = false;
    bool                     snapshot_blocked_                   = false;
    bool                     release_snapshot_                   = false;
    bool                     block_cancel_                       = false;
    bool                     release_cancel_                     = false;
    bool                     cancelled_                          = false;
    size_t                   cancel_count_                       = 0;
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

class CountingReporter final: public KVCacheEventReporter {
public:
    bool post(const std::string&, const std::string& request, std::string& response) noexcept override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            mutation_count_ += countOccurrences(request, "EVENT_BLOCK_ADD");
        }
        response = request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos ?
                       std::string(R"({"header":{"status":{"code":"OK"}},"committed_snapshot_version":")")
                           + kSnapshotVersion + R"("})" :
                       R"({"header":{"status":{"code":"OK"}}})";
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
    explicit LocalHttpStub(std::string blocked_event = "EVENT_BLOCK_SNAPSHOT"):
        blocked_event_(std::move(blocked_event)) {
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
        releaseBlockedRequest();
        stopping_.store(true, std::memory_order_release);
        // Deterministically wake accept() even when the process cannot create
        // another file descriptor for a loopback self-connect.
        (void)::shutdown(listen_fd_, SHUT_RDWR);
        if (worker_.joinable()) {
            worker_.join();
        }
        ::close(listen_fd_);
    }

    std::string endpoint() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    bool waitUntilBlockedRequestIsInFlight(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        (void)cv_.wait_for(lock, timeout, [this] { return blocked_request_in_flight_ || !server_error_.empty(); });
        return blocked_request_in_flight_;
    }

    bool waitForBody(const std::string& text, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        (void)cv_.wait_for(lock, timeout, [this, &text] {
            return !server_error_.empty()
                   || std::any_of(request_bodies_.begin(), request_bodies_.end(), [&text](const auto& body) {
                          return body.find(text) != std::string::npos;
                      });
        });
        return std::any_of(request_bodies_.begin(), request_bodies_.end(), [&text](const auto& body) {
            return body.find(text) != std::string::npos;
        });
    }

    std::string serverError() const {
        std::lock_guard<std::mutex> lock(mu_);
        return server_error_;
    }

    void releaseBlockedRequest() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            release_blocked_request_ = true;
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

    static void writeSuccess(int fd, bool snapshot) {
        const std::string response_body =
            snapshot ? std::string(R"({"header":{"status":{"code":"OK"}},"committed_snapshot_version":")")
                           + kSnapshotVersion + R"("})" :
                       R"({"header":{"status":{"code":"OK"}}})";
        const std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                     + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n"
                                     + response_body;
        size_t sent = 0;
        while (sent < response.size()) {
            const auto bytes = ::send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
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
                const int accept_error = errno;
                if (stopping_.load(std::memory_order_acquire)) {
                    break;
                }
                if (accept_error == EINTR) {
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    server_error_ = "accept failed: " + std::string(std::strerror(accept_error));
                }
                cv_.notify_all();
                break;
            }
            if (stopping_.load(std::memory_order_acquire)) {
                ::close(client_fd);
                break;
            }

            try {
                std::string body;
                if (readRequest(client_fd, body)) {
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                        request_bodies_.push_back(body);
                    }
                    cv_.notify_all();
                    if (!blocked_event_.empty() && body.find(blocked_event_) != std::string::npos) {
                        std::unique_lock<std::mutex> lock(mu_);
                        blocked_request_in_flight_ = true;
                        cv_.notify_all();
                        cv_.wait(lock, [this] {
                            return release_blocked_request_ || stopping_.load(std::memory_order_acquire);
                        });
                    } else {
                        writeSuccess(client_fd, body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos);
                    }
                }
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(mu_);
                server_error_ = "request handling failed: " + std::string(error.what());
                cv_.notify_all();
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu_);
                server_error_ = "request handling failed with an unknown exception";
                cv_.notify_all();
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
    std::string              server_error_;
    std::string              blocked_event_;
    bool                     blocked_request_in_flight_{false};
    bool                     release_blocked_request_{false};
};

KVCacheEventPublisherContext makeContext() {
    KVCacheEventPublisherContext context;
    context.instance_group    = "test_group";
    context.instance_id       = "test_instance";
    context.host_ip_port      = "127.0.0.1:9000";
    context.model_name        = "test_model";
    context.dtype             = "BF16";
    context.spec_name         = "rtp_llm_hbm_64";
    context.location_uri      = "rtp-llm://127.0.0.1:9000/hbm?size=4096";
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

TEST(KVCacheEventPublisherTest, RecordingReporterScriptsAreRequestScopedAndPreserveEmptyBodies) {
    RecordingReporter reporter;
    reporter.respondNextBodyContaining("EVENT_BLOCK_SNAPSHOT", "");
    reporter.respondNextBodyContaining("EVENT_HEARTBEAT", "heartbeat-response");

    std::string response;
    EXPECT_TRUE(reporter.post("/api/reportEvent", R"({"event_type":"EVENT_HEARTBEAT"})", response));
    EXPECT_EQ("heartbeat-response", response);

    response = "stale-response";
    EXPECT_TRUE(reporter.post("/api/reportEvent", R"({"event_type":"EVENT_BLOCK_SNAPSHOT"})", response));
    EXPECT_TRUE(response.empty());
}

TEST(KVCacheEventPublisherTest, BlockingReporterSupportsSequentialOneShotBlocks) {
    BlockingReporter reporter;
    for (const int64_t key : {10, 20}) {
        SCOPED_TRACE(key);
        reporter.blockNextMutation();
        std::string response;
        auto        request = std::async(std::launch::async, [&reporter, &response, key] {
            return reporter.post("/api/reportEvent",
                                 "{\"event_type\":\"EVENT_BLOCK_ADD\",\"block_key\":\"" + std::to_string(key) + "\"}",
                                 response);
        });

        const bool blocked = reporter.waitUntilMutationBlocked(kAsyncTestTimeout);
        if (!blocked) {
            reporter.releaseMutation();
        }
        ASSERT_TRUE(blocked);
        EXPECT_EQ(std::future_status::timeout, request.wait_for(std::chrono::milliseconds(100)));
        reporter.releaseMutation();
        ASSERT_EQ(std::future_status::ready, request.wait_for(kAsyncTestTimeout));
        EXPECT_TRUE(request.get());
        EXPECT_FALSE(response.empty());
    }
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
    publisher     = createKVCacheEventPublisher(config, context, [] { return KVCacheSnapshot{{}}; }, reporter);
    ASSERT_NE(nullptr, publisher);
    EXPECT_TRUE(publisher->enabled());
    ASSERT_TRUE(publisher->start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(*publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher->stop();

    config.type = "unsupported";
    publisher   = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_FALSE(publisher->enabled());
}

TEST(KVCacheEventPublisherTest, KVCMEndpointNormalizationIsStable) {
    EXPECT_EQ("", detail::normalizeKVCacheEventEndpoint(""));
    EXPECT_EQ("kvcm-meta:56020///", detail::normalizeKVCacheEventEndpoint("kvcm-meta:56020///"));
    EXPECT_EQ("http://kvcm-meta:56020", detail::normalizeKVCacheEventEndpoint("http://kvcm-meta:56020/"));
    EXPECT_EQ("https://kvcm-meta.example", detail::normalizeKVCacheEventEndpoint("https://kvcm-meta.example///"));
    EXPECT_EQ("http://", detail::normalizeKVCacheEventEndpoint("http://"));

    for (const auto& test_case : kValidationCases) {
        const bool actual =
            test_case.target == ValidationTarget::ENDPOINT ? detail::isValidKVCacheEventEndpoint(test_case.value) :
            test_case.target == ValidationTarget::HOST     ? detail::isValidKVCacheEventHostIpPort(test_case.value) :
                                                             detail::isValidKVCacheEventIdentity(test_case.value);
        EXPECT_EQ(test_case.expected_valid, actual) << (test_case.target == ValidationTarget::ENDPOINT ? "endpoint" :
                                                        test_case.target == ValidationTarget::HOST     ? "host" :
                                                                                                         "identity")
                                                    << '=' << test_case.value;
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRejectsMalformedBuiltinEndpointBeforeStartingWorker) {
    KVCacheEventPublisherConfig config;
    config.type             = "kvcm";
    config.manager_endpoint = "ftp://kvcm-meta:56020";

    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; });
    EXPECT_FALSE(publisher.start());
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
}

TEST(KVCacheEventPublisherTest, KVCMResponseCodeMappingCoversEveryPublishedProtocolCode) {
    struct CodeCase {
        int64_t                  numeric;
        const char*              name;
        detail::KVCMResponseCode expected;
    };
    const std::vector<CodeCase> cases = {
        {0, "UNSPECIFIED", detail::KVCMResponseCode::UNSPECIFIED},
        {1, "OK", detail::KVCMResponseCode::OK},
        {2, "UNSUPPORTED", detail::KVCMResponseCode::UNSUPPORTED},
        {3, "INTERNAL_ERROR", detail::KVCMResponseCode::INTERNAL_ERROR},
        {4, "SERVICE_NOT_READY", detail::KVCMResponseCode::SERVICE_NOT_READY},
        {5, "INVALID_ARGUMENT", detail::KVCMResponseCode::INVALID_ARGUMENT},
        {6, "DUPLICATE_ENTITY", detail::KVCMResponseCode::DUPLICATE_ENTITY},
        {7, "REACH_MAX_ENTITY_CAPACITY", detail::KVCMResponseCode::REACH_MAX_ENTITY_CAPACITY},
        {8, "INSTANCE_NOT_EXIST", detail::KVCMResponseCode::INSTANCE_NOT_EXIST},
        {9, "SERVER_NOT_LEADER", detail::KVCMResponseCode::SERVER_NOT_LEADER},
        {10, "NODE_NOT_REGISTERED", detail::KVCMResponseCode::NODE_NOT_REGISTERED},
        {11, "SNAPSHOT_IN_PROGRESS", detail::KVCMResponseCode::SNAPSHOT_IN_PROGRESS},
        {13, "SNAPSHOT_RATE_LIMITED", detail::KVCMResponseCode::SNAPSHOT_RATE_LIMITED},
        {14, "SNAPSHOT_REQUIRED", detail::KVCMResponseCode::SNAPSHOT_REQUIRED},
        {20, "IO_ERROR", detail::KVCMResponseCode::IO_ERROR},
        {100, "UNKNOWN_ERROR", detail::KVCMResponseCode::UNKNOWN_ERROR},
        {65535, "ERROR_MAX", detail::KVCMResponseCode::ERROR_MAX},
    };
    const auto parseCode = [](const std::string& json_code) {
        return detail::parseKVCMResponse("{\"header\":{\"status\":{\"code\":" + json_code + "}}}");
    };

    for (const auto& code_case : cases) {
        SCOPED_TRACE(code_case.name);
        for (const auto& representation : {std::to_string(code_case.numeric),
                                           "\"" + std::to_string(code_case.numeric) + "\"",
                                           "\"" + std::string(code_case.name) + "\""}) {
            const auto parsed = parseCode(representation);
            ASSERT_TRUE(parsed.parsed);
            EXPECT_FALSE(parsed.has_unrecognized_code);
            EXPECT_EQ(code_case.expected, parsed.header_code);
            EXPECT_EQ(code_case.expected == detail::KVCMResponseCode::OK, parsed.ok());
        }
    }

    for (const auto& representation : {std::string("12"), std::string("\"12\""), std::string("\"FUTURE_CODE\"")}) {
        const auto parsed = parseCode(representation);
        ASSERT_TRUE(parsed.parsed);
        EXPECT_TRUE(parsed.has_unrecognized_code);
        EXPECT_EQ(detail::KVCMResponseCode::UNRECOGNIZED, parsed.header_code);
        EXPECT_FALSE(parsed.ok());
    }
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
        {R"({"header":{"status":{"code":true}}})", false},
        {R"({"header":{"status":{"code":"OK"}},"item_results":[{}]})", false},
        {R"({"header":{"status":{}}})", false},
        {R"({"header":{}})", false},
        {R"({"header":{"status":{"code":"OK"}})", false},
        {"not-json", false},
    };

    for (const auto& [response, expected] : cases) {
        EXPECT_EQ(expected, detail::parseKVCMResponse(response).ok()) << response;
    }

    // The parser is also a hard allocation boundary for injected and future
    // reporters that do not use the built-in curl receive callback.
    EXPECT_FALSE(detail::parseKVCMResponse(std::string(kKVCacheEventMaxResponseBytes + 1, 'x')).parsed);

    const auto nested_response = [](size_t depth) {
        std::string response = R"({"header":{"status":{"code":"OK"}},"padding":)";
        response.append(depth, '[');
        response.push_back('0');
        response.append(depth, ']');
        response.push_back('}');
        return response;
    };
    EXPECT_TRUE(detail::parseKVCMResponse(nested_response(8)).ok());
    EXPECT_FALSE(detail::parseKVCMResponse(nested_response(128)).parsed);

    std::string embedded_nul = R"({"header":{"status":{"code":"OK"}}})";
    embedded_nul.push_back('\0');
    embedded_nul += "{}";
    EXPECT_FALSE(detail::parseKVCMResponse(embedded_nul).parsed);

    const auto parsed = detail::parseKVCMResponse(R"({"header":{"status":{"code":"OK"}},"itemResults":["OK",1],)"
                                                  R"("committedSnapshotVersion":"generation-7","retryAfterMs":"123",)"
                                                  R"("snapshotRequired":true})");
    ASSERT_TRUE(parsed.parsed);
    EXPECT_TRUE(parsed.ok());
    EXPECT_EQ("generation-7", parsed.committed_snapshot_version);
    EXPECT_EQ(123u, parsed.retry_after_ms);
    EXPECT_TRUE(parsed.requestsSnapshot());

    const auto node_missing =
        detail::parseKVCMResponse(R"({"header":{"status":{"code":"OK"}},"item_results":["NODE_NOT_REGISTERED"]})");
    ASSERT_TRUE(node_missing.parsed);
    EXPECT_FALSE(node_missing.ok());
    EXPECT_TRUE(node_missing.requiresRegistration());

    const auto instance_missing = detail::parseKVCMResponse(R"({"header":{"status":{"code":"INSTANCE_NOT_EXIST"}}})");
    ASSERT_TRUE(instance_missing.parsed);
    EXPECT_FALSE(instance_missing.ok());
    EXPECT_TRUE(instance_missing.requiresRegistration());

    const auto rate_limited = detail::parseKVCMResponse(R"({"header":{"status":{"code":13}},"retry_after_ms":25})");
    ASSERT_TRUE(rate_limited.parsed);
    EXPECT_FALSE(rate_limited.ok());
    EXPECT_FALSE(rate_limited.requiresRegistration());

    const auto snapshot_required = detail::parseKVCMResponse(R"({"header":{"status":{"code":"SNAPSHOT_REQUIRED"}}})");
    ASSERT_TRUE(snapshot_required.parsed);
    EXPECT_TRUE(snapshot_required.requestsSnapshot());
    EXPECT_FALSE(snapshot_required.requiresRegistration());

    const auto mixed_registration =
        detail::parseKVCMResponse(R"({"header":{"status":{"code":"OK"}},)"
                                  R"("item_results":["INTERNAL_ERROR","NODE_NOT_REGISTERED"]})");
    ASSERT_TRUE(mixed_registration.parsed);
    EXPECT_EQ(detail::KVCMResponseCode::INTERNAL_ERROR, mixed_registration.firstFailure());
    EXPECT_TRUE(mixed_registration.requiresRegistration());
    EXPECT_FALSE(mixed_registration.hasPermanentFailure());

    const auto mixed_snapshot =
        detail::parseKVCMResponse(R"({"header":{"status":{"code":"OK"}},)"
                                  R"("item_results":["SERVICE_NOT_READY","SNAPSHOT_REQUIRED"]})");
    ASSERT_TRUE(mixed_snapshot.parsed);
    EXPECT_EQ(detail::KVCMResponseCode::SERVICE_NOT_READY, mixed_snapshot.firstFailure());
    EXPECT_TRUE(mixed_snapshot.requestsSnapshot());

    const auto mixed_permanent = detail::parseKVCMResponse(R"({"header":{"status":{"code":"OK"}},)"
                                                           R"("item_results":["INTERNAL_ERROR","INVALID_ARGUMENT"]})");
    ASSERT_TRUE(mixed_permanent.parsed);
    EXPECT_EQ(detail::KVCMResponseCode::INTERNAL_ERROR, mixed_permanent.firstFailure());
    EXPECT_TRUE(mixed_permanent.hasPermanentFailure());

    const auto future_code = detail::parseKVCMResponse(
        R"({"header":{"status":{"code":4242}},"item_results":["FUTURE_CODE"],)"
        R"("committed_snapshot_version":"generation-8","retry_after_ms":"321","snapshot_required":true})");
    ASSERT_TRUE(future_code.parsed);
    EXPECT_TRUE(future_code.has_unrecognized_code);
    EXPECT_FALSE(future_code.ok());
    EXPECT_EQ(detail::KVCMResponseCode::UNRECOGNIZED, future_code.firstFailure());
    EXPECT_EQ("generation-8", future_code.committed_snapshot_version);
    EXPECT_EQ(321u, future_code.retry_after_ms);
    EXPECT_TRUE(future_code.requestsSnapshot());
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRejectsInvalidIdentity) {
    using Mutator                                            = std::function<void(KVCacheEventPublisherContext&)>;
    const std::vector<std::pair<const char*, Mutator>> cases = {
        {"empty_instance_id", [](auto& context) { context.instance_id.clear(); }},
        {"spaced_instance_id", [](auto& context) { context.instance_id = "instance id"; }},
        {"non_ascii_instance_id", [](auto& context) { context.instance_id = "instance-\xc3\xa9"; }},
        {"spaced_instance_group", [](auto& context) { context.instance_group = "instance group"; }},
    };

    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        KVCacheEventPublisherConfig config;
        config.type  = "kvcm";
        auto context = makeContext();
        mutate(context);
        auto          reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(config, context, [] { return KVCacheSnapshot{}; }, reporter);

        EXPECT_FALSE(publisher.start());
        EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
        EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 1, 0}));
        EXPECT_TRUE(reporter->requests().empty());
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRejectsHostThatCannotFormKVCMLocationId) {
    KVCacheEventPublisherConfig config;
    config.type            = "kvcm";
    auto context           = makeContext();
    context.host_ip_port   = "127.0.0.1#9000";
    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, context, [] { return KVCacheSnapshot{}; }, reporter);

    EXPECT_FALSE(publisher.start());
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 1, 0}));
    EXPECT_TRUE(reporter->requests().empty());
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRejectsInvalidParallelIdentity) {
    using Mutator                                            = std::function<void(KVCacheEventPublisherContext&)>;
    const std::vector<std::pair<const char*, Mutator>> cases = {
        {"tp_size", [](auto& context) { context.tp_size = 0; }},
        {"dp_size", [](auto& context) { context.dp_size = 0; }},
        {"pp_size", [](auto& context) { context.pp_size = 0; }},
        {"unsupported_pp_size", [](auto& context) { context.pp_size = 2; }},
        {"negative_dp_rank", [](auto& context) { context.dp_rank = -1; }},
        {"out_of_range_dp_rank", [](auto& context) { context.dp_rank = context.dp_size; }},
    };

    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        KVCacheEventPublisherConfig config;
        config.type  = "kvcm";
        auto context = makeContext();
        mutate(context);
        auto          reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(config, context, [] { return KVCacheSnapshot{}; }, reporter);

        EXPECT_FALSE(publisher.start());
        EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
        EXPECT_TRUE(reporter->requests().empty());
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDefensivelyRejectsNonPositiveRuntimeSettings) {
    using Mutator                                            = std::function<void(KVCacheEventPublisherConfig&)>;
    const std::vector<std::pair<const char*, Mutator>> cases = {
        {"queue_capacity", [](auto& config) { config.queue_capacity = 0; }},
        {"report_batch_size", [](auto& config) { config.report_batch_size = 0; }},
        {"flush_interval_ms", [](auto& config) { config.flush_interval_ms = 0; }},
        {"heartbeat_interval_ms", [](auto& config) { config.heartbeat_interval_ms = 0; }},
        {"request_timeout_ms", [](auto& config) { config.request_timeout_ms = 0; }},
        {"snapshot_timeout_ms", [](auto& config) { config.snapshot_timeout_ms = 0; }},
        {"retry_interval_ms", [](auto& config) { config.retry_interval_ms = 0; }},
        {"snapshot_interval_ms", [](auto& config) { config.snapshot_interval_ms = 0; }},
        {"snapshot_max_keys", [](auto& config) { config.snapshot_max_keys = 0; }},
        {"snapshot_max_bytes", [](auto& config) { config.snapshot_max_bytes = 0; }},
        {"report_max_bytes", [](auto& config) { config.report_max_bytes = 0; }},
    };

    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        KVCacheEventPublisherConfig config;
        config.type = "kvcm";
        mutate(config);
        auto          reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter);

        EXPECT_FALSE(publisher.start());
        EXPECT_FALSE(publisher.enabled());
        EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
        EXPECT_TRUE(reporter->requests().empty());
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDefensivelyRejectsOversizedRuntimeSettings) {
    using Mutator                                            = std::function<void(KVCacheEventPublisherConfig&)>;
    const std::vector<std::pair<const char*, Mutator>> cases = {
        {"report_batch_size", [](auto& config) { config.report_batch_size = kKVCacheEventMaxReportBatchSize + 1; }},
        {"snapshot_max_keys", [](auto& config) { config.snapshot_max_keys = kKVCacheEventMaxSnapshotKeys + 1; }},
        {"snapshot_max_bytes", [](auto& config) { config.snapshot_max_bytes = kKVCacheEventMaxSnapshotBytes + 1; }},
        {"report_max_bytes", [](auto& config) { config.report_max_bytes = kKVCacheEventMaxReportBytes + 1; }},
    };

    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        KVCacheEventPublisherConfig config;
        config.type = "kvcm";
        mutate(config);
        auto          reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter);

        EXPECT_FALSE(publisher.start());
        EXPECT_FALSE(publisher.enabled());
        EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
        EXPECT_TRUE(reporter->requests().empty());
    }
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

    auto publisher = createKVCacheEventPublisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; });
    ASSERT_TRUE(publisher->enabled());
    ASSERT_TRUE(publisher->start());
    if (!server.waitUntilBlockedRequestIsInFlight(std::chrono::seconds(5))) {
        publisher->stop();
        FAIL() << "real curl snapshot request did not reach the local HTTP stub: " << server.serverError();
    }

    auto       stopped     = std::async(std::launch::async, [&] { publisher->stop(); });
    const auto stop_status = stopped.wait_for(std::chrono::seconds(5));
    server.releaseBlockedRequest();
    ASSERT_EQ(std::future_status::ready, stop_status)
        << "stop waited for the 30 second snapshot timeout instead of cancelling curl";
    stopped.get();

    const auto request_bodies = server.requestBodies();
    ASSERT_GE(request_bodies.size(), 3u);
    EXPECT_NE(std::string::npos, request_bodies[0].find("\"instance_group\""));
    EXPECT_NE(std::string::npos, request_bodies[1].find("EVENT_NODE_REGISTER"));
    EXPECT_NE(std::string::npos, request_bodies[2].find("EVENT_BLOCK_SNAPSHOT"));
}

TEST(KVCacheEventPublisherTest, RealCurlMutationRequestIsCancelledDuringStop) {
    LocalHttpStub               server("EVENT_BLOCK_ADD");
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.manager_endpoint      = server.endpoint();
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.request_timeout_ms    = 30000;
    config.snapshot_timeout_ms   = 2000;
    config.retry_interval_ms     = 1;
    config.snapshot_interval_ms  = 60000;

    auto publisher = createKVCacheEventPublisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; });
    ASSERT_TRUE(publisher->enabled());
    ASSERT_TRUE(publisher->start());
    ASSERT_TRUE(server.waitForBody("EVENT_BLOCK_SNAPSHOT", std::chrono::seconds(5)));
    ASSERT_TRUE(waitForState(*publisher, PublisherState::READY, std::chrono::seconds(5)));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    if (!server.waitUntilBlockedRequestIsInFlight(std::chrono::seconds(5))) {
        publisher->stop();
        FAIL() << "real curl mutation request did not reach the local HTTP stub: " << server.serverError();
    }

    auto       stopped     = std::async(std::launch::async, [&] { publisher->stop(); });
    const auto stop_status = stopped.wait_for(std::chrono::seconds(5));
    server.releaseBlockedRequest();
    ASSERT_EQ(std::future_status::ready, stop_status)
        << "stop waited for the 30 second mutation timeout instead of cancelling curl";
    stopped.get();

    const auto request_bodies = server.requestBodies();
    EXPECT_TRUE(std::any_of(request_bodies.begin(), request_bodies.end(), [](const auto& body) {
        return body.find("EVENT_BLOCK_ADD") != std::string::npos;
    }));
}

TEST(KVCacheEventPublisherTest, LogPublisherAcceptsEventsAsynchronously) {
    KVCacheEventPublisherConfig config;
    config.type              = "log";
    config.queue_capacity    = 8;
    config.report_batch_size = 8;
    config.flush_interval_ms = 1;

    LogPublisher publisher(config, makeContext());
    EXPECT_TRUE(publisher.enabled());
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 41, 0}));
    EXPECT_EQ(0, publisher.status().accepted_count);
    EXPECT_EQ(0, publisher.status().dropped_count);
    ASSERT_TRUE(publisher.start());
    EXPECT_TRUE(publisher.enabled());
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    publisher.stop();
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
    EXPECT_EQ(1u, publisher.status().queue_high_watermark);
    EXPECT_EQ(1, publisher.status().accepted_count);
    EXPECT_EQ(0, publisher.status().dropped_count);
}

TEST(KVCacheEventPublisherTest, LogPublisherDefensivelyRejectsOversizedBatch) {
    KVCacheEventPublisherConfig config;
    config.type              = "log";
    config.queue_capacity    = 8;
    config.report_batch_size = kKVCacheEventMaxReportBatchSize + 1;

    LogPublisher publisher(config, makeContext());
    EXPECT_FALSE(publisher.start());
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
}

TEST(KVCacheEventPublisherTest, LogPublisherQueueOverflowPermanentlyOpensCircuit) {
    KVCacheEventPublisherConfig config;
    config.type                   = "log";
    config.queue_capacity         = 1;
    config.report_batch_size      = 1;
    config.flush_interval_ms      = 1;
    config.log_max_keys_per_batch = 0;

    LogPublisher publisher(config, makeContext());
    ASSERT_TRUE(publisher.start());

    // The producer is deliberately faster than the worker's log sink. A
    // bounded exporter must fail closed as soon as that sink falls behind,
    // without ever blocking the cache-mutation path.
    PublishResult result = PublishResult::ACCEPTED;
    for (int64_t key = 0; key < 10000 && result != PublishResult::QUEUE_FULL; ++key) {
        result = publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0});
    }

    ASSERT_EQ(PublishResult::QUEUE_FULL, result);
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(1u, publisher.status().dropped_count);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 10001, 0}));
    EXPECT_FALSE(publisher.start());

    // Circuit state makes the backlog unactionable immediately; status must
    // not expose a transient stale depth while worker cleanup catches up.
    EXPECT_EQ(0u, publisher.status().queue_size);

    publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
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
    EXPECT_FALSE(log_publisher.enabled());
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
    KVCMPublisher kvcm_publisher(kvcm_config, makeContext(), [] { return KVCacheSnapshot{{}}; }, reporter);
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

TEST(KVCacheEventPublisherTest, StopBeforeStartPermanentlyClosesPublisher) {
    KVCacheEventPublisherConfig log_config;
    log_config.type = "log";
    LogPublisher log_publisher(log_config, makeContext());
    log_publisher.stop();
    EXPECT_FALSE(log_publisher.start());
    EXPECT_FALSE(log_publisher.enabled());
    EXPECT_EQ(PublisherState::STOPPED, log_publisher.status().state);

    KVCacheEventPublisherConfig kvcm_config;
    kvcm_config.type             = "kvcm";
    kvcm_config.manager_endpoint = "http://127.0.0.1:1";
    auto          reporter       = std::make_shared<RecordingReporter>();
    KVCMPublisher kvcm_publisher(kvcm_config, makeContext(), [] { return KVCacheSnapshot{{}}; }, reporter);
    kvcm_publisher.stop();
    EXPECT_FALSE(kvcm_publisher.start());
    EXPECT_FALSE(kvcm_publisher.enabled());
    EXPECT_EQ(PublisherState::STOPPED, kvcm_publisher.status().state);
    EXPECT_TRUE(reporter->requests().empty());
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
        std::thread       starter([&] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            (void)publisher.start();
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

        // Either lifecycle call may acquire the mutex first. If stop wins,
        // start rejects the already-closed one-shot publisher; if start wins,
        // stop joins its worker. Both orders end in STOPPED.
        EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
    }
}

TEST(KVCacheEventPublisherTest, ConcurrentPublishingAndStopPreservesAccountingAndFinalState) {
    constexpr size_t kProducerCount = 8;

    KVCacheEventPublisherConfig config;
    config.type                   = "log";
    config.queue_capacity         = 65536;
    config.report_batch_size      = 1024;
    config.flush_interval_ms      = 1;
    config.log_max_keys_per_batch = 0;

    LogPublisher publisher(config, makeContext());
    ASSERT_TRUE(publisher.start());

    std::atomic<bool>        go{false};
    std::atomic<size_t>      ready{0};
    std::atomic<size_t>      accepted{0};
    std::atomic<size_t>      unexpected{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_id = 0; producer_id < kProducerCount; ++producer_id) {
        producers.emplace_back([&, producer_id] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            size_t ordinal = 0;
            for (;;) {
                const auto key    = static_cast<int64_t>(producer_id * config.queue_capacity + ordinal++);
                const auto result = publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0});
                if (result == PublishResult::ACCEPTED) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                } else if (result == PublishResult::NOT_RUNNING) {
                    break;
                } else {
                    unexpected.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kProducerCount) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (accepted.load(std::memory_order_relaxed) < 1000 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool produced_enough = accepted.load(std::memory_order_relaxed) >= 1000;

    publisher.stop();
    const auto status_at_stop = publisher.status();
    for (auto& producer : producers) {
        producer.join();
    }

    ASSERT_TRUE(produced_enough);
    const auto status = publisher.status();
    EXPECT_EQ(PublisherState::STOPPED, status.state);
    EXPECT_EQ(status_at_stop.accepted_count, status.accepted_count);
    EXPECT_EQ(status_at_stop.dropped_count, status.dropped_count);
    EXPECT_EQ(accepted.load(std::memory_order_relaxed), status.accepted_count);
    EXPECT_EQ(0u, status.queue_size);
    EXPECT_EQ(0u, unexpected.load(std::memory_order_relaxed));
    EXPECT_EQ(0u, status.dropped_count);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, -1, 0}));
}

TEST(KVCacheEventPublisherTest, ConcurrentKVCMPublishingAndStopPreservesAccountingAndFinalState) {
    constexpr size_t kProducerCount = 8;

    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 65536;
    config.report_batch_size     = 1024;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    std::atomic<bool>        go{false};
    std::atomic<size_t>      ready{0};
    std::atomic<size_t>      accepted{0};
    std::atomic<size_t>      unexpected{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_id = 0; producer_id < kProducerCount; ++producer_id) {
        producers.emplace_back([&, producer_id] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            size_t ordinal = 0;
            for (;;) {
                const auto key    = static_cast<int64_t>(producer_id * config.queue_capacity + ordinal++);
                const auto result = publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0});
                if (result == PublishResult::ACCEPTED) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                } else if (result == PublishResult::NOT_RUNNING) {
                    break;
                } else {
                    unexpected.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kProducerCount) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (accepted.load(std::memory_order_relaxed) < 1000 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool produced_enough = accepted.load(std::memory_order_relaxed) >= 1000;

    publisher.stop();
    const auto status_at_stop = publisher.status();
    for (auto& producer : producers) {
        producer.join();
    }

    ASSERT_TRUE(produced_enough);
    const auto status = publisher.status();
    EXPECT_EQ(PublisherState::STOPPED, status.state);
    EXPECT_EQ(status_at_stop.accepted_count, status.accepted_count);
    EXPECT_EQ(status_at_stop.dropped_count, status.dropped_count);
    EXPECT_EQ(accepted.load(std::memory_order_relaxed), status.accepted_count);
    EXPECT_EQ(0u, unexpected.load(std::memory_order_relaxed));
    EXPECT_EQ(0u, status.dropped_count);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, -1, 0}));
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

    auto reporter = std::make_shared<RecordingReporter>();
    auto publisher =
        std::make_shared<KVCMPublisher>(config, makeContext(), [] { return KVCacheSnapshot{{10, 20}}; }, reporter);

    ASSERT_TRUE(publisher->start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(*publisher, PublisherState::READY, kAsyncTestTimeout));
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
    EXPECT_NE(std::string::npos, requests.front().body.find("\"size\":4096"));
    EXPECT_NE(std::string::npos, requests.front().body.find("\"pp_size\":1"));
    EXPECT_EQ(1, countOccurrences(requests.front().body, "\"dtype\""));

    bool   saw_node_register = false;
    bool   saw_snapshot      = false;
    bool   saw_snapshot_item = false;
    bool   saw_add           = false;
    bool   saw_delete        = false;
    bool   saw_sized_uri     = false;
    size_t host_down_count   = 0;
    for (const auto& request : requests) {
        saw_node_register = saw_node_register || request.body.find("EVENT_NODE_REGISTER") != std::string::npos;
        saw_snapshot      = saw_snapshot || request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        saw_snapshot_item =
            saw_snapshot_item || request.body.find("\"block_key\":\"10\",\"medium\":\"hbm\"") != std::string::npos;
        saw_add       = saw_add || request.body.find("EVENT_BLOCK_ADD") != std::string::npos;
        saw_delete    = saw_delete || request.body.find("EVENT_BLOCK_DELETE") != std::string::npos;
        saw_sized_uri = saw_sized_uri
                        || request.body.find("\"uri\":\"rtp-llm://127.0.0.1:9000/hbm?size=4096\"") != std::string::npos;
        host_down_count += request.body.find("EVENT_HOST_DOWN") != std::string::npos;
    }
    EXPECT_TRUE(saw_node_register);
    EXPECT_TRUE(saw_snapshot);
    EXPECT_TRUE(saw_snapshot_item);
    EXPECT_TRUE(saw_add);
    EXPECT_TRUE(saw_delete);
    EXPECT_TRUE(saw_sized_uri);
    // HOST_DOWN is terminal. Registration and recovery use NODE_REGISTER plus
    // an authoritative snapshot rather than pretending the live engine exited.
    EXPECT_EQ(1, host_down_count);
    EXPECT_EQ(3, publisher->status().accepted_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherControlDeadlinesPreemptLongBatchWait) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 60000;
    config.heartbeat_interval_ms = 30;
    config.snapshot_interval_ms  = 40;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    // Neither deadline may inherit the 60-second mutation batching wait.
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 1, std::chrono::seconds(1)));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, std::chrono::seconds(1)));
    publisher.stop();
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
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherHonorsRegisterInstanceRetryAfter) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->respondNextBodyContaining("\"instance_group\"",
                                        R"({"header":{"status":{"code":"SERVICE_NOT_READY"}},"retry_after_ms":80})");
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    publisher.stop();

    std::vector<RecordingReporter::Request> registrations;
    for (const auto& request : reporter->requests()) {
        if (request.route == "/api/registerInstance") {
            registrations.push_back(request);
        }
    }
    ASSERT_GE(registrations.size(), 2u);
    const auto retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(registrations[1].recorded_at
                                                                                   - registrations[0].recorded_at);
    EXPECT_GE(retry_delay.count(), 65);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherHonorsNodeRegisterRetryAfter) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->respondNextBodyContaining("EVENT_NODE_REGISTER",
                                        R"({"header":{"status":{"code":"SERVICE_NOT_READY"}},"retry_after_ms":"80"})");
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    publisher.stop();

    std::optional<RecordingReporter::Request> first_node_registration;
    std::vector<RecordingReporter::Request>   registrations;
    for (const auto& request : reporter->requests()) {
        if (!first_node_registration && request.body.find("EVENT_NODE_REGISTER") != std::string::npos) {
            first_node_registration = request;
        }
        if (request.route == "/api/registerInstance") {
            registrations.push_back(request);
        }
    }
    ASSERT_TRUE(first_node_registration.has_value());
    ASSERT_GE(registrations.size(), 2u);
    const auto retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        registrations[1].recorded_at - first_node_registration->recorded_at);
    EXPECT_GE(retry_delay.count(), 65);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherFailsOpenWhenInitialSnapshotProviderThrows) {
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
        [&snapshot_attempts]() -> KVCacheSnapshot {
            snapshot_attempts.fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error("injected snapshot failure");
        },
        reporter);

    EXPECT_FALSE(publisher.start());
    EXPECT_FALSE(publisher.start());
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(1u, snapshot_attempts.load(std::memory_order_relaxed));
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 10, 0}));
    EXPECT_TRUE(reporter->requests().empty());
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDiscardsIngressWhenConcurrentInitialSnapshotFails) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    std::promise<void> snapshot_entered;
    auto               snapshot_entered_future = snapshot_entered.get_future();
    std::promise<void> release_snapshot;
    auto               release_snapshot_future = release_snapshot.get_future();
    auto               reporter                = std::make_shared<RecordingReporter>();
    KVCMPublisher      publisher(
        config,
        makeContext(),
        [&snapshot_entered, &release_snapshot_future]() -> KVCacheSnapshot {
            snapshot_entered.set_value();
            release_snapshot_future.wait();
            throw std::runtime_error("injected concurrent snapshot failure");
        },
        reporter);

    auto start_result = std::async(std::launch::async, [&publisher] { return publisher.start(); });
    if (snapshot_entered_future.wait_for(kAsyncTestTimeout) != std::future_status::ready) {
        release_snapshot.set_value();
        (void)start_result.get();
        FAIL() << "initial snapshot provider did not start";
    }

    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 10, 0}));
    EXPECT_EQ(1u, publisher.status().queue_size);
    release_snapshot.set_value();

    EXPECT_FALSE(start_result.get());
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(1u, publisher.status().accepted_count);
    EXPECT_EQ(0u, publisher.status().queue_size);
    EXPECT_FALSE(publisher.enabled());
    EXPECT_TRUE(reporter->requests().empty());
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherFailsOpenWhenInitialSnapshotExceedsKeyLimit) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;
    config.snapshot_max_keys     = 1;

    std::atomic<size_t> snapshot_calls{0};
    auto                reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher       publisher(
        config,
        makeContext(),
        [&snapshot_calls] {
            snapshot_calls.fetch_add(1, std::memory_order_relaxed);
            return KVCacheSnapshot{{10, 20}};
        },
        reporter);

    EXPECT_FALSE(publisher.start());
    EXPECT_FALSE(publisher.start());
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(1u, snapshot_calls.load(std::memory_order_relaxed));
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    EXPECT_TRUE(reporter->requests().empty());
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDisablesExporterWhenSnapshotPayloadExceedsByteLimit) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;
    config.snapshot_max_bytes    = 64;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HOST_DOWN", 1, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::CIRCUIT_OPEN, kAsyncTestTimeout));
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    for (const auto& request : reporter->requests()) {
        EXPECT_EQ(std::string::npos, request.body.find("EVENT_BLOCK_SNAPSHOT"));
    }
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDisablesExporterWhenMutationPayloadExceedsByteLimit) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;
    // Registration, control events, and the empty initial snapshot fit, but a
    // mutation with the repeated location spec does not. A test-only context
    // exercises the production hard bound without allocating 16 MiB.
    config.report_max_bytes = 1024;

    auto reporter        = std::make_shared<RecordingReporter>();
    auto context         = makeContext();
    context.location_uri = "rtp-llm://" + std::string(2048, 'x');
    KVCMPublisher publisher(config, context, [] { return KVCacheSnapshot{{}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HOST_DOWN", 1, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::CIRCUIT_OPEN, kAsyncTestTimeout));
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 20, 0}));

    size_t mutation_count = 0;
    for (const auto& request : reporter->requests()) {
        mutation_count += request.body.find("EVENT_BLOCK_ADD") != std::string::npos;
    }
    EXPECT_EQ(0u, mutation_count);
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherReusesSnapshotPayloadAndExponentiallyBacksOff) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 20;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 40;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->failNextBodiesContaining("EVENT_BLOCK_SNAPSHOT", 3);
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10, 20}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 4, kAsyncTestTimeout));
    publisher.stop();

    std::vector<RecordingReporter::Request> snapshot_requests;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshot_requests.push_back(request);
        }
    }
    ASSERT_GE(snapshot_requests.size(), 4u);
    // Retrying an upload must reuse the already captured and serialized
    // snapshot (including its trace id) instead of recopying the cache.
    EXPECT_EQ(snapshot_requests[0].body, snapshot_requests[1].body);
    EXPECT_EQ(snapshot_requests[1].body, snapshot_requests[2].body);
    EXPECT_EQ(snapshot_requests[2].body, snapshot_requests[3].body);

    const auto first_retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        snapshot_requests[1].recorded_at - snapshot_requests[0].recorded_at);
    const auto second_retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        snapshot_requests[2].recorded_at - snapshot_requests[1].recorded_at);
    const auto third_retry_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        snapshot_requests[3].recorded_at - snapshot_requests[2].recorded_at);
    EXPECT_GE(first_retry_delay.count(), 30);
    EXPECT_GE(second_retry_delay.count(), 60);
    EXPECT_GE(third_retry_delay.count(), 130);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherUsesSnapshotRateLimitWithoutReregistering) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->respondNextBodyContaining(
        "EVENT_BLOCK_SNAPSHOT", R"({"header":{"status":{"code":"SNAPSHOT_RATE_LIMITED"}},"retry_after_ms":40})");
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    publisher.stop();

    std::vector<RecordingReporter::Request> snapshots;
    size_t                                  registration_count = 0;
    for (const auto& request : reporter->requests()) {
        if (request.route == "/api/registerInstance") {
            ++registration_count;
        }
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshots.push_back(request);
        }
    }
    ASSERT_GE(snapshots.size(), 2u);
    EXPECT_EQ(1u, registration_count);
    EXPECT_EQ(snapshots[0].body, snapshots[1].body);
    const auto retry_delay =
        std::chrono::duration_cast<std::chrono::milliseconds>(snapshots[1].recorded_at - snapshots[0].recorded_at);
    EXPECT_GE(retry_delay.count(), 35);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherSnapshotsWithoutReregisteringWhenServerRequiresIt) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining(
        "EVENT_BLOCK_ADD", R"({"header":{"status":{"code":"SNAPSHOT_REQUIRED"}},"snapshot_required":true})");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t registration_count = 0;
    bool   snapshot_has_key   = false;
    for (const auto& request : reporter->requests()) {
        registration_count += request.route == "/api/registerInstance";
        snapshot_has_key = snapshot_has_key
                           || (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos
                               && request.body.find("\"block_key\":\"20\"") != std::string::npos);
    }
    EXPECT_EQ(1u, registration_count);
    EXPECT_TRUE(snapshot_has_key);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDrainsMutationsIntoMirrorDuringRetryBackoff) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 4;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 200;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->failNextBodyContaining("EVENT_BLOCK_ADD");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"20\"", kAsyncTestTimeout));

    // The worker must keep draining the bounded producer queue while network
    // retry backoff is active. These events update only its in-memory mirror;
    // the authoritative recovery snapshot sends their final state later.
    for (int64_t key = 21; key <= 24; ++key) {
        ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0}));
    }
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_EQ(0u, publisher.status().dropped_count);
    publisher.stop();

    std::string recovery_snapshot;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            recovery_snapshot = request.body;
        }
    }
    ASSERT_FALSE(recovery_snapshot.empty());
    for (int64_t key = 20; key <= 24; ++key) {
        EXPECT_NE(std::string::npos, recovery_snapshot.find("\"block_key\":\"" + std::to_string(key) + "\""));
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromMalformedMutationResponseWithAuthoritativeSnapshot) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining("EVENT_BLOCK_ADD", "");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_TRUE(publisher.enabled());
    publisher.stop();

    bool recovered_key = false;
    for (const auto& request : reporter->requests()) {
        recovered_key = recovered_key
                        || (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos
                            && request.body.find("\"block_key\":\"20\"") != std::string::npos);
    }
    EXPECT_TRUE(recovered_key);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromMismatchedItemResultsWithAuthoritativeSnapshot) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    // A non-empty item_results array must have one entry per event. Treat a
    // truncated/oversized protocol response as unknown outcome and reconcile
    // from the worker-owned mirror instead of acknowledging it.
    reporter->respondNextBodyContaining("EVENT_BLOCK_ADD",
                                        R"({"header":{"status":{"code":"OK"}},"item_results":["OK","OK"]})");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    bool recovered_key = false;
    for (const auto& request : reporter->requests()) {
        recovered_key = recovered_key
                        || (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos
                            && request.body.find("\"block_key\":\"20\"") != std::string::npos);
    }
    EXPECT_TRUE(recovered_key);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherReregistersAndSnapshotsAfterNodeLoss) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining("EVENT_BLOCK_ADD",
                                        R"({"header":{"status":{"code":"NODE_NOT_REGISTERED"}},)"
                                        R"("item_results":["NODE_NOT_REGISTERED"]})");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_NODE_REGISTER", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    bool recovered_key = false;
    for (const auto& request : reporter->requests()) {
        recovered_key = recovered_key
                        || (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos
                            && request.body.find("\"block_key\":\"20\"") != std::string::npos);
    }
    EXPECT_TRUE(recovered_key);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRebuildsPendingSnapshotAfterReregistration) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 200;
    config.retry_interval_ms     = 300;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining(
        "EVENT_BLOCK_SNAPSHOT",
        R"({"header":{"status":{"code":"INSTANCE_NOT_EXIST"}},"item_results":["INSTANCE_NOT_EXIST"]})");
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::DEGRADED, kAsyncTestTimeout));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
    publisher.stop();

    std::vector<RecordingReporter::Request> snapshots;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshots.push_back(request);
        }
    }
    ASSERT_GE(snapshots.size(), 3u);
    EXPECT_NE(snapshots[1].body, snapshots[2].body);
    EXPECT_NE(std::string::npos, snapshots[2].body.find("\"block_key\":\"20\""));
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDiscardsPendingSnapshotWhenHeartbeatForcesReregistration) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 100;
    config.snapshot_interval_ms  = 200;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    // Retain snapshot #2 behind a long server-directed retry, then let the
    // heartbeat discover that the reporter lifecycle disappeared. Successful
    // re-registration must discard both the old payload and its old deadline;
    // otherwise snapshot #3 would remain delayed for a full minute.
    reporter->respondNextBodyContaining("EVENT_BLOCK_SNAPSHOT",
                                        R"({"header":{"status":{"code":"SNAPSHOT_RATE_LIMITED"}},)"
                                        R"("item_results":["SNAPSHOT_RATE_LIMITED"],"retry_after_ms":"60000"})");
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::DEGRADED, kAsyncTestTimeout));

    // Put a new key in the mirror before simulating instance loss so the
    // post-registration snapshot proves the stale pending payload was not
    // reused. Script the heartbeat only after the rate-limited snapshot has
    // been observed; the scenario no longer depends on transport call order.
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    reporter->respondNextBodyContaining("EVENT_HEARTBEAT",
                                        R"({"header":{"status":{"code":"INSTANCE_NOT_EXIST"}},)"
                                        R"("item_results":["INSTANCE_NOT_EXIST"]})");
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::DEGRADED, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
    publisher.stop();

    std::vector<RecordingReporter::Request> snapshots;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshots.push_back(request);
        }
    }
    ASSERT_GE(snapshots.size(), 3u);
    EXPECT_NE(snapshots[1].body, snapshots[2].body);
    EXPECT_NE(std::string::npos, snapshots[2].body.find("\"block_key\":\"20\""));
}

TEST(KVCacheEventPublisherTest, KVCMPublisherHonorsSuccessfulSnapshotAdvisory) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining("EVENT_BLOCK_ADD",
                                        R"({"header":{"status":{"code":"OK"}},"snapshot_required":true})");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRetriesSuccessfulSnapshotWithoutValidCommittedGeneration) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->respondNextBodyContaining("EVENT_BLOCK_SNAPSHOT", R"({"header":{"status":{"code":"OK"}}})");
    reporter->respondNextBodyContaining(
        "EVENT_BLOCK_SNAPSHOT",
        R"({"header":{"status":{"code":"OK"}},"committed_snapshot_version":"not-a-generation"})");
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_EQ(2u, publisher.status().request_failure_count);
    publisher.stop();

    std::vector<std::string> snapshots;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshots.push_back(request.body);
        }
    }
    ASSERT_GE(snapshots.size(), 3u);
    EXPECT_EQ(snapshots[0], snapshots[1]);
    EXPECT_EQ(snapshots[1], snapshots[2]);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDisablesOnlyExporterOnPermanentProtocolFailure) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining("EVENT_BLOCK_ADD",
                                        R"({"header":{"status":{"code":"OK"}},"item_results":["INVALID_ARGUMENT"]})");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HOST_DOWN", 1, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::CIRCUIT_OPEN, kAsyncTestTimeout));
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 21, 0}));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecreatesInstanceAfterReportEventLosesIt) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining("EVENT_BLOCK_ADD", R"({"header":{"status":{"code":"INSTANCE_NOT_EXIST"}}})");
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_NODE_REGISTER", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_TRUE(publisher.enabled());
    publisher.stop();

    bool recovered_key = false;
    for (const auto& request : reporter->requests()) {
        recovered_key = recovered_key
                        || (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos
                            && request.body.find("\"block_key\":\"20\"") != std::string::npos);
    }
    EXPECT_TRUE(recovered_key);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRetriesFullRegistrationWhenNodeRegisterLosesInstance) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->respondNextBodyContaining(
        "EVENT_NODE_REGISTER",
        R"({"header":{"status":{"code":"INSTANCE_NOT_EXIST"}},"item_results":["INSTANCE_NOT_EXIST"]})");
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("\"instance_group\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_NODE_REGISTER", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_TRUE(publisher.enabled());
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDisablesExporterWhenRegisterCannotCreateInstance) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->respondNextBodyContaining("\"instance_group\"", R"({"header":{"status":{"code":"INSTANCE_NOT_EXIST"}}})");
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    EXPECT_TRUE(waitForState(publisher, PublisherState::CIRCUIT_OPEN, kAsyncTestTimeout));
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    const auto requests = reporter->requests();
    EXPECT_EQ(1u, std::count_if(requests.begin(), requests.end(), [](const auto& request) {
                  return request.body.find("\"instance_group\"") != std::string::npos;
              }));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRetriesHeartbeatFailureWithoutResnapshot) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 100;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    reporter->failNextBodyContaining("EVENT_HEARTBEAT");
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_EQ(1u, publisher.status().request_failure_count);
    publisher.stop();

    size_t snapshot_count = 0;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
    }
    EXPECT_EQ(1u, snapshot_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherPreservesSnapshotAdvisoryFromUnknownHeartbeatCode) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 20;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->respondNextBodyContaining(
        "EVENT_HEARTBEAT", R"({"header":{"status":{"code":4242}},"retry_after_ms":1,"snapshot_required":true})");
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t registration_count = 0;
    for (const auto& request : reporter->requests()) {
        registration_count += request.route == "/api/registerInstance";
    }
    EXPECT_EQ(1u, registration_count);
    EXPECT_GE(publisher.status().request_failure_count, 1u);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDrainsIngressWhileInitialSnapshotIsInFlight) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 2;
    config.report_batch_size     = 2;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }

    for (int64_t key = 20; key < 28; key += 2) {
        EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0}));
        EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key + 1, 0}));
        const auto drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
        while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < drain_deadline) {
            std::this_thread::yield();
        }
        ASSERT_EQ(0u, publisher.status().queue_size);
        ASSERT_TRUE(publisher.enabled());
    }
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    const auto delete_drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
    while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < delete_drain_deadline) {
        std::this_thread::yield();
    }
    ASSERT_EQ(0u, publisher.status().queue_size);

    reporter->releaseSnapshot();
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"27\"", kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t                      snapshot_count  = 0;
    bool                        replayed_delete = false;
    std::unordered_set<int64_t> replayed_keys;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        replayed_delete = replayed_delete
                          || (request.find("EVENT_BLOCK_DELETE") != std::string::npos
                              && request.find("\"block_key\":\"10\"") != std::string::npos);
        for (int64_t key = 20; key < 28; ++key) {
            if (request.find("EVENT_BLOCK_ADD") != std::string::npos
                && request.find("\"block_key\":\"" + std::to_string(key) + "\"") != std::string::npos) {
                replayed_keys.insert(key);
            }
        }
    }
    EXPECT_EQ(1u, snapshot_count);
    EXPECT_EQ(8u, replayed_keys.size());
    EXPECT_TRUE(replayed_delete);
    EXPECT_EQ(0u, publisher.status().dropped_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherPreservesMutationsCreatedWhileSnapshotIsInFlight) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 2;
    config.report_batch_size     = 2;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 50;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    reporter->blockNextSnapshot();
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "periodic snapshot request did not reach the blocking reporter";
    }

    // The snapshot transport is deliberately blocked. The worker must still
    // drain ingress into its mirror so bursts larger than queue capacity do
    // not disable an otherwise healthy exporter.
    for (int64_t key = 20; key < 28; key += 2) {
        EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0}));
        EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key + 1, 0}));
        const auto drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
        while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < drain_deadline) {
            std::this_thread::yield();
        }
        ASSERT_EQ(0u, publisher.status().queue_size);
        ASSERT_TRUE(publisher.enabled());
    }
    reporter->releaseSnapshot();
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"27\"", kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t                      snapshot_count = 0;
    std::unordered_set<int64_t> replayed_keys;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        for (int64_t key = 20; key < 28; ++key) {
            if (request.find("EVENT_BLOCK_ADD") != std::string::npos
                && request.find("\"block_key\":\"" + std::to_string(key) + "\"") != std::string::npos) {
                replayed_keys.insert(key);
            }
        }
    }
    EXPECT_EQ(2u, snapshot_count);
    EXPECT_EQ(8u, replayed_keys.size());
    EXPECT_EQ(0u, publisher.status().dropped_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherHeartbeatsWhileSnapshotIsInFlight) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 2;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 20;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 3, kAsyncTestTimeout));
    EXPECT_TRUE(publisher.enabled());

    reporter->releaseSnapshot();
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherVerifiesInFlightHeartbeatAdvisoryAfterSnapshotCommit) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 2;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 20;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    reporter->requestSnapshotOnNextHeartbeat();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 1, kAsyncTestTimeout));
    reporter->releaseSnapshot();
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t snapshot_count = 0;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
    }
    EXPECT_EQ(1u, snapshot_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherFoldsSnapshotInFlightTransitionsBackToCapturedState) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 2;
    config.report_batch_size     = 2;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }

    for (size_t iteration = 0; iteration < 4; ++iteration) {
        ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
        ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 10, 0}));
        const auto drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
        while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < drain_deadline) {
            std::this_thread::yield();
        }
        ASSERT_EQ(0u, publisher.status().queue_size);
    }

    reporter->releaseSnapshot();
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t snapshot_count = 0;
    size_t mutation_count = 0;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        mutation_count += request.find("EVENT_BLOCK_ADD") != std::string::npos;
        mutation_count += request.find("EVENT_BLOCK_DELETE") != std::string::npos;
    }
    EXPECT_EQ(1u, snapshot_count);
    EXPECT_EQ(0u, mutation_count);
    EXPECT_EQ(0u, publisher.status().dropped_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherDrainsIngressBetweenSnapshotReplayBatches) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 2;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }
    for (int64_t key = 20; key < 24; ++key) {
        ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0}));
        const auto drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
        while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < drain_deadline) {
            std::this_thread::yield();
        }
        ASSERT_EQ(0u, publisher.status().queue_size);
    }

    reporter->blockNextMutation();
    reporter->releaseSnapshot();
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "snapshot replay did not reach the blocking reporter";
    }
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    reporter->releaseMutation();

    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"31\"", kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_TRUE(publisher.enabled());
    EXPECT_EQ(0u, publisher.status().dropped_count);
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherCatchUpDiffAvoidsAnotherFullSnapshot) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 2;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    const auto initial_drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
    while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < initial_drain_deadline) {
        std::this_thread::yield();
    }
    ASSERT_EQ(0u, publisher.status().queue_size);

    reporter->blockNextMutation();
    reporter->releaseSnapshot();
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "first snapshot diff did not reach the blocking reporter";
    }

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    reporter->releaseMutation();
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"30\"", kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    size_t snapshot_count = 0;
    bool   caught_up_key  = false;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        caught_up_key = caught_up_key
                        || (request.find("EVENT_BLOCK_ADD") != std::string::npos
                            && request.find("\"block_key\":\"30\"") != std::string::npos);
    }
    EXPECT_EQ(1u, snapshot_count);
    EXPECT_TRUE(caught_up_key);
    EXPECT_EQ(0u, publisher.status().dropped_count);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherResnapshotsWhenCatchUpInvalidatesRemoteBaseline) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 2;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    const auto drain_deadline = std::chrono::steady_clock::now() + kAsyncTestTimeout;
    while (publisher.status().queue_size != 0 && std::chrono::steady_clock::now() < drain_deadline) {
        std::this_thread::yield();
    }
    ASSERT_EQ(0u, publisher.status().queue_size);

    // A successful replay delta can still report snapshot_required when KVCM
    // has lost the reporter generation. The delta alone cannot reconstruct
    // keys outside that request, so the remembered remote baseline is no
    // longer safe for another diff.
    reporter->requestSnapshotOnNextMutation();
    reporter->releaseSnapshot();
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();

    std::vector<std::string> snapshots;
    for (const auto& request : reporter->requests()) {
        if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshots.push_back(request);
        }
    }
    ASSERT_EQ(2u, snapshots.size());
    EXPECT_NE(std::string::npos, snapshots[1].find("\"block_key\":\"10\""));
    EXPECT_NE(std::string::npos, snapshots[1].find("\"block_key\":\"20\""));
    EXPECT_EQ(0u, publisher.status().dropped_count);
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
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

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
        if (request.find("\"block_add\":{\"block_key\":\"42\"") != std::string::npos) {
            saw_final_add_42 = true;
        }
        if (request.find("\"block_key\":\"42\"") != std::string::npos) {
            EXPECT_EQ(std::string::npos, request.find("\"block_delete\":{\"block_key\":\"42\""));
        }
        if (request.find("\"block_delete\":{\"block_key\":\"43\"") != std::string::npos) {
            saw_final_delete_43 = true;
        }
        if (request.find("\"block_key\":\"43\"") != std::string::npos) {
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
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{}}; }, reporter);
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

TEST(KVCacheEventPublisherTest, KVCMPublisherDisablesExporterWhenMirrorExceedsKeyLimit) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;
    config.snapshot_max_keys     = 1;

    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HOST_DOWN", 1, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::CIRCUIT_OPEN, kAsyncTestTimeout));
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(1u, publisher.status().accepted_count);
    EXPECT_EQ(0u, publisher.status().dropped_count);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversQueueOverflowFromAuthoritativeCacheSnapshot) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 1;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    std::atomic<size_t> snapshot_calls{0};
    std::atomic<bool>   overflow_state{false};
    auto                reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher       publisher(
        config,
        makeContext(),
        [&snapshot_calls, &overflow_state] {
            snapshot_calls.fetch_add(1, std::memory_order_relaxed);
            return overflow_state.load(std::memory_order_acquire) ? KVCacheSnapshot{{20, 30, 31}} :
                                                                          KVCacheSnapshot{{10, 20}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    overflow_state.store(true, std::memory_order_release);
    EXPECT_EQ(PublishResult::DROPPED_RECOVERABLE, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    EXPECT_EQ(1u, publisher.status().queue_high_watermark);
    EXPECT_EQ(1, publisher.status().dropped_count);
    EXPECT_EQ(PublisherState::RESYNCING, publisher.status().state);
    EXPECT_TRUE(publisher.enabled());

    reporter->releaseMutation();
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_GE(snapshot_calls.load(std::memory_order_relaxed), 2u);
    EXPECT_EQ(1u, publisher.status().overflow_recovery_count);
    EXPECT_GE(publisher.status().snapshot_attempt_count, 2u);
    EXPECT_GE(publisher.status().snapshot_commit_count, 2u);
    EXPECT_EQ(0u, publisher.status().queue_size);

    std::string recovered_snapshot;
    for (const auto& request : reporter->requests()) {
        if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            recovered_snapshot = request;
        }
    }
    ASSERT_FALSE(recovered_snapshot.empty());
    EXPECT_EQ(std::string::npos, recovered_snapshot.find("\"block_key\":\"10\""));
    EXPECT_NE(std::string::npos, recovered_snapshot.find("\"block_key\":\"20\""));
    EXPECT_NE(std::string::npos, recovered_snapshot.find("\"block_key\":\"30\""));
    EXPECT_NE(std::string::npos, recovered_snapshot.find("\"block_key\":\"31\""));

    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 32, 0}));
    EXPECT_TRUE(reporter->waitForBody("\"block_key\":\"32\"", kAsyncTestTimeout));
    publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoverySnapshotFailureOnlyOpensExporterCircuit) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 1;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    std::atomic<size_t> snapshot_calls{0};
    auto                reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher       publisher(
        config,
        makeContext(),
        [&snapshot_calls]() -> KVCacheSnapshot {
            if (snapshot_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
                return KVCacheSnapshot{{10}};
            }
            throw std::runtime_error("injected overflow recovery snapshot failure");
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 21, 0}));
    ASSERT_EQ(PublishResult::DROPPED_RECOVERABLE, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    reporter->releaseMutation();

    ASSERT_TRUE(waitForState(publisher, PublisherState::CIRCUIT_OPEN, kAsyncTestTimeout));
    EXPECT_FALSE(publisher.enabled());
    EXPECT_EQ(2u, snapshot_calls.load(std::memory_order_relaxed));
    EXPECT_EQ(2u, publisher.status().accepted_count);
    EXPECT_EQ(1u, publisher.status().dropped_count);
    EXPECT_EQ(0u, publisher.status().overflow_recovery_count);
    EXPECT_EQ(0u, publisher.status().queue_size);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));

    publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRepeatsSnapshotHandoffWhenRecoveryIngressOverflowsAgain) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 1;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    std::mutex              snapshot_mu;
    std::condition_variable snapshot_cv;
    size_t                  snapshot_calls        = 0;
    bool                    second_snapshot_ready = false;
    bool                    release_second        = false;
    std::atomic<int>        authoritative_phase{0};
    auto                    reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher           publisher(
        config,
        makeContext(),
        [&] {
            std::unique_lock<std::mutex> lock(snapshot_mu);
            ++snapshot_calls;
            if (snapshot_calls == 2) {
                second_snapshot_ready = true;
                snapshot_cv.notify_all();
                snapshot_cv.wait(lock, [&] { return release_second; });
            }
            const auto phase = authoritative_phase.load(std::memory_order_acquire);
            snapshot_cv.notify_all();
            if (phase == 0) {
                return KVCacheSnapshot{{10}};
            }
            if (phase == 1) {
                return KVCacheSnapshot{{20, 21}};
            }
            return KVCacheSnapshot{{30, 40}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 21, 0}));
    authoritative_phase.store(1, std::memory_order_release);
    ASSERT_EQ(PublishResult::DROPPED_RECOVERABLE, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    reporter->releaseMutation();

    {
        std::unique_lock<std::mutex> lock(snapshot_mu);
        if (!snapshot_cv.wait_for(lock, kAsyncTestTimeout, [&] { return second_snapshot_ready; })) {
            release_second = true;
            snapshot_cv.notify_all();
            lock.unlock();
            publisher.stop();
            FAIL() << "overflow recovery did not enter its authoritative snapshot provider";
        }
    }

    // The fresh admission epoch is live before the cache snapshot lock is
    // taken. Overflow it again while that snapshot is blocked; the worker
    // must discard the new epoch and capture a third, newer baseline.
    const auto second_epoch_add = publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0});
    authoritative_phase.store(2, std::memory_order_release);
    const auto second_epoch_overflow = publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 20, 0});
    {
        std::lock_guard<std::mutex> lock(snapshot_mu);
        release_second = true;
    }
    snapshot_cv.notify_all();
    ASSERT_EQ(PublishResult::ACCEPTED, second_epoch_add);
    ASSERT_EQ(PublishResult::DROPPED_RECOVERABLE, second_epoch_overflow);

    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    {
        std::unique_lock<std::mutex> lock(snapshot_mu);
        EXPECT_TRUE(snapshot_cv.wait_for(lock, kAsyncTestTimeout, [&] { return snapshot_calls >= 3; }));
    }
    EXPECT_EQ(2u, publisher.status().dropped_count);
    EXPECT_EQ(2u, publisher.status().overflow_recovery_count);
    EXPECT_GE(publisher.status().snapshot_attempt_count, 2u);
    EXPECT_GE(publisher.status().snapshot_commit_count, 2u);

    std::string recovered_snapshot;
    for (const auto& request : reporter->requests()) {
        if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            recovered_snapshot = request;
        }
    }
    EXPECT_NE(std::string::npos, recovered_snapshot.find("\"block_key\":\"30\""));
    EXPECT_NE(std::string::npos, recovered_snapshot.find("\"block_key\":\"40\""));
    EXPECT_EQ(std::string::npos, recovered_snapshot.find("\"block_key\":\"20\""));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherQueueOverflowNeverCallsTransportFromProducer) {
    KVCacheEventPublisherConfig config;
    config.type                  = "kvcm";
    config.queue_capacity        = 1;
    config.report_batch_size     = 1;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 60000;
    config.request_timeout_ms    = 30000;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto          reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{{10}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));

    reporter->blockCancel();
    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 20, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseCancel();
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 21, 0}));
    auto overflow =
        std::async(std::launch::async, [&] { return publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}); });
    const auto overflow_status = overflow.wait_for(kAsyncTestTimeout);

    // Always unblock teardown before asserting so a regression reports a
    // bounded failure instead of hanging the test process.
    reporter->releaseCancel();
    reporter->releaseMutation();

    ASSERT_EQ(std::future_status::ready, overflow_status)
        << "queue overflow called the reporter's deliberately blocking cancel() on the producer thread";
    EXPECT_EQ(PublishResult::DROPPED_RECOVERABLE, overflow.get());
    EXPECT_EQ(0u, reporter->cancelCount());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_EQ(0u, publisher.status().queue_size);

    publisher.stop();
    EXPECT_EQ(1u, reporter->cancelCount());
}

}  // namespace
}  // namespace rtp_llm
