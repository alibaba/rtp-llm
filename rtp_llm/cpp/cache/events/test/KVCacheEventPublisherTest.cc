// Publisher-specific behavior tests are owned by the cache/events subsystem.
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

    bool waitForOccurrenceCount(const std::string& text, size_t expected_count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            size_t count = 0;
            for (const auto& request : requests_) {
                size_t pos = 0;
                while ((pos = request.body.find(text, pos)) != std::string::npos) {
                    ++count;
                    pos += text.size();
                }
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
        {R"({"header":{"status":{"code":"OK"}},"item_results":[{"code":1},{"status":{"code":"OK"}}]})", true},
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
    auto                        context = makeContext();
    context.instance_id.clear();
    auto          reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(
        config, context, [] { return KVCacheSnapshot{}; }, reporter);

    EXPECT_FALSE(publisher.start());
    EXPECT_EQ(PublisherState::DEGRADED, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 1, 0}));
    EXPECT_TRUE(reporter->requests().empty());
}

TEST(KVCacheEventPublisherTest, PublisherLifecycleIsIdempotent) {
    KVCacheEventPublisherConfig kvcm_config;
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

TEST(KVCacheEventPublisherTest, StopBeforeStartPermanentlyStopsPublisher) {
    KVCacheEventPublisherConfig config;
    KVCMPublisher               kvcm_publisher(config, makeContext(), [] { return KVCacheSnapshot{}; });
    kvcm_publisher.stop();
    EXPECT_FALSE(kvcm_publisher.start());
    EXPECT_EQ(PublisherState::STOPPED, kvcm_publisher.status().state);
}

TEST(KVCacheEventPublisherTest, ConcurrentStartAndStopAlwaysLeavesJoinedWorkers) {
    for (size_t iteration = 0; iteration < 32; ++iteration) {
        KVCacheEventPublisherConfig config;
        config.queue_capacity        = 8;
        config.report_batch_size     = 8;
        config.flush_interval_ms     = 1;
        config.heartbeat_interval_ms = 60000;
        config.snapshot_interval_ms  = 60000;

        auto          reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(
            config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter);
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

        EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRegistersSnapshotsAndReportsDeltas) {
    KVCacheEventPublisherConfig config;
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

TEST(KVCacheEventPublisherTest, KVCMPublisherPreservesMutationsCreatedWhileSnapshotIsInFlight) {
    KVCacheEventPublisherConfig config;
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

TEST(KVCacheEventPublisherTest, KVCMPublisherRecoversFromQueueOverflowWithSnapshot) {
    KVCacheEventPublisherConfig config;
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
