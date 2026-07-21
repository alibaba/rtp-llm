// Publisher-specific behavior tests are owned by the cache/events subsystem.
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherFactory.h"
#include "rtp_llm/cpp/cache/events/LogPublisher.h"
#include "rtp_llm/cpp/cache/events/NullPublisher.h"

#include <atomic>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

class RecordingReporter final: public KVCacheEventReporter {
public:
    struct Request {
        std::string route;
        std::string body;
    };

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        bool fail_request = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            requests_.push_back({route, request});
            if (!fail_next_body_.empty() && request.find(fail_next_body_) != std::string::npos) {
                fail_request = true;
                fail_next_body_.clear();
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
        std::lock_guard<std::mutex> lock(mu_);
        fail_next_body_ = std::move(text);
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
    std::string             fail_next_body_;
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

    config.type = "none";
    auto publisher = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_FALSE(publisher->enabled());

    config.type = "log";
    publisher   = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_TRUE(publisher->enabled());

    config.type = "unsupported";
    publisher   = createKVCacheEventPublisher(config, context);
    ASSERT_NE(nullptr, publisher);
    EXPECT_FALSE(publisher->enabled());
}

TEST(KVCacheEventPublisherTest, KVCMPublisherRejectsIncompleteIdentity) {
    KVCacheEventPublisherConfig config;
    config.type = "kvcm";
    auto context = makeContext();
    context.instance_id.clear();
    auto reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, context, [] { return KVCacheSnapshot{}; }, reporter);

    EXPECT_FALSE(publisher.start());
    EXPECT_EQ(PublisherState::DEGRADED, publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 1, 0}));
    EXPECT_TRUE(reporter->requests().empty());
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
    EXPECT_EQ(PublishResult::ACCEPTED,
              log_publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    log_publisher.stop();
    log_publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, log_publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING,
              log_publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 43, 0}));

    KVCacheEventPublisherConfig kvcm_config;
    kvcm_config.type                  = "kvcm";
    kvcm_config.queue_capacity        = 8;
    kvcm_config.report_batch_size     = 8;
    kvcm_config.flush_interval_ms     = 1;
    kvcm_config.heartbeat_interval_ms = 60000;
    kvcm_config.snapshot_interval_ms  = 60000;
    kvcm_config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher kvcm_publisher(
        kvcm_config, makeContext(), [] { return KVCacheSnapshot{1, {}}; }, reporter);
    EXPECT_TRUE(kvcm_publisher.start());
    EXPECT_TRUE(kvcm_publisher.start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", std::chrono::seconds(2)));
    kvcm_publisher.stop();
    kvcm_publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, kvcm_publisher.status().state);
    EXPECT_EQ(PublishResult::NOT_RUNNING,
              kvcm_publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 44, 0}));
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
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", std::chrono::seconds(2)));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"30\"", std::chrono::seconds(2)));

    reporter->failNextBodyContaining("EVENT_BLOCK_ADD");
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"31\"", std::chrono::seconds(2)));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, std::chrono::seconds(2)));
    publisher->stop();

    const auto requests = reporter->requests();
    ASSERT_GE(requests.size(), 5);
    EXPECT_EQ("/api/registerInstance", requests.front().route);
    EXPECT_NE(std::string::npos, requests.front().body.find("rtp_llm_hbm_64"));
    EXPECT_NE(std::string::npos, requests.front().body.find("location_spec_groups"));
    EXPECT_EQ(1, countOccurrences(requests.front().body, "\"dtype\""));

    bool saw_node_register = false;
    bool saw_snapshot      = false;
    bool saw_add           = false;
    bool saw_delete        = false;
    size_t host_down_count = 0;
    for (const auto& request : requests) {
        saw_node_register = saw_node_register || request.body.find("EVENT_NODE_REGISTER") != std::string::npos;
        saw_snapshot      = saw_snapshot || request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        saw_add           = saw_add || request.body.find("EVENT_BLOCK_ADD") != std::string::npos;
        saw_delete        = saw_delete || request.body.find("EVENT_BLOCK_DELETE") != std::string::npos;
        host_down_count += request.body.find("EVENT_HOST_DOWN") != std::string::npos;
    }
    EXPECT_TRUE(saw_node_register);
    EXPECT_TRUE(saw_snapshot);
    EXPECT_TRUE(saw_add);
    EXPECT_TRUE(saw_delete);
    // HOST_DOWN is terminal. Registration and recovery use NODE_REGISTER plus
    // an authoritative snapshot rather than pretending the live engine exited.
    EXPECT_EQ(1, host_down_count);
    EXPECT_EQ(3, publisher->status().accepted_count);
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

    auto reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{1, {}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, std::chrono::seconds(2)));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 999, 0}));
    if (!reporter->waitUntilMutationBlocked(std::chrono::seconds(2))) {
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

    ASSERT_TRUE(reporter->waitForBodyCount("\"block_key\":\"42\"", 1, std::chrono::seconds(2)));
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

    auto reporter = std::make_shared<CountingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{1, {}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (publisher.status().state != PublisherState::READY && std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::yield();
    }
    ASSERT_EQ(PublisherState::READY, publisher.status().state);

    std::atomic<size_t> accepted_count{0};
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
    KVCMPublisher publisher(
        config,
        makeContext(),
        [&snapshot_version] {
            return KVCacheSnapshot{snapshot_version.load(std::memory_order_relaxed), {10, 20, 30, 31}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, std::chrono::seconds(2)));

    reporter->blockNextMutation();
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    if (!reporter->waitUntilMutationBlocked(std::chrono::seconds(2))) {
        reporter->releaseMutation();
        publisher.stop();
        FAIL() << "mutation request did not reach the blocking reporter";
    }

    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    const auto overflow_start = std::chrono::steady_clock::now();
    EXPECT_EQ(PublishResult::QUEUE_FULL, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    const auto overflow_cost = std::chrono::steady_clock::now() - overflow_start;
    EXPECT_LT(overflow_cost, std::chrono::milliseconds(50));
    EXPECT_EQ(1, publisher.status().dropped_count);

    snapshot_version.store(2, std::memory_order_relaxed);
    reporter->releaseMutation();
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, std::chrono::seconds(2)));
    publisher.stop();
    EXPECT_EQ(PublisherState::STOPPED, publisher.status().state);
}

}  // namespace
}  // namespace rtp_llm
