// Publisher-specific behavior tests are owned by the cache/events subsystem.
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <gtest/gtest.h>
#include <mutex>
#include <rapidjson/document.h>
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
        std::string selected_response = R"({"header":{"status":{"code":"OK"}}})";
        {
            std::lock_guard<std::mutex> lock(mu_);
            requests_.push_back({route, request, std::chrono::steady_clock::now()});
            if (reply_body_count_ > 0 && !reply_body_.empty() && request.find(reply_body_) != std::string::npos) {
                selected_response = reply_response_;
                fail_request = reply_failure_;
                if (--reply_body_count_ == 0) {
                    reply_body_.clear();
                }
            }
            if (fail_body_count_ > 0 && !fail_body_.empty() && request.find(fail_body_) != std::string::npos) {
                fail_request = true;
                --fail_body_count_;
                if (fail_body_count_ == 0) {
                    fail_body_.clear();
                }
            }
        }
        response = std::move(selected_response);
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

    void replyNextBodyContaining(std::string text, std::string response, bool failure) {
        replyNextBodiesContaining(std::move(text), std::move(response), failure, 1);
    }

    void replyNextBodiesContaining(std::string text, std::string response, bool failure, size_t count) {
        std::lock_guard<std::mutex> lock(mu_);
        reply_body_ = std::move(text);
        reply_response_ = std::move(response);
        reply_failure_ = failure;
        reply_body_count_ = count;
    }

    bool waitForReplyConsumed(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return reply_body_count_ == 0; });
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
    std::string             reply_body_;
    std::string             reply_response_;
    bool                    reply_failure_{false};
    size_t                  reply_body_count_{0};
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
        if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshot_committed_ = true;
        }
        // The Manager repeats this advisory on successful heartbeats before
        // the first snapshot commits.
        response = snapshot_committed_ ? R"({"header":{"status":{"code":"OK"}}})" :
                                         R"({"header":{"status":{"code":"OK"}},"snapshot_required":true})";
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
    bool                     snapshot_committed_  = false;
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
    context.specs = {{"rtp_llm_hbm_64_full", "event_report://127.0.0.1:9000/hbm"},
                     {"rtp_llm_hbm_64_tail", "event_report://127.0.0.1:9000/hbm"}};
    context.block_size_tokens = 64;
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

KVCacheEventPublisherContext makeControlContext() {
    auto context = makeContext();
    context.specs.clear();
    context.register_instance_request = R"({"instance_group":"test_group","instance_id":"test_instance",
        "block_size":64,"default_query_type":2,
        "model_deployment":{"model_name":"remote_model","dtype":"fp16","tp_size":2,"dp_size":1,"pp_size":1},
        "location_spec_infos":[{"name":"remote_TP0","size":4096},{"name":"remote_TP1","size":4096}],
        "location_spec_groups":[{"name":"remote_group","spec_names":["remote_TP0","remote_TP1"]}]})";
    return context;
}

TEST(KVCacheEventPublisherTest, ParsesRemoteCacheControlWithoutChangingEventAckSemantics) {
    for (const bool enabled : {false, true}) {
        const std::string response =
            std::string(R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":)")
            + (enabled ? "true" : "false") + R"(}"})";
        const auto feedback = detail::parseKVCMReportFeedback(response);
        ASSERT_TRUE(feedback.valid);
        ASSERT_TRUE(feedback.ok);
        ASSERT_TRUE(feedback.remote_cache_enabled.has_value());
        EXPECT_EQ(*feedback.remote_cache_enabled, enabled);
    }
    const auto camel_case = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":1}},"extraInfo":"{\"rtp_remote_cache_enabled\":false}"})");
    ASSERT_TRUE(camel_case.remote_cache_enabled.has_value());
    EXPECT_FALSE(*camel_case.remote_cache_enabled);
    for (const auto* extra : {R"(null)", R"({})", R"("")", R"("not-json")", R"("[]")",
                              R"("{\"metadata_backend_mode\":4}")",
                              R"("{\"rtp_remote_cache_enabled\":0}")",
                              R"("{\"rtp_remote_cache_enabled\":\"false\"}")"}) {
        const std::string response =
            std::string(R"({"header":{"status":{"code":"OK"}},"snapshot_required":true,"extra_info":)") + extra + "}";
        const auto feedback = detail::parseKVCMReportFeedback(response);
        EXPECT_TRUE(feedback.valid) << response;
        EXPECT_TRUE(feedback.ok) << response;
        EXPECT_TRUE(feedback.snapshot_required) << response;
        EXPECT_FALSE(feedback.remote_cache_enabled.has_value()) << response;
    }
}

class RemoteCacheControlUpdates {
public:
    void apply(bool enabled) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            values_.push_back(enabled);
        }
        cv_.notify_all();
    }

    bool waitFor(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, kAsyncTestTimeout, [&] { return values_.size() >= count; });
    }

    std::vector<bool> values() {
        std::lock_guard<std::mutex> lock(mutex_);
        return values_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<bool> values_;
};

TEST(KVCacheEventPublisherTest, ControlOnlyKeepsHeartbeatsAndControlWithoutCacheEvents) {
    KVCacheEventPublisherConfig config;
    config.publish_cache_events = false;
    config.heartbeat_interval_ms = 5;
    config.snapshot_interval_ms = 0;
    // Control-only construction must not allocate the configured event ring.
    config.queue_capacity = std::numeric_limits<size_t>::max();
    auto reporter = std::make_shared<RecordingReporter>();
    RemoteCacheControlUpdates updates;
    reporter->replyNextBodyContaining("EVENT_NODE_REGISTER",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})", false);
    std::atomic<size_t> captures{0};
    KVCMPublisher publisher(config, makeControlContext(), [&] {
        ++captures;
        return KVCacheSnapshot{};
    }, reporter, [&](bool enabled) { updates.apply(enabled); });
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(updates.waitFor(1));
    EXPECT_EQ(updates.values(), (std::vector<bool>{false}));
    EXPECT_EQ(publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}), PublishResult::DISABLED);
    reporter->replyNextBodiesContaining("EVENT_HEARTBEAT",
        R"({"header":{"status":{"code":"OK"}},"snapshot_required":true,
            "extra_info":"{\"rtp_remote_cache_enabled\":true}"})", false, 3);
    ASSERT_TRUE(updates.waitFor(2));
    ASSERT_TRUE(reporter->waitForReplyConsumed(kAsyncTestTimeout));
    reporter->replyNextBodyContaining("EVENT_HEARTBEAT",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})", false);
    ASSERT_TRUE(updates.waitFor(5));
    publisher.stop();
    EXPECT_FALSE(updates.values().back());
    EXPECT_EQ(captures.load(), 0u);
    EXPECT_EQ(publisher.status().queue_size, 0u);
    EXPECT_EQ(publisher.status().accepted_count, 0u);
    EXPECT_EQ(publisher.status().dropped_count, 0u);
    const auto requests = reporter->requests();
    ASSERT_FALSE(requests.empty());
    EXPECT_NE(requests.back().body.find("EVENT_HOST_DOWN"), std::string::npos);
    for (const auto& request : requests) {
        EXPECT_EQ(request.body.find("EVENT_BLOCK_"), std::string::npos);
        rapidjson::Document body;
        body.Parse(request.body.c_str());
        ASSERT_FALSE(body.HasParseError());
        if (request.route == "/api/registerInstance") {
            EXPECT_STREQ(body["instance_id"].GetString(), "test_instance");
            EXPECT_EQ(body["location_spec_infos"].Size(), 2u);
            EXPECT_EQ(body["location_spec_infos"][0]["size"].GetInt(), 4096);
            EXPECT_STREQ(body["location_spec_groups"][0]["name"].GetString(), "remote_group");
            EXPECT_STREQ(body["model_deployment"]["model_name"].GetString(), "remote_model");
            EXPECT_TRUE(body.HasMember("trace_id"));
        } else if (request.body.find("EVENT_NODE_REGISTER") != std::string::npos) {
            EXPECT_TRUE(body["events"][0]["node_register"]["mediums"].Empty());
        }
    }
}

TEST(KVCacheEventPublisherTest, ControlOnlyReregistersWithoutSnapshotProvider) {
    KVCacheEventPublisherConfig config;
    config.publish_cache_events = false;
    config.heartbeat_interval_ms = 5;
    config.retry_interval_ms = 1;
    auto reporter = std::make_shared<RecordingReporter>();
    reporter->failNextBodyContaining("\"location_spec_infos\"");
    KVCMPublisher publisher(config, makeControlContext(), {}, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("\"location_spec_infos\"", 2, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBody("EVENT_HEARTBEAT", kAsyncTestTimeout));
    reporter->replyNextBodyContaining("EVENT_HEARTBEAT",
        R"({"header":{"status":{"code":"NODE_NOT_REGISTERED"}}})", false);
    ASSERT_TRUE(reporter->waitForBodyCount("\"location_spec_infos\"", 3, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_NODE_REGISTER", 2, kAsyncTestTimeout));
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
    for (const auto& request : reporter->requests()) {
        EXPECT_EQ(request.body.find("EVENT_BLOCK_"), std::string::npos);
    }
}

TEST(KVCacheEventPublisherTest, ControlOnlyRejectsInvalidOrMismatchedRegistration) {
    KVCacheEventPublisherConfig config;
    config.publish_cache_events = false;
    for (const auto* request : {"broken", "{}",
         R"({"instance_id":"other","instance_group":"test_group","location_spec_infos":[{"name":"s"}]})",
         R"({"instance_id":"test_instance","instance_group":"other","location_spec_infos":[{"name":"s"}]})",
         R"({"instance_id":"test_instance","instance_group":"test_group","location_spec_infos":[]})"}) {
        auto context = makeControlContext();
        context.register_instance_request = request;
        auto reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(config, context, {}, reporter);
        EXPECT_FALSE(publisher.start()) << request;
        publisher.stop();
        EXPECT_TRUE(reporter->requests().empty());
    }
}

TEST(KVCacheEventPublisherTest, RemoteCacheControlUsesRegistrationAndHeartbeatsWhileEventsContinue) {
    KVCacheEventPublisherConfig config;
    config.heartbeat_interval_ms = 5;
    auto reporter = std::make_shared<RecordingReporter>();
    RemoteCacheControlUpdates updates;
    reporter->replyNextBodyContaining("\"location_spec_infos\"",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})", false);
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter,
                             [&](bool enabled) { updates.apply(enabled); });
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(updates.waitFor(1));
    EXPECT_EQ(updates.values(), std::vector<bool>{false});
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    reporter->replyNextBodyContaining("EVENT_BLOCK_ADD",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":true}"})", false);
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    ASSERT_TRUE(reporter->waitForReplyConsumed(kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 3, kAsyncTestTimeout));
    EXPECT_EQ(updates.values(), std::vector<bool>{false});

    reporter->replyNextBodyContaining("EVENT_HEARTBEAT",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":true}"})", false);
    ASSERT_TRUE(updates.waitFor(2));
    reporter->replyNextBodyContaining("EVENT_HEARTBEAT",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})", false);
    ASSERT_TRUE(updates.waitFor(3));
    publisher.stop();
    EXPECT_EQ(updates.values(), (std::vector<bool>{false, true, false}));
    EXPECT_NE(reporter->requests().back().body.find("EVENT_HOST_DOWN"), std::string::npos);
}

TEST(KVCacheEventPublisherTest, RemoteCacheControlIgnoresFailedHeartbeatsAndInvalidFields) {
    KVCacheEventPublisherConfig config;
    config.heartbeat_interval_ms = 5;
    auto reporter = std::make_shared<RecordingReporter>();
    RemoteCacheControlUpdates updates;
    reporter->replyNextBodyContaining("EVENT_NODE_REGISTER",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})", false);
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter,
                             [&](bool enabled) { updates.apply(enabled); });
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(updates.waitFor(1));
    const std::vector<std::pair<std::string, bool>> ignored{
        {R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":true}"})", true},
        {R"({"header":{"status":{"code":"FAILED"}},"extra_info":"{\"rtp_remote_cache_enabled\":true}"})", false},
        {R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":\"true\"}"})", false},
        {R"({"header":{"status":{"code":"OK"}},"extra_info":"broken"})", false},
    };
    for (const auto& [response, failure] : ignored) {
        reporter->replyNextBodyContaining("EVENT_HEARTBEAT", response, failure);
        ASSERT_TRUE(reporter->waitForReplyConsumed(kAsyncTestTimeout));
    }
    reporter->replyNextBodyContaining("EVENT_HEARTBEAT",
        R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":true}"})", false);
    ASSERT_TRUE(updates.waitFor(2));
    publisher.stop();
    EXPECT_EQ(updates.values(), (std::vector<bool>{false, true}));
}

class DelayedControlHeartbeatReporter final: public KVCacheEventReporter {
public:
    RecordingReporter requests;
    std::promise<void> heartbeat_entered;
    std::promise<void> release_heartbeat;
    std::shared_future<void> released = release_heartbeat.get_future().share();

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        requests.post(route, request, response);
        if (request.find("EVENT_NODE_REGISTER") != std::string::npos) {
            response = ++registrations_ == 1 ?
                R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})" :
                R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":true}"})";
        } else if (request.find("EVENT_HEARTBEAT") != std::string::npos && !heartbeat_delayed_.exchange(true)) {
            response = R"({"header":{"status":{"code":"OK"}},"extra_info":"{\"rtp_remote_cache_enabled\":false}"})";
            heartbeat_entered.set_value();
            return released.wait_for(kAsyncTestTimeout) == std::future_status::ready;
        } else if (request.find("EVENT_BLOCK_ADD") != std::string::npos && !registration_lost_.exchange(true)) {
            response = R"({"header":{"status":{"code":"NODE_NOT_REGISTERED"}}})";
        }
        return true;
    }

    void release() {
        if (!released_.exchange(true)) {
            release_heartbeat.set_value();
        }
    }

private:
    std::atomic<size_t> registrations_{0};
    std::atomic<bool> heartbeat_delayed_{false};
    std::atomic<bool> registration_lost_{false};
    std::atomic<bool> released_{false};
};

TEST(KVCacheEventPublisherTest, DelayedHeartbeatCannotOverwriteNewRegistrationControl) {
    KVCacheEventPublisherConfig config;
    config.heartbeat_interval_ms = 5;
    config.retry_interval_ms = 1;
    auto reporter = std::make_shared<DelayedControlHeartbeatReporter>();
    auto entered = reporter->heartbeat_entered.get_future();
    RemoteCacheControlUpdates updates;
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter,
                             [&](bool enabled) { updates.apply(enabled); });
    struct ReleaseOnExit {
        std::shared_ptr<DelayedControlHeartbeatReporter> reporter;
        ~ReleaseOnExit() {
            reporter->release();
        }
    } release_on_exit{reporter};
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(updates.waitFor(1));
    ASSERT_EQ(entered.wait_for(kAsyncTestTimeout), std::future_status::ready);
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 99, 0}));
    ASSERT_TRUE(updates.waitFor(2));
    reporter->release();
    ASSERT_TRUE(reporter->requests.waitForBodyCount("EVENT_HEARTBEAT", 3, kAsyncTestTimeout));
    publisher.stop();
    EXPECT_EQ(updates.values(), (std::vector<bool>{false, true}));
}

TEST(KVCacheEventPublisherTest, ParsesSnapshotFeedbackOnSuccessAndFailure) {
    const auto success = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":"OK"}},"snapshot_required":true,"retry_after_ms":"2500"})");
    EXPECT_TRUE(success.valid);
    EXPECT_TRUE(success.ok);
    EXPECT_TRUE(success.snapshot_required);
    EXPECT_FALSE(success.registration_required);

    const auto throttled = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":"SNAPSHOT_RATE_LIMITED"}},"retryAfterMs":125})");
    EXPECT_TRUE(throttled.valid);
    EXPECT_FALSE(throttled.ok);
    EXPECT_FALSE(throttled.registration_required);

    const auto partial = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":"OK"}},"item_results":[1,{"code":"SNAPSHOT_REQUIRED"}]})");
    EXPECT_FALSE(partial.ok);
    EXPECT_TRUE(partial.snapshot_required);
}

TEST(KVCacheEventPublisherTest, NestedReportStatusPreservesRegistrationAndSnapshotSignals) {
    const auto success = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":{"code":"OK"}}}})");
    EXPECT_TRUE(success.valid);
    EXPECT_TRUE(success.ok);
    EXPECT_FALSE(success.registration_required);
    EXPECT_FALSE(success.snapshot_required);

    const auto snapshot = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":{"status":{"code":"SNAPSHOT_REQUIRED"}}}}})");
    EXPECT_TRUE(snapshot.valid);
    EXPECT_FALSE(snapshot.ok);
    EXPECT_FALSE(snapshot.registration_required);
    EXPECT_TRUE(snapshot.snapshot_required);

    const auto unregistered = detail::parseKVCMReportFeedback(
        R"({"header":{"status":{"code":{"code":10}}}})");
    EXPECT_TRUE(unregistered.valid);
    EXPECT_FALSE(unregistered.ok);
    EXPECT_TRUE(unregistered.registration_required);
    EXPECT_FALSE(unregistered.snapshot_required);
}

TEST(KVCacheEventPublisherTest, SnapshotRateLimitDropsRoundWithoutBlockingIncrementals) {
    KVCacheEventPublisherConfig config;
    config.flush_interval_ms = 1;
    config.heartbeat_interval_ms = 5;
    config.snapshot_interval_ms = 60000;
    auto reporter = std::make_shared<RecordingReporter>();
    reporter->replyNextBodyContaining("EVENT_BLOCK_SNAPSHOT",
        R"({"header":{"status":{"code":"SNAPSHOT_RATE_LIMITED"}},"retry_after_ms":"18446744073709551615"})",
        false);
    std::atomic<size_t> captures{0};
    KVCMPublisher publisher(config, makeContext(), [&] {
        const auto version = ++captures;
        return KVCacheSnapshot{static_cast<int64_t>(version), {{static_cast<int64_t>(version), {0}}}};
    }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_ADD", kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 10, kAsyncTestTimeout));
    EXPECT_FALSE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, std::chrono::milliseconds(50)));

    // A new successful mutation hint requests a fresh capture, even after a
    // rate limit. It does not replay the old snapshot or the mutation.
    reporter->replyNextBodyContaining("EVENT_BLOCK_ADD",
        R"({"header":{"status":{"code":"OK"}},"snapshot_required":true})", false);
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    publisher.stop();
    EXPECT_EQ(captures.load(), 2u);
    const auto requests = reporter->requests();
    EXPECT_NE(requests.back().body.find("EVENT_HOST_DOWN"), std::string::npos);
    size_t snapshots = 0;
    for (const auto& request : requests) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            EXPECT_NE(request.body.find("\"block_key\":\"" + std::to_string(++snapshots) + "\""), std::string::npos);
        }
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
            return KVCacheSnapshot{7, {{10, {0, 1}}, {20, {0}}}};
        },
        reporter);

    ASSERT_TRUE(publisher->start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 30, 1}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 1}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"30\"", kAsyncTestTimeout));

    reporter->failNextBodyContaining("EVENT_BLOCK_ADD");
    EXPECT_EQ(PublishResult::ACCEPTED, publisher->tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    ASSERT_TRUE(reporter->waitForBody("\"block_key\":\"31\"", kAsyncTestTimeout));
    EXPECT_FALSE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, std::chrono::milliseconds(50)));
    publisher->stop();
    EXPECT_EQ(publisher->status().dropped_count, 1u);

    const auto requests = reporter->requests();
    ASSERT_GE(requests.size(), 5);
    EXPECT_EQ("/api/registerInstance", requests.front().route);
    EXPECT_NE(std::string::npos, requests.front().body.find("rtp_llm_hbm_64"));
    EXPECT_NE(std::string::npos, requests.front().body.find("location_spec_groups"));
    EXPECT_EQ(1, countOccurrences(requests.front().body, "\"dtype\""));
    rapidjson::Document registration;
    registration.Parse(requests.front().body.c_str());
    ASSERT_FALSE(registration.HasParseError());
    ASSERT_EQ(registration["location_spec_infos"].Size(), 2u);
    EXPECT_STREQ(registration["location_spec_infos"][0]["name"].GetString(), "rtp_llm_hbm_64_full");
    EXPECT_FALSE(registration["location_spec_infos"][0].HasMember("size"));
    EXPECT_STREQ(registration["location_spec_infos"][1]["name"].GetString(), "rtp_llm_hbm_64_tail");
    EXPECT_FALSE(registration["location_spec_infos"][1].HasMember("size"));
    EXPECT_EQ(registration["location_spec_groups"][0]["spec_names"].Size(), 2u);
    EXPECT_EQ(requests.front().body.find("size="), std::string::npos);

    bool   saw_node_register = false;
    bool   saw_snapshot      = false;
    bool   saw_snapshot_item = false;
    bool   saw_add           = false;
    bool   saw_delete        = false;
    size_t host_down_count   = 0;
    for (const auto& request : requests) {
        if (request.route == "/api/reportEvent") {
            rapidjson::Document report;
            report.Parse(request.body.c_str());
            ASSERT_FALSE(report.HasParseError());
            for (const auto& event : report["events"].GetArray()) {
                if (event.HasMember("block_snapshot")) {
                    const auto& blocks = event["block_snapshot"]["blocks"];
                    ASSERT_EQ(blocks.Size(), 2u);
                    EXPECT_STREQ(blocks[0]["block_key"].GetString(), "10");
                    ASSERT_EQ(blocks[0]["specs"].Size(), 2u);
                    EXPECT_STREQ(blocks[0]["specs"][0]["name"].GetString(), "rtp_llm_hbm_64_full");
                    EXPECT_STREQ(blocks[0]["specs"][1]["name"].GetString(), "rtp_llm_hbm_64_tail");
                    EXPECT_EQ(blocks[1]["specs"].Size(), 1u);
                } else if (event.HasMember("block_delete")) {
                    const auto& names = event["block_delete"]["spec_names"];
                    ASSERT_EQ(names.Size(), 1u);
                    EXPECT_STREQ(names[0].GetString(), "rtp_llm_hbm_64_tail");
                }
            }
        }
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
            return KVCacheSnapshot{1, {{10, {0}}}};
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
    config.snapshot_interval_ms  = 100;
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
            return KVCacheSnapshot{2, {{10, {0}}, {20, {0}}}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_ADD", kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    EXPECT_GE(snapshot_attempts.load(std::memory_order_relaxed), 2);
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, SnapshotFailuresWaitForPeriodAndCaptureFreshPayloads) {
    KVCacheEventPublisherConfig config;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms = 40;
    auto reporter = std::make_shared<RecordingReporter>();
    reporter->failNextBodiesContaining("EVENT_BLOCK_SNAPSHOT", 2);
    std::atomic<size_t> captures{0};
    KVCMPublisher publisher(config, makeContext(), [&] {
        const auto version = ++captures;
        return KVCacheSnapshot{static_cast<int64_t>(version), {{static_cast<int64_t>(version), {0}}}};
    }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
    publisher.stop();
    std::vector<RecordingReporter::Request> snapshots;
    for (const auto& request : reporter->requests()) {
        if (request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos) {
            snapshots.push_back(request);
        }
    }
    ASSERT_GE(snapshots.size(), 3u);
    EXPECT_GE(captures.load(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NE(snapshots[i].body.find("\"block_key\":\"" + std::to_string(i + 1) + "\""), std::string::npos);
        if (i != 0) {
            EXPECT_NE(snapshots[i].body, snapshots[i - 1].body);
        }
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherHeartbeatsWhileSnapshotUploadKeepsFailing) {
    KVCacheEventPublisherConfig config;
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 10;
    config.snapshot_interval_ms  = 30;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<RecordingReporter>();
    reporter->failNextBodiesContaining("EVENT_BLOCK_SNAPSHOT", 100);
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {{10, {0}}, {20, {0}}}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 2, kAsyncTestTimeout));
    ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_ADD", kAsyncTestTimeout));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, KVCMPublisherHeartbeatsWhileSnapshotUploadIsSlow) {
    KVCacheEventPublisherConfig config;
    config.queue_capacity        = 8;
    config.report_batch_size     = 8;
    config.flush_interval_ms     = 1;
    config.heartbeat_interval_ms = 10;
    config.snapshot_interval_ms  = 60000;
    config.retry_interval_ms     = 1;

    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(
        config,
        makeContext(),
        [] {
            return KVCacheSnapshot{1, {{10, {0}}, {20, {0}}}};
        },
        reporter);

    ASSERT_TRUE(publisher.start());
    if (!reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout)) {
        reporter->releaseSnapshot();
        publisher.stop();
        FAIL() << "initial snapshot request did not reach the blocking reporter";
    }
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 2, kAsyncTestTimeout));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_ADD", 1, kAsyncTestTimeout));
    reporter->releaseSnapshot();
    EXPECT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_ADD", 1, kAsyncTestTimeout));
    publisher.stop();
    size_t snapshot_count = 0;
    for (const auto& request : reporter->requests()) {
        snapshot_count += request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
    }
    EXPECT_GE(snapshot_count, 1u);
    EXPECT_LE(snapshot_count, 2u);
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
            return KVCacheSnapshot{1, {{10, {0}}}};
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

    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 10, 1}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_ADD", 1, kAsyncTestTimeout));
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_DELETE", 1, kAsyncTestTimeout));
    reporter->releaseSnapshot();
    publisher.stop();
    for (const auto& request : reporter->requests()) {
        rapidjson::Document report;
        report.Parse(request.c_str());
        ASSERT_FALSE(report.HasParseError());
        if (!report.HasMember("events")) {
            continue;
        }
        for (const auto& event : report["events"].GetArray()) {
            if (event.HasMember("block_add")) {
                EXPECT_STREQ(event["block_add"]["block_key"].GetString(), "10");
                EXPECT_STREQ(event["block_add"]["specs"][0]["name"].GetString(), "rtp_llm_hbm_64_tail");
            } else if (event.HasMember("block_delete")) {
                EXPECT_STREQ(event["block_delete"]["block_key"].GetString(), "10");
                ASSERT_EQ(event["block_delete"]["spec_names"].Size(), 1u);
                EXPECT_STREQ(event["block_delete"]["spec_names"][0].GetString(), "rtp_llm_hbm_64_full");
            }
        }
    }
}

TEST(KVCacheEventPublisherTest, KVCMPublisherCoalescesEachComponentToItsLastMutation) {
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
    // only the final state for each (key, group) in this queued batch.
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 42, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 43, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 43, 0}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 42, 1}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 42, 1}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 43, 1}));
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 43, 1}));
    reporter->releaseMutation();

    ASSERT_TRUE(reporter->waitForBodyCount("\"block_key\":\"42\"", 1, kAsyncTestTimeout));
    publisher.stop();

    size_t component_events = 0;
    for (const auto& request : reporter->requests()) {
        rapidjson::Document report;
        report.Parse(request.c_str());
        ASSERT_FALSE(report.HasParseError());
        if (!report.HasMember("events")) {
            continue;
        }
        for (const auto& event : report["events"].GetArray()) {
            const bool add = event.HasMember("block_add");
            if (!add && !event.HasMember("block_delete")) {
                continue;
            }
            const auto& mutation = event[add ? "block_add" : "block_delete"];
            const std::string key = mutation["block_key"].GetString();
            if (key == "999") {
                continue;
            }
            ++component_events;
            ASSERT_TRUE(key == "42" || key == "43");
            const auto& specs = mutation[add ? "specs" : "spec_names"];
            ASSERT_EQ(specs.Size(), 1u);
            const auto* name = add ? specs[0]["name"].GetString() : specs[0].GetString();
            const bool full = std::string(name) == "rtp_llm_hbm_64_full";
            EXPECT_EQ(add, (key == "42") == full);
        }
    }
    EXPECT_EQ(component_events, 4u);
}

TEST(KVCacheEventPublisherTest, QueueOverflowDoesNotTriggerAnImmediateSnapshot) {
    KVCacheEventPublisherConfig config;
    config.queue_capacity = 1;
    config.report_batch_size = 1;
    config.flush_interval_ms = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms = 60000;
    auto reporter = std::make_shared<BlockingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{1, {}}; }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 1, kAsyncTestTimeout));
    reporter->blockNextMutation();
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    if (!reporter->waitUntilMutationBlocked(kAsyncTestTimeout)) {
        reporter->releaseMutation();
        FAIL() << "mutation did not reach reporter";
    }
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 31, 0}));
    EXPECT_EQ(PublishResult::QUEUE_FULL, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 10, 0}));
    EXPECT_EQ(publisher.status().dropped_count, 1u);
    reporter->releaseMutation();
    ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_ADD", 2, kAsyncTestTimeout));
    EXPECT_FALSE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, std::chrono::milliseconds(100)));
    publisher.stop();
}

TEST(KVCacheEventPublisherTest, PeriodicCaptureContinuesWhileSnapshotAckIsBlocked) {
    KVCacheEventPublisherConfig config;
    config.flush_interval_ms = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms = 150;
    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    std::mutex mu;
    std::condition_variable cv;
    size_t captures = 0;
    KVCMPublisher publisher(config, makeContext(), [&] {
        std::lock_guard<std::mutex> lock(mu);
        const auto version = static_cast<int64_t>(++captures);
        cv.notify_all();
        return KVCacheSnapshot{version, {{version, {0}}}};
    }, reporter);
    ASSERT_TRUE(publisher.start());
    const bool blocked = reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout);
    bool captured_while_blocked = false;
    if (blocked) {
        std::unique_lock<std::mutex> lock(mu);
        captured_while_blocked = cv.wait_for(lock, kAsyncTestTimeout, [&] { return captures >= 4; });
    }
    EXPECT_TRUE(blocked);
    EXPECT_TRUE(captured_while_blocked);
    reporter->releaseSnapshot();
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    publisher.stop();
    size_t snapshots = 0;
    for (const auto& request : reporter->requests()) {
        if (request.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos && ++snapshots == 2) {
            // The unsent second capture was superseded before the ACK.
            rapidjson::Document report;
            report.Parse(request.c_str());
            EXPECT_GE(std::stoll(report["events"][0]["block_snapshot"]["blocks"][0]["block_key"].GetString()), 3);
        }
    }
}

TEST(KVCacheEventPublisherTest, PendingSnapshotSignalsCoalesceDuringCapture) {
    KVCacheEventPublisherConfig config;
    config.report_batch_size = 1;
    config.flush_interval_ms = 1;
    config.heartbeat_interval_ms = 60000;
    config.snapshot_interval_ms = 60000;
    auto reporter = std::make_shared<RecordingReporter>();
    std::mutex mu;
    std::condition_variable cv;
    size_t captures = 0;
    bool release = false;
    KVCMPublisher publisher(config, makeContext(), [&] {
        std::unique_lock<std::mutex> lock(mu);
        const auto version = static_cast<int64_t>(++captures);
        cv.notify_all();
        if (version == 2) {
            cv.wait(lock, [&] { return release; });
        }
        return KVCacheSnapshot{version, {}};
    }, reporter);
    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_SNAPSHOT", kAsyncTestTimeout));
    reporter->replyNextBodiesContaining("EVENT_BLOCK_ADD",
        R"({"header":{"status":{"code":"OK"}},"snapshot_required":true})", false, 5);
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
    bool blocked;
    {
        std::unique_lock<std::mutex> lock(mu);
        blocked = cv.wait_for(lock, kAsyncTestTimeout, [&] { return captures == 2; });
    }
    EXPECT_TRUE(blocked);
    for (int64_t key = 31; key < 35; ++key) {
        EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, key, 0}));
    }
    EXPECT_TRUE(reporter->waitForReplyConsumed(kAsyncTestTimeout));
    // This later request proves the serial mutation worker consumed all hints.
    EXPECT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_DELETE, 999, 0}));
    EXPECT_TRUE(reporter->waitForBody("EVENT_BLOCK_DELETE", kAsyncTestTimeout));
    {
        std::lock_guard<std::mutex> lock(mu);
        release = true;
    }
    cv.notify_all();
    EXPECT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
    {
        std::unique_lock<std::mutex> lock(mu);
        EXPECT_TRUE(cv.wait_for(lock, kAsyncTestTimeout, [&] { return captures == 3; }));
    }
    EXPECT_FALSE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 4, std::chrono::milliseconds(50)));
    publisher.stop();
    EXPECT_EQ(captures, 3u);
}

TEST(KVCacheEventPublisherTest, StopWaitsForSnapshotBeforeFinalHostDown) {
    KVCacheEventPublisherConfig config;
    config.heartbeat_interval_ms = 60000;
    auto reporter = std::make_shared<BlockingReporter>();
    reporter->blockNextSnapshot();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{}; }, reporter);
    ASSERT_TRUE(publisher.start());
    const bool blocked = reporter->waitUntilSnapshotBlocked(kAsyncTestTimeout);
    std::thread stopper([&] { publisher.stop(); });
    reporter->releaseSnapshot();
    stopper.join();
    EXPECT_TRUE(blocked);
    const auto requests = reporter->requests();
    ASSERT_FALSE(requests.empty());
    EXPECT_NE(requests.back().find("EVENT_HOST_DOWN"), std::string::npos);
    EXPECT_EQ(publisher.status().state, PublisherState::STOPPED);
}

TEST(KVCacheEventPublisherTest, AckSilenceEscalatesOnceAndOnlyNormalAcksRearmIt) {
    for (const bool recover_with_mutation : {false, true}) {
        SCOPED_TRACE(recover_with_mutation);
        KVCacheEventPublisherConfig config;
        config.flush_interval_ms      = 1;
        config.heartbeat_interval_ms  = 5;
        config.snapshot_interval_ms   = 60000;
        config.retry_interval_ms      = 1;
        config.ack_recovery_timeout_ms = 40;
        auto reporter = std::make_shared<RecordingReporter>();
        KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{1, {}}; }, reporter);

        ASSERT_TRUE(publisher.start());
        ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
        reporter->replyNextBodiesContaining("EVENT_HEARTBEAT", "", true, 100000);
        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_NODE_REGISTER", 2, kAsyncTestTimeout));
        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 2, kAsyncTestTimeout));
        ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", 40, kAsyncTestTimeout));

        // Empty queue polls and successful recovery requests must not rearm.
        size_t registrations = 0;
        size_t heartbeats = 0;
        for (const auto& request : reporter->requests()) {
            registrations += request.route == "/api/registerInstance";
            heartbeats += request.body.find("EVENT_HEARTBEAT") != std::string::npos;
        }
        EXPECT_EQ(registrations, 2u);

        if (recover_with_mutation) {
            ASSERT_EQ(PublishResult::ACCEPTED, publisher.tryPublish({KVCacheEventType::BLOCK_ADD, 30, 0}));
            ASSERT_TRUE(reporter->waitForBody("EVENT_BLOCK_ADD", kAsyncTestTimeout));
        } else {
            reporter->replyNextBodyContaining("EVENT_HEARTBEAT", R"({"header":{"status":{"code":"OK"}}})", false);
            ASSERT_TRUE(reporter->waitForReplyConsumed(kAsyncTestTimeout));
            reporter->replyNextBodiesContaining("EVENT_HEARTBEAT", "", true, 100000);
        }

        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_NODE_REGISTER", 3, kAsyncTestTimeout));
        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_SNAPSHOT", 3, kAsyncTestTimeout));
        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_HEARTBEAT", heartbeats + 40, kAsyncTestTimeout));
        publisher.stop();

        registrations = 0;
        size_t snapshots = 0;
        for (const auto& request : reporter->requests()) {
            registrations += request.route == "/api/registerInstance";
            snapshots += request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        }
        EXPECT_EQ(registrations, 3u);
        EXPECT_EQ(snapshots, 3u);
    }
}

TEST(KVCacheEventPublisherTest, MutationAcksPreventEscalationWhileHeartbeatsFail) {
    KVCacheEventPublisherConfig config;
    config.report_batch_size      = 1;
    config.flush_interval_ms      = 1;
    config.heartbeat_interval_ms  = 5;
    config.snapshot_interval_ms   = 60000;
    config.retry_interval_ms      = 1;
    config.ack_recovery_timeout_ms = 500;
    auto reporter = std::make_shared<RecordingReporter>();
    KVCMPublisher publisher(config, makeContext(), [] { return KVCacheSnapshot{1, {}}; }, reporter);

    ASSERT_TRUE(publisher.start());
    ASSERT_TRUE(waitForState(publisher, PublisherState::READY, kAsyncTestTimeout));
    reporter->replyNextBodiesContaining("EVENT_HEARTBEAT", "", true, 100000);
    for (size_t i = 0; i < 24; ++i) {
        ASSERT_EQ(PublishResult::ACCEPTED,
                  publisher.tryPublish({KVCacheEventType::BLOCK_ADD, static_cast<int64_t>(i), 0}));
        ASSERT_TRUE(reporter->waitForBodyCount("EVENT_BLOCK_ADD", i + 1, kAsyncTestTimeout));
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    publisher.stop();

    size_t registrations = 0;
    size_t snapshots = 0;
    for (const auto& request : reporter->requests()) {
        registrations += request.route == "/api/registerInstance";
        snapshots += request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
    }
    EXPECT_EQ(registrations, 1u);
    EXPECT_EQ(snapshots, 1u);
}

}  // namespace
}  // namespace rtp_llm
