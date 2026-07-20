#include "rtp_llm/cpp/cache/KVCacheEventPublisher.h"

#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
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

    bool saw_node_register = false;
    bool saw_snapshot      = false;
    bool saw_add           = false;
    bool saw_delete        = false;
    for (const auto& request : requests) {
        saw_node_register = saw_node_register || request.body.find("EVENT_NODE_REGISTER") != std::string::npos;
        saw_snapshot      = saw_snapshot || request.body.find("EVENT_BLOCK_SNAPSHOT") != std::string::npos;
        saw_add           = saw_add || request.body.find("EVENT_BLOCK_ADD") != std::string::npos;
        saw_delete        = saw_delete || request.body.find("EVENT_BLOCK_DELETE") != std::string::npos;
    }
    EXPECT_TRUE(saw_node_register);
    EXPECT_TRUE(saw_snapshot);
    EXPECT_TRUE(saw_add);
    EXPECT_TRUE(saw_delete);
    EXPECT_EQ(3, publisher->status().accepted_count);
}

}  // namespace
}  // namespace rtp_llm
