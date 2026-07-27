#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

namespace rtp_llm {
namespace transfer {
namespace {

class FakeSender: public IKVCacheSender {
public:
    bool regMem(const BlockInfo&, uint64_t) override {
        return true;
    }

    void send(const SendRequest&, std::function<void(TransferErrorCode, const std::string&)> callback) override {
        callback(TransferErrorCode::OK, "");
    }
};

class FakeReceiver: public IKVCacheReceiver {
public:
    bool regMem(const BlockInfo&, uint64_t) override {
        return true;
    }

    IKVCacheRecvTaskPtr recv(const RecvRequest&) override {
        return nullptr;
    }

    void stealTask(const std::string&) override {}

    IKVCacheRecvTaskPtr getTask(const std::string&) override {
        return nullptr;
    }
};

class FakeLifecycleSender: public FakeSender, public ITransferBackendLifecycle {
public:
    bool supportsTransportOnlyCheckpoint() const override {
        return true;
    }

    bool stopTransportForCheckpoint() override {
        ++stop_count_;
        return true;
    }

    bool restoreTransportAfterCheckpoint() override {
        ++restore_count_;
        return true;
    }

    bool resume() override {
        ++resume_count_;
        return true;
    }

    int stop_count_{0};
    int restore_count_{0};
    int resume_count_{0};
};

TEST(TransferBackendLifecycleTest, StatelessBackendLifecycleIsNoop) {
    TransferBackendPair backend{std::make_shared<FakeSender>(), std::make_shared<FakeReceiver>()};

    // No lifecycle endpoint: transport-only checkpoint is unsupported and the
    // stop/restore hooks are no-ops that report failure; resume() is a benign no-op.
    EXPECT_FALSE(backend.supportsTransportOnlyCheckpoint());
    EXPECT_FALSE(backend.stopTransportForCheckpoint());
    EXPECT_FALSE(backend.restoreTransportAfterCheckpoint());
    EXPECT_TRUE(backend.resume());
}

TEST(TransferBackendLifecycleTest, DelegatesLifecycleToBackendEndpoint) {
    auto                sender = std::make_shared<FakeLifecycleSender>();
    TransferBackendPair backend{sender, std::make_shared<FakeReceiver>()};

    EXPECT_TRUE(backend.supportsTransportOnlyCheckpoint());
    EXPECT_TRUE(backend.stopTransportForCheckpoint());
    EXPECT_TRUE(backend.restoreTransportAfterCheckpoint());
    EXPECT_TRUE(backend.resume());
    EXPECT_EQ(sender->stop_count_, 1);
    EXPECT_EQ(sender->restore_count_, 1);
    EXPECT_EQ(sender->resume_count_, 1);
}

TEST(TransferBackendLifecycleTest, PreservesTwoElementStructuredBinding) {
    auto                expected_sender   = std::make_shared<FakeSender>();
    auto                expected_receiver = std::make_shared<FakeReceiver>();
    TransferBackendPair backend{expected_sender, expected_receiver};

    auto [sender, receiver] = backend;

    EXPECT_EQ(sender, expected_sender);
    EXPECT_EQ(receiver, expected_receiver);
}

}  // namespace
}  // namespace transfer
}  // namespace rtp_llm
