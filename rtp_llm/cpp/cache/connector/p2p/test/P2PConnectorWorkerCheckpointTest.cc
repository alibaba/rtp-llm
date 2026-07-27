// Integration test for the sleep-mode Level3 transport-only checkpoint path.
//
// Level3 PD-disaggregation over RDMA must stop the KV-cache transport before the
// CUDA process checkpoint and restore it afterwards. The capability is advertised
// through ITransferBackendLifecycle::supportsTransportOnlyCheckpoint() and driven
// by stopTransportForCheckpoint()/restoreTransportAfterCheckpoint(). The Barex RDMA
// backend overrides these (delegating to a shared RdmaBackendLifecycle); the base
// default returns false.
//
// This test wires a fake lifecycle-aware transfer backend into the *real*
// P2PConnectorWorker (the same injection point that P2PConnectorWorker::init()
// populates via createTransferBackend) and proves the worker-side contract:
//   (a) the worker reports supportsTransportOnlyCheckpoint() from its backend;
//   (b) the Level3 suspend path (teardownRdmaTransports) calls
//       stopTransportForCheckpoint exactly once and is fail-closed;
//   (c) the wake path (rebuildRdmaTransports) calls restoreTransportAfterCheckpoint
//       before transport is resumed and before it is usable again;
//   (d) transport is stopped (and KV-pointer-holding logical state is dropped)
//       before the checkpointable point.
//
// The Barex RdmaBackendLifecycle itself is unit-tested with a mocked verbs/memory
// layer in
//   internal_source/.../barex_rdma/test/RdmaBackendLifecycleTest.cc
// so no RDMA hardware is required here either — this test covers the
// factory->worker->lifecycle *delegation* that the Barex test does not exercise.

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

namespace rtp_llm {
namespace test {
namespace {

// Ordered, thread-safe log of lifecycle transitions plus test-injected markers.
class EventLog {
public:
    void push(const std::string& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }

    std::vector<std::string> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }

    // Index of the first occurrence of `event`, or -1 if absent.
    int indexOf(const std::string& event) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < static_cast<int>(events_.size()); ++i) {
            if (events_[i] == event) {
                return i;
            }
        }
        return -1;
    }

    int count(const std::string& event) const {
        std::lock_guard<std::mutex> lock(mutex_);
        int                         n = 0;
        for (const auto& e : events_) {
            if (e == event) {
                ++n;
            }
        }
        return n;
    }

private:
    mutable std::mutex       mutex_;
    std::vector<std::string> events_;
};

// A minimal receiver with no lifecycle behaviour — used for the "default backend"
// (capability == false) case and as the passive endpoint of the fake pair.
class FakeReceiver: public transfer::IKVCacheReceiver {
public:
    bool regMem(const BlockInfo&, uint64_t) override {
        return true;
    }
    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest&) override {
        return nullptr;
    }
    void                          stealTask(const std::string&) override {}
    transfer::IKVCacheRecvTaskPtr getTask(const std::string&) override {
        return nullptr;
    }
};

// A sender that ALSO implements the transport-only checkpoint lifecycle, mirroring
// how the Barex RdmaKVCacheSender derives from ITransferBackendLifecycle and
// delegates to the shared RdmaBackendLifecycle. TransferBackendPair::getLifecycle()
// discovers this via dynamic_cast on the sender endpoint.
class FakeLifecycleSender: public transfer::IKVCacheSender, public transfer::ITransferBackendLifecycle {
public:
    explicit FakeLifecycleSender(std::shared_ptr<EventLog> log): log_(std::move(log)) {}

    // --- IKVCacheSender ---
    bool regMem(const BlockInfo&, uint64_t) override {
        return true;
    }
    void send(const transfer::SendRequest&,
              std::function<void(transfer::TransferErrorCode, const std::string&)> callback) override {
        callback(transfer::TransferErrorCode::OK, "");
    }

    // --- ITransferBackendLifecycle: Level3 transport-only checkpoint capability ---
    bool supportsTransportOnlyCheckpoint() const override {
        return supports_transport_only_;
    }
    bool stopTransportForCheckpoint() override {
        log_->push("stop_transport");
        ++stop_count_;
        return stop_result_;
    }
    bool restoreTransportAfterCheckpoint() override {
        log_->push("restore_transport");
        ++restore_count_;
        return restore_result_;
    }
    bool resume() override {
        log_->push("resume");
        return true;
    }

    // Test knobs.
    void setSupportsTransportOnly(bool v) {
        supports_transport_only_ = v;
    }
    void setStopResult(bool v) {
        stop_result_ = v;
    }
    void setRestoreResult(bool v) {
        restore_result_ = v;
    }
    int stopCount() const {
        return stop_count_.load();
    }
    int restoreCount() const {
        return restore_count_.load();
    }

private:
    std::shared_ptr<EventLog> log_;
    bool                      supports_transport_only_{true};
    bool                      stop_result_{true};
    bool                      restore_result_{true};
    std::atomic<int>          stop_count_{0};
    std::atomic<int>          restore_count_{0};
};

// LayerBlockConverter with no buffers — regMem loop in worker paths becomes a no-op,
// keeping the whole test CPU-only (matches the existing P2PConnectorWorkerTest style).
class EmptyLayerBlockConverter: public LayerBlockConverter {
public:
    std::vector<BlockInfo> convertIndexToBuffer(int, int, int, int) const override {
        return {};
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

class P2PConnectorWorkerCheckpointTest: public ::testing::Test {
protected:
    void SetUp() override {
        // Match the CPU-only config used by the sibling P2PConnectorWorkerTest.
        config_.transfer_backend_config.cache_store_rdma_mode        = true;
        config_.transfer_backend_config.messager_io_thread_count     = 1;
        config_.transfer_backend_config.messager_worker_thread_count = 1;
        config_.transfer_backend_config.cache_store_listen_port      = 0;
        config_.tp_size                                              = 2;
        config_.tp_rank                                              = 0;
        config_.layer_all_num                                        = 2;

        converter_ = std::make_shared<EmptyLayerBlockConverter>();
        log_       = std::make_shared<EventLog>();
    }

    // Build a worker whose transfer backend is the lifecycle-aware fake, with logical
    // state already established (as after a successful init()).
    std::unique_ptr<P2PConnectorWorker> makeRunningWorker(std::shared_ptr<FakeLifecycleSender>& sender_out) {
        auto sender   = std::make_shared<FakeLifecycleSender>(log_);
        auto receiver = std::make_shared<FakeReceiver>();
        sender_out    = sender;

        auto worker               = std::make_unique<P2PConnectorWorker>(config_, converter_, nullptr);
        worker->transfer_backend_ = transfer::TransferBackendPair{sender, receiver};
        EXPECT_TRUE(worker->rebuildLogicalStateAfterRestore());
        EXPECT_NE(worker->prefill_, nullptr);
        EXPECT_NE(worker->decode_, nullptr);
        return worker;
    }

    P2PConnectorWorkerConfig             config_;
    std::shared_ptr<LayerBlockConverter> converter_;
    std::shared_ptr<EventLog>            log_;
};

// (a) Capability reporting.
TEST_F(P2PConnectorWorkerCheckpointTest, ReportsCapabilityFromBackend) {
    // Lifecycle-aware backend that advertises support -> true.
    {
        auto               sender   = std::make_shared<FakeLifecycleSender>(log_);
        auto               receiver = std::make_shared<FakeReceiver>();
        P2PConnectorWorker worker(config_, converter_, nullptr);
        worker.transfer_backend_ = transfer::TransferBackendPair{sender, receiver};
        EXPECT_TRUE(worker.transfer_backend_.supportsTransportOnlyCheckpoint());
    }

    // Same backend but capability disabled -> false.
    {
        auto sender = std::make_shared<FakeLifecycleSender>(log_);
        sender->setSupportsTransportOnly(false);
        auto               receiver = std::make_shared<FakeReceiver>();
        P2PConnectorWorker worker(config_, converter_, nullptr);
        worker.transfer_backend_ = transfer::TransferBackendPair{sender, receiver};
        EXPECT_FALSE(worker.transfer_backend_.supportsTransportOnlyCheckpoint());
    }

    // Default (stateless / no-lifecycle) backend -> false, matching TransferBackendFactory.h base default.
    // Here the sender does NOT implement ITransferBackendLifecycle, so getLifecycle() finds none.
    {
        struct PlainSender: public transfer::IKVCacheSender {
            bool regMem(const BlockInfo&, uint64_t) override {
                return true;
            }
            void send(const transfer::SendRequest&,
                      std::function<void(transfer::TransferErrorCode, const std::string&)> cb) override {
                cb(transfer::TransferErrorCode::OK, "");
            }
        };
        P2PConnectorWorker worker(config_, converter_, nullptr);
        worker.transfer_backend_ =
            transfer::TransferBackendPair{std::make_shared<PlainSender>(), std::make_shared<FakeReceiver>()};
        EXPECT_FALSE(worker.transfer_backend_.supportsTransportOnlyCheckpoint());
    }
}

// (b)+(d) Suspend path: stopTransportForCheckpoint called exactly once, logical state
// dropped, and transport stopped before the checkpointable point.
TEST_F(P2PConnectorWorkerCheckpointTest, SuspendStopsTransportBeforeCheckpointablePoint) {
    std::shared_ptr<FakeLifecycleSender> sender;
    auto                                 worker = makeRunningWorker(sender);

    ASSERT_TRUE(worker->teardownRdmaTransports());

    // stopTransportForCheckpoint invoked exactly once.
    EXPECT_EQ(sender->stopCount(), 1);
    EXPECT_EQ(log_->count("stop_transport"), 1);

    // (d) KV-pointer-holding logical wrappers are gone BEFORE the checkpointable point,
    // and transport is stopped. Inject the "checkpointable point" marker now (this is
    // where the sleep controller would run the CUDA process checkpoint).
    EXPECT_EQ(worker->prefill_, nullptr);
    EXPECT_EQ(worker->decode_, nullptr);
    log_->push("CHECKPOINT_POINT");

    const int stop_idx       = log_->indexOf("stop_transport");
    const int checkpoint_idx = log_->indexOf("CHECKPOINT_POINT");
    ASSERT_GE(stop_idx, 0);
    ASSERT_GE(checkpoint_idx, 0);
    EXPECT_LT(stop_idx, checkpoint_idx) << "transport must be stopped before the checkpointable point";
}

// (b) Fail-closed: capability advertised but stopTransportForCheckpoint returns false
// -> teardownRdmaTransports returns false so the controller will NOT checkpoint.
TEST_F(P2PConnectorWorkerCheckpointTest, SuspendFailsClosedWhenStopReturnsFalse) {
    std::shared_ptr<FakeLifecycleSender> sender;
    auto                                 worker = makeRunningWorker(sender);
    sender->setStopResult(false);

    EXPECT_FALSE(worker->teardownRdmaTransports());
    EXPECT_EQ(sender->stopCount(), 1);  // attempted exactly once
}

// (b) Fail-closed gate: a backend that does NOT support transport-only checkpoint must
// short-circuit — teardownRdmaTransports returns false WITHOUT calling stopTransport.
TEST_F(P2PConnectorWorkerCheckpointTest, SuspendRefusedWhenBackendUnsupported) {
    std::shared_ptr<FakeLifecycleSender> sender;
    auto                                 worker = makeRunningWorker(sender);
    sender->setSupportsTransportOnly(false);

    EXPECT_FALSE(worker->teardownRdmaTransports());
    EXPECT_EQ(sender->stopCount(), 0) << "must not stop transport when capability is not advertised";
    EXPECT_EQ(log_->count("stop_transport"), 0);
}

// (c) Wake path: restoreTransportAfterCheckpoint runs before logical state is rebuilt
// and before resume(); resume happens only on the separate resumeRdmaTransports call.
TEST_F(P2PConnectorWorkerCheckpointTest, WakeRestoresTransportBeforeResume) {
    std::shared_ptr<FakeLifecycleSender> sender;
    auto                                 worker = makeRunningWorker(sender);

    // Suspend first so we are in the checkpointed state.
    ASSERT_TRUE(worker->teardownRdmaTransports());
    ASSERT_EQ(worker->prefill_, nullptr);

    // Wake: rebuild transports. This must restore transport THEN rebuild logical state,
    // and must NOT resume yet.
    ASSERT_TRUE(worker->rebuildRdmaTransports());
    EXPECT_EQ(sender->restoreCount(), 1);
    EXPECT_NE(worker->prefill_, nullptr);
    EXPECT_NE(worker->decode_, nullptr);
    EXPECT_EQ(log_->count("resume"), 0) << "rebuild must not resume transport";

    // Resume is a distinct step.
    ASSERT_TRUE(worker->resumeRdmaTransports());
    EXPECT_EQ(log_->count("resume"), 1);

    const int restore_idx = log_->indexOf("restore_transport");
    const int resume_idx  = log_->indexOf("resume");
    ASSERT_GE(restore_idx, 0);
    ASSERT_GE(resume_idx, 0);
    EXPECT_LT(restore_idx, resume_idx) << "transport must be restored before it is resumed";
}

// (c) Wake fail path: if restoreTransportAfterCheckpoint fails, rebuild reports failure
// and does NOT rebuild logical state (nothing may observe stale/unregistered MRs).
TEST_F(P2PConnectorWorkerCheckpointTest, WakeFailsWhenRestoreTransportFails) {
    std::shared_ptr<FakeLifecycleSender> sender;
    auto                                 worker = makeRunningWorker(sender);

    ASSERT_TRUE(worker->teardownRdmaTransports());
    ASSERT_EQ(worker->prefill_, nullptr);

    sender->setRestoreResult(false);
    EXPECT_FALSE(worker->rebuildRdmaTransports());
    EXPECT_EQ(sender->restoreCount(), 1);
    EXPECT_EQ(worker->prefill_, nullptr) << "logical state must stay down when transport restore fails";
    EXPECT_EQ(worker->decode_, nullptr);
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
