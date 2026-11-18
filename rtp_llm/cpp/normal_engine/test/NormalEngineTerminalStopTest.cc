#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "autil/EnvUtil.h"
#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#ifdef RTP_LLM_TEST_PUBLISHER_SHUTDOWN
#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include <pybind11/embed.h>
#endif

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

// Only the executor/transport is substituted: the real engine loop, FIFO,
// arm/cancel/commit APIs and thread joins run on both ranks.
struct TerminalTransport {
    std::mutex                      mutex;
    std::condition_variable         changed;
    std::vector<bool>               inputs;
    bool                            hold_process    = false;
    bool                            shutdown        = false;
    bool                            release_on_stop = true;
    std::atomic<int>                stop_notifications{0};
    std::array<std::atomic<int>, 2> entered{{0, 0}};
    std::array<std::atomic<int>, 2> completed{{0, 0}};
    std::atomic<int>                terminals{0};
};

class TerminalExecutor: public Executor {
public:
    TerminalExecutor(TerminalTransport& transport, int rank): transport_(transport), rank_(rank) {}
    bool supportsTerminalStop() const override {
        return true;
    }
    void requestTerminalStop() override {
        requested_ = true;
    }
    bool terminalStopObserved() const override {
        return observed_;
    }
    void notifyStop() override {
        std::lock_guard<std::mutex> lock(transport_.mutex);
        ++transport_.stop_notifications;
        if (!transport_.release_on_stop) {
            return;
        }
        transport_.shutdown = true;
        transport_.changed.notify_all();
    }
    absl::Status process(const std::list<GenerateStreamPtr>&, int64_t = 0) override {
        if (!error.ok()) {
            return error;
        }
        if (throw_error) {
            throw std::runtime_error("injected process exception");
        }
        std::unique_lock<std::mutex> lock(transport_.mutex);
        if (rank_ == 0) {
            transport_.inputs.push_back(requested_);
            if (requested_) {
                ++transport_.terminals;
            }
            transport_.changed.notify_all();
        }
        transport_.changed.wait(lock, [&]() { return transport_.shutdown || index_ < transport_.inputs.size(); });
        if (transport_.shutdown) {
            return absl::OkStatus();
        }
        const bool terminal = transport_.inputs[index_++];
        ++transport_.entered[rank_];
        transport_.changed.notify_all();
        transport_.changed.wait(lock, [&]() { return transport_.shutdown || !transport_.hold_process; });
        observed_ = terminal;
        ++transport_.completed[rank_];
        return absl::OkStatus();
    }
    absl::Status error       = absl::OkStatus();
    bool         throw_error = false;

private:
    TerminalTransport& transport_;
    const int          rank_;
    size_t             index_     = 0;
    bool               requested_ = false;
    bool               observed_  = false;
};

class NormalEngineTerminalStopTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        autil::EnvGuard startup_warmup("RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS", "");
        for (int rank = 0; rank < 2; ++rank) {
            ModelConfig   model;
            RuntimeConfig runtime;
            KVCacheConfig cache;
            auto          params               = createEngineInitParams(CustomConfig{}, model, runtime, cache);
            NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
                return std::make_unique<MockModel>(model.vocab_size);
            };
            try {
                engines_[rank] = std::make_shared<NormalEngine>(params, nullptr, true);
            } catch (...) {
                NormalExecutor::test_model_factory = nullptr;
                throw;
            }
            NormalExecutor::test_model_factory = nullptr;
            // Model construction is single-device; only executor transport is
            // replaced for the two-rank engine control-plane test.
            auto& pc   = engines_[rank]->parallelism_config;
            pc.tp_size = pc.world_size = pc.ep_size = pc.ffn_tp_size = 2;
            pc.tp_rank = pc.world_rank = pc.ep_rank = pc.ffn_tp_rank = rank;
            pc.prefill_cp_config.prefill_cp_size                     = 2;
            auto executor             = std::make_unique<TerminalExecutor>(transport_, rank);
            executors_[rank]          = executor.get();
            engines_[rank]->executor_ = std::move(executor);
        }
    }

    void TearDown() override {
        // Failure cleanup only; successful late-root tests must finish their
        // cancellation ACKs and loop joins with transport stop-release disabled.
        {
            std::lock_guard<std::mutex> lock(transport_.mutex);
            transport_.release_on_stop = true;
        }
        for (auto& engine : engines_) {
            if (engine) {
                engine->requestForceStop();
            }
        }
        for (auto& thread : controls_) {
            thread.join();
        }
        for (auto& engine : engines_) {
            if (engine) {
                (void)engine->forceStop();
                engine.reset();
            }
        }
        DeviceTestBase::TearDown();
    }

    std::future<absl::Status> arm(int rank) {
        std::packaged_task<absl::Status()> task([&, rank]() { return engines_[rank]->armStopAtStep(0, 2000); });
        auto                               result = task.get_future();
        controls_.emplace_back(std::move(task));
        return result;
    }
    void expectReady(std::future<absl::Status>& result) {
        ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
        EXPECT_TRUE(result.get().ok());
    }
    void start() {
        for (auto& engine : engines_) {
            ASSERT_TRUE(engine->startLoop().ok());
        }
    }
    void commit() {
        for (auto& engine : engines_) {
            EXPECT_TRUE(engine->stopAtStep(0).ok());
        }
    }

    TerminalTransport                            transport_;
    std::array<std::shared_ptr<NormalEngine>, 2> engines_;
    std::array<TerminalExecutor*, 2>             executors_;
    std::vector<std::thread>                     controls_;
};

#ifdef RTP_LLM_TEST_PUBLISHER_SHUTDOWN
// Isolated target: real RtpLLMOp/service/engine shutdown and publisher worker,
// with the existing terminal transport and external-store fixture substituted.
class NormalEnginePublisherShutdownTest: public NormalEngineTerminalStopTest {
public:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<pybind11::scoped_interpreter>();
        pybind11::module_::import("torch");
        release_ = std::make_unique<pybind11::gil_scoped_release>();
    }
    static void TearDownTestSuite() {
        release_.reset();
        interpreter_.reset();
    }
    void SetUp() override {
        NormalEngineTerminalStopTest::SetUp();
        pybind11::gil_scoped_acquire gil;
        pybind11::exec(R"PY(
import os
import runpy
import types
import weakref
from pathlib import Path
from threading import Event
from unittest.mock import patch

# Reuse the existing store contract; do not initialize its extension/model fixture.
fixture_path = (Path(os.environ['TEST_SRCDIR']) / os.environ['TEST_WORKSPACE'] /
                'rtp_llm/cpp/models/test/pywrapped_model_cache_store_integration_test.py')
fixture = runpy.run_path(str(fixture_path), run_name='publisher_shutdown_fixture')
MockStore, MockConfig = fixture['MockStore'], fixture['MockConfig']
put_entered, allow_put = Event(), Event()
flush_entered, allow_flush = Event(), Event()
shutdown_events, gate_timeouts, payload_refs = [], [], []
retain_sources = False

class ShutdownStore(MockStore):
    def put_batch(self, **payload):
        shutdown_events.append('put_enter')
        put_entered.set()
        if not allow_put.wait(15):
            gate_timeouts.append('put')
            raise RuntimeError('shutdown test did not release PUT')
        metadata = super().put_batch(**payload)
        payload_refs.extend(weakref.ref(tensor) for tensor in payload['hidden_states'])
        shutdown_events.append('put_done')
        return metadata
    def flush(self):
        shutdown_events.append('flush_enter')
        flush_entered.set()
        if not allow_flush.wait(15):
            gate_timeouts.append('flush')
            raise RuntimeError('shutdown test did not release flush')
        super().flush()
        shutdown_events.append('flush_done')
    def query_keys(self, keys):
        if retain_sources and 'put_done' in shutdown_events:
            return {key: not key.endswith('_abandoned') for key in keys}
        return super().query_keys(keys)
    def close(self):
        shutdown_events.append('close')
        super().close()

mooncake = types.ModuleType('torchspec.transfer.mooncake')
mooncake.MooncakeConfig, mooncake.EagleMooncakeStore = MockConfig, ShutdownStore
module_patch = patch.dict('sys.modules', {
    'torchspec': types.ModuleType('torchspec'),
    'torchspec.transfer': types.ModuleType('torchspec.transfer'),
    'torchspec.transfer.mooncake': mooncake,
})
module_patch.start()
)PY");
    }
    void TearDown() override {
        // Release gates before any future/join or engine destruction, including
        // failed assertions. A timed-out gate is a test failure, never a pass.
        exec("allow_put.set(); allow_flush.set()");
        if (stop_future_.valid()) {
            EXPECT_NO_THROW(stop_future_.get());
        }
        NormalEngineTerminalStopTest::TearDown();
        pybind11::gil_scoped_acquire gil;
        op_.reset();
        pybind11::exec("module_patch.stop()");
    }

    void runShutdown(bool retain_sources) {
        class PublisherExecutor: public TerminalExecutor {
        public:
            PublisherExecutor(TerminalTransport&                           transport,
                              std::unique_ptr<HiddenStateCapturePublisher> publisher,
                              std::shared_ptr<std::promise<void>>          destruction_started):
                TerminalExecutor(transport, 0),
                publisher_(std::move(publisher)),
                destruction_started_(std::move(destruction_started)) {}
            ~PublisherExecutor() override {
                destruction_started_->set_value();
                // The real publisher destructor joins/drains or quarantines.
                publisher_.reset();
            }

        private:
            std::unique_ptr<HiddenStateCapturePublisher> publisher_;
            std::shared_ptr<std::promise<void>>          destruction_started_;
        };
        exec(retain_sources ? "retain_sources = True" : "retain_sources = False");
        auto  publisher          = std::make_unique<HiddenStateCapturePublisher>(2,
                                                                       2,
                                                                       HiddenStateCaptureDtype::BF16,
                                                                       torch::kFloat16,
                                                                       true,
                                                                       0,
                                                                       nullptr,
                                                                       HiddenStateCaptureLimits{4, 1000, 100, 2, 5},
                                                                       std::vector<int64_t>{1, 0});
        auto* observed_publisher = publisher.get();
        ASSERT_TRUE(publisher->flushPending());
        ASSERT_TRUE(publisher->shouldPublish());
        // Only weak Python references remain: fixture-owned instances must not
        // manufacture the store retention we assert after full destruction.
        exec("assert len(MockStore.instances) == 1; store_ref = weakref.ref(MockStore.instances.pop())");
        auto destruction_started = std::make_shared<std::promise<void>>();
        auto destroying          = destruction_started->get_future();
        auto executor = std::make_shared<PublisherExecutor>(transport_, std::move(publisher), destruction_started);
        executors_[0] = executor.get();
        engines_[0]->executor_ = std::move(executor);
        {
            pybind11::gil_scoped_acquire gil;
            op_                     = std::make_unique<RtpLLMOp>();
            auto server             = std::make_shared<LocalRpcServer>();
            server->engine_         = engines_[0];
            auto service            = std::make_unique<LocalRpcServiceImpl>();
            service->local_server_  = std::move(server);
            op_->model_rpc_service_ = std::move(service);
        }
        auto admission = observed_publisher->tryReserve(42, 2);
        ASSERT_TRUE(admission.ok()) << admission.status().ToString();
        ASSERT_NE(admission.value(), nullptr);
        auto packed =
            torch::arange(12, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)).reshape({2, 6});
        ASSERT_TRUE(observed_publisher->trySubmit(packed,
                                                  torch::tensor({1, 2}, torch::kInt32),
                                                  torch::tensor({2}, torch::kInt32),
                                                  torch::tensor({42}, torch::kInt64),
                                                  2,
                                                  "engine shutdown"));
        packed = torch::Tensor();
        ASSERT_TRUE(eval("put_entered.wait(3)"));
        admission.value()->cancel();
        EXPECT_EQ(observed_publisher->quotaSnapshot().samples, 1);

        transport_.release_on_stop = false;
        start();
        auto root     = arm(0);
        auto follower = arm(1);
        ASSERT_EQ(root.wait_for(3s), std::future_status::ready);
        ASSERT_TRUE(root.get().ok());
        ASSERT_EQ(follower.wait_for(3s), std::future_status::ready);
        ASSERT_TRUE(follower.get().ok());
        EXPECT_TRUE(eval("shutdown_events == ['put_enter']"));
        {
            pybind11::gil_scoped_acquire gil;
            op_->prepareStop(true, 0);
            op_->prepareStop(true, 0);
            EXPECT_TRUE(op_->is_engine_stopped_);
            EXPECT_NE(op_->model_rpc_service_, nullptr);
        }
        ASSERT_TRUE(engines_[1]->stopAtStep(0).ok());
        EXPECT_EQ(transport_.terminals, 1);
        EXPECT_EQ(transport_.stop_notifications, 0);
        EXPECT_TRUE(eval("shutdown_events == ['put_enter'] and not store_ref().closed"));
        EXPECT_EQ(observed_publisher->quotaSnapshot().samples, 1);

        const std::weak_ptr<NormalEngine> engine = engines_[0];
        engines_[0].reset();
        executors_[0] = nullptr;
        stop_future_  = std::async(std::launch::async, [this] {
            pybind11::gil_scoped_acquire gil;
            op_->stop();
        });
        ASSERT_EQ(destroying.wait_for(3s), std::future_status::ready);
        EXPECT_EQ(stop_future_.wait_for(50ms), std::future_status::timeout);
        EXPECT_TRUE(eval("shutdown_events == ['put_enter'] and not store_ref().closed"));
        exec("allow_put.set()");
        ASSERT_TRUE(eval("flush_entered.wait(3)"));
        EXPECT_EQ(stop_future_.wait_for(50ms), std::future_status::timeout);
        EXPECT_TRUE(eval("'put_done' in shutdown_events and 'close' not in shutdown_events"));
        exec("allow_flush.set()");
        ASSERT_EQ(stop_future_.wait_for(3s), std::future_status::ready);
        EXPECT_NO_THROW(stop_future_.get());
        EXPECT_TRUE(engine.expired());
        EXPECT_TRUE(eval("not gate_timeouts and 'flush_done' in shutdown_events"));
        if (retain_sources) {
            EXPECT_TRUE(eval("'close' not in shutdown_events and store_ref() is not None and not store_ref().closed"));
            EXPECT_TRUE(eval("bool(payload_refs) and all(ref() is not None for ref in payload_refs)"));
            EXPECT_TRUE(eval("len(store_ref().batches) == 1"));
        } else {
            EXPECT_TRUE(eval(
                "shutdown_events.count('close') == 1 and shutdown_events.index('flush_done') < shutdown_events.index('close')"));
            EXPECT_TRUE(eval("store_ref() is None and all(ref() is None for ref in payload_refs)"));
        }
        {
            pybind11::gil_scoped_acquire gil;
            op_->stop();
            op_.reset();
        }
        EXPECT_TRUE(eval(retain_sources ? "'close' not in shutdown_events" : "shutdown_events.count('close') == 1"));
        EXPECT_NO_THROW(admission.value()->cancel());
    }

private:
    static void exec(const char* code) {
        pybind11::gil_scoped_acquire gil;
        pybind11::exec(code);
    }
    static bool eval(const char* expression) {
        pybind11::gil_scoped_acquire gil;
        return pybind11::eval(expression).cast<bool>();
    }
    inline static std::unique_ptr<pybind11::scoped_interpreter> interpreter_;
    inline static std::unique_ptr<pybind11::gil_scoped_release> release_;
    std::unique_ptr<RtpLLMOp>                                   op_;
    std::future<void>                                           stop_future_;
};

TEST_F(NormalEnginePublisherShutdownTest, FullStopDrainsInflightPublisherBeforeClosingStore) {
    runShutdown(false);
}

TEST_F(NormalEnginePublisherShutdownTest, FullStopRetainsStoreAndPayloadForUnconsumedSources) {
    runShutdown(true);
}
#endif

TEST_F(NormalEngineTerminalStopTest, IdleFifoStopsWithOneTerminalAndNoIdleBudget) {
    start();
    EXPECT_EQ(engines_[0]->completedSteps(), EngineBase::kTerminalStopCapability);
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    EXPECT_TRUE(engines_[0]->stopAtStep(0).ok());
}

TEST_F(NormalEngineTerminalStopTest, TerminalCommitRetainsCaptureOwnerUntilExecutorDestruction) {
    // Probe only the engine/executor ownership boundary. Store completion and
    // GIL-aware publisher draining remain the publisher fixture's responsibility.
    struct CaptureOwner {
        explicit CaptureOwner(std::shared_ptr<int> closes): closes(std::move(closes)) {}
        ~CaptureOwner() {
            ++*closes;
        }
        std::shared_ptr<int> closes;
        std::shared_ptr<int> submitted_source = std::make_shared<int>(1);
        int                  reserved         = 1;
    };
    class CaptureTerminalExecutor: public TerminalExecutor {
    public:
        CaptureTerminalExecutor(TerminalTransport& transport, int rank, std::shared_ptr<CaptureOwner> owner):
            TerminalExecutor(transport, rank), owner_(std::move(owner)) {}

    private:
        std::shared_ptr<CaptureOwner> owner_;
    };
    std::array<std::shared_ptr<int>, 2>             closes{std::make_shared<int>(0), std::make_shared<int>(0)};
    std::array<std::weak_ptr<CaptureOwner>, 2>      owners;
    std::array<std::weak_ptr<int>, 2>               sources;
    std::array<HiddenStateCaptureReservationPtr, 2> reservations;
    for (int rank = 0; rank < 2; ++rank) {
        auto owner                = std::make_shared<CaptureOwner>(closes[rank]);
        owners[rank]              = owner;
        sources[rank]             = owner->submitted_source;
        reservations[rank]        = std::make_shared<HiddenStateCaptureReservation>([weak = owners[rank]] {
            if (auto owner = weak.lock()) {
                --owner->reserved;
            }
        });
        auto executor             = std::make_shared<CaptureTerminalExecutor>(transport_, rank, std::move(owner));
        executors_[rank]          = executor.get();
        engines_[rank]->executor_ = std::move(executor);
    }
    transport_.release_on_stop = false;
    start();
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.stop_notifications, 0);
    for (int rank = 0; rank < 2; ++rank) {
        ASSERT_TRUE(engines_[rank]->stopAtStep(0).ok());
        ASSERT_TRUE(engines_[rank]->stop().ok());
        EXPECT_EQ(engines_[rank]->executor_.get(), executors_[rank]);
        EXPECT_FALSE(owners[rank].expired());
        EXPECT_FALSE(sources[rank].expired());
        EXPECT_EQ(*closes[rank], 0);
    }
    reservations[0]->cancel();
    reservations[0]->cancel();
    ASSERT_NE(owners[0].lock(), nullptr);
    EXPECT_EQ(owners[0].lock()->reserved, 0);
    EXPECT_FALSE(sources[0].expired());
    for (int rank = 0; rank < 2; ++rank) {
        engines_[rank].reset();
        executors_[rank] = nullptr;
        EXPECT_TRUE(owners[rank].expired());
        EXPECT_TRUE(sources[rank].expired());
        EXPECT_EQ(*closes[rank], 1);
        EXPECT_NO_THROW(reservations[rank]->cancel());
        EXPECT_FALSE(reservations[rank]->active());
        EXPECT_EQ(*closes[rank], 1);
    }
}

TEST_F(NormalEngineTerminalStopTest, DisabledCpIgnoresStaleSizeAndStillUsesTerminalHandshake) {
    for (auto& engine : engines_) {
        auto& cp            = engine->parallelism_config.prefill_cp_config;
        cp.method           = CPRotateMethod::DISABLED;
        cp.kv_cache_sharded = false;
        cp.prefill_cp_size  = 4;
        EXPECT_EQ(engine->completedSteps(), EngineBase::kTerminalStopCapability);
    }
    start();
    // A follower timeout must not turn a healthy pure-TP topology into a local stop.
    EXPECT_EQ(engines_[1]->armStopAtStep(0, 10).code(), absl::StatusCode::kDeadlineExceeded);
    EXPECT_TRUE(engines_[1]->running_);
    EXPECT_EQ(transport_.terminals, 0);
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
}

TEST_F(NormalEngineTerminalStopTest, ActiveUnshardedCpIgnoresStaleSizeAndStillUsesTerminalHandshake) {
    for (auto& engine : engines_) {
        auto& cp            = engine->parallelism_config.prefill_cp_config;
        cp.method           = CPRotateMethod::ALL_GATHER;
        cp.kv_cache_sharded = false;
        cp.prefill_cp_size  = 4;
        ASSERT_TRUE(cp.is_enabled());
        ASSERT_EQ(engine->parallelism_config.tp_size, 2);
        EXPECT_TRUE(engine->validateTerminalStop().ok());
        EXPECT_EQ(engine->completedSteps(), EngineBase::kTerminalStopCapability);
    }
    transport_.release_on_stop = false;
    start();
    // A missing root must time out, not fall back to a local stop.
    EXPECT_EQ(engines_[1]->armStopAtStep(0, 10).code(), absl::StatusCode::kDeadlineExceeded);
    EXPECT_TRUE(engines_[1]->running_);
    EXPECT_EQ(transport_.terminals, 0);
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    EXPECT_EQ(transport_.stop_notifications, 0);
    for (auto& engine : engines_) {
        EXPECT_FALSE(engine->running_);
        EXPECT_EQ(engine->loop_thread_, nullptr);
        EXPECT_TRUE(engine->stopAtStep(0).ok());
    }
}

TEST_F(NormalEngineTerminalStopTest, CpSizeValidationAppliesOnlyToSharding) {
    auto& engine = *engines_[0];
    auto& cp     = engine.parallelism_config.prefill_cp_config;
    for (auto method : {CPRotateMethod::DISABLED,
                        CPRotateMethod::ALL_GATHER,
                        CPRotateMethod::ALL_GATHER_WITH_OVERLAP,
                        CPRotateMethod::ALLTOALL,
                        CPRotateMethod::PREFILL_CP,
                        CPRotateMethod::UNKNOWN}) {
        for (bool sharded : {false, true}) {
            for (int size : {-1, 0, 1, 2, 4}) {
                SCOPED_TRACE(::testing::Message()
                             << "method=" << static_cast<int>(method) << " sharded=" << sharded << " size=" << size);
                cp.method           = method;
                cp.kv_cache_sharded = sharded;
                cp.prefill_cp_size  = size;
                // Ignoring an unused field here does not make it a valid sharded CP size.
                const bool valid    = !sharded || (size >= 0 && size <= 2);
                EXPECT_EQ(engine.validateTerminalStop().ok(), valid);
                EXPECT_EQ(engine.completedSteps(), EngineBase::kTerminalStopCapability);
                if (!valid) {
                    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
                    EXPECT_EQ(engine.stopAtStep(0).code(), absl::StatusCode::kFailedPrecondition);
                }
            }
        }
    }
    EXPECT_EQ(transport_.terminals, 0);
}

TEST_F(NormalEngineTerminalStopTest, PausedEnginesReachTerminalBoundaryOnArm) {
    for (auto& engine : engines_) {
        engine->pause();
    }
    start();
    std::this_thread::sleep_for(20ms);
    EXPECT_EQ(transport_.completed[0], 0);
    EXPECT_EQ(transport_.completed[1], 0);
    auto follower = arm(1);
    auto root     = arm(0);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
}

TEST_F(NormalEngineTerminalStopTest, FollowerTimeoutAndLateRootAbortReachCommonBoundaryBeforeCancellation) {
    transport_.release_on_stop = false;
    start();
    // Both loops are live: follower is in input sync and root is in idle FIFO.
    ASSERT_EQ(engines_[1]->armStopAtStep(0, 20).code(), absl::StatusCode::kDeadlineExceeded);
    EXPECT_EQ(transport_.terminals, 0);
    EXPECT_EQ(engines_[0]->cancelArmedStop(20).code(), absl::StatusCode::kFailedPrecondition);

    // Shared abort recovery re-arms the timed-out follower and the late root.
    // No notifyStop/force-stop can release the transport in this test.
    auto follower = arm(1);
    EXPECT_EQ(follower.wait_for(20ms), std::future_status::timeout);
    engines_[0]->pause();  // abort recovery must also release a paused late root
    auto root = arm(0);
    ASSERT_EQ(root.wait_for(3s), std::future_status::ready);
    ASSERT_TRUE(root.get().ok());
    ASSERT_EQ(follower.wait_for(3s), std::future_status::ready);
    ASSERT_TRUE(follower.get().ok());
    for (auto& engine : engines_) {
        ASSERT_TRUE(engine->cancelArmedStop(2000).ok());
    }
    EXPECT_EQ(transport_.stop_notifications, 0);
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    for (auto& engine : engines_) {
        EXPECT_TRUE(engine->forceStop().ok());
    }
    EXPECT_FALSE(transport_.shutdown);  // loop joins did not unblock a fake collective
}

TEST_F(NormalEngineTerminalStopTest, ArmWaitsForInflightProcessBeforeTerminalBroadcast) {
    transport_.hold_process = true;
    start();
    static_cast<FIFOScheduler&>(engines_[0]->getScheduler()).notifyControlWakeup();
    {
        std::unique_lock<std::mutex> lock(transport_.mutex);
        ASSERT_TRUE(transport_.changed.wait_for(
            lock, 2s, [&]() { return transport_.entered[0] == 1 && transport_.entered[1] == 1; }));
    }
    auto root     = arm(0);
    auto follower = arm(1);
    EXPECT_EQ(root.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(follower.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(transport_.terminals, 0);
    EXPECT_EQ(transport_.completed[0], 0);
    {
        std::lock_guard<std::mutex> lock(transport_.mutex);
        transport_.hold_process = false;
        transport_.changed.notify_all();
    }
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 2);
    EXPECT_EQ(transport_.completed[1], 2);
}

TEST_F(NormalEngineTerminalStopTest, LateFollowerArmAndCommitDoNotReenterProcess) {
    start();
    auto root = arm(0);
    expectReady(root);
    // Root's broadcast may beat this rank's control-plane arm.
    auto follower = arm(1);
    expectReady(follower);
    EXPECT_TRUE(engines_[0]->stopAtStep(0).ok());
    std::this_thread::sleep_for(20ms);
    EXPECT_EQ(transport_.completed[1], 1);
    EXPECT_TRUE(engines_[1]->armStopAtStep(0, 20).ok());
    EXPECT_TRUE(engines_[1]->stopAtStep(0).ok());
    EXPECT_EQ(transport_.terminals, 1);
}

TEST_F(NormalEngineTerminalStopTest, CancellationParksAtBoundaryUntilForceStop) {
    start();
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    for (auto& engine : engines_) {
        EXPECT_TRUE(engine->cancelArmedStop(2000).ok());
        EXPECT_FALSE(engine->stopAtStep(0).ok());
        EXPECT_FALSE(engine->armStopAtStep(0, 20).ok());
    }
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    for (auto& engine : engines_) {
        EXPECT_TRUE(engine->forceStop().ok());
        EXPECT_FALSE(engine->stopAtStep(0).ok());  // stale observation cannot authorize a new commit
    }
    EXPECT_EQ(transport_.terminals, 1);
}

TEST_F(NormalEngineTerminalStopTest, InvalidTopologyAndOtherModesRejectArmWithoutExecuting) {
    auto&      engine                                = *engines_[0];
    const auto pc                                    = engine.parallelism_config;
    const auto runtime                               = engine.runtime_config;
    engine.running_                                  = true;
    const std::vector<std::function<void()>> invalid = {
        [&]() { engine.parallelism_config.world_size = 4; },
        [&]() { engine.parallelism_config.pp_size = 2; },
        [&]() { engine.parallelism_config.dp_size = 2; },
        [&]() { engine.parallelism_config.tp_rank = 1; },
        [&]() { engine.parallelism_config.ep_size = 4; },
        [&]() { engine.parallelism_config.ffn_tp_size = 0; },
        [&]() {
            engine.parallelism_config.prefill_cp_config.method          = CPRotateMethod::ALL_GATHER;
            engine.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
            engine.parallelism_config.prefill_cp_config.prefill_cp_size = 4;
        },
        [&]() {
            engine.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
            engine.parallelism_config.prefill_cp_config.prefill_cp_size  = 4;
        },
        [&]() { engine.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = true; },
        [&]() { engine.runtime_config.use_batch_decode_scheduler = true; },
        [&]() { engine.runtime_config.use_gather_batch_scheduler = true; },
    };
    for (const auto& change : invalid) {
        engine.parallelism_config = pc;
        engine.runtime_config     = runtime;
        change();
        EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
        EXPECT_EQ(engine.stopAtStep(0).code(), absl::StatusCode::kFailedPrecondition);
    }
    engine.parallelism_config                              = pc;
    engine.runtime_config                                  = runtime;
    engine.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    EXPECT_FALSE(engine.armStopAtStep(0, 1).ok());
    engine.ffn_disaggregate_config.enable_ffn_disaggregate = false;
    class UnsupportedExecutor: public Executor {
    public:
        absl::Status process(const std::list<GenerateStreamPtr>&, int64_t = 0) override {
            return absl::InternalError("unsupported executor must not run");
        }
    };
    auto executor    = std::move(engine.executor_);
    engine.executor_ = std::make_unique<UnsupportedExecutor>();
    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
    engine.executor_ = std::move(executor);
    auto scheduler   = std::move(engine.scheduler_);
    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
    engine.scheduler_ = std::move(scheduler);
    EXPECT_EQ(engine.armStopAtStep(1, 1).code(), absl::StatusCode::kInvalidArgument);
    engine.running_ = false;
    EXPECT_FALSE(engine.armStopAtStep(0, 1).ok());  // loop not started
    EXPECT_EQ(transport_.terminals, 0);
    engine.propose_params_            = std::make_unique<ProposeModelEngineInitParams>();
    engine.published_completed_steps_ = 64;
    EXPECT_EQ(engine.completedSteps(), 64);  // MTP retains step snapshots.
    engine.propose_params_.reset();
}

TEST_F(NormalEngineTerminalStopTest, ProcessErrorsCannotAcknowledgeTerminalOrCommit) {
    auto& engine    = *engines_[0];
    engine.running_ = true;
    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kDeadlineExceeded);
    executors_[0]->error = absl::InternalError("injected process failure");
    EXPECT_EQ(engine.step().code(), absl::StatusCode::kInternal);
    EXPECT_FALSE(engine.stopAtStep(0).ok());
    EXPECT_EQ(engine.cancelArmedStop(1).code(), absl::StatusCode::kDeadlineExceeded);
    executors_[0]->error       = absl::OkStatus();
    executors_[0]->throw_error = true;
    static_cast<FIFOScheduler&>(engine.getScheduler()).notifyControlWakeup();
    EXPECT_THROW(engine.step(), std::runtime_error);
    EXPECT_EQ(transport_.terminals, 0);
    engine.requestForceStop();
    EXPECT_FALSE(engine.armStopAtStep(0, 1).ok());
    EXPECT_FALSE(engine.stopAtStep(0).ok());
}

}  // namespace
}  // namespace rtp_llm
