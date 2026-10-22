#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>

#include <c10/core/CPUAllocator.h>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace py = pybind11;

namespace rtp_llm {
namespace {

// Publisher-only CPU tests: real ledger, worker, preparation and shutdown, with
// only the external store mocked. No PyWrappedModel is required; the producer
// event regression additionally exercises CUDA when that backend/device is available.
class PublisherPythonEnvironment final: public ::testing::Environment {
public:
    void SetUp() override {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
        release_     = std::make_unique<py::gil_scoped_release>();
    }
    void TearDown() override {
        release_.reset();
        interpreter_.reset();
    }

private:
    std::unique_ptr<py::scoped_interpreter> interpreter_;
    std::unique_ptr<py::gil_scoped_release> release_;
};

[[maybe_unused]] auto* environment = ::testing::AddGlobalTestEnvironment(new PublisherPythonEnvironment());

// Fail the real request-metadata clone inside trySubmit, not a wrapper around
// the call. Only this thread's next allocation is affected; the idle publisher
// worker delegates to the original allocator. Restore before executing checks
// or submitting work that outlives this scope.
class SubmissionAllocationFailure final: public c10::Allocator {
public:
    explicit SubmissionAllocationFailure(std::exception_ptr error):
        original_(c10::GetCPUAllocator()), thread_(std::this_thread::get_id()), error_(std::move(error)) {
        c10::SetCPUAllocator(this);
    }
    ~SubmissionAllocationFailure() override {
        c10::SetCPUAllocator(original_);
    }
    c10::DataPtr allocate(size_t bytes) override {
        if (std::this_thread::get_id() == thread_ && !triggered_) {
            triggered_ = true;
            std::rethrow_exception(error_);
        }
        return original_->allocate(bytes);
    }
    void copy_data(void* destination, const void* source, size_t bytes) const override {
        original_->copy_data(destination, source, bytes);
    }
    bool triggered() const {
        return triggered_;
    }

private:
    c10::Allocator*    original_;
    std::thread::id    thread_;
    std::exception_ptr error_;
    bool               triggered_{false};
};

// Inject into the actual worker's first tensor allocation, while the submitting
// thread continues to use the real allocator. flushPending() completes before
// restoration so no worker can retain this allocator object.
class WorkerAllocationFailure final: public c10::Allocator {
public:
    explicit WorkerAllocationFailure(std::exception_ptr error):
        original_(c10::GetCPUAllocator()), submitting_thread_(std::this_thread::get_id()), error_(std::move(error)) {
        c10::SetCPUAllocator(this);
    }
    ~WorkerAllocationFailure() override {
        c10::SetCPUAllocator(original_);
    }
    c10::DataPtr allocate(size_t bytes) override {
        if (std::this_thread::get_id() != submitting_thread_ && !triggered_.exchange(true)) {
            std::rethrow_exception(error_);
        }
        return original_->allocate(bytes);
    }
    void copy_data(void* destination, const void* source, size_t bytes) const override {
        original_->copy_data(destination, source, bytes);
    }
    bool triggered() const {
        return triggered_.load();
    }

private:
    c10::Allocator*    original_;
    std::thread::id    submitting_thread_;
    std::exception_ptr error_;
    std::atomic<bool>  triggered_{false};
};

class HiddenStateCapturePublisherTest: public ::testing::Test {
protected:
    void SetUp() override {
        py::gil_scoped_acquire gil;
        py::exec(R"PY(
import sys
import types
import threading
import torch

store_keys = set()
put_count = 0
flush_count = 0
close_count = 0
clear_on_flush = False
accepted_ids = []
accepted_hidden = []
accepted_capture_metadata = []
fail_init = False
fail_put = False
flush_error = None
async_error_mode = ''
close_error = False
query_mode = ''
put_started = threading.Event()
allow_put = threading.Event()
allow_put.set()

class Config:
    hidden_dim = 2
    num_aux_layers = 2
    store_key_namespace = 'PublisherTest'
    @classmethod
    def from_env(cls):
        return cls()
    def make_store_key(self, request_id):
        return f'rtp_PublisherTest_{request_id}'

class Store:
    def __init__(self, config):
        pass
    def setup(self, device):
        if fail_init:
            raise RuntimeError('injected store setup failure')
    def warmup_rdma(self):
        pass
    def put_batch(self, **payload):
        global put_count
        put_count += 1
        put_started.set()
        if not allow_put.wait(timeout=10):
            raise RuntimeError('test did not unblock PUT')
        accepted_ids.extend(tensor.tolist() for tensor in payload['input_ids'])
        accepted_hidden.extend(tensor.float().tolist() for tensor in payload['hidden_states'])
        accepted_capture_metadata.append(payload['capture_metadata'])
        for key in payload['request_ids']:
            store_keys.update([key + '_hs', key + '_ids', key + '_lhs', key + '_ready'])
            if 'hidden_states_scale' in payload:
                store_keys.add(key + '_hss')
            if fail_put:
                raise RuntimeError('injected partial PUT with visible ready')
        return [{} for _ in payload['request_ids']]
    def flush(self):
        global flush_count
        flush_count += 1
        if flush_error is not None:
            raise flush_error
        if clear_on_flush:
            store_keys.clear()
    def take_async_errors(self):
        if async_error_mode == 'malformed':
            return 0
        if async_error_mode == 'exception':
            raise RuntimeError('injected take_async_errors failure')
        return []
    def query_keys(self, keys):
        if query_mode == 'error':
            raise RuntimeError('injected query failure')
        if query_mode == 'incomplete':
            return {}
        if query_mode == 'integer':
            return {key: 0 for key in keys}
        return {key: key in store_keys for key in keys}
    def close(self):
        global close_count
        close_count += 1
        if close_error:
            raise RuntimeError('injected close failure')

module = types.ModuleType('torchspec.transfer.mooncake')
module.MooncakeConfig = Config
module.EagleMooncakeStore = Store
for name in ('torchspec', 'torchspec.transfer', 'torchspec.utils'):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules[module.__name__] = module
fp8 = types.ModuleType('torchspec.utils.fp8')
fp8.quantize_aux_hidden_states = lambda *args: (_ for _ in ()).throw(RuntimeError('injected quantizer failure'))
sys.modules[fp8.__name__] = fp8
)PY");
    }

    void TearDown() override {
        {
            py::gil_scoped_acquire gil;
            py::exec(
                "allow_put.set(); store_keys.clear(); query_mode = ''; async_error_mode = ''; close_error = False");
        }
        for (auto& publisher : publishers_) {
            publisher->flushAndClose();
        }
        publishers_.clear();
    }

    HiddenStateCapturePublisher& publisher(HiddenStateCaptureLimits limits    = {},
                                           bool                     owner     = true,
                                           bool                     fp8       = false,
                                           std::vector<int64_t>     layer_ids = {1, 0}) {
        publishers_.push_back(std::make_unique<HiddenStateCapturePublisher>(2,
                                                                            2,
                                                                            fp8 ? HiddenStateCaptureDtype::FP8_E4M3 :
                                                                                  HiddenStateCaptureDtype::BF16,
                                                                            torch::kFloat16,
                                                                            owner,
                                                                            0,
                                                                            nullptr,
                                                                            limits,
                                                                            std::move(layer_ids)));
        publishers_.back()->flushPending();
        return *publishers_.back();
    }

    bool submit(HiddenStateCapturePublisher& publisher, int64_t id, int32_t tokens = 2) {
        auto packed = torch::arange(tokens * 6, torch::kFloat16).reshape({tokens, 6});
        return publisher.trySubmit(packed,
                                   torch::arange(tokens, torch::kInt32),
                                   torch::tensor({tokens}, torch::kInt32),
                                   torch::tensor({id}, torch::kInt64),
                                   tokens,
                                   "publisher test");
    }

    template<typename Predicate>
    bool waitFor(Predicate predicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!predicate()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    }

    int64_t evalInt(const char* expression) {
        py::gil_scoped_acquire gil;
        return py::eval(expression).cast<int64_t>();
    }

    std::vector<std::unique_ptr<HiddenStateCapturePublisher>> publishers_;
};

TEST_F(HiddenStateCapturePublisherTest, PayloadLimitsAndIdempotentCancellation) {
    auto& p = publisher({2, 100, 5, 1, 5});
    EXPECT_FALSE(p.tryReserve(-1, 1));
    EXPECT_FALSE(p.tryReserve(1, 0));
    EXPECT_FALSE(p.tryReserve(1, std::numeric_limits<int64_t>::max()));
    ASSERT_TRUE(p.tryReserve(1, 2));
    EXPECT_FALSE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.tryReserve(2, 3));
    auto quota = p.quotaSnapshot();
    EXPECT_EQ(quota.samples, 2);
    EXPECT_EQ(quota.bytes, 100);
    EXPECT_EQ(quota.tokens, 5);
    EXPECT_FALSE(p.tryReserve(3, 1));
    p.cancelReserved(1);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 60);
    EXPECT_EQ(p.quotaSnapshot().tokens, 3);
    EXPECT_TRUE(p.tryReserve(3, 2));
    EXPECT_FALSE(p.tryReserve(4, 1));
}

TEST_F(HiddenStateCapturePublisherTest, EachLimitIndependentlyRejectsWithoutChangingLedger) {
    for (const auto limits : {HiddenStateCaptureLimits{1, 1000, 100, 1, 5},
                              HiddenStateCaptureLimits{10, 40, 100, 1, 5},
                              HiddenStateCaptureLimits{10, 1000, 2, 1, 5}}) {
        auto& p = publisher(limits);
        ASSERT_TRUE(p.tryReserve(1, 2));
        EXPECT_FALSE(p.tryReserve(2, 1));
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
        EXPECT_EQ(p.quotaSnapshot().bytes, 40);
        EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    }
}

TEST_F(HiddenStateCapturePublisherTest, Fp8IncludesScaleAndBf16FinalInBudget) {
    auto& p = publisher({1, 48, 2, 1, 5}, true, true);
    ASSERT_TRUE(p.tryReserve(1, 2));
    EXPECT_EQ(p.quotaSnapshot().bytes, 48);
    EXPECT_FALSE(p.tryReserve(2, 1));
}

TEST_F(HiddenStateCapturePublisherTest, OwnerOnlyAndInvalidLimitsDisableCapture) {
    auto& non_owner = publisher({}, false);
    EXPECT_FALSE(non_owner.tryReserve(1, 2));
    EXPECT_EQ(non_owner.quotaSnapshot().samples, 0);
    for (int index = 0; index < 5; ++index) {
        HiddenStateCaptureLimits limits;
        switch (index) {
            case 0:
                limits.max_inflight_samples = 0;
                break;
            case 1:
                limits.max_inflight_bytes = -1;
                break;
            case 2:
                limits.max_inflight_tokens = 0;
                break;
            case 3:
                limits.max_pending_batches = 0;
                break;
            case 4:
                limits.poll_interval_ms = 0;
                break;
        }
        auto& invalid = publisher(limits);
        EXPECT_FALSE(invalid.shouldPublish());
        EXPECT_FALSE(invalid.tryReserve(1, 1));
        EXPECT_FALSE(invalid.takeDeferredError().has_value());
    }
}

TEST_F(HiddenStateCapturePublisherTest, ConcurrentReservationsNeverOversubscribe) {
    auto&                    p = publisher({4, 80, 4, 2, 5});
    std::atomic<int>         accepted{0};
    std::vector<std::thread> threads;
    for (int id = 0; id < 16; ++id) {
        threads.emplace_back([&p, &accepted, id] { accepted += p.tryReserve(id, 1); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(accepted.load(), 4);
    EXPECT_EQ(p.quotaSnapshot().bytes, 80);
    EXPECT_EQ(p.quotaSnapshot().tokens, 4);
}

TEST_F(HiddenStateCapturePublisherTest, SubmittedAndPublishedQuotaWaitsForEverySourceKey) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    ASSERT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    p.cancelReserved(1);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.discard('rtp_PublisherTest_1_ready')");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.clear()");
    }
    ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 0);
    EXPECT_EQ(p.quotaSnapshot().tokens, 0);
}

TEST_F(HiddenStateCapturePublisherTest, WorkerOwnsPayloadAndSnapshotsMutableHostMetadata) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.tryReserve(2, 3));
    {
        // Holding the GIL deterministically delays worker preparation. Neither
        // strong ownership nor CPU metadata snapshots may depend on its timing.
        py::gil_scoped_acquire gil;
        auto                   packed   = torch::arange(30, torch::kFloat16).reshape({5, 6});
        auto                   ids      = torch::arange(5, torch::kInt32);
        auto                   lengths  = torch::tensor({2, 3}, torch::kInt32);
        auto                   requests = torch::tensor({1, 2}, torch::kInt64);
        ASSERT_TRUE(p.trySubmit(packed, ids, lengths, requests, 5, "owned payload"));
        ids.fill_(99);
        lengths.fill_(99);
        requests.fill_(99);
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 2);
    EXPECT_EQ(p.quotaSnapshot().tokens, 5);
    EXPECT_EQ(p.quotaSnapshot().bytes, 100);
    EXPECT_EQ(evalInt("accepted_ids == [[0, 1], [2, 3, 4]]"), 1);
    EXPECT_EQ(evalInt("accepted_hidden[0] == [[0, 1, 2, 3], [6, 7, 8, 9]]"), 1);
}

#if USING_CUDA
TEST_F(HiddenStateCapturePublisherTest, ProducerEventProtectsNonDefaultStreamPayloadAfterServingCancellation) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "The producer-event regression requires a CUDA device";
        }
    }
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    {
        py::gil_scoped_acquire     gil;
        c10::cuda::CUDAGuard       device_guard(0);
        auto                       stream = c10::cuda::getStreamFromPool(false, 0);
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        auto                       options = torch::TensorOptions().device(torch::kCUDA, 0);
        auto                       packed  = torch::arange(12, options.dtype(torch::kFloat16)).reshape({2, 6});
        // Delay the producer before its final write. The publisher uses a
        // different stream and must honor the captured producer event.
        py::module_::import("torch").attr("cuda").attr("_sleep")(100000000);
        packed.add_(1);
        auto ids = torch::arange(2, options.dtype(torch::kInt32));
        ASSERT_TRUE(p.trySubmit(packed,
                                ids,
                                torch::tensor({2}, torch::kInt32),
                                torch::tensor({1}, torch::kInt64),
                                2,
                                "non-default producer stream"));
        p.cancelReserved(1);
        // Local handles disappear while device writes/copies can still be in flight.
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("accepted_ids == [[0, 1]]"), 1);
    EXPECT_EQ(evalInt("accepted_hidden == [[[1, 2, 3, 4], [7, 8, 9, 10]]]"), 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
}
#endif

TEST_F(HiddenStateCapturePublisherTest, WorkerPassesExactLayerOrderWithoutInventingLayoutMetadata) {
    for (const bool fp8 : {false, true}) {
        {
            py::gil_scoped_acquire gil;
            py::exec("fp8.quantize_aux_hidden_states = lambda hidden, layers: "
                     "(hidden.to(torch.float8_e4m3fn), torch.ones((hidden.shape[0], layers), dtype=torch.float32))");
        }
        auto&         p          = publisher({}, true, fp8, {7, 2});
        const int64_t request_id = fp8 ? 2 : 1;
        ASSERT_TRUE(p.tryReserve(request_id, 2));
        const auto previous_puts = evalInt("put_count");
        ASSERT_TRUE(submit(p, request_id));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), previous_puts + 1);
        EXPECT_EQ(evalInt("accepted_capture_metadata[-1] == {'capture_layer_ids': [7, 2]}"), 1);
        EXPECT_EQ(evalInt("set(accepted_capture_metadata[-1]) == {'capture_layer_ids'}"), 1);
        EXPECT_EQ(evalInt("any(key.endswith('_metadata') for key in store_keys)"), 0);
        EXPECT_EQ(evalInt("any(key.endswith('_hss') for key in store_keys)"), static_cast<int>(fp8));
    }
}

TEST_F(HiddenStateCapturePublisherTest, WorkerRejectsMissingOrMismatchedLayerOrderBeforePut) {
    for (const auto& ids : {std::vector<int64_t>{}, std::vector<int64_t>{7}, std::vector<int64_t>{7, 2, 1}}) {
        auto& p = publisher({}, true, false, ids);
        ASSERT_TRUE(p.tryReserve(1, 2));
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), 0);
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_FALSE(p.shouldPublish());
        EXPECT_FALSE(p.takeDeferredError().has_value());
    }
}

TEST_F(HiddenStateCapturePublisherTest, ServingCancellationAfterSubmitDoesNotSuppressPublication) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    {
        // The worker cannot enter Python until this scope exits. submit() also
        // drops its caller-owned tensors before serving cancellation occurs.
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(submit(p, 1));
        p.cancelReserved(1);
        p.cancelReserved(1);
        const auto quota = p.quotaSnapshot();
        EXPECT_EQ(quota.samples, 1);
        EXPECT_EQ(quota.bytes, 40);
        EXPECT_EQ(quota.tokens, 2);
        EXPECT_EQ(quota.reserved_samples, 0);
        EXPECT_EQ(quota.cleanup_pending_samples, 0);
        EXPECT_EQ(evalInt("put_count"), 0);
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("flush_count"), 1);
    EXPECT_EQ(evalInt("accepted_ids == [[0, 1]]"), 1);
    EXPECT_EQ(evalInt("accepted_hidden == [[[0, 1, 2, 3], [6, 7, 8, 9]]]"), 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
    EXPECT_FALSE(p.takeDeferredError().has_value());
}

TEST_F(HiddenStateCapturePublisherTest, ServingCancellationOfOneSubmittedRequestDoesNotDropBatch) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.tryReserve(2, 3));
    {
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(p.trySubmit(torch::arange(30, torch::kFloat16).reshape({5, 6}),
                                torch::arange(5, torch::kInt32),
                                torch::tensor({2, 3}, torch::kInt32),
                                torch::tensor({1, 2}, torch::kInt64),
                                5,
                                "serving cancellation"));
        p.cancelReserved(1);
        EXPECT_EQ(p.quotaSnapshot().samples, 2);
        EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("accepted_ids == [[0, 1], [2, 3, 4]]"), 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 2);
    EXPECT_EQ(p.quotaSnapshot().bytes, 100);
    EXPECT_EQ(p.quotaSnapshot().tokens, 5);
}

TEST_F(HiddenStateCapturePublisherTest, MalformedOrExceptionFromTakeAsyncErrorsIsRecoverable) {
    for (const auto mode : {"malformed", "exception"}) {
        SCOPED_TRACE(mode);
        const int64_t first_id      = mode[0] == 'm' ? 1 : 3;
        const auto    previous_puts = evalInt("put_count");
        auto&         p             = publisher({4, 1000, 100, 2, 5});
        {
            py::gil_scoped_acquire gil;
            py::globals()["async_error_mode"] = mode;
        }
        ASSERT_TRUE(p.tryReserve(first_id, 2));
        ASSERT_TRUE(submit(p, first_id));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), previous_puts);
        EXPECT_EQ(p.stats().failure_count, 1);
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_TRUE(p.shouldPublish());
        EXPECT_FALSE(p.takeDeferredError().has_value());

        {
            py::gil_scoped_acquire gil;
            py::exec("async_error_mode = ''");
        }
        ASSERT_TRUE(p.tryReserve(first_id + 1, 2));
        ASSERT_TRUE(submit(p, first_id + 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), previous_puts + 1);
        EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
        EXPECT_FALSE(p.takeDeferredError().has_value());
    }
}

TEST_F(HiddenStateCapturePublisherTest, CloseExceptionDoesNotEscapeOrBreakWorkerJoin) {
    auto& p = publisher();
    {
        py::gil_scoped_acquire gil;
        py::exec("close_error = True");
    }
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_EQ(evalInt("close_count"), 1);
    EXPECT_GE(p.stats().failure_count, 1);
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_FALSE(p.shouldPublish());
    EXPECT_FALSE(p.takeDeferredError().has_value());
}

TEST_F(HiddenStateCapturePublisherTest, IdleWorkerCanBeJoinedAndCloseIsIdempotent) {
    auto& p = publisher();
    ASSERT_TRUE(p.flushPending());
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_EQ(evalInt("close_count"), 1);
    EXPECT_FALSE(p.tryReserve(1, 2));
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
}

TEST_F(HiddenStateCapturePublisherTest, CloseWithGilDrainsAcceptedTaskBeforeReleasingOwners) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    py::gil_scoped_acquire gil;
    py::exec("clear_on_flush = True");
    ASSERT_TRUE(submit(p, 1));
    p.cancelReserved(1);
    // close must release our GIL so the delayed worker can publish, flush,
    // confirm absence, and close Python resources before the join completes.
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("flush_count"), 1);
    EXPECT_EQ(evalInt("close_count"), 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, QueryFailureOrMalformedAnswerIsNotAbsence) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    for (const auto mode : {"error", "incomplete", "integer"}) {
        {
            py::gil_scoped_acquire gil;
            py::globals()["query_mode"] = mode;
            py::exec("store_keys.clear()");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
    }
    {
        py::gil_scoped_acquire gil;
        py::exec("query_mode = ''");
    }
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
}

TEST_F(HiddenStateCapturePublisherTest, SlowPutDoesNotBlockSubmitAndQueueFullDoesNotRefundSubmittedWork) {
    auto& p = publisher({4, 1000, 100, 1, 5});
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.clear()");
    }
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.tryReserve(2, 2));
    ASSERT_TRUE(p.tryReserve(3, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(waitFor([&] { return evalInt("put_started.is_set()") != 0; }));
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 3);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
    ASSERT_TRUE(submit(p, 2));
    EXPECT_FALSE(submit(p, 3));
    EXPECT_EQ(evalInt("flush_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    p.cancelReserved(3);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    // Even dropped exports retain their source until the local fence completes.
    EXPECT_EQ(p.quotaSnapshot().samples, 3);
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.set()");
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 2);
    EXPECT_EQ(p.quotaSnapshot().samples, 2);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 2);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, PartialPutNeverForceDeletesVisibleSourceOrReturnsServingError) {
    auto& p = publisher({1, 1000, 100, 1, 5});
    {
        py::gil_scoped_acquire gil;
        py::exec("fail_put = True");
    }
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_GT(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    p.cancelReserved(1);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    EXPECT_FALSE(p.tryReserve(2, 2));
    EXPECT_FALSE(p.takeDeferredError().has_value());
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.clear()");
    }
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
}

TEST_F(HiddenStateCapturePublisherTest, QuantizerFailureWithoutSourceObjectsRefundsAfterLocalCompletion) {
    auto& p = publisher({1, 1000, 100, 1, 5}, true, true);
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
    EXPECT_FALSE(p.takeDeferredError().has_value());
    EXPECT_TRUE(p.tryReserve(2, 2));
}

TEST_F(HiddenStateCapturePublisherTest, FatalQuantizerErrorsStopAdmissionAndSkipAlreadyQueuedCapture) {
    struct FailureCase {
        const char*                 expression;
        HiddenStateCaptureFatalKind kind;
    };
    const std::vector<FailureCase> cases{
        {"torch.OutOfMemoryError('injected quantizer allocation failure')", HiddenStateCaptureFatalKind::OUT_OF_MEMORY},
        {"torch.AcceleratorError('injected quantizer device failure')", HiddenStateCaptureFatalKind::ACCELERATOR},
        {"MemoryError('injected quantizer host allocation failure')", HiddenStateCaptureFatalKind::ALLOCATION},
        {"RuntimeError('CUDA error: an illegal memory access was encountered')",
         HiddenStateCaptureFatalKind::ACCELERATOR},
        {"RuntimeError('HIP out of memory. injected failure')", HiddenStateCaptureFatalKind::OUT_OF_MEMORY}};
    for (const auto& item : cases) {
        SCOPED_TRACE(item.expression);
        {
            py::gil_scoped_acquire gil;
            py::globals()["quantizer_error"] = py::eval(item.expression);
            py::exec(R"PY(
quantizer_calls = 0
def injected_quantizer(hidden, layers):
    global quantizer_calls
    quantizer_calls += 1
    raise quantizer_error
fp8.quantize_aux_hidden_states = injected_quantizer
)PY");
        }
        auto& p = publisher({4, 1000, 100, 2, 5}, true, true);
        ASSERT_TRUE(p.tryReserve(1, 2));
        ASSERT_TRUE(p.tryReserve(2, 2));
        {
            // Delay worker Python execution until both tasks are submitted.
            py::gil_scoped_acquire gil;
            ASSERT_TRUE(submit(p, 1));
            ASSERT_TRUE(submit(p, 2));
        }
        ASSERT_TRUE(p.flushPending());
        const auto fatal = p.fatalState();
        EXPECT_EQ(fatal.kind, item.kind);
        EXPECT_EQ(fatal.stage, HiddenStateCaptureFailureStage::QUANTIZE);
        EXPECT_FALSE(fatal.message.empty());
        EXPECT_EQ(p.stats().fatal_kind, item.kind);
        EXPECT_EQ(evalInt("quantizer_calls"), 1);
        EXPECT_EQ(evalInt("put_count"), 0);
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_FALSE(p.shouldPublish());
        EXPECT_FALSE(p.tryReserve(3, 2));
        EXPECT_FALSE(p.takeDeferredError().has_value());
        EXPECT_FALSE(p.takeDeferredError().has_value());
        EXPECT_EQ(p.fatalState().message, fatal.message);
        p.flushAndClose();
        EXPECT_EQ(p.fatalState().kind, fatal.kind);
        EXPECT_EQ(p.fatalState().stage, fatal.stage);
    }
}

TEST_F(HiddenStateCapturePublisherTest, RecoverableQuantizerErrorAllowsLaterHealthyPublication) {
    {
        py::gil_scoped_acquire gil;
        py::exec(R"PY(
quantizer_calls = 0
quantizer_error = RuntimeError('CUDA quantizer temporarily unavailable')
def recoverable_quantizer(hidden, layers):
    global quantizer_calls
    quantizer_calls += 1
    if quantizer_error is not None:
        raise quantizer_error
    return hidden.to(torch.float8_e4m3fn), torch.ones((hidden.shape[0], layers), dtype=torch.float32)
fp8.quantize_aux_hidden_states = recoverable_quantizer
)PY");
    }
    auto& p = publisher({2, 1000, 100, 1, 5}, true, true);
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_EQ(p.stats().failure_count, 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
    EXPECT_TRUE(p.shouldPublish());
    EXPECT_EQ(evalInt("put_count"), 0);
    {
        py::gil_scoped_acquire gil;
        py::exec("quantizer_error = None");
    }
    ASSERT_TRUE(p.tryReserve(2, 2));
    ASSERT_TRUE(submit(p, 2));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("quantizer_calls"), 2);
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_FALSE(p.takeDeferredError().has_value());
}

TEST_F(HiddenStateCapturePublisherTest, NativeWorkerAllocationFailuresLatchActualExceptionKinds) {
    const std::vector<std::pair<std::exception_ptr, HiddenStateCaptureFatalKind>> cases{
        {std::make_exception_ptr(c10::Error("injected worker tensor failure")), HiddenStateCaptureFatalKind::TENSOR},
        {std::make_exception_ptr(c10::OutOfMemoryError("injected worker allocation failure")),
         HiddenStateCaptureFatalKind::OUT_OF_MEMORY},
        {std::make_exception_ptr(
             c10::AcceleratorError(c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
                                   700,
                                   "injected worker accelerator failure")),
         HiddenStateCaptureFatalKind::ACCELERATOR},
        {std::make_exception_ptr(std::bad_alloc()), HiddenStateCaptureFatalKind::ALLOCATION}};
    for (const auto& item : cases) {
        auto& p = publisher();
        ASSERT_TRUE(p.tryReserve(1, 2));
        bool triggered = false;
        {
            py::gil_scoped_acquire  gil;
            WorkerAllocationFailure failure(item.first);
            ASSERT_TRUE(submit(p, 1));
            ASSERT_TRUE(p.flushPending());
            triggered = failure.triggered();
        }
        ASSERT_TRUE(triggered);
        EXPECT_EQ(p.fatalState().kind, item.second);
        EXPECT_EQ(p.fatalState().stage, HiddenStateCaptureFailureStage::PREPARE);
        EXPECT_FALSE(p.fatalState().message.empty());
        EXPECT_FALSE(p.tryReserve(2, 2));
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_EQ(evalInt("put_count"), 0);
        EXPECT_FALSE(p.takeDeferredError().has_value());
        p.flushAndClose();
    }
}

TEST_F(HiddenStateCapturePublisherTest, RecoverableNativeWorkerFailureAllowsLaterHealthyPublication) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(1, 2));
    bool triggered = false;
    {
        py::gil_scoped_acquire  gil;
        WorkerAllocationFailure failure(
            std::make_exception_ptr(std::runtime_error("temporary capture preparation failure")));
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        triggered = failure.triggered();
    }
    ASSERT_TRUE(triggered);
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_TRUE(p.shouldPublish());
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
    EXPECT_EQ(evalInt("put_count"), 0);
    ASSERT_TRUE(p.tryReserve(2, 2));
    ASSERT_TRUE(submit(p, 2));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_FALSE(p.takeDeferredError().has_value());
}

TEST_F(HiddenStateCapturePublisherTest, FatalStoreFlushRetainsUnknownCompletionAndSourceQuota) {
    auto& p = publisher({2, 1000, 100, 1, 5});
    {
        py::gil_scoped_acquire gil;
        py::exec("flush_error = torch.OutOfMemoryError('injected staging allocation failure')");
    }
    auto packed = torch::arange(12, torch::kFloat16).reshape({2, 6});
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::arange(2, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "unknown staging completion"));
    ASSERT_TRUE(p.flushPending());
    const auto fatal = p.fatalState();
    EXPECT_EQ(fatal.kind, HiddenStateCaptureFatalKind::OUT_OF_MEMORY);
    EXPECT_EQ(fatal.stage, HiddenStateCaptureFailureStage::STORE);
    EXPECT_GT(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_FALSE(p.tryReserve(2, 2));
    p.cancelReserved(1);
    {
        py::gil_scoped_acquire gil;
        // Even confirmed object absence cannot release unconfirmed local work.
        py::exec("store_keys.clear(); flush_error = None");
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    EXPECT_EQ(p.fatalState().message, fatal.message);
    EXPECT_FALSE(p.takeDeferredError().has_value());
    p.flushAndClose();
    EXPECT_EQ(evalInt("close_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    // Unknown staging work still owns its source after publisher destruction;
    // retaining it must not require a fresh allocation during fatal shutdown.
    publishers_.pop_back();
    EXPECT_GT(packed.use_count(), 1);
}

TEST_F(HiddenStateCapturePublisherTest, SubmitRequiresMatchingReservationAndRetainsCanonicalByIdentity) {
    auto& p = publisher();
    EXPECT_FALSE(submit(p, 1));
    ASSERT_TRUE(p.tryReserve(1, 1));
    EXPECT_FALSE(submit(p, 1, 2));
    EXPECT_EQ(p.quotaSnapshot().reserved_samples, 1);
    EXPECT_EQ(evalInt("put_count"), 0);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, FatalSubmissionExceptionsPropagateWithoutChangingQuotaOrType) {
    const std::vector<std::exception_ptr> errors{
        std::make_exception_ptr(c10::Error("injected tensor failure")),
        std::make_exception_ptr(c10::OutOfMemoryError("injected allocation failure")),
        std::make_exception_ptr(
            c10::AcceleratorError(c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
                                  700,
                                  "injected event-record failure")),
        std::make_exception_ptr(std::bad_alloc())};
    // Both entry points must preserve the original exception, not only the
    // standalone classifier. publish() must not shield serving from trySubmit.
    for (const bool publish_api : {false, true}) {
        for (const auto& error : errors) {
            auto& p = publisher();
            ASSERT_TRUE(p.tryReserve(1, 2));
            auto               packed    = torch::arange(12, torch::kFloat16).reshape({2, 6});
            auto               ids       = torch::arange(2, torch::kInt32);
            auto               lengths   = torch::tensor({2}, torch::kInt32);
            auto               requests  = torch::tensor({1}, torch::kInt64);
            auto               canonical = torch::ones({2, 2}, torch::kFloat16);
            std::exception_ptr observed;
            bool               triggered = false;
            {
                py::gil_scoped_acquire      gil;
                SubmissionAllocationFailure failure(error);
                ASSERT_EQ(c10::GetCPUAllocator(), &failure);
                try {
                    if (publish_api) {
                        (void)p.publish(packed, ids, lengths, requests, 2, "fatal submit", torch::kCPU, canonical);
                    } else {
                        (void)p.trySubmit(packed, ids, lengths, requests, 2, "fatal submit");
                    }
                } catch (...) {
                    observed = std::current_exception();
                }
                triggered = failure.triggered();
            }
            ASSERT_TRUE(triggered);
            ASSERT_TRUE(observed == error) << "Submission must propagate the same exception object";
            EXPECT_EQ(p.quotaSnapshot().reserved_samples, 1);
            EXPECT_EQ(p.quotaSnapshot().pending_batches, 0);
            EXPECT_EQ(p.quotaSnapshot().bytes, 40);
            EXPECT_EQ(p.quotaSnapshot().tokens, 2);
            EXPECT_EQ(p.stats().failure_count, 0);
            EXPECT_TRUE(p.shouldPublish());
            EXPECT_FALSE(p.takeDeferredError().has_value());
            EXPECT_EQ(evalInt("put_count"), 0);
            p.cancelReserved(1);
            EXPECT_EQ(p.quotaSnapshot().samples, 0);
        }
    }
}

TEST_F(HiddenStateCapturePublisherTest, RecoverableSubmissionExceptionsFailOpenEvenWithCudaInMessage) {
    for (const bool publish_api : {false, true}) {
        auto&         p             = publisher();
        const int64_t request_id    = publish_api ? 2 : 1;
        const auto    previous_puts = evalInt("put_count");
        ASSERT_TRUE(p.tryReserve(request_id, 2));
        auto packed    = torch::arange(12, torch::kFloat16).reshape({2, 6});
        auto ids       = torch::arange(2, torch::kInt32);
        auto lengths   = torch::tensor({2}, torch::kInt32);
        auto requests  = torch::tensor({request_id}, torch::kInt64);
        auto canonical = torch::ones({2, 2}, torch::kFloat16);
        auto error     = std::make_exception_ptr(std::runtime_error("CUDA capture submission temporarily unavailable"));
        bool accepted  = true;
        bool triggered = false;
        torch::Tensor result;
        {
            py::gil_scoped_acquire      gil;
            SubmissionAllocationFailure failure(error);
            ASSERT_EQ(c10::GetCPUAllocator(), &failure);
            EXPECT_NO_THROW({
                if (publish_api) {
                    result = p.publish(packed, ids, lengths, requests, 2, "recoverable submit", torch::kCPU, canonical);
                } else {
                    accepted = p.trySubmit(packed, ids, lengths, requests, 2, "recoverable submit");
                }
            });
            triggered = failure.triggered();
        }
        ASSERT_TRUE(triggered);
        if (publish_api) {
            ASSERT_TRUE(result.defined());
            EXPECT_EQ(result.data_ptr(), canonical.data_ptr());
        } else {
            EXPECT_FALSE(accepted);
        }
        EXPECT_EQ(p.stats().failure_count, 1);
        EXPECT_TRUE(p.shouldPublish());
        EXPECT_EQ(p.quotaSnapshot().reserved_samples, 1);
        EXPECT_EQ(p.quotaSnapshot().pending_batches, 0);
        EXPECT_FALSE(p.takeDeferredError().has_value());
        EXPECT_EQ(evalInt("put_count"), previous_puts);
        // Only the unsubmitted reservation remains; a later healthy submit
        // must still succeed on the same publisher after allocator restoration.
        ASSERT_TRUE(submit(p, request_id));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), previous_puts + 1);
    }
}

TEST_F(HiddenStateCapturePublisherTest, CanonicalOutputSurvivesMalformedCapture) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(1, 2));
    auto canonical = torch::ones({2, 2}, torch::kFloat16);
    auto result    = p.publish(torch::ones({2, 5}, torch::kFloat16),
                            torch::arange(2, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "malformed capture",
                            torch::kCPU,
                            canonical);
    EXPECT_EQ(result.data_ptr(), canonical.data_ptr());
    EXPECT_TRUE(torch::equal(result, canonical));
    EXPECT_FALSE(p.takeDeferredError().has_value());
    EXPECT_EQ(evalInt("put_count"), 0);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, InitializationFailureDisablesOnlyCaptureAndCloseWithGilJoins) {
    {
        py::gil_scoped_acquire gil;
        py::exec("fail_init = True");
    }
    auto& p = publisher();
    EXPECT_FALSE(p.shouldPublish());
    EXPECT_FALSE(p.tryReserve(1, 2));
    EXPECT_FALSE(p.takeDeferredError().has_value());
    py::gil_scoped_acquire gil;
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_GT(py::eval("close_count").cast<int>(), 0);
}

}  // namespace
}  // namespace rtp_llm
