#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <exception>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

#include <c10/core/CPUAllocator.h>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

namespace py = pybind11;

namespace rtp_llm {
namespace test {
struct HiddenStateCapturePublisherTestPeer {
    static HiddenStateCaptureFatalKind classify(const std::exception& error) {
        return HiddenStateCapturePublisher::nativeFatalKind(error);
    }
    static void replaceProtectiveFence(HiddenStateCapturePublisher& publisher, void (*operation)()) {
        std::lock_guard<std::mutex> lock(publisher.state_mutex_);
        publisher.protective_fence_override_ = operation;
    }
    static std::weak_ptr<void> retainedTask(const HiddenStateCapturePublisher& publisher, int64_t id) {
        std::lock_guard<std::mutex> lock(publisher.state_mutex_);
        return publisher.reservations_.at(id).task;
    }
    static bool localComplete(const HiddenStateCapturePublisher& publisher, int64_t id) {
        std::lock_guard<std::mutex> lock(publisher.state_mutex_);
        return publisher.reservations_.at(id).local_complete;
    }
    static bool nativeQuiescent(const HiddenStateCapturePublisher& publisher, int64_t id) {
        std::lock_guard<std::mutex> lock(publisher.state_mutex_);
        return publisher.reservations_.at(id).native_quiescent;
    }
    static std::optional<c10::Stream> workerStream(const HiddenStateCapturePublisher& publisher) {
        return publisher.worker_stream_;
    }
    static std::weak_ptr<void> workerStreamOwner(const HiddenStateCapturePublisher& publisher) {
        return publisher.worker_stream_owner_;
    }
};
}  // namespace test
namespace {
using PublisherPeer = test::HiddenStateCapturePublisherTestPeer;

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
pending_native_keys = set()
defer_native_put = False
put_acceptance_mode = ''
async_error_on_flush = False
query_count = 0
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
put_timed_out = False
last_setup_device = None
expected_worker_stream = None
observed_worker_streams = []
sdk_calls = []
async_errors = []
put_started = threading.Event()
allow_put = threading.Event()
allow_put.set()

def complete_pending_native():
    store_keys.update(pending_native_keys)
    pending_native_keys.clear()

def observe_worker_stream(phase):
    global expected_worker_stream
    sdk_calls.append(phase)
    if last_setup_device is not None and last_setup_device.startswith('cuda:'):
        device = torch.device(last_setup_device)
        assert torch.cuda.current_device() == device.index
        native_handle = torch.cuda.current_stream().cuda_stream
        if expected_worker_stream is None:
            expected_worker_stream = native_handle
        assert native_handle == expected_worker_stream
        observed_worker_streams.append((phase, native_handle))

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
    capture_protocol_version = 1
    def __init__(self, config):
        pass
    def setup(self, device):
        global last_setup_device
        last_setup_device = str(device)
        observe_worker_stream('setup')
        if fail_init:
            raise RuntimeError('injected store setup failure')
    def warmup_rdma(self):
        observe_worker_stream('warmup')
    def put_batch(self, **payload):
        global put_count, put_timed_out
        observe_worker_stream('put')
        put_count += 1
        put_started.set()
        if not allow_put.wait(timeout=10):
            put_timed_out = True
            raise RuntimeError('test did not unblock PUT')
        accepted_ids.extend(tensor.tolist() for tensor in payload['input_ids'])
        accepted_hidden.extend(tensor.float().tolist() for tensor in payload['hidden_states'])
        accepted_capture_metadata.append(payload['capture_metadata'])
        for key in payload['request_ids']:
            # Absent keys can still have accepted INPROGRESS native work.
            destination = pending_native_keys if defer_native_put else store_keys
            destination.update([key + '_hs', key + '_ids', key + '_lhs', key + '_ready'])
            if 'hidden_states_scale' in payload:
                destination.add(key + '_hss')
            if fail_put:
                sdk_calls.append('fault:put')
                raise RuntimeError('injected unknown partial PUT')
        if put_acceptance_mode == 'incomplete':
            sdk_calls.append('fault:acceptance')
            return []
        return [{} for _ in payload['request_ids']]
    def flush(self):
        observe_worker_stream('flush')
        global flush_count
        flush_count += 1
        if flush_error is not None:
            sdk_calls.append('fault:flush')
            raise flush_error
        complete_pending_native()
        if async_error_on_flush:
            async_errors.append(RuntimeError('injected asynchronous storage failure'))
        if clear_on_flush:
            store_keys.clear()
    def take_async_errors(self):
        observe_worker_stream('async_errors')
        if async_error_mode == 'malformed':
            return 0
        if async_error_mode == 'exception':
            raise RuntimeError('injected take_async_errors failure')
        result = list(async_errors)
        async_errors.clear()
        if result:
            sdk_calls.append('fault:async_errors')
        return result
    def query_keys(self, keys):
        global query_count
        query_count += 1
        observe_worker_stream('query')
        if query_mode == 'error':
            raise RuntimeError('injected query failure')
        if query_mode == 'incomplete':
            return {}
        if query_mode == 'integer':
            return {key: 0 for key in keys}
        return {key: key in store_keys for key in keys}
    def close(self):
        observe_worker_stream('close')
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
                                           std::vector<int64_t>     layer_ids = {1, 0},
                                           c10::Device              device    = c10::Device(torch::kCPU)) {
        publishers_.push_back(std::make_unique<HiddenStateCapturePublisher>(2,
                                                                            2,
                                                                            fp8 ? HiddenStateCaptureDtype::FP8_E4M3 :
                                                                                  HiddenStateCaptureDtype::BF16,
                                                                            torch::kFloat16,
                                                                            owner,
                                                                            0,
                                                                            nullptr,
                                                                            limits,
                                                                            std::move(layer_ids),
                                                                            device));
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

    void execute(const char* code) {
        py::gil_scoped_acquire gil;
        py::exec(code);
    }

    int64_t evalInt(const char* expression) {
        py::gil_scoped_acquire gil;
        return py::eval(expression).cast<int64_t>();
    }

    void expectQuarantinedSdkCalls(const char* expected) {
        py::gil_scoped_acquire gil;
        // A fresh request requires exactly one preflight query before PUT.
        // The exact suffix also rejects maintenance, retry, diagnostics or close
        // after the fault; idle diagnostics before preflight are immaterial.
        ASSERT_EQ(py::eval("query_count").cast<int64_t>(), 1);
        auto calls = py::eval("sdk_calls[sdk_calls.index('query'):]");
        EXPECT_TRUE(calls.equal(py::eval(expected))) << py::str(calls).cast<std::string>();
    }

    std::vector<std::unique_ptr<HiddenStateCapturePublisher>> publishers_;
};

TEST_F(HiddenStateCapturePublisherTest, LegacyTorch26C10BaseDeclarationsRemainExecutionFatal) {
    class LegacyDeviceError final: public c10::Error {
    public:
        using c10::Error::Error;
    };
    static_assert(std::is_base_of_v<c10::Error, c10::OutOfMemoryError>);
    EXPECT_EQ(PublisherPeer::classify(c10::Error("legacy execution failure")), HiddenStateCaptureFatalKind::TENSOR);
    EXPECT_EQ(PublisherPeer::classify(LegacyDeviceError("legacy derived device failure")),
              HiddenStateCaptureFatalKind::TENSOR);
    EXPECT_EQ(PublisherPeer::classify(c10::OutOfMemoryError("legacy allocation failure")),
              HiddenStateCaptureFatalKind::OUT_OF_MEMORY);
    EXPECT_EQ(PublisherPeer::classify(std::runtime_error("ordinary storage failure")),
              HiddenStateCaptureFatalKind::NONE);
}

TEST_F(HiddenStateCapturePublisherTest, ProtectiveFenceFailureIsStickyFatalForEveryExceptionAndRetainsOwners) {
    using Fence = void (*)();
    const std::vector<Fence> failures{
        +[] { throw std::runtime_error("untyped native wrapper failure"); },
        +[] { throw c10::Error("typed native completion failure"); },
        +[] { throw std::bad_alloc(); },
        +[] { throw 42; },
    };
    for (auto failure : failures) {
        auto& p = publisher();
        execute("fail_put = True");
        PublisherPeer::replaceProtectiveFence(p, failure);
        ASSERT_TRUE(p.tryReserve(1, 2));
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ACCELERATOR);
        EXPECT_EQ(p.fatalState().stage, HiddenStateCaptureFailureStage::PREPARE);
        EXPECT_FALSE(p.shouldPublish());
        EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
        EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
        EXPECT_GT(p.quotaSnapshot().retained_bytes, 0);
        auto owner = PublisherPeer::retainedTask(p, 1);
        execute("store_keys.clear()");
        ASSERT_TRUE(p.flushPending());
        p.cancelReserved(1);
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
        p.flushAndClose();
        EXPECT_EQ(evalInt("close_count"), 0);
        publishers_.pop_back();
        EXPECT_FALSE(owner.expired());  // Destructor quarantines the unresolved task/payload owners.
    }
}

TEST_F(HiddenStateCapturePublisherTest, PayloadQuotaAndCancellationAreBoundedAndIdempotent) {
    auto& p = publisher({2, 100, 5, 2, 1000});
    EXPECT_TRUE(p.tryReserve(1, 2));
    EXPECT_TRUE(p.tryReserve(2, 3));
    EXPECT_FALSE(p.tryReserve(1, 2));
    EXPECT_FALSE(p.tryReserve(3, 1));
    EXPECT_EQ(p.quotaSnapshot().bytes, 100);
    p.cancelReserved(1);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 60);
    EXPECT_EQ(p.quotaSnapshot().tokens, 3);
}

TEST_F(HiddenStateCapturePublisherTest, InvalidLimitsOrMissingProtocolDisableCaptureOnly) {
    for (const char* code : {"Store.capture_protocol_version = 0",
                             "Store.capture_protocol_version = True",
                             "del Store.capture_protocol_version"}) {
        execute(code);
        auto& p = publisher();
        EXPECT_FALSE(p.shouldPublish());
        EXPECT_FALSE(p.tryReserve(1, 2));
        EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    }
    execute("Store.capture_protocol_version = 1; fail_init = True");
    auto& p = publisher();
    EXPECT_FALSE(p.shouldPublish());
    EXPECT_EQ(evalInt("put_count"), 0);
}

TEST_F(HiddenStateCapturePublisherTest, MissingQueryCapabilityNeverFallsBackToLegacyPublication) {
    execute("del Store.query_keys");
    auto& p = publisher();
    EXPECT_FALSE(p.shouldPublish());
    EXPECT_FALSE(p.tryReserve(1, 2));
    EXPECT_EQ(evalInt("put_count"), 0);
}

TEST_F(HiddenStateCapturePublisherTest, ConcurrentReservationsNeverOversubscribe) {
    auto&                    p = publisher({2, 80, 4, 2, 1000});
    std::atomic<int>         accepted{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&, i] { accepted += p.tryReserve(i, 2); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(accepted.load(), 2);
    EXPECT_EQ(p.quotaSnapshot().bytes, 80);
    EXPECT_EQ(p.quotaSnapshot().tokens, 4);
}

TEST_F(HiddenStateCapturePublisherTest, ActiveAndQueuedTasksShareOneLimitAndNoCleanupOverflowQueue) {
    auto& p = publisher({4, 1000, 100, 1, 1000});
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.tryReserve(2, 2));
    execute("allow_put.clear()");
    ASSERT_TRUE(submit(p, 1));
    {
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(py::eval("put_started.wait(timeout=2)").cast<bool>());
    }
    EXPECT_FALSE(submit(p, 2));
    p.cancelReserved(2);
    auto quota = p.quotaSnapshot();
    EXPECT_EQ(quota.retained_tasks, 1);
    EXPECT_EQ(quota.pending_batches, 0);
    EXPECT_EQ(quota.cleanup_pending_samples, 0);
    EXPECT_GT(quota.retained_bytes, 0);
    execute("allow_put.set()");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 0);
    EXPECT_EQ(p.quotaSnapshot().retained_bytes, 0);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
}

TEST_F(HiddenStateCapturePublisherTest, RetainedBudgetRejectsBeforeAllocatingTaskSnapshot) {
    HiddenStateCaptureLimits limits;
    limits.max_retained_bytes = 1;
    auto& p                   = publisher(limits);
    ASSERT_TRUE(p.tryReserve(1, 2));
    EXPECT_FALSE(submit(p, 1));
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 0);
    EXPECT_EQ(p.quotaSnapshot().retained_bytes, 0);
    EXPECT_EQ(evalInt("put_count"), 0);
    p.cancelReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, SnapshotsProtectPayloadAndHostMetadataFromInPlaceMutation) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(7, 2));
    auto packed   = torch::arange(12, torch::kFloat16).reshape({2, 6});
    auto ids      = torch::tensor({1, 2}, torch::kInt32);
    auto lengths  = torch::tensor({2}, torch::kInt32);
    auto requests = torch::tensor({7}, torch::kInt64);
    {
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(p.trySubmit(packed, ids, lengths, requests, 2, "snapshot"));
        packed.fill_(99);
        ids.fill_(99);
        lengths.fill_(99);
        requests.fill_(99);
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("accepted_ids == [[1, 2]]"), 1);
    EXPECT_EQ(evalInt("accepted_hidden == [[[0, 1, 2, 3], [6, 7, 8, 9]]]"), 1);
    EXPECT_EQ(evalInt("accepted_capture_metadata == [{'capture_layer_ids': [1, 0]}]"), 1);
    p.cancelReserved(7);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
}

TEST_F(HiddenStateCapturePublisherTest, QueryUnknownOrPartialManifestNeverRefundsSourceQuota) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_TRUE(PublisherPeer::localComplete(p, 1));
    EXPECT_TRUE(PublisherPeer::nativeQuiescent(p, 1));
    EXPECT_TRUE(PublisherPeer::retainedTask(p, 1).expired());
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 0);
    execute("store_keys.discard('rtp_PublisherTest_1_ready')");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    for (const char* mode : {"query_mode = 'error'", "query_mode = 'incomplete'", "query_mode = 'integer'"}) {
        execute(mode);
        execute("store_keys.clear()");
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    }
    execute("query_mode = ''");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, PartialPutQuarantinesSourceWithoutRejectingServing) {
    auto& p = publisher();
    execute("fail_put = True");
    ASSERT_TRUE(p.tryReserve(1, 2));
    EXPECT_NO_THROW(EXPECT_TRUE(submit(p, 1)));
    ASSERT_TRUE(p.flushPending());
    expectQuarantinedSdkCalls("['query', 'put', 'fault:put']");
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(evalInt("'rtp_PublisherTest_1_ready' in store_keys"), 1);
    EXPECT_FALSE(p.shouldPublish());
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_TRUE(PublisherPeer::localComplete(p, 1));
    EXPECT_FALSE(PublisherPeer::nativeQuiescent(p, 1));
    execute("store_keys.clear(); fail_put = False");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("flush_count"), 0);  // No retry flush can turn unknown PUT into terminal proof.
    expectQuarantinedSdkCalls("['query', 'put', 'fault:put']");
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
    p.flushAndClose();
    expectQuarantinedSdkCalls("['query', 'put', 'fault:put']");
}

TEST_F(HiddenStateCapturePublisherTest, HistoricalStorageDiagnosticDoesNotInferFatalFromText) {
    auto& p = publisher();
    execute("async_errors.append(RuntimeError('CUDA storage temporarily unavailable'))");
    ASSERT_TRUE(p.flushPending());
    EXPECT_GT(p.stats().failure_count, 0);
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_TRUE(p.shouldPublish());
}

TEST_F(HiddenStateCapturePublisherTest, UnknownNativePutOrFlushCannotRefundAbsentButInProgressSources) {
    for (const char* failure : {"fail_put = True",
                                "flush_error = RuntimeError('unknown native completion')",
                                "put_acceptance_mode = 'incomplete'"}) {
        SCOPED_TRACE(failure);
        execute("fail_put = False; flush_error = None; put_acceptance_mode = ''; "
                "defer_native_put = True; store_keys.clear(); pending_native_keys.clear(); "
                "flush_count = 0; query_count = 0");
        auto& p = publisher({4, 1000, 100, 1, 1000});
        execute(failure);
        ASSERT_TRUE(p.tryReserve(1, 2));
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_TRUE(PublisherPeer::localComplete(p, 1));
        EXPECT_FALSE(PublisherPeer::nativeQuiescent(p, 1));
        EXPECT_EQ(evalInt("len(store_keys)"), 0);
        EXPECT_GT(evalInt("len(pending_native_keys)"), 0);
        EXPECT_EQ(evalInt("all(not value for value in Store(Config()).query_keys(list(pending_native_keys)).values())"),
                  1);  // Strict all-false is NOT native quiescence.
        auto owner = PublisherPeer::retainedTask(p, 1);
        EXPECT_FALSE(owner.expired());
        EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
        EXPECT_FALSE(p.shouldPublish());
        EXPECT_FALSE(p.tryReserve(2, 2));
        EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
        EXPECT_EQ(p.quotaSnapshot().bytes, 40);
        EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
        EXPECT_GT(p.quotaSnapshot().retained_bytes, 0);
        const auto flushes = evalInt("flush_count");
        const auto queries = evalInt("query_count");
        execute("fail_put = False; flush_error = None; put_acceptance_mode = ''");
        ASSERT_TRUE(p.flushPending());
        p.cancelReserved(1);
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
        execute("complete_pending_native()");  // The accepted native operation can still write later.
        EXPECT_GT(evalInt("len(store_keys)"), 0);
        execute("store_keys.clear()");
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
        EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
        EXPECT_EQ(evalInt("flush_count"), flushes);
        EXPECT_EQ(evalInt("query_count"), queries);
        p.flushAndClose();
        EXPECT_EQ(evalInt("close_count"), 0);
        publishers_.pop_back();
        EXPECT_FALSE(owner.expired());  // Even a successful GPU fence cannot free unknown native owners.
    }
}

TEST_F(HiddenStateCapturePublisherTest, NativeQuarantineStillRollsBackCompletedNeverPutQueuedTask) {
    auto& p = publisher({4, 1000, 100, 2, 1000});
    execute("defer_native_put = True; fail_put = True; allow_put.clear()");
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.tryReserve(2, 2));
    ASSERT_TRUE(submit(p, 1));
    {
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(py::eval("put_started.wait(timeout=2)").cast<bool>());
    }
    ASSERT_TRUE(submit(p, 2));
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 2);
    execute("allow_put.set()");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("flush_count"), 0);
    expectQuarantinedSdkCalls("['query', 'put', 'fault:put']");
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 0);
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_FALSE(p.shouldPublish());
    p.cancelReserved(2);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    ASSERT_TRUE(p.flushPending());
    expectQuarantinedSdkCalls("['query', 'put', 'fault:put']");
    p.flushAndClose();
    expectQuarantinedSdkCalls("['query', 'put', 'fault:put']");
}

TEST_F(HiddenStateCapturePublisherTest, AsyncErrorAfterFlushCannotEstablishNativeQuiescence) {
    auto& p = publisher();
    execute("async_error_on_flush = True");
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    expectQuarantinedSdkCalls("['query', 'put', 'flush', 'async_errors', 'fault:async_errors']");
    EXPECT_TRUE(PublisherPeer::localComplete(p, 1));
    EXPECT_FALSE(PublisherPeer::nativeQuiescent(p, 1));
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_FALSE(p.shouldPublish());
    execute("store_keys.clear(); async_error_on_flush = False");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
    EXPECT_EQ(evalInt("flush_count"), 1);
    expectQuarantinedSdkCalls("['query', 'put', 'flush', 'async_errors', 'fault:async_errors']");
    p.flushAndClose();
    expectQuarantinedSdkCalls("['query', 'put', 'flush', 'async_errors', 'fault:async_errors']");
}

TEST_F(HiddenStateCapturePublisherTest, CanonicalAndPrenormOutputsRemainUnchangedAfterExportFailure) {
    auto&                     p         = publisher();
    auto                      canonical = torch::tensor({1, 2, 3, 4}, torch::kFloat16).reshape({2, 2});
    auto                      prenorm   = canonical + 100;
    torch_ext::PyModelOutputs output(canonical, prenorm);
    output.captured_hidden_states = torch::cat({canonical + 10, canonical + 20, canonical}, 1);
    const auto canonical_before   = output.hidden_states.clone();
    const auto prenorm_before     = output.mtp_target_hidden_states.clone();
    execute("fail_put = True");
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(p.trySubmit(output.captured_hidden_states,
                            torch::tensor({1, 2}, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "canonical ABI"));
    ASSERT_TRUE(p.flushPending());
    EXPECT_TRUE(torch::equal(output.hidden_states, canonical_before));
    EXPECT_TRUE(torch::equal(output.mtp_target_hidden_states, prenorm_before));
    EXPECT_EQ(output.hidden_states.data_ptr(), canonical.data_ptr());
    EXPECT_EQ(output.mtp_target_hidden_states.data_ptr(), prenorm.data_ptr());
    EXPECT_GT(p.stats().failure_count, 0);
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
}

TEST_F(HiddenStateCapturePublisherTest, CpuQuantizerRuntimeErrorDropsOnlyExport) {
    auto& p = publisher({}, true, true);
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_GT(p.stats().failure_count, 0);
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_TRUE(p.shouldPublish());
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, TypedAsyncAllocationFailureIsFatalNotInferredFromText) {
    auto& p = publisher();
    execute("async_errors.append(MemoryError('typed allocation failure'))");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ALLOCATION);
    EXPECT_FALSE(p.shouldPublish());
}

TEST_F(HiddenStateCapturePublisherTest, NativeSubmissionErrorsPreserveExceptionIdentityAndQuota) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(1, 2));
    for (auto error : {std::make_exception_ptr(c10::Error("tensor execution failure")),
                       std::make_exception_ptr(c10::OutOfMemoryError("allocation failure")),
                       std::make_exception_ptr(std::bad_alloc())}) {
        auto               packed   = torch::arange(12, torch::kFloat16).reshape({2, 6});
        auto               ids      = torch::arange(2, torch::kInt32);
        auto               lengths  = torch::tensor({2}, torch::kInt32);
        auto               requests = torch::tensor({1}, torch::kInt64);
        std::exception_ptr observed;
        {
            SubmissionAllocationFailure failure(error);
            try {
                (void)p.trySubmit(packed, ids, lengths, requests, 2, "typed submission");
            } catch (...) {
                observed = std::current_exception();
            }
            EXPECT_TRUE(failure.triggered());
        }
        EXPECT_EQ(observed, error);
        EXPECT_EQ(p.quotaSnapshot().reserved_samples, 1);
        EXPECT_EQ(p.quotaSnapshot().retained_tasks, 0);
    }
}

TEST_F(HiddenStateCapturePublisherTest, NoAdmissionUntilInitializationHasCompleted) {
    py::gil_scoped_acquire gil;
    auto                   p = std::make_unique<HiddenStateCapturePublisher>(2,
                                                           2,
                                                           HiddenStateCaptureDtype::BF16,
                                                           torch::kFloat16,
                                                           true,
                                                           0,
                                                           nullptr,
                                                           HiddenStateCaptureLimits{},
                                                           std::vector<int64_t>{1, 0},
                                                           c10::Device(torch::kCPU));
    EXPECT_FALSE(p->tryReserve(1, 2));  // Worker cannot enter Python while this thread holds the GIL.
    ASSERT_TRUE(p->flushPending());
    EXPECT_TRUE(p->tryReserve(1, 2));
    p->cancelReserved(1);
    p->flushAndClose();
}

TEST_F(HiddenStateCapturePublisherTest, OwnerOnlyAndInvalidLimitsHaveNoWorkerMaintenanceHang) {
    auto& non_owner = publisher({}, false);
    EXPECT_FALSE(non_owner.tryReserve(1, 2));
    EXPECT_TRUE(non_owner.flushPending());
    HiddenStateCaptureLimits limits;
    limits.max_inflight_samples = 0;
    auto& invalid               = publisher(limits);
    EXPECT_FALSE(invalid.shouldPublish());
    EXPECT_TRUE(invalid.flushPending());
}

TEST_F(HiddenStateCapturePublisherTest, WorkerNativeAllocationFailureLatchesFatalAndRetainsNoUnsubmittedSource) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(1, 2));
    {
        py::gil_scoped_acquire  gil;
        WorkerAllocationFailure failure(std::make_exception_ptr(std::bad_alloc()));
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_TRUE(failure.triggered());
    }
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ALLOCATION);
    EXPECT_FALSE(p.shouldPublish());
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, CloseWithGilDrainsAcceptedTaskAndIsIdempotent) {
    auto& p = publisher();
    execute("clear_on_flush = True");
    ASSERT_TRUE(p.tryReserve(1, 2));
    py::gil_scoped_acquire gil;
    ASSERT_TRUE(submit(p, 1));
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_EQ(py::eval("close_count").cast<int64_t>(), 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, ShutdownDoesNotCloseLiveSourceOrLoseItsOwner) {
    auto& p = publisher();
    ASSERT_TRUE(p.tryReserve(1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    p.flushAndClose();
    EXPECT_EQ(evalInt("close_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
}

TEST_F(HiddenStateCapturePublisherTest, SharedAdmissionCancellationRunsExactlyOnceWithoutOwningPublisher) {
    int  cancelled = 0;
    auto token     = std::make_shared<HiddenStateCaptureReservation>([&] { ++cancelled; });
    auto copy      = token;
    token->cancel();
    copy->cancel();
    token.reset();
    copy.reset();
    EXPECT_EQ(cancelled, 1);
    auto                                       p    = std::make_shared<HiddenStateCapturePublisher>(2,
                                                           2,
                                                           HiddenStateCaptureDtype::BF16,
                                                           torch::kFloat16,
                                                           true,
                                                           0,
                                                           nullptr,
                                                           HiddenStateCaptureLimits{},
                                                           std::vector<int64_t>{1, 0},
                                                           c10::Device(torch::kCPU));
    std::weak_ptr<HiddenStateCapturePublisher> weak = p;
    ASSERT_TRUE(p->flushPending());
    ASSERT_TRUE(p->tryReserve(2, 2));
    auto late = std::make_shared<HiddenStateCaptureReservation>([weak] {
        if (auto owner = weak.lock()) {
            owner->cancelReserved(2);
        }
    });
    p.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_NO_THROW(late->cancel());
}

#if USING_CUDA
class ScopedNativeStreamGate final {
public:
    explicit ScopedNativeStreamGate(cudaStream_t stream): stream_(stream) {
        const auto status = cudaLaunchHostFunc(stream_, &ScopedNativeStreamGate::wait, this);
        if (status != cudaSuccess) {
            throw std::runtime_error("failed to enqueue native stream gate");
        }
    }
    ~ScopedNativeStreamGate() {
        release();
        (void)cudaStreamSynchronize(stream_);
    }
    bool entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, std::chrono::seconds(3), [&] { return entered_; });
    }
    bool timedOut() {
        std::lock_guard<std::mutex> lock(mutex_);
        return timed_out_;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        changed_.notify_all();
    }

private:
    static void CUDART_CB wait(void* opaque) {
        auto&                        gate = *static_cast<ScopedNativeStreamGate*>(opaque);
        std::unique_lock<std::mutex> lock(gate.mutex_);
        gate.entered_ = true;
        gate.changed_.notify_all();
        // No CUDA APIs are called from the callback; failures cannot hang the test.
        gate.timed_out_ = !gate.changed_.wait_for(lock, std::chrono::seconds(8), [&] { return gate.released_; });
    }
    cudaStream_t            stream_;
    std::mutex              mutex_;
    std::condition_variable changed_;
    bool                    entered_{false};
    bool                    released_{false};
    bool                    timed_out_{false};
};

TEST_F(HiddenStateCapturePublisherTest, UnknownGpuCompletionQuarantinesNativeStreamAndTaskOwners) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "native stream quarantine requires CUDA";
        }
    }
    c10::cuda::CUDAGuard guard(0);
    auto&                p      = publisher({}, true, false, {1, 0}, c10::Device(torch::kCUDA, 0));
    auto                 stream = PublisherPeer::workerStream(p);
    ASSERT_TRUE(stream.has_value());
    const auto native       = c10::cuda::CUDAStream(*stream).stream();
    auto       stream_owner = PublisherPeer::workerStreamOwner(p);
    PublisherPeer::replaceProtectiveFence(
        p, +[] { throw std::runtime_error("native completion wrapper failed"); });
    execute("fail_put = True");
    ASSERT_TRUE(p.tryReserve(1, 2));
    auto packed =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0)).reshape({2, 6});
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::tensor({1, 2}, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "native stream quarantine"));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ACCELERATOR);
    auto task_owner = PublisherPeer::retainedTask(p, 1);
    execute("store_keys.clear()");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
    p.flushAndClose();
    publishers_.pop_back();
    EXPECT_FALSE(stream_owner.expired());
    EXPECT_FALSE(task_owner.expired());
    unsigned int flags = 0;
    EXPECT_EQ(cudaStreamGetFlags(native, &flags), cudaSuccess);
    EXPECT_NE(flags & cudaStreamNonBlocking, 0u);
}

TEST_F(HiddenStateCapturePublisherTest, UnknownNativeWriteRetainsPrivateStreamDespiteSuccessfulGpuFence) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "native-write stream quarantine requires CUDA";
        }
    }
    c10::cuda::CUDAGuard guard(0);
    auto&                p      = publisher({}, true, false, {1, 0}, c10::Device(torch::kCUDA, 0));
    auto                 stream = PublisherPeer::workerStream(p);
    ASSERT_TRUE(stream.has_value());
    const auto native       = c10::cuda::CUDAStream(*stream).stream();
    auto       stream_owner = PublisherPeer::workerStreamOwner(p);
    execute("defer_native_put = True; flush_error = RuntimeError('native write still in progress')");
    ASSERT_TRUE(p.tryReserve(1, 2));
    auto packed =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0)).reshape({2, 6});
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::tensor({1, 2}, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "native-write quarantine"));
    ASSERT_TRUE(p.flushPending());
    expectQuarantinedSdkCalls("['query', 'put', 'flush', 'fault:flush']");
    EXPECT_TRUE(PublisherPeer::localComplete(p, 1));
    EXPECT_FALSE(PublisherPeer::nativeQuiescent(p, 1));
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_FALSE(p.shouldPublish());
    auto task_owner = PublisherPeer::retainedTask(p, 1);
    execute("store_keys.clear(); flush_error = None");
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 1);
    expectQuarantinedSdkCalls("['query', 'put', 'flush', 'fault:flush']");
    p.flushAndClose();
    publishers_.pop_back();
    expectQuarantinedSdkCalls("['query', 'put', 'flush', 'fault:flush']");
    EXPECT_FALSE(stream_owner.expired());
    EXPECT_FALSE(task_owner.expired());
    EXPECT_EQ(evalInt("close_count"), 0);
    unsigned int flags = 0;
    EXPECT_EQ(cudaStreamGetFlags(native, &flags), cudaSuccess);
    EXPECT_NE(flags & cudaStreamNonBlocking, 0u);
}

TEST_F(HiddenStateCapturePublisherTest, PublishedSourceRetainsNativeStreamAfterCompletedTasksAreReleased) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "native source ownership requires CUDA";
        }
    }
    c10::cuda::CUDAGuard guard(0);
    auto&                p      = publisher({}, true, false, {1, 0}, c10::Device(torch::kCUDA, 0));
    auto                 stream = PublisherPeer::workerStream(p);
    ASSERT_TRUE(stream.has_value());
    const auto native = c10::cuda::CUDAStream(*stream).stream();
    auto       owner  = PublisherPeer::workerStreamOwner(p);
    ASSERT_TRUE(p.tryReserve(1, 2));
    auto packed =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0)).reshape({2, 6});
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::tensor({1, 2}, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "published source stream ownership"));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().retained_tasks, 0);
    EXPECT_TRUE(PublisherPeer::retainedTask(p, 1).expired());
    p.flushAndClose();
    publishers_.pop_back();
    EXPECT_FALSE(owner.expired());  // The source/store lease alone must retain the stream.
    EXPECT_EQ(evalInt("close_count"), 0);
    unsigned int flags = 0;
    EXPECT_EQ(cudaStreamGetFlags(native, &flags), cudaSuccess);
}

TEST_F(HiddenStateCapturePublisherTest, GpuQuantizerRuntimeErrorIsFatalByBoundaryWithoutMessageHeuristics) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "GPU quantizer boundary requires CUDA";
        }
    }
    auto& p = publisher({}, true, true, {1, 0}, c10::Device(torch::kCUDA, 0));
    ASSERT_TRUE(p.tryReserve(1, 2));
    auto packed =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0)).reshape({2, 6});
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::tensor({1, 2}, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "GPU quantizer boundary"));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::TENSOR);
    EXPECT_EQ(p.fatalState().stage, HiddenStateCaptureFailureStage::QUANTIZE);
    EXPECT_FALSE(p.shouldPublish());
}

TEST_F(HiddenStateCapturePublisherTest, NonzeroDeviceAndNondefaultProducerStreamCoverAllWorkerPhases) {
    {
        py::gil_scoped_acquire gil;
        if (py::module_::import("torch").attr("cuda").attr("device_count")().cast<int>() < 2) {
            if (std::getenv("CAPTURE_TEST_REQUIRE_TWO_GPUS")) {
                FAIL() << "multi-GPU target was scheduled without two visible CUDA devices";
            }
            GTEST_SKIP() << "requires two CUDA devices to verify the nonzero device context";
        }
    }
    auto& p      = publisher({}, true, false, {1, 0}, c10::Device(torch::kCUDA, 1));
    auto  worker = PublisherPeer::workerStream(p);
    ASSERT_TRUE(worker.has_value());
    const auto worker_native = c10::cuda::CUDAStream(*worker).stream();
    auto       stream_owner  = PublisherPeer::workerStreamOwner(p);
    {
        c10::cuda::CUDAGuard guard(1);
        unsigned int         flags = 0;
        ASSERT_EQ(cudaStreamGetFlags(worker_native, &flags), cudaSuccess);
        EXPECT_NE(flags & cudaStreamNonBlocking, 0u);
        EXPECT_NE(worker_native, c10::cuda::getDefaultCUDAStream(1).stream());
        // Exhaust a full pool wraparound; compare CUDA handles, not Torch IDs.
        for (int i = 0; i < 64; ++i) {
            EXPECT_NE(worker_native, c10::cuda::getStreamFromPool(false, 1).stream());
        }
        const auto producer = c10::cuda::getStreamFromPool(false, 1);
        ASSERT_NE(producer.stream(), worker_native);
        c10::cuda::CUDAStreamGuard producer_guard(producer);
        auto probe = torch::ones({16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1));
        probe.add_(1);  // Warm allocations/kernel loading before blocking the worker.
        producer.synchronize();
        {
            ScopedNativeStreamGate gate(worker_native);
            ASSERT_TRUE(gate.entered());
            probe.add_(1);
            producer.synchronize();  // The zero-token serving fence must not join worker work.
            EXPECT_FALSE(gate.timedOut());
            gate.release();
        }
        EXPECT_TRUE(probe.cpu().eq(3).all().item<bool>());
    }
    ASSERT_TRUE(p.tryReserve(1, 2));
    {
        c10::cuda::CUDAGuard       device_guard(1);
        auto                       stream = c10::cuda::getStreamFromPool(false, 1);
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        auto                       packed =
            torch::arange(12, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 1)).reshape({2, 6});
        auto ids = torch::tensor({1, 2}, torch::kInt32).to(packed.device());
        ASSERT_TRUE(p.trySubmit(
            packed, ids, torch::tensor({2}, torch::kInt32), torch::tensor({1}, torch::kInt64), 2, "nonzero producer"));
        packed.fill_(99);
        ids.fill_(99);
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("accepted_ids == [[1, 2]]"), 1);
    EXPECT_EQ(evalInt("accepted_hidden == [[[0, 1, 2, 3], [6, 7, 8, 9]]]"), 1);
    execute("store_keys.clear()");
    ASSERT_TRUE(p.flushPending());
    p.flushAndClose();
    EXPECT_EQ(evalInt("close_count"), 1);
    EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_TRUE(stream_owner.expired());
    {
        py::gil_scoped_acquire gil;
        EXPECT_TRUE(
            py::eval(
                "{'setup', 'warmup', 'put', 'flush', 'query', 'close'}.issubset({phase for phase, _ in observed_worker_streams})")
                .cast<bool>());
        EXPECT_TRUE(
            py::eval("all(handle == expected_worker_stream for _, handle in observed_worker_streams)").cast<bool>());
        EXPECT_EQ(py::eval("expected_worker_stream").cast<std::uintptr_t>(),
                  reinterpret_cast<std::uintptr_t>(worker_native));
    }
}

TEST_F(HiddenStateCapturePublisherTest, NormalAndZeroTokenFinishWhileStorePutIsBlocked) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "executor integration requires CUDA";
        }
    }
    py::gil_scoped_acquire executor_gil;
    initRuntime(0, false, false, MlaOpsType::AUTO);
    class PublishingModel final: public ModelBase {
    public:
        explicit PublishingModel(HiddenStateCapturePublisher& publisher): publisher_(publisher) {}
        GptModelOutputs forward(const GptModelInputs& inputs) override {
            EXPECT_TRUE(inputs.capture_hidden_states);
            EXPECT_TRUE(inputs.capture_input_lengths_cpu.device().is_cpu());
            EXPECT_TRUE(publisher_.trySubmit(torch::arange(12, torch::kFloat16).reshape({2, 6}),
                                             torch::tensor({1, 2}, torch::kInt32),
                                             inputs.capture_input_lengths_cpu,
                                             inputs.capture_request_ids_cpu,
                                             2,
                                             "executor integration"));
            GptModelOutputs output;
            if (!inputs.skip_lm_head) {
                output.logits =
                    torch::zeros({1, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                output.logits.select(1, 17).fill_(100);
            }
            return output;
        }

    private:
        HiddenStateCapturePublisher& publisher_;
    };
    for (const bool fail_put : {false, true}) {
        for (int generated : {1, 0}) {
            SCOPED_TRACE(generated);
            execute(fail_put ? "fail_put = True" : "fail_put = False");
            // Unknown PUT disables this publisher, not the next independent case.
            auto&            p = publisher();
            EngineInitParams params;
            params.model_config_.max_seq_len = 128;
            params.model_config_.num_layers  = 1;
            params.model_config_.vocab_size = params.model_config_.input_vocab_size = 128;
            params.model_config_.hidden_size                                        = 128;
            params.model_config_.attn_config.head_num = params.model_config_.attn_config.kv_head_num = 2;
            params.model_config_.attn_config.size_per_head                                           = 64;
            params.model_config_.attn_config.tokens_per_block =
                params.model_config_.attn_config.kernel_tokens_per_block = 2;
            params.model_config_.hidden_state_capture_layer_ids          = {0};
            params.pd_sep_config.role_type                               = RoleType::PDFUSION;
            {
                py::gil_scoped_acquire gil;
                params.py_model = py::none();
            }
            NormalExecutor executor(params, nullptr);
            executor.setModel(std::make_unique<PublishingModel>(p));
            auto input                                    = std::make_shared<GenerateInput>();
            input->request_id                             = 100 + generated + (fail_put ? 10 : 0);
            input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);
            input->generate_config                        = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens        = generated;
            input->generate_config->capture_hidden_states = generated != 0;
            input->generate_config->reuse_cache           = false;
            input->generate_config->top_k                 = 1;
            ResourceContext resource;
            resource.role_type = RoleType::PDFUSION;
            auto stream        = std::make_shared<NormalGenerateStream>(
                input, params.model_config_, params.runtime_config, resource, nullptr);
            ASSERT_TRUE(p.tryReserve(stream->streamId(), 2));
            stream->setHiddenStateCaptureReservation(std::make_shared<HiddenStateCaptureReservation>(
                [&p, id = stream->streamId()] { p.cancelReserved(id); }));
            execute("allow_put.clear(); put_started.clear()");
            ASSERT_TRUE(executor.process({stream}).ok());
            {
                py::gil_scoped_acquire gil;
                ASSERT_TRUE(py::eval("put_started.wait(timeout=2)").cast<bool>());
            }
            ASSERT_FALSE(stream->hasError());
            auto output = stream->nextOutput(1000);
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1);
            EXPECT_TRUE(output.value().generate_outputs[0].finished);
            EXPECT_EQ(stream->outputTokenLen(), generated);
            EXPECT_EQ(evalInt("put_timed_out"), 0);
            EXPECT_EQ(evalInt("allow_put.is_set()"), 0);
            execute("allow_put.set()");
            ASSERT_TRUE(p.flushPending());
            execute("store_keys.clear()");
            ASSERT_TRUE(p.flushPending());
            EXPECT_EQ(p.stats().fatal_kind, HiddenStateCaptureFatalKind::NONE);
            EXPECT_EQ(p.quotaSnapshot().samples, fail_put ? 1 : 0);
            p.flushAndClose();
        }
    }
}
#endif

}  // namespace
}  // namespace rtp_llm
