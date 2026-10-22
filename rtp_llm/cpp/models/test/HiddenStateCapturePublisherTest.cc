#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <new>
#include <torch/version.h>
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

class HiddenStateCapturePublisherTestPeer {
public:
    static void failMaintenanceAllocation(HiddenStateCapturePublisher& publisher) {
        std::lock_guard<std::mutex> lock(publisher.state_mutex_);
        publisher.maintenance_allocation_probe_ = [] { throw std::bad_alloc(); };
    }
};

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

// Observe real ATen allocation boundaries without calling Python or replacing
// tensor behavior. The BF16 path includes the int32 -> int64 input-ID conversion.
class WorkerAllocationGilProbe final: public c10::Allocator {
public:
    WorkerAllocationGilProbe():
        original_(c10::GetCPUAllocator()), submitting_thread_(std::this_thread::get_id()) {
        c10::SetCPUAllocator(this);
    }
    ~WorkerAllocationGilProbe() override {
        c10::SetCPUAllocator(original_);
    }
    c10::DataPtr allocate(size_t bytes) override {
        if (std::this_thread::get_id() != submitting_thread_) {
            ++worker_allocations;
            if (PyGILState_Check()) {
                ++gil_held_allocations;
            }
        }
        return original_->allocate(bytes);
    }
    void copy_data(void* destination, const void* source, size_t bytes) const override {
        original_->copy_data(destination, source, bytes);
    }

    std::atomic<int> worker_allocations{0};
    std::atomic<int> gil_held_allocations{0};

private:
    c10::Allocator* original_;
    std::thread::id submitting_thread_;
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
failed_flush_count = 0
close_count = 0
clear_on_flush = False
accepted_ids = []
accepted_hidden = []
accepted_last_hidden = []
accepted_capture_metadata = []
fail_init = False
fail_put = False
flush_error = None
async_error_mode = ''
close_error = False
query_mode = ''
query_error_after_put = 0
cleanup_mode = ''
cleanup_errors = {}
last_query_bases = []
reclaim_count = 0
removed_keys = []
gpu_direct = False
send_buffer_present = False
send_buffer_size = 0
configured_pool_size = 8
host_capacity = None
host_reported_pool_size = 8
fail_put_at = 0
bad_metadata_at = 0
bad_metadata_kind = 'shape'
flush_fail_after_put = 0
fail_flush_once_at_put = 0
chunk_events = []
chunk_ids = []
chunk_lengths = []
put_request_ids = []
put_started = threading.Event()
allow_put = threading.Event()
allow_put.set()

class Config:
    hidden_dim = 2
    num_aux_layers = 2
    store_key_namespace = 'PublisherTest'
    @property
    def async_put_pool_size(self):
        return configured_pool_size
    @classmethod
    def from_env(cls):
        return cls()
    def make_store_key(self, request_id):
        return f'rtp_PublisherTest_{request_id}'

class Store:
    def __init__(self, config):
        self.config = config
        self._gpu_direct_available = gpu_direct
        self._gpu_send_buffer = types.SimpleNamespace(size=send_buffer_size) if send_buffer_present else None
        self._async_put_manager = object()
        self._host_buffer_pool = types.SimpleNamespace(
            pool_size=host_reported_pool_size,
            _buffers=[object() for _ in range(host_capacity or 0)])
    def setup(self, device):
        if fail_init:
            raise RuntimeError('injected store setup failure')
    def warmup_rdma(self):
        pass
    def put_batch(self, **payload):
        global put_count
        put_count += 1
        chunk_events.append('put')
        chunk_ids.append(payload['batch_id'])
        put_request_ids.append(list(payload['request_ids']))
        chunk_lengths.append([tensor.shape[0] for tensor in payload['hidden_states']])
        if not (self._gpu_direct_available and self._gpu_send_buffer is not None):
            capacity = len(getattr(self._host_buffer_pool, '_buffers', [])) or self._host_buffer_pool.pool_size
            if len(payload['request_ids']) > capacity:
                raise ValueError('host batch exceeds actual pool capacity; no work accepted')
        put_started.set()
        if not allow_put.wait(timeout=10):
            raise RuntimeError('test did not unblock PUT')
        accepted_ids.extend(tensor.tolist() for tensor in payload['input_ids'])
        accepted_hidden.extend(tensor.float().tolist() for tensor in payload['hidden_states'])
        accepted_last_hidden.extend(tensor.float().tolist() for tensor in payload['last_hidden_states'])
        accepted_capture_metadata.append(payload['capture_metadata'])
        for key in payload['request_ids']:
            store_keys.update([key + '_hs', key + '_ids', key + '_lhs', key + '_ready'])
            if 'hidden_states_scale' in payload:
                store_keys.add(key + '_hss')
            if fail_put or put_count == fail_put_at:
                raise RuntimeError('injected partial PUT with visible ready')
        names = ['hidden_states', 'input_ids', 'last_hidden_states']
        if 'hidden_states_scale' in payload:
            names.append('hidden_states_scale')
        metadata = [
            {'shapes': {name: tuple(payload[name][i].shape) for name in names},
             'dtypes': {name: payload[name][i].dtype for name in names}}
            for i in range(len(payload['request_ids']))
        ]
        if put_count == bad_metadata_at:
            if bad_metadata_kind == 'count':
                return []
            if bad_metadata_kind == 'item':
                return [None] * len(metadata)
            if bad_metadata_kind == 'dtype':
                metadata[0]['dtypes']['input_ids'] = torch.int32
            elif bad_metadata_kind == 'scale':
                metadata[0]['shapes']['hidden_states_scale'] = (999, 2)
            else:
                metadata[0]['shapes']['hidden_states'] = (999, 4)
        return metadata
    def flush(self):
        global flush_count, failed_flush_count, fail_flush_once_at_put
        flush_count += 1
        chunk_events.append('flush')
        if fail_flush_once_at_put and put_count == fail_flush_once_at_put:
            fail_flush_once_at_put = 0
            raise RuntimeError('injected once-only later flush failure')
        if flush_fail_after_put and put_count >= flush_fail_after_put:
            failed_flush_count += 1
            raise RuntimeError('injected later chunk drain failure')
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
        global last_query_bases
        bases = [key[:-len('_abandoned')] for key in keys if key.endswith('_abandoned')]
        if bases:
            last_query_bases = bases
        if query_mode == 'error' or (query_error_after_put and put_count >= query_error_after_put):
            raise RuntimeError('injected query failure')
        if query_mode == 'incomplete':
            return {}
        if query_mode == 'integer':
            return {key: 0 for key in keys}
        return {key: key in store_keys for key in keys}
    def reclaim_abandoned_capture(self, key):
        global reclaim_count
        reclaim_count += 1
        if key + '_abandoned' not in store_keys:
            return False
        if key in cleanup_errors:
            raise cleanup_errors[key]
        if cleanup_mode == 'error':
            raise RuntimeError('injected source cleanup failure')
        if cleanup_mode == 'incomplete':
            return False
        if cleanup_mode == 'integer':
            return 1
        for suffix in ('_hs', '_hss', '_ids', '_lhs', '_tgt', '_ready', '_abandoned'):
            source = key + suffix
            store_keys.discard(source)
            removed_keys.append(source)
        return True
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

    void configureDirectPlanner(int64_t send_bytes, int64_t marker_bytes) {
        py::gil_scoped_acquire gil;
        py::globals()["send_buffer_size"] = send_bytes;
        py::globals()["marker_size"] = marker_bytes;
        py::exec(R"PY(
gpu_direct = send_buffer_present = True
host_capacity = host_reported_pool_size = configured_pool_size = 1
encoded_metadata = []
# Planner-only boundary: real publisher calls this encoder, but these bytes
# neither validate the JSON codec nor perform a real GPUDirect transfer.
def encode_planner_metadata(plan, actual):
    assert plan == {'capture_layer_ids': [1, 0]}
    assert set(actual) == {'shapes', 'dtypes'}
    encoded_metadata.append(actual)
    return b'm' * marker_size
metadata_module = types.ModuleType('torchspec.transfer.mooncake.capture_metadata')
metadata_module.encode_capture_metadata = encode_planner_metadata
sys.modules[metadata_module.__name__] = metadata_module
)PY");
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

    // Legacy bool assertions still need a serving owner for every accepted token.
    bool reserve(HiddenStateCapturePublisher& publisher, int64_t id, int64_t tokens) {
        auto reservation = publisher.tryReserve(id, tokens);
        if (!reservation) {
            return false;
        }
        reservations_.push_back(std::move(reservation));
        return true;
    }

    std::vector<HiddenStateCaptureReservationPtr>             reservations_;
    std::vector<std::unique_ptr<HiddenStateCapturePublisher>> publishers_;
};

TEST_F(HiddenStateCapturePublisherTest, ReservationDestructionRefundsAndOldGenerationCannotCancelReusedId) {
    auto& p = publisher({1, 1000, 100, 2, 5});
    {
        auto temporary = p.tryReserve(1, 2);
        ASSERT_TRUE(temporary);
        EXPECT_TRUE(temporary->active());
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
    }
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
    for (const bool explicit_cancel : {false, true}) {
        auto old = p.tryReserve(1, 2);
        ASSERT_TRUE(old);
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        {
            py::gil_scoped_acquire gil;
            py::exec("store_keys.clear()");
        }
        ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
        // A was consumed, not cancelled: its callback must really execute now.
        ASSERT_TRUE(old->active());
        auto current = p.tryReserve(1, 3);
        ASSERT_TRUE(current);
        if (explicit_cancel) {
            old->cancel();
            EXPECT_FALSE(old->active());
        }
        old.reset();
        EXPECT_TRUE(current->active());
        EXPECT_EQ(p.quotaSnapshot().reserved_samples, 1);
        EXPECT_EQ(p.quotaSnapshot().tokens, 3);
        EXPECT_EQ(p.quotaSnapshot().bytes, 60);
        current->cancel();
        current->cancel();
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
    }
}

TEST_F(HiddenStateCapturePublisherTest, OrphanTokenAndConcurrentCancelDuringPublisherDestructionAreSafe) {
    auto& p      = publisher();
    auto  orphan = p.tryReserve(1, 2);
    ASSERT_TRUE(orphan);
    auto owned = std::move(publishers_.back());
    publishers_.pop_back();
    owned.reset();
    EXPECT_TRUE(orphan->active());
    EXPECT_NO_THROW(orphan->cancel());
    EXPECT_NO_THROW(orphan.reset());

    auto& concurrent = publisher();
    auto  token      = concurrent.tryReserve(2, 3);
    ASSERT_TRUE(token);
    auto destruct_only = concurrent.tryReserve(3, 2);
    ASSERT_TRUE(destruct_only);
    owned = std::move(publishers_.back());
    publishers_.pop_back();
    std::promise<void>       start;
    auto                     ready = start.get_future().share();
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([token, ready]() mutable {
            ready.wait();
            token->cancel();
            token.reset();
        });
    }
    threads.emplace_back([reservation = std::move(destruct_only), ready]() mutable {
        ready.wait();
        reservation.reset();
    });
    threads.emplace_back([publisher = std::move(owned), ready]() mutable {
        ready.wait();
        publisher.reset();
    });
    start.set_value();
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_FALSE(token->active());
    EXPECT_NO_THROW(token.reset());
}

TEST_F(HiddenStateCapturePublisherTest, ConcurrentTokenCancellationInvokesCallbackExactlyOnce) {
    std::atomic<int>         callbacks{0};
    auto                     token = std::make_shared<HiddenStateCaptureReservation>([&] { ++callbacks; });
    std::promise<void>       start;
    auto                     ready = start.get_future().share();
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([token, ready, i]() mutable {
            ready.wait();
            if (i % 2 == 0) {
                token->cancel();
            }
            token.reset();
        });
    }
    start.set_value();
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(callbacks.load(), 1);
    EXPECT_FALSE(token->active());
    token.reset();
    EXPECT_EQ(callbacks.load(), 1);
}

TEST_F(HiddenStateCapturePublisherTest, ThrowingTokenCancellationRemainsInactiveAndDoesNotEscapeDestruction) {
    for (const auto& error : {std::make_exception_ptr(std::bad_alloc()), std::make_exception_ptr(42)}) {
        int callbacks = 0;
        {
            HiddenStateCaptureReservation token([&] {
                ++callbacks;
                std::rethrow_exception(error);
            });
            EXPECT_NO_THROW(token.cancel());
            EXPECT_FALSE(token.active());
            EXPECT_NO_THROW(token.cancel());
        }
        EXPECT_EQ(callbacks, 1);
        {
            HiddenStateCaptureReservation token([&] {
                ++callbacks;
                std::rethrow_exception(error);
            });
        }
        EXPECT_EQ(callbacks, 2);
    }
}

TEST_F(HiddenStateCapturePublisherTest, SubmittedReservationTokenCannotRefundStoreOwnedPayload) {
    auto& p     = publisher({1, 1000, 100, 1, 5});
    auto  token = p.tryReserve(1, 2);
    ASSERT_TRUE(token);
    ASSERT_TRUE(submit(p, 1));
    token->cancel();
    token.reset();
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    EXPECT_EQ(evalInt("put_count"), 1);
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.add('rtp_PublisherTest_1_abandoned')");
    }
    ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
}

TEST_F(HiddenStateCapturePublisherTest, LaterChunkMetadataMustMatchEveryActualTensorIncludingFp8Scale) {
    for (const auto& kind : {"count", "item", "shape", "dtype", "scale"}) {
        SCOPED_TRACE(kind);
        {
            py::gil_scoped_acquire gil;
            py::exec(R"PY(
gpu_direct = False
send_buffer_present = False
host_capacity = 1
put_count = flush_count = 0
bad_metadata_at = 2
store_keys.clear()
chunk_events.clear()
fp8.quantize_aux_hidden_states = lambda hidden, layers: (
    hidden.to(torch.float8_e4m3fn), torch.ones((hidden.shape[0], layers), dtype=torch.float32))
)PY");
            py::globals()["bad_metadata_kind"] = kind;
        }
        auto& p = publisher({}, true, true, {7, 2});
        ASSERT_TRUE(reserve(p, 1, 1));
        ASSERT_TRUE(reserve(p, 2, 2));
        ASSERT_TRUE(p.trySubmit(torch::ones({3, 6}, torch::kFloat16),
                                torch::arange(3, torch::kInt32),
                                torch::tensor({1, 2}, torch::kInt32),
                                torch::tensor({1, 2}, torch::kInt64),
                                3,
                                "FP8 chunk metadata"));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), 2);
        EXPECT_GT(p.stats().failure_count, 0);
        EXPECT_EQ(p.quotaSnapshot().samples, 2);
        EXPECT_EQ(evalInt("chunk_events[:4] == ['put', 'flush', 'put', 'flush']"), 1);
        {
            py::gil_scoped_acquire gil;
            py::exec("store_keys.clear()");
        }
        ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
        p.flushAndClose();
    }
}

TEST_F(HiddenStateCapturePublisherTest, ActualHostCapacityBoundsChunksAndPreservesVariableSlicesAndLayerOrder) {
    // Includes N=P, N=P+1, actual capacity below/above config, a pool-size-only
    // fallback and receive-only GPU capability without a send buffer.
    struct Case {
        int  capacity;
        int  configured;
        int  requests;
        bool direct;
        bool send;
        bool buffers;
    };
    const std::vector<Case> cases{{1, 8, 3, false, false, true},
                                  {2, 2, 2, false, false, true},
                                  {2, 2, 3, false, false, true},
                                  {3, 1, 3, false, false, true},
                                  {1, 8, 3, true, false, true},
                                  {2, 8, 3, false, false, false}};
    for (const bool fp8_mode : {false, true}) {
        for (const auto& item : cases) {
            SCOPED_TRACE(::testing::Message() << "capacity=" << item.capacity << " config=" << item.configured
                                              << " requests=" << item.requests << " direct=" << item.direct << " send="
                                              << item.send << " buffers=" << item.buffers << " fp8=" << fp8_mode);
            {
                py::gil_scoped_acquire gil;
                auto                   globals     = py::globals();
                globals["gpu_direct"]              = item.direct;
                globals["send_buffer_present"]     = item.send;
                globals["configured_pool_size"]    = item.configured;
                globals["host_reported_pool_size"] = item.buffers ? 9 : item.capacity;
                globals["host_capacity"]           = item.buffers ? py::cast(item.capacity) : py::none();
                py::exec(R"PY(
store_keys.clear()
accepted_ids.clear()
accepted_hidden.clear()
accepted_last_hidden.clear()
accepted_capture_metadata.clear()
chunk_events.clear()
chunk_ids.clear()
chunk_lengths.clear()
# This test covers publisher dtype/scale plumbing, not quantization accuracy.
fp8.quantize_aux_hidden_states = lambda hidden, layers: (
    hidden.to(torch.float8_e4m3fn), torch.ones((hidden.shape[0], layers), dtype=torch.float32))
)PY");
            }
            auto&                p = publisher({}, true, fp8_mode, {7, 2});
            std::vector<int32_t> lengths;
            std::vector<int64_t> requests;
            int                  total = 0;
            for (int i = 0; i < item.requests; ++i) {
                lengths.push_back(i + 1);
                requests.push_back(100 + i);
                total += i + 1;
                ASSERT_TRUE(reserve(p, 100 + i, i + 1));
            }
            // Small integral values are exactly representable in BF16 and E4M3.
            auto packed = torch::arange(total, torch::kFloat16).unsqueeze(1) + torch::arange(6, torch::kFloat16);
            ASSERT_TRUE(p.trySubmit(packed,
                                    torch::arange(total, torch::kInt32),
                                    torch::tensor(lengths, torch::kInt32),
                                    torch::tensor(requests, torch::kInt64),
                                    total,
                                    "host capacity"));
            ASSERT_TRUE(p.flushPending());
            EXPECT_EQ(p.stats().failure_count, 0);
            EXPECT_EQ(p.quotaSnapshot().published_samples, item.requests);
            const int chunk_size = item.direct && item.send ? item.requests : item.capacity;
            const int chunks     = (item.requests + chunk_size - 1) / chunk_size;
            EXPECT_EQ(evalInt("len(chunk_ids)"), chunks);
            EXPECT_EQ(evalInt("len(set(chunk_ids)) == len(chunk_ids)"), 1);
            {
                py::gil_scoped_acquire gil;
                py::globals()["expected_requests"]   = item.requests;
                py::globals()["expected_chunk_size"] = chunk_size;
                py::exec(R"PY(
lengths = list(range(1, expected_requests + 1))
assert chunk_lengths == [lengths[i:i + expected_chunk_size] for i in range(0, len(lengths), expected_chunk_size)]
assert chunk_events[:2 * len(chunk_lengths)] == ['put', 'flush'] * len(chunk_lengths)
assert accepted_capture_metadata == [{'capture_layer_ids': [7, 2]}] * len(chunk_lengths)
offset = 0
for length, ids, hidden, last in zip(lengths, accepted_ids, accepted_hidden, accepted_last_hidden):
    assert ids == list(range(offset, offset + length))
    assert hidden == [[float(row + col) for col in range(4)] for row in range(offset, offset + length)]
    assert last == [[float(row + col) for col in (4, 5)] for row in range(offset, offset + length)]
    offset += length
store_keys.clear()
)PY");
            }
            ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
            p.flushAndClose();
        }
    }
}

TEST_F(HiddenStateCapturePublisherTest, ActualDirectPlannerIgnoresHostPoolCountAndReusesPayloadCapacity) {
    // BF16 requests occupy 20, 40 and 60 bytes. Each fits the reusable 60-byte
    // send buffer; their sum need not fit. All three 20-byte markers fit exactly.
    configureDirectPlanner(60, 20);
    auto& p = publisher();
    for (int64_t id = 1; id <= 3; ++id) {
        ASSERT_TRUE(reserve(p, id, id));
    }
    ASSERT_TRUE(p.trySubmit(torch::ones({6, 6}, torch::kFloat16),
                            torch::arange(6, torch::kInt32),
                            torch::tensor({1, 2, 3}, torch::kInt32),
                            torch::tensor({1, 2, 3}, torch::kInt64), 6, "direct payload reuse"));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.stats().failure_count, 0);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 3);
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("chunk_lengths == [[1, 2, 3]]"), 1);
    EXPECT_EQ(evalInt("[m['shapes']['hidden_states'][0] for m in encoded_metadata] == [1, 2, 3]"), 1);
    EXPECT_EQ(evalInt("chunk_events[:2] == ['put', 'flush']"), 1);
}

TEST_F(HiddenStateCapturePublisherTest, ActualDirectPlannerChunksByAggregateMarkerBytes) {
    // Same per-request payload limit, but only two 30-byte markers fit together.
    // Host capacity is one: a two-request first chunk proves this is direct.
    configureDirectPlanner(60, 30);
    auto& p = publisher();
    for (int64_t id = 1; id <= 3; ++id) {
        ASSERT_TRUE(reserve(p, id, id));
    }
    ASSERT_TRUE(p.trySubmit(torch::ones({6, 6}, torch::kFloat16),
                            torch::arange(6, torch::kInt32),
                            torch::tensor({1, 2, 3}, torch::kInt32),
                            torch::tensor({1, 2, 3}, torch::kInt64), 6, "direct marker chunks"));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.stats().failure_count, 0);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 3);
    EXPECT_EQ(evalInt("put_count"), 2);
    EXPECT_EQ(evalInt("chunk_lengths == [[1, 2], [3]]"), 1);
    EXPECT_EQ(evalInt("len(set(chunk_ids))"), 2);
    EXPECT_EQ(evalInt("chunk_events[:4] == ['put', 'flush', 'put', 'flush']"), 1);
    EXPECT_EQ(evalInt("accepted_ids == [[0], [1, 2], [3, 4, 5]]"), 1);
    EXPECT_EQ(evalInt("{m['shapes']['hidden_states'][0] for m in encoded_metadata} == {1, 2, 3}"), 1);
}

TEST_F(HiddenStateCapturePublisherTest, ActualDirectPlannerRejectsOversizedRequestBeforePutAndRefundsLocally) {
    for (const bool oversized_marker : {false, true}) {
        SCOPED_TRACE(oversized_marker);
        // Two tokens require 40 payload bytes: first reject a 39-byte buffer,
        // then fit the payload exactly but reject a single 41-byte ready marker.
        configureDirectPlanner(oversized_marker ? 40 : 39, oversized_marker ? 41 : 8);
        auto& p = publisher();
        ASSERT_TRUE(reserve(p, 1, 2));
        auto packed = torch::ones({2, 6}, torch::kFloat16);
        c10::weak_intrusive_ptr<c10::TensorImpl> source_owner(packed.getIntrusivePtr());
        ASSERT_TRUE(p.trySubmit(packed, torch::arange(2, torch::kInt32),
                                torch::tensor({2}, torch::kInt32), torch::tensor({1}, torch::kInt64),
                                2, "direct oversized request"));
        packed = torch::Tensor();
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("len(encoded_metadata)"), 1);
        EXPECT_EQ(evalInt("put_count"), 0);
        EXPECT_EQ(evalInt("len(store_keys)"), 0);
        EXPECT_GT(p.stats().failure_count, 0);
        EXPECT_TRUE(p.shouldPublish());
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_EQ(p.quotaSnapshot().bytes, 0);
        EXPECT_EQ(p.quotaSnapshot().tokens, 0);
        EXPECT_TRUE(source_owner.expired());
        p.flushAndClose();
    }
}

TEST_F(HiddenStateCapturePublisherTest, LaterChunkFailuresNeverInheritEarlierFlushAndOnlySafeSuffixIsRefunded) {
    for (const bool failed_drain : {false, true}) {
        for (const int fault : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message() << "failed_drain=" << failed_drain << " fault=" << fault);
            {
                py::gil_scoped_acquire gil;
                py::exec(R"PY(
gpu_direct = False
send_buffer_present = False
host_capacity = 1
configured_pool_size = 8
store_keys.clear()
removed_keys.clear()
chunk_events.clear()
put_count = flush_count = 0
)PY");
                py::globals()["fail_put_at"]          = fault == 0 ? 2 : 0;
                py::globals()["bad_metadata_at"]      = fault == 1 ? 2 : 0;
                py::globals()["flush_fail_after_put"] = failed_drain ? 2 : 0;
                // One failure after a successful PUT exercises recovery drain;
                // failed_drain keeps every later attempt unconfirmed instead.
                py::globals()["fail_flush_once_at_put"] = fault == 2 ? 2 : 0;
                py::globals()["query_error_after_put"] = 2;
            }
            auto& p = publisher({3, 1000, 100, 2, 5});
            ASSERT_TRUE(reserve(p, 1, 1));
            ASSERT_TRUE(reserve(p, 2, 2));
            ASSERT_TRUE(reserve(p, 3, 3));
            auto                                     packed = torch::arange(36, torch::kFloat16).reshape({6, 6});
            c10::weak_intrusive_ptr<c10::TensorImpl> source_owner(packed.getIntrusivePtr());
            ASSERT_TRUE(p.trySubmit(packed,
                                    torch::arange(6, torch::kInt32),
                                    torch::tensor({1, 2, 3}, torch::kInt32),
                                    torch::tensor({1, 2, 3}, torch::kInt64),
                                    6,
                                    "later chunk"));
            packed = torch::Tensor();
            EXPECT_EQ(p.flushPending(), !failed_drain);
            EXPECT_EQ(evalInt("put_count"), 2);
            EXPECT_GT(p.stats().failure_count, 0);
            EXPECT_EQ(evalInt("chunk_events[:3] == ['put', 'flush', 'put']"), 1);
            EXPECT_EQ(evalInt("all('PublisherTest_3' not in key for key in store_keys)"), 1);
            // All post-PUT queries fail here. Refunding the untouched suffix
            // must follow local completion, not a fictitious missing-key reply.
            EXPECT_EQ(p.quotaSnapshot().samples, failed_drain ? 3 : 2);
            EXPECT_EQ(p.quotaSnapshot().tokens, failed_drain ? 6 : 3);
            EXPECT_EQ(source_owner.expired(), !failed_drain);
            {
                py::gil_scoped_acquire gil;
                py::exec("query_error_after_put = 0; store_keys.add('rtp_PublisherTest_2_abandoned')");
            }
            EXPECT_EQ(p.flushPending(), !failed_drain);
            if (failed_drain) {
                // Earlier chunk's flush cannot authorize freeing this task's
                // tensor owner, reclaiming the failed PUT or refunding its suffix.
                EXPECT_FALSE(source_owner.expired());
                EXPECT_EQ(p.quotaSnapshot().samples, 3);
                EXPECT_EQ(p.quotaSnapshot().tokens, 6);
                EXPECT_EQ(evalInt("len(removed_keys)"), 0);
                EXPECT_EQ(evalInt("'rtp_PublisherTest_2_abandoned' in store_keys"), 1);
            } else {
                ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 1; }));
                EXPECT_TRUE(source_owner.expired());
                EXPECT_EQ(p.quotaSnapshot().tokens, 1);
                EXPECT_EQ(p.quotaSnapshot().bytes, 20);
                EXPECT_EQ(
                    evalInt(
                        "store_keys == {'rtp_PublisherTest_1' + suffix for suffix in ('_hs', '_ids', '_lhs', '_ready')}"),
                    1);
                EXPECT_EQ(evalInt("all('PublisherTest_1' not in key for key in removed_keys)"), 1);
            }
            // Stop injecting failures for teardown; an unconfirmed task must
            // still retain its owners rather than treating shutdown as a fence.
            {
                py::gil_scoped_acquire gil;
                py::exec("flush_fail_after_put = 0; store_keys.clear()");
            }
            p.flushAndClose();
        }
    }
}

TEST_F(HiddenStateCapturePublisherTest, SubmittedCompletionRetryDrainsWithoutReputAndRecoversCapture) {
    for (const bool partial_put : {true, false}) {
        SCOPED_TRACE(partial_put);
        {
            py::gil_scoped_acquire gil;
            py::exec(R"PY(
host_capacity = 1
put_count = flush_count = failed_flush_count = 0
chunk_events.clear()
put_request_ids.clear()
removed_keys.clear()
store_keys.clear()
flush_fail_after_put = 2
query_error_after_put = 2
)PY");
            py::globals()["fail_put_at"] = partial_put ? 2 : 0;
        }
        // Explicit maintenance calls control retries; no short-period polling.
        auto& p = publisher({3, 1000, 100, 1, 60000});
        for (int64_t id = 1; id <= 3; ++id) {
            ASSERT_TRUE(reserve(p, id, id));
        }
        auto packed = torch::ones({6, 6}, torch::kFloat16);
        c10::weak_intrusive_ptr<c10::TensorImpl> owner(packed.getIntrusivePtr());
        ASSERT_TRUE(p.trySubmit(packed, torch::arange(6, torch::kInt32),
                                torch::tensor({1, 2, 3}, torch::kInt32),
                                torch::tensor({1, 2, 3}, torch::kInt64), 6, "retry submitted completion"));
        packed = torch::Tensor();
        EXPECT_FALSE(p.flushPending());
        EXPECT_FALSE(p.flushPending());
        ASSERT_GE(evalInt("failed_flush_count"), 2);
        EXPECT_EQ(evalInt("put_count"), 2);
        EXPECT_EQ(evalInt("chunk_events[:3] == ['put', 'flush', 'put']"), 1);
        EXPECT_FALSE(owner.expired());
        EXPECT_EQ(p.quotaSnapshot().samples, 3);
        EXPECT_EQ(p.quotaSnapshot().tokens, 6);
        EXPECT_EQ(p.quotaSnapshot().bytes, 120);
        EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
        EXPECT_TRUE(p.shouldPublish());
        {
            py::gil_scoped_acquire gil;
            py::exec("store_keys.add('rtp_PublisherTest_2_abandoned')");
        }
        EXPECT_FALSE(p.flushPending());
        EXPECT_EQ(evalInt("len(removed_keys)"), 0);
        EXPECT_FALSE(owner.expired());
        const auto attempts = evalInt("put_count");
        const auto flushes = evalInt("flush_count");
        {
            py::gil_scoped_acquire gil;
            py::exec("flush_fail_after_put = 0");
        }
        ASSERT_TRUE(p.flushPending());
        EXPECT_GT(evalInt("flush_count"), flushes);
        EXPECT_EQ(evalInt("put_count"), attempts);
        EXPECT_TRUE(owner.expired());
        // Source queries still fail, so only the never-attempted suffix can be
        // refunded. A completion retry cannot treat the two attempted IDs as new.
        EXPECT_EQ(p.quotaSnapshot().samples, 2);
        EXPECT_EQ(p.quotaSnapshot().tokens, 3);
        EXPECT_EQ(p.quotaSnapshot().bytes, 60);
        EXPECT_EQ(evalInt("put_request_ids == [['rtp_PublisherTest_1'], ['rtp_PublisherTest_2']]"), 1);
        EXPECT_TRUE(p.shouldPublish());
        auto next = p.tryReserve(4, 1);
        ASSERT_TRUE(next);
        next->cancel();
        {
            py::gil_scoped_acquire gil;
            py::exec(R"PY(
query_error_after_put = 0
cleanup_mode = 'integer'
store_keys.add('rtp_PublisherTest_1_abandoned')
)PY");
        }
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(p.quotaSnapshot().samples, 2);
        EXPECT_EQ(evalInt("len(removed_keys)"), 0);
        {
            py::gil_scoped_acquire gil;
            py::exec("cleanup_mode = ''");
        }
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_EQ(p.quotaSnapshot().bytes, 0);
        EXPECT_EQ(p.quotaSnapshot().tokens, 0);
        EXPECT_EQ(evalInt("len(store_keys)"), 0);
        EXPECT_EQ(evalInt("put_count"), attempts);
        EXPECT_EQ(evalInt("all('PublisherTest_3' not in key for key in removed_keys)"), 1);
        EXPECT_EQ(p.stats().broken_rejection_count, 0);
        p.flushAndClose();
    }
}

TEST_F(HiddenStateCapturePublisherTest, CompletionRetryRefundsQueueDroppedOwnerWithoutPublishingIt) {
    {
        py::gil_scoped_acquire gil;
        py::exec(R"PY(
host_capacity = 1
fail_put_at = 2
flush_fail_after_put = 2
allow_put.clear()
)PY");
    }
    auto& p = publisher({5, 1000, 100, 1, 60000});
    for (int64_t id = 1; id <= 5; ++id) {
        ASSERT_TRUE(reserve(p, id, id <= 3 ? id : 1));
    }
    auto packed = torch::ones({6, 6}, torch::kFloat16);
    auto queued = torch::ones({1, 6}, torch::kFloat16);
    auto dropped = torch::ones({1, 6}, torch::kFloat16);
    c10::weak_intrusive_ptr<c10::TensorImpl> owner(packed.getIntrusivePtr());
    c10::weak_intrusive_ptr<c10::TensorImpl> queued_owner(queued.getIntrusivePtr());
    c10::weak_intrusive_ptr<c10::TensorImpl> dropped_owner(dropped.getIntrusivePtr());
    ASSERT_TRUE(p.trySubmit(packed, torch::arange(6, torch::kInt32),
                            torch::tensor({1, 2, 3}, torch::kInt32),
                            torch::tensor({1, 2, 3}, torch::kInt64), 6, "blocked submitted completion"));
    {
        py::gil_scoped_acquire gil;
        // Event.wait releases the GIL; the worker is now blocked in the real PUT
        // callback, making the pending-queue and dropped paths deterministic.
        ASSERT_TRUE(py::eval("put_started.wait(timeout=10)").cast<bool>());
    }
    ASSERT_TRUE(p.trySubmit(queued, torch::tensor({6}, torch::kInt32),
                            torch::tensor({1}, torch::kInt32), torch::tensor({4}, torch::kInt64),
                            1, "queued behind blocked PUT"));
    EXPECT_FALSE(p.trySubmit(dropped, torch::tensor({7}, torch::kInt32),
                             torch::tensor({1}, torch::kInt32), torch::tensor({5}, torch::kInt64),
                             1, "queue-full dropped completion"));
    packed = queued = dropped = torch::Tensor();
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 5);
    EXPECT_FALSE(owner.expired());
    EXPECT_FALSE(queued_owner.expired());
    EXPECT_FALSE(dropped_owner.expired());
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.set()");
    }
    EXPECT_FALSE(p.flushPending());
    EXPECT_FALSE(p.flushPending());
    ASSERT_GE(evalInt("failed_flush_count"), 2);
    EXPECT_TRUE(p.shouldPublish());
    EXPECT_FALSE(owner.expired());
    EXPECT_FALSE(queued_owner.expired());
    // CPU dropped work has a constructible local-completion path, but no PUT
    // to drain. It must refund independently of other tasks' unresolved drains.
    EXPECT_TRUE(dropped_owner.expired());
    EXPECT_EQ(p.quotaSnapshot().samples, 4);
    EXPECT_EQ(p.quotaSnapshot().tokens, 7);
    EXPECT_EQ(evalInt("put_request_ids == [['rtp_PublisherTest_1'], ['rtp_PublisherTest_2'], ['rtp_PublisherTest_4']]"), 1);
    const auto attempts = evalInt("put_count");
    EXPECT_EQ(attempts, 3);
    {
        py::gil_scoped_acquire gil;
        py::exec("flush_fail_after_put = 0; query_mode = 'error'");
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_TRUE(owner.expired());
    EXPECT_TRUE(queued_owner.expired());
    EXPECT_EQ(p.quotaSnapshot().samples, 3);
    EXPECT_EQ(p.quotaSnapshot().tokens, 4);
    EXPECT_EQ(p.quotaSnapshot().bytes, 80);
    EXPECT_EQ(evalInt("put_count"), attempts);
    {
        py::gil_scoped_acquire gil;
        py::exec(R"PY(
query_mode = ''
store_keys.update('rtp_PublisherTest_' + str(i) + '_abandoned' for i in (1, 2, 4))
)PY");
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
    EXPECT_EQ(p.quotaSnapshot().bytes, 0);
    EXPECT_EQ(p.quotaSnapshot().tokens, 0);
    EXPECT_EQ(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(evalInt("put_count"), attempts);
    EXPECT_EQ(evalInt("all('PublisherTest_3' not in key and 'PublisherTest_5' not in key for key in removed_keys)"), 1);
    EXPECT_TRUE(p.shouldPublish());
}

TEST_F(HiddenStateCapturePublisherTest, PayloadLimitsAndIdempotentCancellation) {
    auto& p = publisher({2, 100, 5, 1, 5});
    EXPECT_FALSE(p.tryReserve(-1, 1));
    EXPECT_FALSE(p.tryReserve(1, 0));
    EXPECT_FALSE(p.tryReserve(1, std::numeric_limits<int64_t>::max()));
    ASSERT_TRUE(reserve(p, 1, 2));
    EXPECT_FALSE(p.tryReserve(1, 2));
    ASSERT_TRUE(reserve(p, 2, 3));
    auto quota = p.quotaSnapshot();
    EXPECT_EQ(quota.samples, 2);
    EXPECT_EQ(quota.bytes, 100);
    EXPECT_EQ(quota.tokens, 5);
    EXPECT_FALSE(p.tryReserve(3, 1));
    p.discardReserved(1);
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 60);
    EXPECT_EQ(p.quotaSnapshot().tokens, 3);
    EXPECT_TRUE(reserve(p, 3, 2));
    EXPECT_FALSE(p.tryReserve(4, 1));
}

TEST_F(HiddenStateCapturePublisherTest, EachLimitIndependentlyRejectsWithoutChangingLedger) {
    for (const auto limits : {HiddenStateCaptureLimits{1, 1000, 100, 1, 5},
                              HiddenStateCaptureLimits{10, 40, 100, 1, 5},
                              HiddenStateCaptureLimits{10, 1000, 2, 1, 5}}) {
        auto& p = publisher(limits);
        ASSERT_TRUE(reserve(p, 1, 2));
        EXPECT_FALSE(p.tryReserve(2, 1));
        EXPECT_EQ(p.quotaSnapshot().samples, 1);
        EXPECT_EQ(p.quotaSnapshot().bytes, 40);
        EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    }
}

TEST_F(HiddenStateCapturePublisherTest, Fp8IncludesScaleAndBf16FinalInBudget) {
    auto& p = publisher({1, 48, 2, 1, 5}, true, true);
    ASSERT_TRUE(reserve(p, 1, 2));
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
    }
}

TEST_F(HiddenStateCapturePublisherTest, ConcurrentReservationsNeverOversubscribe) {
    auto&                    p = publisher({4, 80, 4, 2, 5});
    std::atomic<int>         accepted{0};
    std::vector<std::thread> threads;
    std::vector<HiddenStateCaptureReservationPtr> reservations(16);
    for (int id = 0; id < 16; ++id) {
        threads.emplace_back([&p, &accepted, &reservations, id] {
            reservations[id] = p.tryReserve(id, 1);
            accepted += bool(reservations[id]);
        });
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
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    ASSERT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    p.discardReserved(1);
    p.discardReserved(1);
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
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 0);
    EXPECT_EQ(p.quotaSnapshot().tokens, 0);
}

TEST_F(HiddenStateCapturePublisherTest, WorkerOwnsPayloadAndSnapshotsMutableHostMetadata) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(reserve(p, 2, 3));
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
    ASSERT_TRUE(reserve(p, 1, 2));
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
        p.discardReserved(1);
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
        ASSERT_TRUE(reserve(p, request_id, 2));
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
        ASSERT_TRUE(reserve(p, 1, 2));
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), 0);
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_FALSE(p.shouldPublish());
    }
}

TEST_F(HiddenStateCapturePublisherTest, ServingCancellationAfterSubmitDoesNotSuppressPublication) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(reserve(p, 1, 2));
    {
        // The worker cannot enter Python until this scope exits. submit() also
        // drops its caller-owned tensors before serving cancellation occurs.
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(submit(p, 1));
        p.discardReserved(1);
        p.discardReserved(1);
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
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, ServingCancellationOfOneSubmittedRequestDoesNotDropBatch) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(reserve(p, 2, 3));
    {
        py::gil_scoped_acquire gil;
        ASSERT_TRUE(p.trySubmit(torch::arange(30, torch::kFloat16).reshape({5, 6}),
                                torch::arange(5, torch::kInt32),
                                torch::tensor({2, 3}, torch::kInt32),
                                torch::tensor({1, 2}, torch::kInt64),
                                5,
                                "serving cancellation"));
        p.discardReserved(1);
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
        ASSERT_TRUE(reserve(p, first_id, 2));
        ASSERT_TRUE(submit(p, first_id));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), previous_puts);
        EXPECT_EQ(p.stats().failure_count, 1);
        EXPECT_EQ(p.quotaSnapshot().samples, 0);
        EXPECT_TRUE(p.shouldPublish());

        {
            py::gil_scoped_acquire gil;
            py::exec("async_error_mode = ''");
        }
        ASSERT_TRUE(reserve(p, first_id + 1, 2));
        ASSERT_TRUE(submit(p, first_id + 1));
        ASSERT_TRUE(p.flushPending());
        EXPECT_EQ(evalInt("put_count"), previous_puts + 1);
        EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
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
    ASSERT_TRUE(reserve(p, 1, 2));
    py::gil_scoped_acquire gil;
    py::exec("clear_on_flush = True");
    ASSERT_TRUE(submit(p, 1));
    p.discardReserved(1);
    // close must release our GIL so the delayed worker can publish, flush,
    // confirm absence, and close Python resources before the join completes.
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("flush_count"), 1);
    EXPECT_EQ(evalInt("close_count"), 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, PublishedSourceRetainsStoreOwnershipDuringShutdown) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_GT(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);

    p.flushAndClose();
    EXPECT_EQ(evalInt("close_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
}

TEST_F(HiddenStateCapturePublisherTest, QueryFailureOrMalformedAnswerIsNotAbsence) {
    auto& p = publisher({4, 1000, 100, 2, 5});
    ASSERT_TRUE(reserve(p, 1, 2));
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
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(reserve(p, 2, 2));
    ASSERT_TRUE(reserve(p, 3, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(waitFor([&] { return evalInt("put_started.is_set()") != 0; }));
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 3);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 0);
    ASSERT_TRUE(submit(p, 2));
    EXPECT_FALSE(submit(p, 3));
    EXPECT_EQ(evalInt("flush_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    p.discardReserved(3);
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
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_GT(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    p.discardReserved(1);
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    EXPECT_FALSE(p.tryReserve(2, 2));
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.clear()");
    }
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
}

TEST_F(HiddenStateCapturePublisherTest, AbandonmentBeforeLatePublishWaitsForProducerCompletion) {
    auto& p = publisher({2, 1000, 100, 2, 5});
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.clear()");
    }
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(waitFor([&] { return evalInt("put_started.is_set()") != 0; }));
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.add('rtp_PublisherTest_1_abandoned')");
    }
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(evalInt("reclaim_count"), 0);
    EXPECT_EQ(evalInt("len(removed_keys)"), 0);
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.set()");
    }
    ASSERT_TRUE(p.flushPending());
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(evalInt("removed_keys[-1] == 'rtp_PublisherTest_1_abandoned'"), 1);
}

TEST_F(HiddenStateCapturePublisherTest, AbandonedPartialPutCleanupRetriesAndDoesNotDeleteSuccessfulPeer) {
    auto& p = publisher({2, 1000, 100, 2, 5});
    ASSERT_TRUE(reserve(p, 2, 2));
    ASSERT_TRUE(submit(p, 2));
    ASSERT_TRUE(p.flushPending());
    {
        py::gil_scoped_acquire gil;
        py::exec("fail_put = True; cleanup_mode = 'error'");
    }
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("reclaim_count"), 0);
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.add('rtp_PublisherTest_1_abandoned')");
    }
    ASSERT_TRUE(waitFor([&] { return evalInt("reclaim_count") > 0; }));
    EXPECT_EQ(p.quotaSnapshot().samples, 2);
    EXPECT_EQ(evalInt("len(removed_keys)"), 0);
    {
        py::gil_scoped_acquire gil;
        py::exec("cleanup_mode = ''");
    }
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 1; }));
    EXPECT_EQ(evalInt("store_keys == {'rtp_PublisherTest_2' + s for s in ('_hs', '_ids', '_lhs', '_ready')}"), 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
}

TEST_F(HiddenStateCapturePublisherTest, OperationalCleanupFailureDoesNotBlockLaterCandidates) {
    auto& p = publisher({3, 1000, 100, 3, 5});
    for (const int64_t id : {1, 2, 3}) {
        ASSERT_TRUE(reserve(p, id, 2));
        ASSERT_TRUE(submit(p, id));
        ASSERT_TRUE(p.flushPending());
    }
    {
        py::gil_scoped_acquire gil;
        // Choose from the actual manifest order, never assume unordered_map
        // iteration order. The failing candidate must precede BOTH peers.
        py::exec(R"PY(
assert len(last_query_bases) == 3
blocked_key, consumed_key, abandoned_key = last_query_bases
cleanup_errors[blocked_key] = RuntimeError('persistent per-key remove failure')
store_keys.add(blocked_key + '_abandoned')
store_keys.add(abandoned_key + '_abandoned')
store_keys.difference_update([key for key in store_keys if key.startswith(consumed_key + '_')])
)PY");
    }
    ASSERT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 1; }));
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    EXPECT_EQ(evalInt("all(key.startswith(blocked_key + '_') for key in store_keys)"), 1);
    EXPECT_EQ(evalInt("blocked_key + '_ready' in store_keys and blocked_key + '_abandoned' in store_keys"), 1);
    EXPECT_EQ(evalInt("abandoned_key + '_abandoned' in removed_keys"), 1);
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
    {
        py::gil_scoped_acquire gil;
        py::exec("cleanup_errors.clear()");
    }
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
}

TEST_F(HiddenStateCapturePublisherTest, FatalCandidateCleanupStopsThePollAndLatchesOwnerFailure) {
    auto& p = publisher({2, 1000, 100, 2, 60000});
    for (const int64_t id : {1, 2}) {
        ASSERT_TRUE(reserve(p, id, 2));
        ASSERT_TRUE(submit(p, id));
        ASSERT_TRUE(p.flushPending());
    }
    {
        py::gil_scoped_acquire gil;
        py::exec(R"PY(
assert len(last_query_bases) == 2
blocked_key, abandoned_key = last_query_bases
cleanup_errors[blocked_key] = MemoryError('fatal per-key cleanup allocation')
store_keys.add(blocked_key + '_abandoned')
store_keys.add(abandoned_key + '_abandoned')
)PY");
    }
    // Shutdown triggers the final maintenance poll without waiting for the
    // long timer, so later worker iterations cannot hide same-poll behavior.
    p.flushAndClose();
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ALLOCATION);
    EXPECT_EQ(p.quotaSnapshot().samples, 2);
    EXPECT_EQ(evalInt("len(removed_keys)"), 0);
}

TEST_F(HiddenStateCapturePublisherTest, AbandonMarkerMustBeReclaimedEvenWhenAllSourceKeysAreMissing) {
    auto& p = publisher({1, 1000, 100, 1, 5});
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    {
        py::gil_scoped_acquire gil;
        py::exec("store_keys.clear(); store_keys.add('rtp_PublisherTest_1_abandoned'); cleanup_mode = 'incomplete'");
    }
    ASSERT_TRUE(waitFor([&] { return evalInt("reclaim_count") > 0; }));
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    {
        py::gil_scoped_acquire gil;
        py::exec("cleanup_mode = ''");
    }
    EXPECT_TRUE(waitFor([&] { return p.quotaSnapshot().samples == 0; }));
    EXPECT_EQ(evalInt("len(store_keys)"), 0);
}

TEST_F(HiddenStateCapturePublisherTest, AbandonedCaptureWithUnconfirmedDrainRetainsSourcesAndQuota) {
    auto& p = publisher({1, 1000, 100, 1, 5});
    {
        py::gil_scoped_acquire gil;
        py::exec("flush_error = RuntimeError('drain incomplete'); store_keys.add('rtp_PublisherTest_1_abandoned')");
    }
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    EXPECT_FALSE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(evalInt("reclaim_count"), 0);
    EXPECT_EQ(evalInt("len(removed_keys)"), 0);
    EXPECT_GT(evalInt("len(store_keys)"), 1);
}

TEST_F(HiddenStateCapturePublisherTest, QuantizerFailureWithoutSourceObjectsRefundsAfterLocalCompletion) {
    auto& p = publisher({1, 1000, 100, 1, 5}, true, true);
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
    EXPECT_TRUE(reserve(p, 2, 2));
}

TEST_F(HiddenStateCapturePublisherTest, FatalQuantizerErrorsStopAdmissionAndSkipAlreadyQueuedCapture) {
    struct FailureCase {
        const char*                 expression;
        HiddenStateCaptureFatalKind kind;
    };
    std::vector<FailureCase> cases{
        {"torch.OutOfMemoryError('injected quantizer allocation failure')", HiddenStateCaptureFatalKind::OUT_OF_MEMORY},
        {"MemoryError('injected quantizer host allocation failure')", HiddenStateCaptureFatalKind::ALLOCATION},
        {"RuntimeError('CUDA error: an illegal memory access was encountered')",
         HiddenStateCaptureFatalKind::ACCELERATOR},
        {"RuntimeError('HIP out of memory. injected failure')", HiddenStateCaptureFatalKind::OUT_OF_MEMORY}};
    {
        py::gil_scoped_acquire gil;
        if (py::hasattr(py::module_::import("torch"), "AcceleratorError")) {
            cases.push_back(
                {"torch.AcceleratorError('injected quantizer device failure')", HiddenStateCaptureFatalKind::ACCELERATOR});
        }
    }
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
        ASSERT_TRUE(reserve(p, 1, 2));
        ASSERT_TRUE(reserve(p, 2, 2));
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
    ASSERT_TRUE(reserve(p, 1, 2));
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
    ASSERT_TRUE(reserve(p, 2, 2));
    ASSERT_TRUE(submit(p, 2));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("quantizer_calls"), 2);
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
}

TEST_F(HiddenStateCapturePublisherTest, NativePayloadConversionsReleaseGilAndRestoreItBeforePythonStoreCalls) {
    auto& p = publisher();
    ASSERT_TRUE(reserve(p, 1, 2));
    int allocations = 0;
    int gil_held = 0;
    {
        WorkerAllocationGilProbe probe;
        ASSERT_TRUE(submit(p, 1));
        ASSERT_TRUE(p.flushPending());
        allocations = probe.worker_allocations.load();
        gil_held = probe.gil_held_allocations.load();
    }
    EXPECT_GE(allocations, 4);
    EXPECT_EQ(gil_held, 0);
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::NONE);
    EXPECT_EQ(p.quotaSnapshot().published_samples, 1);
    EXPECT_EQ(evalInt("accepted_ids == [[0, 1]]"), 1);
}

TEST_F(HiddenStateCapturePublisherTest, MaintenanceAllocationFailureDrainsQueuedCpuTasksWithoutPuttingThem) {
    auto& p = publisher({4, 1000, 100, 2, 60000});
    HiddenStateCapturePublisherTestPeer::failMaintenanceAllocation(p);
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.clear()");
    }
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(waitFor([&] { return evalInt("put_started.is_set()") != 0; }));

    // Real submissions queue behind the blocked first PUT. The last request
    // overflows the queue and is owned solely by the completion ledger.
    std::vector<c10::weak_intrusive_ptr<c10::TensorImpl>> owners;
    for (int64_t id = 2; id <= 4; ++id) {
        ASSERT_TRUE(reserve(p, id, 2));
        auto packed = torch::arange(12, torch::kFloat16).reshape({2, 6});
        owners.emplace_back(packed.getIntrusivePtr());
        EXPECT_EQ(p.trySubmit(packed,
                              torch::arange(2, torch::kInt32),
                              torch::tensor({2}, torch::kInt32),
                              torch::tensor({id}, torch::kInt64),
                              2,
                              "maintenance allocation failure"),
                  id != 4);
    }
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 2);
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.set()");
    }
    // Allocation fails at the native maintenance candidate snapshot boundary,
    // outside per-request catches. Queued tasks have not reached completion
    // maintenance, so the exit path must explicitly drain them without PUT.
    EXPECT_TRUE(p.flushPending());
    p.flushAndClose();
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ALLOCATION);
    EXPECT_EQ(p.fatalState().stage, HiddenStateCaptureFailureStage::SHUTDOWN);
    EXPECT_FALSE(p.tryReserve(5, 2));
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);  // First PUT's consumer-owned keys survive.
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("reclaim_count"), 0);
    EXPECT_EQ(evalInt("close_count"), 0);
    for (const auto& owner : owners) {
        EXPECT_TRUE(owner.expired());
    }
    publishers_.pop_back();
    for (const auto& owner : owners) {
        EXPECT_TRUE(owner.expired());  // No CPU-only unresolved_owner self-cycle.
    }
}

#if USING_CUDA
TEST_F(HiddenStateCapturePublisherTest, MaintenanceAllocationFailureQuarantinesQueuedDeviceOwners) {
    {
        py::gil_scoped_acquire gil;
        if (!py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>()) {
            GTEST_SKIP() << "The queued device-owner regression requires a CUDA device";
        }
    }
    auto& p = publisher({3, 1000, 100, 2, 60000});
    HiddenStateCapturePublisherTestPeer::failMaintenanceAllocation(p);
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.clear()");
    }
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(submit(p, 1));
    ASSERT_TRUE(waitFor([&] { return evalInt("put_started.is_set()") != 0; }));
    c10::cuda::CUDAGuard device_guard(0);
    auto packed = torch::ones({2, 6}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat16));
    c10::weak_intrusive_ptr<c10::TensorImpl> owner(packed.getIntrusivePtr());
    ASSERT_TRUE(reserve(p, 2, 2));
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::arange(2, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({2}, torch::kInt64),
                            2,
                            "queued device owner at maintenance failure"));
    packed = torch::Tensor();
    ASSERT_TRUE(reserve(p, 3, 2));
    ASSERT_TRUE(submit(p, 3));  // CPU successor must still be drained.
    {
        py::gil_scoped_acquire gil;
        py::exec("allow_put.set()");
    }
    EXPECT_FALSE(p.flushPending());
    p.flushAndClose();
    EXPECT_EQ(p.fatalState().kind, HiddenStateCaptureFatalKind::ALLOCATION);
    EXPECT_EQ(p.quotaSnapshot().pending_batches, 0);
    EXPECT_EQ(p.quotaSnapshot().samples, 2);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(evalInt("put_count"), 1);
    EXPECT_EQ(evalInt("close_count"), 0);
    EXPECT_FALSE(owner.expired());
    publishers_.pop_back();
    EXPECT_FALSE(owner.expired());  // Fatal exit is not permission to synchronize/free the device owner.
}
#endif

TEST_F(HiddenStateCapturePublisherTest, NativeWorkerAllocationFailuresLatchActualExceptionKinds) {
    const std::vector<std::pair<std::exception_ptr, HiddenStateCaptureFatalKind>> cases{
        {std::make_exception_ptr(c10::Error("injected worker tensor failure")), HiddenStateCaptureFatalKind::TENSOR},
        {std::make_exception_ptr(c10::OutOfMemoryError("injected worker allocation failure")),
         HiddenStateCaptureFatalKind::OUT_OF_MEMORY},
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
        {std::make_exception_ptr(
             c10::AcceleratorError(c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
                                   700,
                                   "injected worker accelerator failure")),
         HiddenStateCaptureFatalKind::ACCELERATOR},
#endif
        {std::make_exception_ptr(std::bad_alloc()), HiddenStateCaptureFatalKind::ALLOCATION}};
    for (const auto& item : cases) {
        auto& p = publisher();
        ASSERT_TRUE(reserve(p, 1, 2));
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
        p.flushAndClose();
    }
}

TEST_F(HiddenStateCapturePublisherTest, RecoverableNativeWorkerFailureAllowsLaterHealthyPublication) {
    auto& p = publisher();
    ASSERT_TRUE(reserve(p, 1, 2));
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
    ASSERT_TRUE(reserve(p, 2, 2));
    ASSERT_TRUE(submit(p, 2));
    ASSERT_TRUE(p.flushPending());
    EXPECT_EQ(evalInt("put_count"), 1);
}

TEST_F(HiddenStateCapturePublisherTest, FatalStoreFlushRetainsUnknownCompletionAndSourceQuota) {
    auto& p = publisher({2, 1000, 100, 1, 5});
    {
        py::gil_scoped_acquire gil;
        py::exec("flush_error = torch.OutOfMemoryError('injected staging allocation failure')");
    }
    auto packed = torch::arange(12, torch::kFloat16).reshape({2, 6});
    ASSERT_TRUE(reserve(p, 1, 2));
    ASSERT_TRUE(p.trySubmit(packed,
                            torch::arange(2, torch::kInt32),
                            torch::tensor({2}, torch::kInt32),
                            torch::tensor({1}, torch::kInt64),
                            2,
                            "unknown staging completion"));
    EXPECT_FALSE(p.flushPending());
    const auto fatal = p.fatalState();
    EXPECT_EQ(fatal.kind, HiddenStateCaptureFatalKind::OUT_OF_MEMORY);
    EXPECT_EQ(fatal.stage, HiddenStateCaptureFailureStage::STORE);
    EXPECT_GT(evalInt("len(store_keys)"), 0);
    EXPECT_EQ(p.quotaSnapshot().cleanup_pending_samples, 1);
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_FALSE(p.tryReserve(2, 2));
    p.discardReserved(1);
    {
        py::gil_scoped_acquire gil;
        // Even confirmed object absence cannot release unconfirmed local work.
        py::exec("store_keys.clear(); flush_error = None");
    }
    EXPECT_FALSE(p.flushPending());
    EXPECT_EQ(p.quotaSnapshot().samples, 1);
    EXPECT_EQ(p.quotaSnapshot().bytes, 40);
    EXPECT_EQ(p.quotaSnapshot().tokens, 2);
    EXPECT_EQ(p.fatalState().message, fatal.message);
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
    ASSERT_TRUE(reserve(p, 1, 1));
    EXPECT_FALSE(submit(p, 1, 2));
    EXPECT_EQ(p.quotaSnapshot().reserved_samples, 1);
    EXPECT_EQ(evalInt("put_count"), 0);
    p.discardReserved(1);
    EXPECT_EQ(p.quotaSnapshot().samples, 0);
}

TEST_F(HiddenStateCapturePublisherTest, FatalSubmissionExceptionsPropagateWithoutChangingQuotaOrType) {
    const std::vector<std::exception_ptr> errors{
        std::make_exception_ptr(c10::Error("injected tensor failure")),
        std::make_exception_ptr(c10::OutOfMemoryError("injected allocation failure")),
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
        std::make_exception_ptr(
            c10::AcceleratorError(c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
                                  700,
                                  "injected event-record failure")),
#endif
        std::make_exception_ptr(std::bad_alloc())};
    // Both entry points must preserve the original exception, not only the
    // standalone classifier. publish() must not shield serving from trySubmit.
    for (const bool publish_api : {false, true}) {
        for (const auto& error : errors) {
            auto& p = publisher();
            ASSERT_TRUE(reserve(p, 1, 2));
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
            EXPECT_EQ(evalInt("put_count"), 0);
            p.discardReserved(1);
            EXPECT_EQ(p.quotaSnapshot().samples, 0);
        }
    }
}

TEST_F(HiddenStateCapturePublisherTest, RecoverableSubmissionExceptionsFailOpenEvenWithCudaInMessage) {
    for (const bool publish_api : {false, true}) {
        auto&         p             = publisher();
        const int64_t request_id    = publish_api ? 2 : 1;
        const auto    previous_puts = evalInt("put_count");
        ASSERT_TRUE(reserve(p, request_id, 2));
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
    ASSERT_TRUE(reserve(p, 1, 2));
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
    EXPECT_EQ(evalInt("put_count"), 0);
    p.discardReserved(1);
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
    py::gil_scoped_acquire gil;
    p.flushAndClose();
    p.flushAndClose();
    EXPECT_GT(py::eval("close_count").cast<int>(), 0);
}

}  // namespace
}  // namespace rtp_llm
