#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <new>
#include <torch/version.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"

namespace py = pybind11;

namespace rtp_llm {

class HiddenStateCapturePublisher::CaptureFailure final: public std::runtime_error {
public:
    CaptureFailure(FailureStage failure_stage, FailureDisposition failure_disposition, std::string error_message):
        std::runtime_error(std::move(error_message)),
        failure_stage_(failure_stage),
        failure_disposition_(failure_disposition) {}

    FailureStage failureStage() const {
        return failure_stage_;
    }

    FailureDisposition failureDisposition() const {
        return failure_disposition_;
    }

private:
    FailureStage       failure_stage_;
    FailureDisposition failure_disposition_;
};

template<typename Operation>
decltype(auto) HiddenStateCapturePublisher::perform(FailureStage       failure_stage,
                                                    FailureDisposition failure_disposition,
                                                    const char*        context,
                                                    Operation&&        operation) {
    try {
        return operation();
    } catch (const CaptureFailure&) {
        throw;
    } catch (const std::exception& e) {
        const auto disposition = classifyWorkerFailure(e, failure_stage, failure_disposition);
        throw CaptureFailure(failure_stage, disposition, std::string(context) + ": " + e.what());
    } catch (...) {
        throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + " with an unknown exception");
    }
}

namespace {

HiddenStateCaptureFatalKind runtimeFatalKind(std::string_view message) noexcept {
    // Only recognizable runtime diagnostics, never an arbitrary CUDA/HIP word.
    for (const auto* marker :
         {"CUDA out of memory", "HIP out of memory", "CUDA error: out of memory", "HIP error: out of memory"}) {
        if (message.find(marker) != std::string::npos) {
            return HiddenStateCaptureFatalKind::OUT_OF_MEMORY;
        }
    }
    for (const auto* marker : {"CUDA error: an illegal memory access was encountered",
                               "CUDA error: device-side assert triggered",
                               "CUDA error: unspecified launch failure",
                               "CUDA error: misaligned address",
                               "HIP error: an illegal memory access was encountered",
                               "HIP error: illegal memory access",
                               "HIP error: device-side assert triggered",
                               "HIP error: launch failure"}) {
        if (message.find(marker) != std::string::npos) {
            return HiddenStateCaptureFatalKind::ACCELERATOR;
        }
    }
    return HiddenStateCaptureFatalKind::NONE;
}

}  // namespace

HiddenStateCaptureFatalKind HiddenStateCapturePublisher::nativeFatalKind(const std::exception& error) noexcept {
    if (dynamic_cast<const c10::OutOfMemoryError*>(&error)) {
        return HiddenStateCaptureFatalKind::OUT_OF_MEMORY;
    }
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
    if (dynamic_cast<const c10::AcceleratorError*>(&error)) {
        return HiddenStateCaptureFatalKind::ACCELERATOR;
    }
#endif
    if (dynamic_cast<const c10::Error*>(&error)) {
        return HiddenStateCaptureFatalKind::TENSOR;
    }
    if (dynamic_cast<const std::bad_alloc*>(&error)) {
        return HiddenStateCaptureFatalKind::ALLOCATION;
    }
    return runtimeFatalKind(error.what());
}

HiddenStateCaptureFatalKind HiddenStateCapturePublisher::pythonFatalKind(py::handle error) {
    if (PyErr_GivenExceptionMatches(error.ptr(), PyExc_MemoryError)) {
        return HiddenStateCaptureFatalKind::ALLOCATION;
    }
    const auto torch = PyDict_GetItemString(PyImport_GetModuleDict(), "torch");
    if (torch) {
        for (const auto& item : {std::make_pair("OutOfMemoryError", HiddenStateCaptureFatalKind::OUT_OF_MEMORY),
                                 std::make_pair("AcceleratorError", HiddenStateCaptureFatalKind::ACCELERATOR)}) {
            PyObject* type = PyObject_GetAttrString(torch, item.first);
            if (type) {
                const bool matches = PyErr_GivenExceptionMatches(error.ptr(), type);
                Py_DECREF(type);
                if (matches) {
                    return item.second;
                }
            } else {
                PyErr_Clear();
            }
        }
    }
    if (PyErr_GivenExceptionMatches(error.ptr(), PyExc_RuntimeError)) {
        return runtimeFatalKind(py::str(error).cast<std::string>());
    }
    return HiddenStateCaptureFatalKind::NONE;
}

void HiddenStateCapturePublisher::latchFatal(FailureStage                stage,
                                             HiddenStateCaptureFatalKind kind,
                                             const char*                 message) noexcept {
    capture_enabled_.store(false, std::memory_order_release);
    try {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (fatal_kind_.load(std::memory_order_relaxed) == HiddenStateCaptureFatalKind::NONE) {
            fatal_stage_ = stage;
            // Publish the allocation-free kind even if retaining diagnostics fails.
            fatal_kind_.store(kind, std::memory_order_release);
            fatal_message_ = message;
        }
    } catch (...) {
        auto expected = HiddenStateCaptureFatalKind::NONE;
        fatal_kind_.compare_exchange_strong(expected, kind, std::memory_order_release);
    }
}

HiddenStateCapturePublisher::FailureDisposition HiddenStateCapturePublisher::classifyWorkerFailure(
    const std::exception& error, FailureStage stage, FailureDisposition fallback) noexcept {
    try {
        HiddenStateCaptureFatalKind kind;
        if (const auto* python_error = dynamic_cast<const py::error_already_set*>(&error)) {
            py::gil_scoped_acquire gil;
            kind = pythonFatalKind(python_error->value());
            if (kind != HiddenStateCaptureFatalKind::NONE) {
                capture_enabled_.store(false, std::memory_order_release);
                latchFatal(stage, kind, python_error->what());
            }
        } else {
            kind = nativeFatalKind(error);
            if (kind != HiddenStateCaptureFatalKind::NONE) {
                latchFatal(stage, kind, error.what());
            }
        }
        return kind == HiddenStateCaptureFatalKind::NONE ? fallback : FailureDisposition::FATAL;
    } catch (const std::bad_alloc&) {
        latchFatal(stage, HiddenStateCaptureFatalKind::ALLOCATION, "allocation failed while classifying worker error");
    } catch (...) {
        latchFatal(stage, HiddenStateCaptureFatalKind::RUNTIME, "unable to classify publisher worker error");
    }
    return FailureDisposition::FATAL;
}

struct HiddenStateCapturePublisher::PublishTask {
    torch::Tensor                 packed;
    torch::Tensor                 input_ids;
    torch::Tensor                 input_lengths;
    torch::Tensor                 request_ids;
    std::vector<int64_t>          ids;
    std::vector<uint64_t>         generations;
    size_t                       confirmed_requests{0};
    std::shared_ptr<torch::Event> ready;
    // Completion retries must fence the stream that actually produced payloads,
    // not a fresh stream obtained from the pool on a later maintenance pass.
    std::optional<c10::Stream>    worker_stream;
    bool                         producer_complete{false};
    // Allocation-free lifetime retention if shutdown cannot confirm completion.
    std::shared_ptr<PublishTask> unresolved_owner;
    // Own converted/quantized payloads even when preparation or PUT throws.
    std::vector<torch::Tensor> payloads;
    bool                       store_attempted{false};
    bool                       succeeded{false};
    bool                       local_complete{false};
    bool                       store_flushed{false};
    bool                       dropped{false};
    bool                       completion_pending{false};
};

HiddenStateCapturePublisher::HiddenStateCapturePublisher(int64_t                      layer_count,
                                                         int64_t                      hidden_size,
                                                         HiddenStateCaptureDtype      capture_dtype,
                                                         c10::ScalarType              model_dtype,
                                                         bool                         owner,
                                                         int                          local_rank,
                                                         kmonitor::MetricsReporterPtr metrics_reporter,
                                                         HiddenStateCaptureLimits     limits,
                                                         std::vector<int64_t>         layer_ids):
    layer_count_(layer_count),
    layer_ids_(std::move(layer_ids)),
    hidden_size_(hidden_size),
    capture_dtype_(capture_dtype),
    model_dtype_(model_dtype),
    owner_(owner),
    local_rank_(local_rank),
    limits_(limits),
    metrics_reporter_(owner ? std::move(metrics_reporter) : nullptr) {
    reservation_lifetime_->publisher = this;
    int64_t bytes = 0;
    if (limits_.max_inflight_samples <= 0 || limits_.max_inflight_bytes <= 0 || limits_.max_inflight_tokens <= 0
        || limits_.max_pending_batches <= 0 || limits_.poll_interval_ms <= 0 || !payloadBytes(1, bytes)) {
        completeFailure("capture requires positive finite limits and a valid model layout",
                        FailureStage::INITIALIZATION,
                        FailureDisposition::HARD_CONTRACT,
                        "configuration");
        return;
    }
    if (owner_) {
        if (!Py_IsInitialized()) {
            completeFailure("Python interpreter is not initialized",
                            FailureStage::INITIALIZATION,
                            FailureDisposition::HARD_CONTRACT,
                            "worker startup");
            return;
        }
        try {
            worker_active_   = true;
            worker_finished_ = false;
            worker_          = std::thread(&HiddenStateCapturePublisher::workerLoop, this);
        } catch (const std::exception& e) {
            worker_active_   = false;
            worker_finished_ = true;
            completeFailure(
                e.what(), FailureStage::INITIALIZATION, FailureDisposition::HARD_CONTRACT, "worker startup");
        }
    }
}

HiddenStateCapturePublisher::~HiddenStateCapturePublisher() {
    {
        // Finish any token callback before invalidating the raw publisher pointer.
        // No callback owns the publisher/model or can race its state destruction.
        std::lock_guard<std::mutex> lock(reservation_lifetime_->mutex);
        reservation_lifetime_->publisher = nullptr;
    }
    flushAndClose();
    // Python objects are normally cleared by the worker under the GIL. A
    // failed local fence must retain store buffers until process teardown.
    (void)mooncake_config_.release();
    (void)store_.release();
    (void)quantize_fn_.release();
    for (const auto& entry : reservations_) {
        if (entry.second.task && !entry.second.task->local_complete) {
            // A failed fence is not permission to free live allocations. Keep
            // at most the bounded tasks alive without allocating during OOM.
            entry.second.task->unresolved_owner = entry.second.task;
        }
    }
}

bool HiddenStateCapturePublisher::payloadBytes(int64_t tokens, int64_t& bytes) const {
    if (tokens <= 0 || layer_count_ <= 0 || hidden_size_ <= 0
        || layer_count_ > (std::numeric_limits<int64_t>::max() - 8) / 4
        || (capture_dtype_ != HiddenStateCaptureDtype::BF16 && capture_dtype_ != HiddenStateCaptureDtype::FP8_E4M3)) {
        return false;
    }
    const int64_t coefficient =
        capture_dtype_ == HiddenStateCaptureDtype::BF16 ? 2 * (layer_count_ + 1) : layer_count_ + 2;
    const int64_t extra = capture_dtype_ == HiddenStateCaptureDtype::BF16 ? 8 : 4 * layer_count_ + 8;
    if (hidden_size_ > (std::numeric_limits<int64_t>::max() - extra) / coefficient) {
        return false;
    }
    const int64_t per_token = coefficient * hidden_size_ + extra;
    if (tokens > std::numeric_limits<int64_t>::max() / per_token) {
        return false;
    }
    bytes = tokens * per_token;
    return true;
}

HiddenStateCaptureReservationPtr HiddenStateCapturePublisher::tryReserve(int64_t request_id, int64_t valid_tokens) {
    // This owner must outlive state_mutex_: a failed map allocation destroys the
    // token and calls cancellation, which takes that same mutex.
    HiddenStateCaptureReservationPtr token;
    try {
        int64_t bytes = 0;
        if (!owner_ || !capture_enabled_.load(std::memory_order_acquire) || request_id < 0
            || !payloadBytes(valid_tokens, bytes)) {
            return {};
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!stopping_ && capture_enabled_.load(std::memory_order_acquire)
                && next_reservation_generation_ != 0 && reservations_.find(request_id) == reservations_.end()
                && reservations_.size() < static_cast<uint64_t>(limits_.max_inflight_samples)
                && bytes <= limits_.max_inflight_bytes - reserved_bytes_
                && valid_tokens <= limits_.max_inflight_tokens - reserved_tokens_) {
                const auto generation = next_reservation_generation_;
                next_reservation_generation_ =
                    generation == std::numeric_limits<uint64_t>::max() ? 0 : generation + 1;
                token = std::make_shared<HiddenStateCaptureReservation>(
                    [lifetime = std::weak_ptr<LifetimeGate>(reservation_lifetime_), request_id, generation]() noexcept {
                        try {
                            if (auto gate = lifetime.lock()) {
                                std::lock_guard<std::mutex> lock(gate->mutex);
                                if (gate->publisher) {
                                    gate->publisher->cancelReserved(request_id, generation);
                                }
                            }
                        } catch (const std::exception& e) {
                            try {
                                RTP_LLM_LOG_WARNING("hidden-state reservation cancellation failed: %s", e.what());
                            } catch (...) {
                                // Logging must not turn cancellation failure into termination.
                            }
                        } catch (...) {
                            try {
                                RTP_LLM_LOG_WARNING("hidden-state reservation cancellation failed with unknown exception");
                            } catch (...) {
                            }
                        }
                    });
                Reservation reservation;
                reservation.generation = generation;
                reservation.tokens     = valid_tokens;
                reservation.bytes      = bytes;
                reservations_.emplace(request_id, std::move(reservation));
                reserved_bytes_ += bytes;
                reserved_tokens_ += valid_tokens;
            }
        }
        RtpLLMHiddenStateCaptureMetricsCollector collector;
        collector.reservation_qps     = static_cast<bool>(token);
        collector.quota_rejection_qps = !token;
        reportMetrics(collector);
        recordQuota();
        return token;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& e) {
        rejectFailure(e.what(), FailureStage::PREPARE, FailureDisposition::OPERATIONAL);
        return {};
    }
}

void HiddenStateCapturePublisher::releaseReservation(std::unordered_map<int64_t, Reservation>::iterator entry) {
    reserved_bytes_ -= entry->second.bytes;
    reserved_tokens_ -= entry->second.tokens;
    reservations_.erase(entry);
}

void HiddenStateCapturePublisher::discardReserved(int64_t request_id) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto                        entry = reservations_.find(request_id);
        if (entry == reservations_.end() || entry->second.state != ReservationState::RESERVED) {
            return;
        }
        releaseReservation(entry);
    }
    state_changed_.notify_all();
    recordQuota();
}

void HiddenStateCapturePublisher::cancelReserved(int64_t request_id, uint64_t generation) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto                        entry = reservations_.find(request_id);
        if (entry == reservations_.end() || entry->second.generation != generation
            || entry->second.state != ReservationState::RESERVED) {
            return;
        }
        releaseReservation(entry);
    }
    state_changed_.notify_all();
    recordQuota();
}

HiddenStateCaptureQuotaSnapshot HiddenStateCapturePublisher::quotaSnapshot() const {
    std::lock_guard<std::mutex>     lock(state_mutex_);
    HiddenStateCaptureQuotaSnapshot snapshot;
    snapshot.samples         = static_cast<int64_t>(reservations_.size());
    snapshot.bytes           = reserved_bytes_;
    snapshot.tokens          = reserved_tokens_;
    snapshot.pending_batches = static_cast<int64_t>(pending_.size());
    for (const auto& entry : reservations_) {
        snapshot.reserved_samples += entry.second.state == ReservationState::RESERVED;
        snapshot.published_samples += entry.second.state == ReservationState::PUBLISHED;
        snapshot.cleanup_pending_samples += entry.second.state == ReservationState::CLEANUP_PENDING;
    }
    return snapshot;
}

void HiddenStateCapturePublisher::recordQuota() {
    const auto                               snapshot = quotaSnapshot();
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.has_quota               = true;
    collector.inflight_samples        = snapshot.samples;
    collector.inflight_bytes          = snapshot.bytes;
    collector.inflight_tokens         = snapshot.tokens;
    collector.pending_batches         = snapshot.pending_batches;
    collector.cleanup_pending_samples = snapshot.cleanup_pending_samples;
    reportMetrics(collector);
}

bool HiddenStateCapturePublisher::trySubmit(torch::Tensor        packed_hidden_states,
                                            const torch::Tensor& input_ids,
                                            const torch::Tensor& input_lengths,
                                            const torch::Tensor& request_ids,
                                            int64_t              expected_rows,
                                            const std::string&   layout_context) {
    if (!owner_ || !capture_enabled_.load(std::memory_order_acquire)) {
        return false;
    }
    try {
        if (!packed_hidden_states.defined()) {
            return rejectLayout(layout_context + " is undefined");
        }
        const auto layout_error =
            validatePackedLayout(packed_hidden_states, expected_rows, packed_hidden_states.device(), layout_context);
        if (!layout_error.empty()) {
            return rejectLayout(layout_error);
        }
        // Host metadata is an admission contract: never do a D2H wait in forward.
        if (!request_ids.defined() || request_ids.dim() != 1 || !request_ids.device().is_cpu()
            || request_ids.scalar_type() != torch::kInt64 || !input_lengths.defined()
            || !input_lengths.device().is_cpu() || input_lengths.dim() != 1
            || input_lengths.scalar_type() != torch::kInt32 || request_ids.numel() == 0
            || request_ids.numel() != input_lengths.numel() || !input_ids.defined() || input_ids.dim() != 1
            || input_ids.numel() != expected_rows) {
            return rejectFailure("capture submit requires CPU int64 request IDs, CPU int32 lengths and aligned IDs",
                                 FailureStage::PREPARE,
                                 FailureDisposition::REQUEST_ERROR);
        }
        auto task           = std::make_shared<PublishTask>();
        bool dropped        = false;
        task->request_ids   = request_ids.clone().contiguous();
        task->input_lengths = input_lengths.clone().contiguous();
        task->input_ids     = input_ids.device().is_cpu() ? input_ids.clone() : input_ids;
        task->packed        = std::move(packed_hidden_states);
        std::unordered_set<int64_t> seen;
        int64_t                     rows = 0;
        for (int64_t i = 0; i < task->request_ids.numel(); ++i) {
            const auto    id     = task->request_ids.data_ptr<int64_t>()[i];
            const int64_t tokens = task->input_lengths.data_ptr<int32_t>()[i];
            if (id < 0 || !seen.insert(id).second || tokens <= 0 || tokens > expected_rows - rows) {
                return false;
            }
            task->ids.push_back(id);
            rows += tokens;
        }
        if (rows != expected_rows || (input_ids.device() != task->packed.device() && !input_ids.device().is_cpu())) {
            return false;
        }
        task->generations.reserve(task->ids.size());
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (stopping_ || !capture_enabled_.load(std::memory_order_acquire)) {
                return false;
            }

            for (size_t i = 0; i < task->ids.size(); ++i) {
                auto entry = reservations_.find(task->ids[i]);
                if (entry == reservations_.end() || entry->second.state != ReservationState::RESERVED
                    || entry->second.tokens != task->input_lengths.data_ptr<int32_t>()[i]) {
                    return false;
                }
                task->generations.push_back(entry->second.generation);
            }
#if USING_CUDA || USING_ROCM
            if (!task->packed.device().is_cpu()) {
                c10::DeviceGuard guard(task->packed.device());
                task->ready = runtimeCreateEvent();
            }
#else
            if (!task->packed.device().is_cpu()) {
                return false;  // No producer-event implementation for this backend.
            }
#endif
            dropped       = pending_.size() >= static_cast<uint64_t>(limits_.max_pending_batches);
            task->dropped = dropped;
            if (task->dropped) {
                // Retain only a local-completion ticket inside the already
                // bounded ledger. Queue overload never launches a PUT or frees
                // producer storage before its outstanding work completes.
                task->completion_pending = true;
                cleanup_requested_       = true;
            } else {
                pending_.push_back(task);
            }
            for (const auto id : task->ids) {
                auto& reservation = reservations_.at(id);
                reservation.state = task->dropped ? ReservationState::CLEANUP_PENDING : ReservationState::SUBMITTED;
                reservation.task  = task;
            }
        }
        state_changed_.notify_all();
        if (dropped) {
            RtpLLMHiddenStateCaptureMetricsCollector collector;
            collector.queue_rejection_qps = true;
            reportMetrics(collector);
        }
        recordQuota();
        return !dropped;
    } catch (const c10::Error&) {
        // Submission runs on the serving thread/device. Allocation, guard and
        // event-record failures may invalidate that shared execution context.
        // Includes OutOfMemoryError and AcceleratorError; preserve their type.
        throw;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& e) {
        return rejectFailure(e.what(), FailureStage::PREPARE, FailureDisposition::OPERATIONAL);
    } catch (...) {
        return rejectFailure(
            "unknown capture submission failure", FailureStage::PREPARE, FailureDisposition::OPERATIONAL);
    }
}

torch::Tensor HiddenStateCapturePublisher::publish(torch::Tensor        packed_hidden_states,
                                                   const torch::Tensor& input_ids,
                                                   const torch::Tensor& input_lengths,
                                                   const torch::Tensor& request_ids,
                                                   int64_t              expected_rows,
                                                   const std::string&   layout_context,
                                                   const c10::Device&   fallback_device,
                                                   const torch::Tensor& canonical_hidden_states) {
    auto normal_output = canonical_hidden_states.defined() ?
                             canonical_hidden_states :
                             makeFallback(packed_hidden_states, expected_rows, fallback_device);
    (void)trySubmit(
        std::move(packed_hidden_states), input_ids, input_lengths, request_ids, expected_rows, layout_context);
    return normal_output;
}

void HiddenStateCapturePublisher::beginForward() {}

bool HiddenStateCapturePublisher::shouldPublish() {
    recordBatch();
    if (capture_enabled_.load(std::memory_order_acquire)) {
        return true;
    }

    recordBrokenRejection();
    return false;
}

int64_t HiddenStateCapturePublisher::packedWidth() const {
    return (layer_count_ + 1) * hidden_size_;
}

std::string HiddenStateCapturePublisher::makeRequestKey(int64_t request_id) const {
    return mooncake_config_.attr("make_store_key")(request_id).cast<std::string>();
}

std::string HiddenStateCapturePublisher::makeBatchId(uint64_t batch_sequence,
                                                     int64_t  first_request_id,
                                                     int64_t  last_request_id) const {
    return std::string("rtp-forward-") + store_key_namespace_ + "-" + std::to_string(batch_sequence) + "-requests-"
           + std::to_string(first_request_id) + "-" + std::to_string(last_request_id);
}

std::string HiddenStateCapturePublisher::validatePackedLayout(const torch::Tensor& tensor,
                                                              int64_t              expected_rows,
                                                              const c10::Device&   expected_device,
                                                              const std::string&   context) const {
    if (!tensor.defined() || tensor.dim() != 2) {
        return context + " must be a rank-2 tensor";
    }
    if (tensor.size(0) != expected_rows) {
        return context + " row count " + std::to_string(tensor.size(0)) + " must match expected row count "
               + std::to_string(expected_rows);
    }
    if (tensor.size(1) != packedWidth()) {
        return context + " width " + std::to_string(tensor.size(1)) + " must match expected packed width "
               + std::to_string(packedWidth());
    }
    if (tensor.scalar_type() != model_dtype_) {
        return context + " dtype must match the model dtype";
    }
    if (tensor.device() != expected_device) {
        return context + " must be on the model input device";
    }
    return {};
}

HiddenStateCaptureStats HiddenStateCapturePublisher::stats() const {
    return {failure_count_.load(std::memory_order_relaxed),
            broken_rejection_count_.load(std::memory_order_relaxed),
            fatal_kind_.load(std::memory_order_acquire)};
}

HiddenStateCaptureFatalState HiddenStateCapturePublisher::fatalState() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return {fatal_kind_.load(std::memory_order_acquire), fatal_stage_, fatal_message_};
}

void HiddenStateCapturePublisher::deferError(std::string error_message, bool disable_capture) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (disable_capture && !broken_reason_.has_value()) {
        broken_reason_ = error_message;
    }

    if (disable_capture) {
        capture_enabled_.store(false, std::memory_order_release);
    }
}

void HiddenStateCapturePublisher::reportMetrics(RtpLLMHiddenStateCaptureMetricsCollector& collector) {
    if (owner_ && metrics_reporter_) {
        try {
            metrics_reporter_->report<RtpLLMHiddenStateCaptureMetrics, RtpLLMHiddenStateCaptureMetricsCollector>(
                nullptr, &collector);
        } catch (...) {
            // Telemetry cannot change admission, ownership, or serving results.
        }
    }
}

void HiddenStateCapturePublisher::recordBatch() {
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.batch_qps = true;
    reportMetrics(collector);
}

void HiddenStateCapturePublisher::recordFailure(FailureStage failure_stage, FailureDisposition failure_disposition) {
    failure_count_.fetch_add(1, std::memory_order_relaxed);
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.failure_qps                = true;
    collector.initialization_failure_qps = failure_stage == FailureStage::INITIALIZATION;
    collector.layout_failure_qps         = failure_stage == FailureStage::LAYOUT;
    collector.prepare_failure_qps        = failure_stage == FailureStage::PREPARE;
    collector.quantize_failure_qps       = failure_stage == FailureStage::QUANTIZE;
    collector.store_failure_qps          = failure_stage == FailureStage::STORE;
    collector.shutdown_failure_qps       = failure_stage == FailureStage::SHUTDOWN;
    collector.hard_contract_failure_qps =
        failure_disposition == FailureDisposition::HARD_CONTRACT || failure_disposition == FailureDisposition::FATAL;
    collector.request_error_failure_qps = failure_disposition == FailureDisposition::REQUEST_ERROR;
    collector.operational_failure_qps   = failure_disposition == FailureDisposition::OPERATIONAL;
    reportMetrics(collector);
}

void HiddenStateCapturePublisher::recordBrokenRejection() {
    broken_rejection_count_.fetch_add(1, std::memory_order_relaxed);
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.broken_rejection_qps = true;
    reportMetrics(collector);
}

void HiddenStateCapturePublisher::recordDuplicateRequestId(int64_t            request_id,
                                                           const std::string& key,
                                                           const char*        source) {
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.duplicate_request_id_qps = true;
    reportMetrics(collector);

    if (!owner_) {
        return;
    }
    RTP_LLM_INTERVAL_LOG(
        300,
        WARN,
        "[TorchSpec hidden-state publisher] rejecting duplicate request id %ld with key [%s] from %s; existing "
        "hidden-state objects will not be overwritten",
        request_id,
        key.c_str(),
        source);
}

void HiddenStateCapturePublisher::recordCaptureStatus() {
    bool broken = false;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        broken = broken_reason_.has_value();
    }
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.has_capture_status = true;
    collector.capture_enabled    = capture_enabled_.load(std::memory_order_acquire) ? 1 : 0;
    collector.capture_broken     = broken ? 1 : 0;
    reportMetrics(collector);
}

void HiddenStateCapturePublisher::recordPublish(bool success, const PublishMetrics& metrics) {
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.publish_success_qps       = success;
    collector.bf16_publish_qps          = success && capture_dtype_ == HiddenStateCaptureDtype::BF16;
    collector.fp8_publish_qps           = success && capture_dtype_ == HiddenStateCaptureDtype::FP8_E4M3;
    collector.has_publish_latency       = true;
    collector.has_quantize_latency      = metrics.has_quantize_latency;
    collector.has_store_put_latency     = metrics.has_store_put_latency;
    collector.has_publish_payload       = success;
    collector.publish_latency_us        = metrics.publish_latency_us;
    collector.quantize_latency_us       = metrics.quantize_latency_us;
    collector.store_put_latency_us      = metrics.store_put_latency_us;
    collector.publish_request_count     = metrics.request_count;
    collector.publish_token_count       = metrics.token_count;
    collector.publish_payload_bytes     = metrics.payload_bytes;
    collector.publish_input_ids_bytes   = metrics.input_ids_bytes;
    collector.publish_aux_hidden_bytes  = metrics.auxiliary_hidden_bytes;
    collector.publish_last_hidden_bytes = metrics.last_hidden_bytes;
    collector.publish_scale_bytes       = metrics.scale_bytes;
    reportMetrics(collector);
}

bool HiddenStateCapturePublisher::observeAsyncErrors(FailureStage failure_stage, const char* phase) {
    RTP_LLM_PROFILE_SCOPE("hidden_state_capture.observe_async_errors");
    bool       success = true;
    py::object result;
    {
        RTP_LLM_PROFILE_SCOPE("hidden_state_capture.take_async_errors");
        result = store_.attr("take_async_errors")();
    }
    if (!py::isinstance<py::list>(result)) {
        throw std::runtime_error("EagleMooncakeStore.take_async_errors() must return a list");
    }

    const auto errors = result.cast<py::list>();
    for (const auto error : errors) {
        success = false;
        // Historical failures never belong to the current stream. A typed
        // allocation/device fatal still has to stop new capture admission.
        const auto kind = pythonFatalKind(error);
        if (kind != HiddenStateCaptureFatalKind::NONE) {
            capture_enabled_.store(false, std::memory_order_release);
        }
        const auto detail = py::str(error).cast<std::string>();
        if (kind != HiddenStateCaptureFatalKind::NONE) {
            latchFatal(failure_stage, kind, detail.c_str());
        }
        recordFailure(failure_stage,
                      kind == HiddenStateCaptureFatalKind::NONE ? FailureDisposition::OPERATIONAL :
                                                                  FailureDisposition::FATAL);
        RTP_LLM_LOG_ERROR(
            "[TorchSpec hidden-state publisher] %s observed an asynchronous Mooncake failure from a previously "
            "accepted batch; the original batch/request IDs are preserved in the error and the current stream "
            "continues: %s",
            phase,
            detail.c_str());
    }
    return success;
}

bool HiddenStateCapturePublisher::rejectFailure(std::string        error_message,
                                                FailureStage       failure_stage,
                                                FailureDisposition failure_disposition) {
    return completeFailure(error_message, failure_stage, failure_disposition, "capture contract validation");
}

bool HiddenStateCapturePublisher::rejectLayout(std::string layout_error) {
    return rejectFailure(std::move(layout_error), FailureStage::LAYOUT, FailureDisposition::HARD_CONTRACT);
}

bool HiddenStateCapturePublisher::completeFailure(const std::string& error_message,
                                                  FailureStage       failure_stage,
                                                  FailureDisposition failure_disposition,
                                                  const char*        phase) {
    if (!owner_) {
        return false;
    }

    recordFailure(failure_stage, failure_disposition);
    const auto detail = error_message.empty() ? "TorchSpec hidden-state publisher failed" : error_message;

    const bool hard_contract =
        failure_disposition == FailureDisposition::HARD_CONTRACT || failure_disposition == FailureDisposition::FATAL;
    deferError(detail, hard_contract);
    recordCaptureStatus();
    if (failure_disposition == FailureDisposition::FATAL) {
        RTP_LLM_LOG_ERROR(
            "[TorchSpec hidden-state publisher] fatal worker failure (kind=%d), capture admission stopped: %s: %s",
            static_cast<int>(fatal_kind_.load(std::memory_order_acquire)),
            phase,
            detail.c_str());
    } else if (hard_contract) {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s violated a hard contract; disabling capture only: %s",
                          phase,
                          detail.c_str());
    } else {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s failed for the current batch; capture remains enabled "
                          "for subsequent batches; serving is unaffected: %s",
                          phase,
                          detail.c_str());
    }
    return false;
}

void HiddenStateCapturePublisher::initialize(int local_rank) {
    if (!owner_) {
        return;
    }

    try {
        py::gil_scoped_acquire gil;
        py::object             mooncake_module = perform(FailureStage::INITIALIZATION,
                                             FailureDisposition::OPERATIONAL,
                                             "failed to import torchspec.transfer.mooncake",
                                             []() { return py::module_::import("torchspec.transfer.mooncake"); });
        mooncake_config_                       = perform(FailureStage::INITIALIZATION,
                                   FailureDisposition::OPERATIONAL,
                                   "MooncakeConfig.from_env failed",
                                   [&]() { return mooncake_module.attr("MooncakeConfig").attr("from_env")(); });
        store_key_namespace_ =
            perform(FailureStage::INITIALIZATION,
                    FailureDisposition::HARD_CONTRACT,
                    "invalid Mooncake store-key namespace contract",
                    [&]() {
                        if (!py::hasattr(mooncake_config_, "make_store_key")
                            || PyCallable_Check(mooncake_config_.attr("make_store_key").ptr()) == 0) {
                            throw std::runtime_error("MooncakeConfig.make_store_key must be callable");
                        }
                        (void)mooncake_config_.attr("make_store_key")(0).cast<std::string>();
                        return mooncake_config_.attr("store_key_namespace").cast<std::string>();
                    });

        perform(FailureStage::INITIALIZATION,
                FailureDisposition::HARD_CONTRACT,
                "invalid Mooncake capture layout contract",
                [&]() {
                    const auto configured_hidden_dim = mooncake_config_.attr("hidden_dim").cast<int64_t>();
                    if (configured_hidden_dim != hidden_size_) {
                        throw std::runtime_error("Mooncake hidden_dim " + std::to_string(configured_hidden_dim)
                                                 + " does not match RTP hidden size " + std::to_string(hidden_size_));
                    }
                    const auto configured_layer_count = mooncake_config_.attr("num_aux_layers").cast<int64_t>();
                    if (configured_layer_count != layer_count_) {
                        throw std::runtime_error("Mooncake num_aux_layers " + std::to_string(configured_layer_count)
                                                 + " does not match RTP capture layer count "
                                                 + std::to_string(layer_count_));
                    }
                });

        store_ = perform(FailureStage::INITIALIZATION,
                         FailureDisposition::OPERATIONAL,
                         "failed to construct EagleMooncakeStore",
                         [&]() { return mooncake_module.attr("EagleMooncakeStore")(mooncake_config_); });
        perform(FailureStage::INITIALIZATION,
                FailureDisposition::HARD_CONTRACT,
                "EagleMooncakeStore does not satisfy the RTP batch publisher contract",
                [&]() {
                    for (const char* method_name :
                         {"put_batch", "take_async_errors", "flush", "query_keys", "reclaim_abandoned_capture", "close"}) {
                        if (!py::hasattr(store_, method_name)
                            || PyCallable_Check(store_.attr(method_name).ptr()) == 0) {
                            throw std::runtime_error(std::string("EagleMooncakeStore.") + method_name
                                                     + " must be callable");
                        }
                    }
                });
        perform(FailureStage::INITIALIZATION,
                FailureDisposition::OPERATIONAL,
                "failed to set up EagleMooncakeStore",
                [&]() {
                    py::object torch_module = py::module_::import("torch");
                    py::object device       = torch_module.attr("device")("cuda:" + std::to_string(local_rank));
                    store_.attr("setup")(device);
                    store_.attr("warmup_rdma")();
                });
        if (capture_dtype_ == HiddenStateCaptureDtype::FP8_E4M3) {
            quantize_fn_ =
                perform(FailureStage::INITIALIZATION,
                        FailureDisposition::OPERATIONAL,
                        "failed to load TorchSpec FP8 quantizer",
                        []() { return py::module_::import("torchspec.utils.fp8").attr("quantize_aux_hidden_states"); });
        }
        RTP_LLM_LOG_INFO("TorchSpec hidden-state publisher initialized on local rank %d", local_rank);
        recordCaptureStatus();
    } catch (const CaptureFailure& failure) {
        const std::string error_message =
            std::string("failed to initialize TorchSpec hidden-state publisher; install the TorchSpec runtime "
                        "providing torchspec.transfer.mooncake (and torchspec.utils.fp8 for FP8 capture) in the "
                        "same Python environment: ")
            + failure.what();
        // setup() or warmup_rdma() may fail after store_ is assigned. Drop
        // partially initialized Python objects here so shutdown cannot retry
        // initialization while attempting to flush them.
        py::gil_scoped_acquire gil;
        if (store_ && !store_.is_none()) {
            try {
                store_.attr("close")();
            } catch (const std::exception& close_error) {
                RTP_LLM_LOG_ERROR("failed to close partially initialized TorchSpec hidden-state store: %s",
                                  close_error.what());
            }
        }
        mooncake_config_ = py::object();
        store_           = py::object();
        quantize_fn_     = py::object();
        completeFailure(error_message,
                        failure.failureStage(),
                        failure.failureDisposition() == FailureDisposition::FATAL ? FailureDisposition::FATAL :
                                                                                    FailureDisposition::HARD_CONTRACT,
                        "publisher initialization");
    }
}

torch::Tensor HiddenStateCapturePublisher::makeFallback(const torch::Tensor& hidden_states,
                                                        int64_t              token_count,
                                                        const c10::Device&   fallback_device) const {
    (void)fallback_device;
    if (hidden_states.defined() && hidden_states.dim() == 2 && hidden_states.size(0) == token_count) {
        if (hidden_states.size(1) == hidden_size_) {
            return hidden_states;
        }
        if (hidden_states.size(1) == packedWidth()) {
            return hidden_states.narrow(1, layer_count_ * hidden_size_, hidden_size_);
        }
    }
    return torch::Tensor();
}

torch::Tensor HiddenStateCapturePublisher::publishOnWorker(PublishTask& task) {
    // Keep exception formatting and Python temporary destruction under the GIL.
    py::gil_scoped_acquire worker_gil;
    if (layer_ids_.size() != static_cast<size_t>(layer_count_)) {
        rejectFailure("capture layer IDs must preserve the configured order and match the layer count",
                      FailureStage::PREPARE,
                      FailureDisposition::HARD_CONTRACT);
        return torch::Tensor();
    }
    RTP_LLM_PROFILE_SCOPE("hidden_state_capture.publish_worker");
    const int64_t publish_start_us     = autil::TimeUtility::currentTimeInMicroSeconds();
    const auto&   packed_hidden_states = task.packed;
    const auto&   input_ids            = task.input_ids;
    const auto&   input_lengths        = task.input_lengths;
    const auto&   request_ids          = task.request_ids;
    // Reserve owner slots before launching conversion work, so retaining a new
    // contiguous/quantized allocation cannot itself fail after the launch.
    task.payloads.reserve(8);
    torch::Tensor last_hidden_states;
    torch::Tensor auxiliary_hidden_view;
    {
        py::gil_scoped_release release;
        last_hidden_states = packed_hidden_states.narrow(1, layer_count_ * hidden_size_, hidden_size_).contiguous();
        task.payloads.push_back(last_hidden_states);
        auxiliary_hidden_view = packed_hidden_states.narrow(1, 0, layer_count_ * hidden_size_);
    }

    torch::Tensor  lengths_host;
    torch::Tensor  request_ids_host;
    int64_t        total_tokens = 0;
    PublishMetrics metrics;
    int64_t        quantize_start_us           = 0;
    int64_t        store_put_start_us          = 0;
    bool           quantize_latency_completed  = false;
    bool           store_put_latency_completed = false;

    const auto finish_failure =
        [&](const std::string& error_message, FailureStage failure_stage, FailureDisposition failure_disposition) {
            if (metrics.has_quantize_latency && !quantize_latency_completed) {
                metrics.quantize_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - quantize_start_us;
            }
            if (metrics.has_store_put_latency && !store_put_latency_completed) {
                metrics.store_put_latency_us += autil::TimeUtility::currentTimeInMicroSeconds() - store_put_start_us;
            }
            // Worker failures are training telemetry, never serving deferred errors.
            completeFailure(error_message, failure_stage, failure_disposition, "runtime publish");
            if (owner_) {
                metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
                metrics.request_count      = lengths_host.defined() ? lengths_host.numel() : 0;
                metrics.token_count        = total_tokens;
                recordPublish(/*success=*/false, metrics);
            }
        };

    if (owner_) {
        try {
            std::optional<ProfilingScope> gil_scope(std::in_place, "hidden_state_capture.gil_acquire");
            py::gil_scoped_acquire        gil;
            gil_scope.reset();

            perform(FailureStage::STORE,
                    FailureDisposition::OPERATIONAL,
                    "TorchSpec hidden-state store async-error polling failed",
                    [&]() { observeAsyncErrors(FailureStage::STORE, "publish start"); });
            if (!capture_enabled_.load(std::memory_order_acquire)) {
                return last_hidden_states;
            }

            // End preparation before admission without shortening any payload lifetime.
            std::optional<ProfilingScope> prepare_scope(std::in_place, "hidden_state_capture.prepare_payload");
            // Only the TP owner publishes. Keep request metadata and store payload
            // materialization owner-local; non-owners only preserve the same packed
            // model-output contract needed by the existing TP/FFN paths.
            perform(
                FailureStage::PREPARE, FailureDisposition::REQUEST_ERROR, "invalid capture request metadata", [&]() {
                    if (!input_lengths.defined() || !request_ids.defined()) {
                        throw std::runtime_error(
                            "captured hidden-state publishing requires input lengths and request ids");
                    }
                    if (input_lengths.dim() != 1 || request_ids.dim() != 1) {
                        throw std::runtime_error("capture input lengths and request ids must be rank-1 tensors");
                    }
                    if (input_lengths.numel() != request_ids.numel()) {
                        throw std::runtime_error("capture input length count " + std::to_string(input_lengths.numel())
                                                 + " must match request id count "
                                                 + std::to_string(request_ids.numel()));
                    }
                    if (!input_ids.defined()) {
                        throw std::runtime_error("captured hidden-state publishing requires input ids");
                    }
                    if (input_ids.dim() != 1) {
                        throw std::runtime_error("capture input ids must be a rank-1 tensor");
                    }
                });

            auto host_metadata =
                perform(FailureStage::PREPARE,
                        FailureDisposition::OPERATIONAL,
                        "failed to prepare capture request metadata on CPU",
                        [&]() {
                            py::gil_scoped_release release;
                            auto prepared_lengths     = input_lengths.is_cuda() ? input_lengths.cpu() : input_lengths;
                            auto prepared_request_ids = request_ids.is_cuda() ? request_ids.cpu() : request_ids;
                            return std::make_pair(prepared_lengths.contiguous(), prepared_request_ids.contiguous());
                        });
            lengths_host     = std::move(host_metadata.first);
            request_ids_host = std::move(host_metadata.second);

            perform(
                FailureStage::PREPARE, FailureDisposition::REQUEST_ERROR, "invalid capture request metadata", [&]() {
                    if (lengths_host.scalar_type() != torch::kInt32) {
                        throw std::runtime_error("capture input lengths must be int32");
                    }
                    if (lengths_host.numel() == 0) {
                        throw std::runtime_error("capture batch must contain at least one request");
                    }
                    if (request_ids_host.scalar_type() != torch::kInt64) {
                        throw std::runtime_error("capture request ids must be int64");
                    }

                    std::unordered_set<int64_t> seen_request_ids;
                    for (int64_t i = 0; i < lengths_host.numel(); ++i) {
                        const auto length     = static_cast<int64_t>(lengths_host.data_ptr<int32_t>()[i]);
                        const auto request_id = request_ids_host.data_ptr<int64_t>()[i];
                        if (length <= 0) {
                            throw std::runtime_error("capture request length must be positive, got "
                                                     + std::to_string(length));
                        }
                        if (request_id < 0) {
                            throw std::runtime_error("capture request id must be non-negative, got "
                                                     + std::to_string(request_id));
                        }
                        if (!seen_request_ids.insert(request_id).second) {
                            const auto key = makeRequestKey(request_id);
                            recordDuplicateRequestId(request_id, key, "publish batch metadata");
                            throw std::runtime_error("duplicate capture request id " + std::to_string(request_id)
                                                     + " in the same publish batch");
                        }
                        if (total_tokens > std::numeric_limits<int64_t>::max() - length) {
                            throw std::runtime_error("capture request lengths overflow int64 token count");
                        }
                        total_tokens += length;
                    }
                    if (total_tokens != input_ids.numel()) {
                        throw std::runtime_error("capture request lengths sum to " + std::to_string(total_tokens)
                                                 + " but input id token count is " + std::to_string(input_ids.numel()));
                    }
                    if (total_tokens != packed_hidden_states.size(0)) {
                        throw std::runtime_error("capture request lengths sum to " + std::to_string(total_tokens)
                                                 + " but hidden-state token count is "
                                                 + std::to_string(packed_hidden_states.size(0)));
                    }
                });

            torch::Tensor stored_auxiliary;
            torch::Tensor stored_last_hidden =
                perform(FailureStage::PREPARE,
                        FailureDisposition::OPERATIONAL,
                        "failed to prepare last hidden states for storage",
                        [&]() {
                            py::gil_scoped_release release;
                            return last_hidden_states.to(torch::kBFloat16).contiguous();
                        });
            task.payloads.push_back(stored_last_hidden);
            torch::Tensor hidden_states_scale;
            if (capture_dtype_ == HiddenStateCaptureDtype::FP8_E4M3) {
                auto quantizer_input = perform(FailureStage::QUANTIZE,
                                               FailureDisposition::OPERATIONAL,
                                               "failed to prepare TorchSpec FP8 quantizer input",
                                               [&]() {
                                                   py::gil_scoped_release release;
                                                   return auxiliary_hidden_view.contiguous();
                                               });
                task.payloads.push_back(quantizer_input);
                RTP_LLM_PROFILE_SCOPE("hidden_state_capture.quantize");
                metrics.has_quantize_latency = true;
                quantize_start_us            = autil::TimeUtility::currentTimeInMicroSeconds();
                py::object quantized_result  = perform(FailureStage::QUANTIZE,
                                                      FailureDisposition::OPERATIONAL,
                                                      "TorchSpec FP8 quantizer execution failed",
                                                      [&]() { return quantize_fn_(quantizer_input, layer_count_); });
                auto       quantized_tensors = perform(
                    FailureStage::QUANTIZE,
                    FailureDisposition::HARD_CONTRACT,
                    "TorchSpec FP8 quantizer returned an invalid result",
                    [&]() {
                        if (!py::isinstance<py::tuple>(quantized_result)) {
                            throw std::runtime_error("TorchSpec FP8 quantizer must return tensor and scale");
                        }
                        py::tuple quantized = quantized_result.cast<py::tuple>();
                        if (quantized.size() != 2) {
                            throw std::runtime_error("TorchSpec FP8 quantizer must return tensor and scale");
                        }
                        auto quantized_hidden = quantized[0].cast<torch::Tensor>();
                        auto quantized_scale  = quantized[1].cast<torch::Tensor>();
                        if (!quantized_hidden.defined() || quantized_hidden.dim() != 2
                            || quantized_hidden.size(0) != quantizer_input.size(0)
                            || quantized_hidden.size(1) != quantizer_input.size(1)
                            || quantized_hidden.scalar_type() != TORCH_FP8_E4M3_TYPE
                            || quantized_hidden.device() != quantizer_input.device()) {
                            throw std::runtime_error("TorchSpec FP8 quantized hidden states must preserve the "
                                                           "auxiliary tensor layout");
                        }
                        if (!quantized_scale.defined() || quantized_scale.dim() != 2
                            || quantized_scale.size(0) != quantizer_input.size(0)
                            || quantized_scale.size(1) != layer_count_
                            || quantized_scale.scalar_type() != torch::kFloat32
                            || quantized_scale.device() != quantizer_input.device()) {
                            throw std::runtime_error("TorchSpec FP8 scale must have shape [T, "
                                                     + std::to_string(layer_count_) + "] on the capture device");
                        }
                        return std::make_pair(std::move(quantized_hidden), std::move(quantized_scale));
                    });
                stored_auxiliary            = std::move(quantized_tensors.first);
                hidden_states_scale         = std::move(quantized_tensors.second);
                metrics.quantize_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - quantize_start_us;
                quantize_latency_completed  = true;
            } else {
                stored_auxiliary = perform(FailureStage::PREPARE,
                                           FailureDisposition::OPERATIONAL,
                                           "failed to prepare BF16 auxiliary hidden states for storage",
                                           [&]() {
                                               py::gil_scoped_release release;
                                               return auxiliary_hidden_view.to(torch::kBFloat16).contiguous();
                                           });
            }

            task.payloads.push_back(stored_auxiliary);
            if (hidden_states_scale.defined()) {
                task.payloads.push_back(hidden_states_scale);
            }
            torch::Tensor stored_input_ids;
            perform(FailureStage::PREPARE,
                    FailureDisposition::OPERATIONAL,
                    "failed to materialize contiguous hidden-state store payloads",
                    [&]() {
                        // Keep the synchronous H2D copy and all owners, but do not
                        // block normal Python forward while ATen waits for the GPU.
                        // Unwinding restores the GIL before perform classifies errors.
                        py::gil_scoped_release release;
                        stored_auxiliary = stored_auxiliary.contiguous();
                        task.payloads.push_back(stored_auxiliary);
                        stored_input_ids =
                            input_ids
                                .to(torch::TensorOptions().device(last_hidden_states.device()).dtype(torch::kInt64))
                                .contiguous();
                        task.payloads.push_back(stored_input_ids);
                        if (hidden_states_scale.defined()) {
                            hidden_states_scale = hidden_states_scale.contiguous();
                            task.payloads.push_back(hidden_states_scale);
                        }
                        metrics.input_ids_bytes        = static_cast<int64_t>(stored_input_ids.nbytes());
                        metrics.auxiliary_hidden_bytes = static_cast<int64_t>(stored_auxiliary.nbytes());
                        metrics.last_hidden_bytes      = static_cast<int64_t>(stored_last_hidden.nbytes());
                        metrics.scale_bytes =
                            hidden_states_scale.defined() ? static_cast<int64_t>(hidden_states_scale.nbytes()) : 0;
                        metrics.payload_bytes = metrics.input_ids_bytes + metrics.auxiliary_hidden_bytes
                                                + metrics.last_hidden_bytes + metrics.scale_bytes;
                    });

            const auto request_count  = lengths_host.numel();
            const auto batch_sequence = batch_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto batch_id = makeBatchId(batch_sequence,
                                              request_ids_host.data_ptr<int64_t>()[0],
                                              request_ids_host.data_ptr<int64_t>()[request_count - 1]);
            int64_t chunk_capacity = request_count;
            int64_t send_bytes     = 0;
            py::object encode_metadata;
            perform(FailureStage::STORE,
                    FailureDisposition::HARD_CONTRACT,
                    "invalid EagleMooncakeStore send-buffer contract",
                    [&]() {
                        // Use the initialized store, not enable_gpu_direct or the
                        // configured pool: GPU setup may have fallen back to host.
                        const bool direct = store_.attr("_gpu_direct_available").cast<bool>()
                                            && !store_.attr("_gpu_send_buffer").is_none();
                        if (direct) {
                            send_bytes = store_.attr("_gpu_send_buffer").attr("size").cast<int64_t>();
                            if (send_bytes <= 0) {
                                throw std::runtime_error("GPU send buffer must have positive capacity");
                            }
                            encode_metadata = py::module_::import("torchspec.transfer.mooncake.capture_metadata")
                                                  .attr("encode_capture_metadata");
                        } else {
                            const auto pool = store_.attr("_host_buffer_pool");
                            if (pool.is_none() || store_.attr("_async_put_manager").is_none()) {
                                throw std::runtime_error("host capture requires a buffer pool and async PUT manager");
                            }
                            const auto allocated = static_cast<int64_t>(py::len(pool.attr("_buffers")));
                            const auto capacity  = allocated ? allocated : pool.attr("pool_size").cast<int64_t>();
                            if (capacity <= 0) {
                                throw std::runtime_error("host capture requires at least one buffer per request");
                            }
                            chunk_capacity = std::min(capacity, request_count);
                        }
                    });

            // Record the source manifest before any PUT. Only attempted chunks
            // can own source keys; an untouched suffix is refunded after fencing.
            py::list source_keys;
            for (size_t i = 0; i < task.ids.size(); ++i) {
                const auto id = task.ids[i];
                auto       key = makeRequestKey(id);
                std::vector<std::string> keys{key + "_hs", key + "_ids", key + "_lhs", key + "_ready"};
                if (hidden_states_scale.defined()) {
                    keys.push_back(key + "_hss");
                }
                for (const auto& source_key : keys) {
                    source_keys.append(source_key);
                }
                std::lock_guard<std::mutex> lock(state_mutex_);
                auto& reservation = reservations_.at(id);
                if (reservation.generation != task.generations[i] || reservation.task.get() != &task) {
                    throw std::runtime_error("capture source reservation identity changed");
                }
                reservation.keys = std::move(keys);
            }
            auto existing = perform(FailureStage::STORE,
                                    FailureDisposition::OPERATIONAL,
                                    "source-key preflight query failed",
                                    [&]() { return store_.attr("query_keys")(source_keys); });
            if (!py::isinstance<py::dict>(existing) || py::len(existing) != py::len(source_keys)) {
                throw CaptureFailure(FailureStage::STORE,
                                     FailureDisposition::HARD_CONTRACT,
                                     "query_keys must return one strict boolean for every source key");
            }
            for (const auto& key : source_keys) {
                auto states = existing.cast<py::dict>();
                if (!states.contains(key) || !py::isinstance<py::bool_>(states[key])) {
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::HARD_CONTRACT,
                                         "query_keys returned invalid source-key state");
                }
                if (states[key].cast<bool>()) {
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::REQUEST_ERROR,
                                         "capture source key already exists; refusing to overwrite it");
                }
            }
            // TorchSpec derives versioned metadata from the actual payloads and
            // writes it into the existing ready key; no additional protocol key.
            py::list capture_layer_ids;
            for (const auto layer_id : layer_ids_) {
                capture_layer_ids.append(layer_id);
            }
            py::dict capture_metadata;
            capture_metadata["capture_layer_ids"] = capture_layer_ids;
            int64_t token_offset = 0;
            for (int64_t chunk_begin = 0; chunk_begin < request_count;) {
                if (!capture_enabled_.load(std::memory_order_acquire)) {
                    return last_hidden_states;
                }
                py::list batch_keys;
                py::list batch_hidden_states;
                py::list batch_input_ids;
                py::list batch_last_hidden_states;
                py::list batch_hidden_states_scale;
                py::list expected_metadata;
                int64_t  marker_bytes = 0;
                int64_t  chunk_count  = 0;
                while (chunk_count < chunk_capacity && chunk_begin + chunk_count < request_count) {
                    const auto i      = chunk_begin + chunk_count;
                    const auto length = lengths_host.data_ptr<int32_t>()[i];
                    py::object hidden = py::cast(stored_auxiliary.narrow(0, token_offset, length));
                    py::object ids    = py::cast(stored_input_ids.narrow(0, token_offset, length));
                    py::object last   = py::cast(stored_last_hidden.narrow(0, token_offset, length));
                    py::object scale;
                    py::dict shapes;
                    py::dict dtypes;
                    int64_t payload_bytes = 0;
                    const auto describe = [&](const char* name, const py::object& tensor) {
                        shapes[name] = tensor.attr("shape");
                        dtypes[name] = tensor.attr("dtype");
                        payload_bytes += static_cast<int64_t>(tensor.cast<torch::Tensor>().nbytes());
                    };
                    describe("hidden_states", hidden);
                    describe("input_ids", ids);
                    describe("last_hidden_states", last);
                    if (hidden_states_scale.defined()) {
                        scale = py::cast(hidden_states_scale.narrow(0, token_offset, length));
                        describe("hidden_states_scale", scale);
                    }
                    py::dict metadata;
                    metadata["shapes"] = shapes;
                    metadata["dtypes"] = dtypes;
                    if (send_bytes > 0) {
                        // The direct path reuses one buffer per request, then
                        // stages ALL ready markers together. Reuse TorchSpec's
                        // encoder rather than guessing JSON size or token-splitting.
                        const auto marker = encode_metadata(capture_metadata, metadata);
                        const auto size   = static_cast<int64_t>(py::len(marker));
                        if (payload_bytes > send_bytes || size > send_bytes) {
                            throw CaptureFailure(FailureStage::STORE,
                                                 FailureDisposition::REQUEST_ERROR,
                                                 "capture request exceeds the GPU send buffer capacity");
                        }
                        if (size > send_bytes - marker_bytes) {
                            break;
                        }
                        marker_bytes += size;
                    }
                    batch_keys.append(makeRequestKey(task.ids[i]));
                    batch_hidden_states.append(hidden);
                    batch_input_ids.append(ids);
                    batch_last_hidden_states.append(last);
                    if (hidden_states_scale.defined()) {
                        batch_hidden_states_scale.append(scale);
                    }
                    expected_metadata.append(metadata);
                    token_offset += length;
                    ++chunk_count;
                }
                const auto chunk_id = chunk_count == request_count ? batch_id :
                                          batch_id + "-chunk-" + std::to_string(chunk_begin) + "-requests-"
                                              + std::to_string(task.ids[chunk_begin]) + "-"
                                              + std::to_string(task.ids[chunk_begin + chunk_count - 1]);
                prepare_scope.reset();
                {
                    ProfilingScope batch_marker(chunk_id.c_str());
                }
                std::optional<ProfilingScope> admit_scope(std::in_place, "hidden_state_capture.put_batch_admit");
                metrics.has_store_put_latency = true;
                store_put_start_us            = autil::TimeUtility::currentTimeInMicroSeconds();
                store_put_latency_completed   = false;
                py::object acceptance;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    for (int64_t i = chunk_begin; i < chunk_begin + chunk_count; ++i) {
                        const auto& reservation = reservations_.at(task.ids[i]);
                        if (reservation.generation != task.generations[i] || reservation.task.get() != &task) {
                            throw std::runtime_error("capture PUT reservation identity changed");
                        }
                    }
                    for (int64_t i = chunk_begin; i < chunk_begin + chunk_count; ++i) {
                        reservations_.at(task.ids[i]).put_attempted = true;
                    }
                }
                task.store_attempted = true;
                // A previous chunk's flush cannot cover a new partial/throwing PUT.
                task.store_flushed = false;
                try {
                    RTP_LLM_PROFILE_SCOPE("hidden_state_capture.put_batch_python");
                    if (hidden_states_scale.defined()) {
                        acceptance = store_.attr("put_batch")(py::arg("batch_id")            = chunk_id,
                                                              py::arg("request_ids")         = batch_keys,
                                                              py::arg("hidden_states")       = batch_hidden_states,
                                                              py::arg("input_ids")           = batch_input_ids,
                                                              py::arg("last_hidden_states")  = batch_last_hidden_states,
                                                              py::arg("hidden_states_scale") = batch_hidden_states_scale,
                                                              py::arg("capture_metadata")    = capture_metadata);
                    } else {
                        acceptance = store_.attr("put_batch")(py::arg("batch_id")           = chunk_id,
                                                              py::arg("request_ids")        = batch_keys,
                                                              py::arg("hidden_states")      = batch_hidden_states,
                                                              py::arg("input_ids")          = batch_input_ids,
                                                              py::arg("last_hidden_states") = batch_last_hidden_states,
                                                              py::arg("capture_metadata")   = capture_metadata);
                    }
                } catch (const py::error_already_set& e) {
                    const auto python_error = std::string(e.what());
                    if (e.matches(PyExc_FileExistsError)) {
                        recordDuplicateRequestId(task.ids[chunk_begin], batch_keys[0].cast<std::string>(),
                                                 "Mooncake batch admission");
                        throw CaptureFailure(FailureStage::STORE,
                                             FailureDisposition::REQUEST_ERROR,
                                             "TorchSpec hidden-state store rejected duplicate request keys for batch ["
                                                 + chunk_id + "]: " + python_error);
                    }
                    throw CaptureFailure(FailureStage::STORE,
                                         classifyWorkerFailure(e, FailureStage::STORE, FailureDisposition::OPERATIONAL),
                                         "failed to admit TorchSpec hidden-state batch [" + chunk_id + "]: " + python_error);
                } catch (const std::exception& e) {
                    throw CaptureFailure(FailureStage::STORE,
                                         classifyWorkerFailure(e, FailureStage::STORE, FailureDisposition::OPERATIONAL),
                                         "failed to admit TorchSpec hidden-state batch [" + chunk_id + "]: " + e.what());
                } catch (...) {
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::OPERATIONAL,
                                         "failed to admit TorchSpec hidden-state batch [" + chunk_id
                                             + "] with an unknown exception");
                }
                perform(FailureStage::STORE,
                        FailureDisposition::HARD_CONTRACT,
                        ("invalid EagleMooncakeStore.put_batch() metadata for batch [" + chunk_id + "]").c_str(),
                        [&]() {
                            if (!py::isinstance<py::list>(acceptance)
                                || py::len(acceptance) != static_cast<py::ssize_t>(chunk_count)) {
                                throw std::runtime_error("expected one accepted metadata item per chunk request");
                            }
                            const auto accepted = acceptance.cast<py::list>();
                            for (int64_t i = 0; i < chunk_count; ++i) {
                                if (!py::isinstance<py::dict>(accepted[i])) {
                                    throw std::runtime_error("accepted metadata must be a dict");
                                }
                                const auto item     = accepted[i].cast<py::dict>();
                                const auto expected = expected_metadata[i].cast<py::dict>();
                                for (const char* field : {"shapes", "dtypes"}) {
                                    if (!item.contains(field) || !py::isinstance<py::dict>(item[field])
                                        || !py::object(item[field]).equal(py::object(expected[field]))) {
                                        throw std::runtime_error("accepted metadata mismatch for "
                                                                 + batch_keys[i].cast<std::string>() + ", " + field);
                                    }
                                }
                            }
                        });
                metrics.store_put_latency_us += autil::TimeUtility::currentTimeInMicroSeconds() - store_put_start_us;
                store_put_latency_completed = true;
                admit_scope.reset();
                const bool flush_succeeded = perform(FailureStage::STORE,
                                                     FailureDisposition::OPERATIONAL,
                                                     "TorchSpec hidden-state store flush failed",
                                                     [&]() { return flushStore(); });
                task.store_flushed = true;
                if (!flush_succeeded) {
                    finish_failure("TorchSpec hidden-state store reported an asynchronous publish failure",
                                   FailureStage::STORE,
                                   FailureDisposition::OPERATIONAL);
                    return last_hidden_states;
                }
                task.confirmed_requests += chunk_count;
                // Reuse host leases only after draining. All owners and quota stay
                // retained until processTask also fences local producer/staging work.
                chunk_begin += chunk_count;
            }
        } catch (const CaptureFailure& failure) {
            finish_failure(failure.what(), failure.failureStage(), failure.failureDisposition());
            return last_hidden_states;
        } catch (const std::exception& e) {
            finish_failure(std::string("unexpected hidden-state publisher failure: ") + e.what(),
                           FailureStage::PREPARE,
                           classifyWorkerFailure(e, FailureStage::PREPARE, FailureDisposition::OPERATIONAL));
            return last_hidden_states;
        } catch (...) {
            finish_failure("unexpected hidden-state publisher failure with an unknown exception",
                           FailureStage::PREPARE,
                           FailureDisposition::OPERATIONAL);
            return last_hidden_states;
        }
    }

    if (owner_) {
        metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
        metrics.request_count      = lengths_host.numel();
        metrics.token_count        = total_tokens;
        recordPublish(/*success=*/true, metrics);
    }
    task.succeeded = true;
    return last_hidden_states;
}

bool HiddenStateCapturePublisher::flushStore() {
    RTP_LLM_PROFILE_SCOPE("hidden_state_capture.flush");
    {
        RTP_LLM_PROFILE_SCOPE("hidden_state_capture.flush_python");
        store_.attr("flush")();
    }
    return observeAsyncErrors(FailureStage::STORE, "after publish flush");
}

void HiddenStateCapturePublisher::processTask(const std::shared_ptr<PublishTask>& task) {
    // This is the only publication entry. Maintenance never re-enters it, even
    // if preparation, a partial PUT, or its first completion attempt failed.
    task->completion_pending = true;
    if (fatal_kind_.load(std::memory_order_acquire) == HiddenStateCaptureFatalKind::NONE) {
        try {
            c10::DeviceGuard                device_guard(task->packed.device());
            std::optional<c10::StreamGuard> stream_guard;
            if (!task->packed.device().is_cpu()) {
                c10::impl::VirtualGuardImpl impl(task->packed.device().type());
                task->worker_stream = impl.getStreamFromGlobalPool(task->packed.device(), false);
                stream_guard.emplace(*task->worker_stream);
            }
            // Strong refs protect allocation lifetime; the producer event
            // establishes visibility on this independent worker stream.
            if (task->ready) {
                task->ready->synchronize();
            }
            task->producer_complete = true;
            if (!task->dropped && capture_enabled_.load(std::memory_order_acquire)) {
                (void)publishOnWorker(*task);
            }
        } catch (const std::exception& e) {
            completeFailure(e.what(),
                            FailureStage::PREPARE,
                            classifyWorkerFailure(e, FailureStage::PREPARE, FailureDisposition::OPERATIONAL),
                            "worker prepare; retaining capture resources until completion");
        } catch (...) {
            latchFatal(FailureStage::PREPARE, HiddenStateCaptureFatalKind::RUNTIME,
                       "unknown worker preparation failure; retaining capture resources");
        }
    }
    completeTask(task);
}

void HiddenStateCapturePublisher::completeTask(const std::shared_ptr<PublishTask>& task) {
    if (!task->completion_pending || task->local_complete) {
        return;
    }
    // Under the CPU capture contract, these owners have no asynchronous
    // dependencies before any PUT. Inspect every retained tensor, not just the
    // packed input: a failed preparation/quantizer may have retained other data.
    // This proof uses metadata only and remains valid after a fatal latch; it
    // must never permit a device/event/stream operation or a fatal store drain.
    const auto cpu_only = [](const torch::Tensor& tensor) {
        return !tensor.defined() || tensor.device().is_cpu();
    };
    bool local_safe = !task->store_attempted && !task->ready && !task->worker_stream
                      && task->packed.defined() && cpu_only(task->packed) && cpu_only(task->input_ids)
                      && cpu_only(task->input_lengths) && cpu_only(task->request_ids)
                      && std::all_of(task->payloads.begin(), task->payloads.end(), cpu_only);
    // A disabled capture contract is not a disabled maintenance path. Only a
    // fatal latch forbids further native operations on potentially unsafe state.
    if (!local_safe && fatal_kind_.load(std::memory_order_acquire) == HiddenStateCaptureFatalKind::NONE) {
        try {
            c10::DeviceGuard                device_guard(task->packed.device());
            std::optional<c10::StreamGuard> stream_guard;
            if (task->worker_stream) {
                stream_guard.emplace(*task->worker_stream);
            }
            if (!task->producer_complete) {
                if (task->ready) {
                    task->ready->synchronize();
                }
                task->producer_complete = true;
            }
            if (task->store_attempted && !task->store_flushed) {
                py::gil_scoped_acquire gil;
                try {
                    (void)flushStore();
                    task->store_flushed = true;
                } catch (const std::exception& e) {
                    completeFailure(e.what(),
                                    FailureStage::STORE,
                                    classifyWorkerFailure(e, FailureStage::STORE, FailureDisposition::OPERATIONAL),
                                    "failed PUT drain; retaining source and staging for completion retry");
                } catch (...) {
                    latchFatal(FailureStage::STORE, HiddenStateCaptureFatalKind::RUNTIME,
                               "unknown PUT drain failure; retaining source and staging");
                }
            }
            if ((!task->store_attempted || task->store_flushed)
                && fatal_kind_.load(std::memory_order_acquire) == HiddenStateCaptureFatalKind::NONE) {
                if (task->worker_stream) {
                    task->worker_stream->synchronize();
                }
#if USING_CUDA || USING_ROCM
                // A partial admission may stage on a store-owned stream before
                // acquiring a Future. Draining alone does not fence that copy.
                if (task->store_attempted && !task->succeeded && !task->packed.device().is_cpu()) {
                    runtimeSyncAndCheck();
                }
#endif
                local_safe = true;
            }
        } catch (const std::exception& e) {
            completeFailure(e.what(),
                            FailureStage::PREPARE,
                            classifyWorkerFailure(e, FailureStage::PREPARE, FailureDisposition::OPERATIONAL),
                            "local completion fence; retaining capture resources");
        } catch (...) {
            latchFatal(FailureStage::PREPARE, HiddenStateCaptureFatalKind::RUNTIME,
                       "unknown local completion failure; retaining capture resources");
        }
    }
    if (local_safe) {
        task->payloads.clear();
        task->packed        = torch::Tensor();
        task->input_ids     = torch::Tensor();
        task->input_lengths = torch::Tensor();
        task->request_ids   = torch::Tensor();
        task->ready.reset();
        task->worker_stream.reset();
        task->completion_pending = false;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        task->local_complete = local_safe;
        for (size_t i = 0; i < task->ids.size(); ++i) {
            auto entry = reservations_.find(task->ids[i]);
            if (entry == reservations_.end() || entry->second.generation != task->generations[i]
                || entry->second.task != task) {
                continue;
            }
            if (local_safe && !entry->second.put_attempted) {
                releaseReservation(entry);
            } else if (!local_safe || i >= task->confirmed_requests
                       || entry->second.state == ReservationState::CLEANUP_PENDING) {
                entry->second.state = ReservationState::CLEANUP_PENDING;
            } else {
                entry->second.state = ReservationState::PUBLISHED;
            }
        }
    }
}

void HiddenStateCapturePublisher::pollSourceKeys() {
    std::vector<std::shared_ptr<PublishTask>> completing;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        std::unordered_set<PublishTask*> seen;
        for (const auto& entry : reservations_) {
            const auto& task = entry.second.task;
            // Queued tasks have completion_pending=false. Dropped tasks enter
            // this path directly; normal tasks enter only after publication ends.
            if (task && task->completion_pending && !task->local_complete && seen.insert(task.get()).second) {
                completing.push_back(task);
            }
        }
    }
    for (const auto& task : completing) {
        // Even after fatal, visit every task for metadata-proven CPU-only
        // release. completeTask still quarantines all possible async owners.
        completeTask(task);
    }
    if (fatal_kind_.load(std::memory_order_acquire) != HiddenStateCaptureFatalKind::NONE) {
        return;  // No source query/reclaim or other native cleanup after fatal.
    }
    struct Candidate {
        int64_t                      id;
        uint64_t                     generation;
        std::shared_ptr<PublishTask>  task;
        std::vector<std::string>      keys;
    };
    std::vector<Candidate> candidates;
    const auto matches = [](const Reservation& reservation, const Candidate& candidate) {
        return reservation.generation == candidate.generation && reservation.task == candidate.task
               && reservation.task && reservation.task->local_complete && reservation.put_attempted
               && reservation.keys == candidate.keys;
    };
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (auto entry = reservations_.begin(); entry != reservations_.end();) {
            const auto& task = entry->second.task;
            if (!task || !task->local_complete) {
                ++entry;
                continue;
            }
            if (!entry->second.put_attempted) {
                auto released = entry++;
                releaseReservation(released);
            } else {
                if (maintenance_allocation_probe_) {
                    maintenance_allocation_probe_();
                }
                candidates.push_back({entry->first, entry->second.generation, task, entry->second.keys});
                ++entry;
            }
        }
    }
    if (candidates.empty()) {
        recordQuota();
        return;
    }
    py::gil_scoped_acquire gil;
    try {
        py::list keys;
        for (const auto& candidate : candidates) {
            for (const auto& key : candidate.keys) {
                keys.append(key);
            }
            keys.append(makeRequestKey(candidate.id) + "_abandoned");
        }
        auto result = store_.attr("query_keys")(keys);
        if (!py::isinstance<py::dict>(result) || py::len(result) != py::len(keys)) {
            throw std::runtime_error("query_keys returned an incomplete source manifest");
        }
        auto states = result.cast<py::dict>();
        // Validate the entire answer before changing the ledger. An exception,
        // missing entry, integer error code, or malformed result is not absence.
        for (const auto& key : keys) {
            if (!states.contains(key) || !py::isinstance<py::bool_>(states[key])) {
                throw std::runtime_error("query_keys must return strict booleans for every requested key");
            }
        }
        for (const auto& candidate : candidates) {
            try {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    auto                        entry = reservations_.find(candidate.id);
                    if (entry == reservations_.end() || !matches(entry->second, candidate)) {
                        continue;
                    }
                }
                const bool missing = !candidate.keys.empty()
                                     && std::all_of(candidate.keys.begin(), candidate.keys.end(), [&](const auto& key) {
                                            return !states[py::str(key)].cast<bool>();
                                        });
                // Only the delivery owner may waive a training read. Timeout,
                // serving cancellation and partial PUT alone never authorize this.
                // Every candidate has completed local copies and drained native PUTs;
                // the single publisher worker cannot publish this request again.
                const auto request_key = makeRequestKey(candidate.id);
                bool       releasable  = missing;
                if (states[py::str(request_key + "_abandoned")].cast<bool>()) {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        auto entry = reservations_.find(candidate.id);
                        if (entry == reservations_.end() || !matches(entry->second, candidate)) {
                            continue;
                        }
                        entry->second.state = ReservationState::CLEANUP_PENDING;
                    }
                    auto reclaimed = store_.attr("reclaim_abandoned_capture")(request_key);
                    if (!py::isinstance<py::bool_>(reclaimed)) {
                        throw std::runtime_error("reclaim_abandoned_capture must return a strict boolean");
                    }
                    releasable = reclaimed.cast<bool>();
                }
                if (releasable) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    auto                        entry = reservations_.find(candidate.id);
                    if (entry != reservations_.end() && matches(entry->second, candidate)) {
                        releaseReservation(entry);
                    }
                }
            } catch (const std::exception& e) {
                const auto disposition = classifyWorkerFailure(e, FailureStage::STORE, FailureDisposition::OPERATIONAL);
                if (disposition == FailureDisposition::FATAL) {
                    // Preserve the owner-level fatal latch and whole-poll
                    // failure handling; never continue native cleanup after it.
                    throw;
                }
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    auto                        entry = reservations_.find(candidate.id);
                    if (entry != reservations_.end() && matches(entry->second, candidate)) {
                        entry->second.state = ReservationState::CLEANUP_PENDING;
                    }
                }
                RtpLLMHiddenStateCaptureMetricsCollector collector;
                collector.cleanup_query_failure_qps = true;
                reportMetrics(collector);
                RTP_LLM_INTERVAL_LOG(300,
                                     WARN,
                                     "[TorchSpec hidden-state publisher] request %ld cleanup unknown; quota retained: %s",
                                     candidate.id,
                                     e.what());
            }
        }
    } catch (const std::exception& e) {
        const auto disposition = classifyWorkerFailure(e, FailureStage::STORE, FailureDisposition::OPERATIONAL);
        if (disposition == FailureDisposition::FATAL) {
            completeFailure(e.what(), FailureStage::STORE, disposition, "source cleanup query");
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (const auto& candidate : candidates) {
                auto entry = reservations_.find(candidate.id);
                if (entry != reservations_.end() && matches(entry->second, candidate)) {
                    entry->second.state = ReservationState::CLEANUP_PENDING;
                }
            }
        }
        RtpLLMHiddenStateCaptureMetricsCollector collector;
        collector.cleanup_query_failure_qps = true;
        reportMetrics(collector);
        RTP_LLM_INTERVAL_LOG(
            300, WARN, "[TorchSpec hidden-state publisher] source cleanup unknown; quota retained: %s", e.what());
    }
    recordQuota();
}

void HiddenStateCapturePublisher::workerLoop() noexcept {
    const auto report_worker_failure = [this](const std::exception& error, const char* phase) noexcept {
        const auto disposition =
            classifyWorkerFailure(error, FailureStage::SHUTDOWN, FailureDisposition::HARD_CONTRACT);
        try {
            completeFailure(error.what(), FailureStage::SHUTDOWN, disposition, phase);
        } catch (const std::exception& reporting_error) {
            const auto kind = nativeFatalKind(reporting_error);
            latchFatal(FailureStage::SHUTDOWN,
                       kind == HiddenStateCaptureFatalKind::NONE ? HiddenStateCaptureFatalKind::RUNTIME : kind,
                       "failed to report publisher worker failure");
        } catch (...) {
            latchFatal(FailureStage::SHUTDOWN,
                       HiddenStateCaptureFatalKind::RUNTIME,
                       "failed to report publisher worker failure");
        }
    };
    // Keep the current task reachable if preparation/completion diagnostics throw.
    std::shared_ptr<PublishTask> task;
    try {
        initialize(local_rank_);
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                worker_active_ = false;
                state_changed_.notify_all();
                state_changed_.wait_for(lock,
                                        std::chrono::milliseconds(std::min<int64_t>(limits_.poll_interval_ms, 60000)),
                                        [this] { return stopping_ || cleanup_requested_ || !pending_.empty(); });
                const bool cleanup_requested = std::exchange(cleanup_requested_, false);
                if (!pending_.empty()) {
                    task = std::move(pending_.front());
                    pending_.pop_front();
                } else if (stopping_ && !cleanup_requested) {
                    break;
                }
                worker_active_ = true;
            }
            if (task) {
                processTask(task);
                task.reset();
            }
            pollSourceKeys();
        }
    } catch (const std::exception& e) {
        report_worker_failure(e, "publisher worker");
    } catch (...) {
        latchFatal(FailureStage::SHUTDOWN, HiddenStateCaptureFatalKind::RUNTIME, "unknown publisher worker failure");
    }
    // No more admission after an unexpected exit. Drain the existing queue one
    // owner at a time: allocating a snapshot here could fail for the same OOM.
    // Never publish these tasks; completeTask alone proves CPU-only safety and
    // quarantines any device/event/PUT dependency after a fatal latch.
    capture_enabled_.store(false, std::memory_order_release);
    for (;;) {
        try {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                stopping_ = true;
                if (!task) {
                    if (pending_.empty()) {
                        break;
                    }
                    task = std::move(pending_.front());
                    pending_.pop_front();
                }
            }
            task->completion_pending = true;
            completeTask(task);
        } catch (const std::exception& e) {
            report_worker_failure(e, "queued task completion after worker exit");
            if (!task) {
                break;  // A failed queue lock cannot be retried in a busy loop.
            }
        } catch (...) {
            latchFatal(FailureStage::SHUTDOWN, HiddenStateCaptureFatalKind::RUNTIME,
                       "unknown queued task completion failure");
            if (!task) {
                break;
            }
        }
        task.reset();
    }
    try {
        if (store_ && !store_.is_none()) {
            pollSourceKeys();
        }
    } catch (const std::exception& e) {
        report_worker_failure(e, "final source cleanup poll");
    } catch (...) {
        latchFatal(
            FailureStage::SHUTDOWN, HiddenStateCaptureFatalKind::RUNTIME, "unknown final source cleanup failure");
    }
    closeStore();
    try {
        std::lock_guard<std::mutex> lock(state_mutex_);
        worker_active_   = false;
        worker_finished_ = true;
    } catch (...) {
        latchFatal(FailureStage::SHUTDOWN, HiddenStateCaptureFatalKind::RUNTIME, "failed to finish publisher worker");
    }
    state_changed_.notify_all();
}

bool HiddenStateCapturePublisher::flushPending() {
    auto wait = [this] {
        std::unique_lock<std::mutex> lock(state_mutex_);
        if (!worker_finished_ && !stopping_) {
            // Request one maintenance pass, not an unbounded wait for every
            // outstanding source/completion to become reclaimable.
            cleanup_requested_ = true;
            state_changed_.notify_all();
        }
        state_changed_.wait(
            lock, [this] { return worker_finished_ || (pending_.empty() && !worker_active_ && !cleanup_requested_); });
        if (!pending_.empty()) {
            return false;
        }
        return std::all_of(reservations_.begin(), reservations_.end(), [](const auto& entry) {
            return !entry.second.task || entry.second.task->local_complete;
        });
    };
    if (Py_IsInitialized() && PyGILState_Check()) {
        py::gil_scoped_release release;
        return wait();
    }
    return wait();
}

void HiddenStateCapturePublisher::closeStore() noexcept {
    try {
        // Unconfirmed asynchronous dependencies must not lose their buffer owners.
        // Retention is bounded by the source ledger; no new captures are admitted.
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (const auto& entry : reservations_) {
                if (entry.second.task && (!entry.second.task->local_complete || entry.second.put_attempted)) {
                    RTP_LLM_LOG_WARNING(
                        "[TorchSpec hidden-state publisher] shutdown retains store ownership for unresolved source keys/dependencies");
                    return;
                }
            }
        }
        py::gil_scoped_acquire gil;
        if (store_ && !store_.is_none()) {
            try {
                store_.attr("close")();
            } catch (const std::exception& e) {
                recordFailure(FailureStage::SHUTDOWN,
                              classifyWorkerFailure(e, FailureStage::SHUTDOWN, FailureDisposition::OPERATIONAL));
                RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] store close failed: %s", e.what());
            }
        }
        store_           = py::object();
        quantize_fn_     = py::object();
        mooncake_config_ = py::object();
    } catch (const std::exception& e) {
        (void)classifyWorkerFailure(e, FailureStage::SHUTDOWN, FailureDisposition::HARD_CONTRACT);
        capture_enabled_.store(false, std::memory_order_release);
    } catch (...) {
        latchFatal(
            FailureStage::SHUTDOWN, HiddenStateCaptureFatalKind::RUNTIME, "unknown publisher store close failure");
    }
}

void HiddenStateCapturePublisher::flushAndClose() noexcept {
    auto close = [this] {
        std::lock_guard<std::mutex> close_lock(close_mutex_);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        state_changed_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        capture_enabled_.store(false, std::memory_order_release);
    };
    try {
        if (Py_IsInitialized() && PyGILState_Check()) {
            py::gil_scoped_release release;
            close();
        } else {
            close();
        }
    } catch (...) {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] failed to join publisher worker");
    }
}

}  // namespace rtp_llm
