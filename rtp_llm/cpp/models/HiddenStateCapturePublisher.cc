#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <ATen/ThreadLocalState.h>
#include <ATen/record_function.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include "autil/TimeUtility.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

// Keep this union in sync with hidden_state_capture.py::_FATAL_CAPTURE_MARKERS.
// Lowercase during comparison, avoiding allocations on allocator failure paths.
bool isFatalCaptureMessage(std::string_view message) noexcept {
    for (const std::string_view marker : {"out of memory",
                                          "outofmemoryerror",
                                          "cuda error",
                                          "cuda driver error",
                                          "cuda runtime error",
                                          "hip error",
                                          "hip runtime error",
                                          "cublas_status_alloc_failed",
                                          "cusparse_status_alloc_failed",
                                          "cudnn_status_alloc_failed",
                                          "hipblas_status_alloc_failed",
                                          "rocblas_status_memory_error",
                                          "cublas_status_execution_failed",
                                          "cusparse_status_execution_failed",
                                          "cudnn_status_execution_failed",
                                          "hipblas_status_execution_failed",
                                          "rocblas_status_internal_error",
                                          "illegal memory access",
                                          "illegal address",
                                          "invalid device pointer",
                                          "misaligned address",
                                          "device-side assert",
                                          "device side assert",
                                          "launch failure",
                                          "launch failed",
                                          "launch timed out",
                                          "launch_failure",
                                          "launch_failed",
                                          "cudaerrorillegaladdress",
                                          "cuda_error_illegal_address",
                                          "hiperrorillegaladdress",
                                          "cudaerrormemoryallocation",
                                          "hiperroroutofmemory",
                                          "nccl error",
                                          "collective failed",
                                          "collective failure",
                                          "collective operation timeout",
                                          "collective timed out"}) {
        if (std::search(message.begin(),
                        message.end(),
                        marker.begin(),
                        marker.end(),
                        [](unsigned char ch, char expected) { return std::tolower(ch) == expected; })
            != message.end()) {
            return true;
        }
    }
    return false;
}

bool isFatalCapturePythonException(py::handle error) {
    py::gil_scoped_acquire gil;
    try {
        if (py::isinstance(error, py::handle(PyExc_MemoryError))
            || (py::isinstance(error, py::handle(PyExc_BaseException))
                && !py::isinstance(error, py::handle(PyExc_Exception)))) {
            return true;
        }
        // py::getattr with a default clears every Python error, not just
        // AttributeError. Only genuinely missing optional types are benign.
        const auto optional_attribute = [](py::handle object, const char* name) -> py::object {
            try {
                return py::getattr(object, name);
            } catch (const py::error_already_set& lookup_error) {
                if (!lookup_error.matches(PyExc_AttributeError)) {
                    throw;
                }
                return py::none();
            }
        };
        const auto torch_module = py::module_::import("torch");
        for (const char* name : {"OutOfMemoryError", "AcceleratorError"}) {
            const auto type = optional_attribute(torch_module, name);
            if (!type.is_none() && py::isinstance(error, type)) {
                return true;
            }
        }
        const auto distributed = optional_attribute(torch_module, "distributed");
        const auto dist_error  = optional_attribute(distributed, "DistError");
        return !dist_error.is_none() && py::isinstance(error, dist_error);
    } catch (...) {
        // Classification itself can fail under memory pressure. Never let a
        // secondary exception escape into an operational-failure catch.
        return true;
    }
}

bool isFatalCapturePythonError(const py::error_already_set& error) {
    return isFatalCapturePythonException(error.value()) || isFatalCaptureMessage(error.what());
}

}  // namespace

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

struct HiddenStateCapturePublisher::PublishTask {
    at::ThreadLocalState          thread_state;
    torch::Tensor                 packed;
    torch::Tensor                 input_ids;
    torch::Tensor                 input_lengths;
    torch::Tensor                 request_ids;
    std::vector<int64_t>          ids;
    int64_t                       expected_rows{0};
    std::string                   layout_context;
    c10::Device                   fallback_device{c10::DeviceType::CPU};
    std::shared_ptr<torch::Event> ready;
    std::optional<c10::Stream>    producer_stream;
    std::vector<torch::Tensor>    payloads;
    bool                          store_attempted{false};
    bool                          store_flushed{false};
    bool                          local_complete{false};
    bool                          local_fenced{false};
    bool                          succeeded{false};
    bool                          dropped{false};
    bool                          cleanup_started{false};
    int64_t                       quota_bytes{0};
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
    } catch (const py::error_already_set& error) {
        if (isFatalCapturePythonError(error)) {
            throw;
        }
        throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + ": " + error.what());
    } catch (const c10::Error&) {
        throw;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& e) {
        throw CaptureFailure(failure_stage,
                             isFatalCaptureMessage(e.what()) ? FailureDisposition::FATAL : failure_disposition,
                             std::string(context) + ": " + e.what());
    } catch (...) {
        throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + " with an unknown exception");
    }
}

HiddenStateCapturePublisher::HiddenStateCapturePublisher(int64_t                      layer_count,
                                                         int64_t                      hidden_size,
                                                         HiddenStateCaptureDtype      capture_dtype,
                                                         c10::ScalarType              model_dtype,
                                                         bool                         owner,
                                                         int                          local_rank,
                                                         kmonitor::MetricsReporterPtr metrics_reporter,
                                                         HiddenStateCaptureLimits     limits,
                                                         std::vector<int64_t>         layer_ids,
                                                         std::string                  initialization_error):
    layer_count_(layer_count),
    layer_ids_(std::move(layer_ids)),
    hidden_size_(hidden_size),
    local_rank_(local_rank),
    limits_(limits),
    capture_dtype_(capture_dtype),
    model_dtype_(model_dtype),
    owner_(owner),
    metrics_reporter_(owner ? std::move(metrics_reporter) : nullptr) {
    reservation_lifetime_->publisher = this;
    int64_t bytes                    = 0;
    if (!initialization_error.empty()) {
        completeFailure(initialization_error,
                        FailureStage::INITIALIZATION,
                        FailureDisposition::HARD_CONTRACT,
                        "publisher configuration");
        return;
    }
    if (limits_.max_inflight_samples <= 0 || limits_.max_inflight_bytes <= 0 || limits_.max_inflight_tokens <= 0
        || limits_.max_pending_batches <= 0 || limits_.poll_interval_ms <= 0 || !payloadBytes(1, bytes)
        || layer_ids_.size() != static_cast<size_t>(layer_count_)
        || std::any_of(layer_ids_.begin(), layer_ids_.end(), [](int64_t id) { return id < 0; })
        || std::unordered_set<int64_t>(layer_ids_.begin(), layer_ids_.end()).size() != layer_ids_.size()) {
        completeFailure("capture requires positive limits and a valid model layout",
                        FailureStage::INITIALIZATION,
                        FailureDisposition::HARD_CONTRACT,
                        "publisher configuration");
        return;
    }
    if (!owner_ || !capture_enabled_.load(std::memory_order_acquire)) {
        return;
    }
    if (!Py_IsInitialized()) {
        completeFailure("Python interpreter is not initialized",
                        FailureStage::INITIALIZATION,
                        FailureDisposition::HARD_CONTRACT,
                        "publisher worker startup");
        return;
    }
    // Preallocate the exceptional lifetime escape hatch before admitting work;
    // shutdown must not need to allocate while handling an allocator failure.
    quarantine_ = std::make_unique<LifetimeQuarantine>();
    try {
        worker_active_ = true;  // Initialization belongs to the worker and is part of drain.
        worker_ = std::thread(&HiddenStateCapturePublisher::workerLoop, this);
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        worker_active_   = false;
        worker_finished_ = true;
        completeFailure(
            error.what(), FailureStage::INITIALIZATION, FailureDisposition::OPERATIONAL, "publisher worker startup");
    }
}

HiddenStateCapturePublisher::~HiddenStateCapturePublisher() {
    {
        // Wait for an in-progress cancellation, then make all surviving tokens
        // inert before destroying any publisher state. No owner is promoted.
        std::lock_guard<std::mutex> lock(reservation_lifetime_->mutex);
        reservation_lifetime_->publisher = nullptr;
    }
    flushAndClose();
    // If shutdown was entered without a live interpreter, intentionally leak the
    // Python handles rather than calling into a finalized interpreter.
    (void)mooncake_config_.release();
    (void)store_.release();
    (void)quantize_fn_.release();
}

bool HiddenStateCapturePublisher::payloadBytes(int64_t tokens, int64_t& bytes, int64_t* store_bytes) const {
    if (tokens <= 0 || layer_count_ <= 0 || hidden_size_ <= 0 || !c10::isFloatingType(model_dtype_)
        || (capture_dtype_ != HiddenStateCaptureDtype::BF16 && capture_dtype_ != HiddenStateCaptureDtype::FP8_E4M3)) {
        return false;
    }
    constexpr int64_t limit = std::numeric_limits<int64_t>::max();
    // Two int64 ID tensors per token: retained input IDs and store input IDs.
    // Per request: int32 length, int64 request-ID tensor and the task's ID ledger.
    constexpr int64_t ids_per_token           = 2 * sizeof(int64_t);
    constexpr int64_t metadata_per_request    = sizeof(int32_t) + 2 * sizeof(int64_t);
    const int64_t     source_element_bytes    = c10::elementSize(model_dtype_);
    const int64_t     auxiliary_element_bytes = capture_dtype_ == HiddenStateCaptureDtype::BF16 ? 2 : 1;
    const int64_t     last_hidden_extra       = 2 - auxiliary_element_bytes;
    if (layer_count_ == limit
        || layer_count_ + 1 > (limit - last_hidden_extra) / (source_element_bytes + auxiliary_element_bytes)) {
        return false;
    }
    const int64_t coefficient =
        (source_element_bytes + auxiliary_element_bytes) * (layer_count_ + 1) + last_hidden_extra;
    int64_t scale_bytes = 0;
    if (capture_dtype_ == HiddenStateCaptureDtype::FP8_E4M3) {
        if (layer_count_ > (limit - ids_per_token) / sizeof(float)) {
            return false;
        }
        scale_bytes = layer_count_ * sizeof(float);
    }
    const int64_t extra = ids_per_token + scale_bytes;
    if (hidden_size_ > (limit - extra) / coefficient) {
        return false;
    }
    const int64_t per_token = coefficient * hidden_size_ + extra;
    if (tokens > (limit - metadata_per_request) / per_token) {
        return false;
    }
    bytes = tokens * per_token + metadata_per_request;
    if (store_bytes) {
        *store_bytes =
            tokens * ((auxiliary_element_bytes * layer_count_ + 2) * hidden_size_ + sizeof(int64_t) + scale_bytes);
    }
    return true;
}

bool HiddenStateCapturePublisher::retainedBytes(const PublishTask& task, int64_t& bytes) const {
    constexpr int64_t limit = std::numeric_limits<int64_t>::max();
    if (task.ids.capacity() > static_cast<size_t>(limit / sizeof(int64_t))) {
        return false;
    }
    bytes = task.ids.capacity() * sizeof(int64_t);
    std::array<const void*, 8> seen{};
    size_t                     count   = 0;
    const auto                 account = [&](const torch::Tensor& tensor) {
        if (!tensor.defined()) {
            return true;
        }
        const auto  storage = tensor.storage();
        const void* id      = storage.unsafeGetStorageImpl();
        if (std::find(seen.begin(), seen.begin() + count, id) != seen.begin() + count) {
            return true;
        }
        if (count == seen.size() || storage.nbytes() > static_cast<size_t>(limit - bytes)) {
            return false;
        }
        seen[count++] = id;
        bytes += storage.nbytes();
        return true;
    };
    if (!account(task.packed) || !account(task.input_ids) || !account(task.input_lengths)
        || !account(task.request_ids)) {
        return false;
    }
    for (const auto& tensor : task.payloads) {
        if (!account(tensor)) {
            return false;
        }
    }
    return true;
}

HiddenStateCaptureReservationPtr HiddenStateCapturePublisher::tryReserve(int64_t request_id, int64_t valid_tokens) {
    int64_t bytes = 0;
    if (!owner_ || !capture_enabled_.load(std::memory_order_acquire) || request_id < 0
        || !payloadBytes(valid_tokens, bytes)) {
        return {};
    }
    // Outlives the lock so allocation failures cannot run its cancellation
    // callback while holding state_mutex_.
    HiddenStateCaptureReservationPtr reservation;
    try {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (stopping_ || !capture_enabled_.load(std::memory_order_acquire)
            || reservations_.find(request_id) != reservations_.end()
            || reservations_.size() >= static_cast<size_t>(limits_.max_inflight_samples)
            || bytes > limits_.max_inflight_bytes - reserved_bytes_
            || valid_tokens > limits_.max_inflight_tokens - reserved_tokens_) {
            return {};
        }
        const auto generation = ++reservation_generation_;
        reservation           = std::make_shared<HiddenStateCaptureReservation>(
            [lifetime = reservation_lifetime_, request_id, generation]() noexcept {
                try {
                    std::lock_guard<std::mutex> lock(lifetime->mutex);
                    if (lifetime->publisher) {
                        lifetime->publisher->cancelReserved(request_id, generation);
                    }
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_WARNING("hidden-state capture reservation cancellation failed: %s", error.what());
                } catch (...) {
                    RTP_LLM_LOG_WARNING("hidden-state capture reservation cancellation failed with unknown exception");
                }
            });
        reservations_.emplace(request_id, Reservation{generation, valid_tokens, bytes, false, nullptr, {}, false});
        reserved_bytes_ += bytes;
        reserved_tokens_ += valid_tokens;
        return reservation;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("hidden-state capture reservation failed: %s", error.what());
        return {};
    }
}

bool HiddenStateCapturePublisher::hasReservation(int64_t request_id) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto                  entry = reservations_.find(request_id);
    return entry != reservations_.end() && !entry->second.submitted;
}

int64_t HiddenStateCapturePublisher::inflightSamples() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return static_cast<int64_t>(reservations_.size());
}

int64_t HiddenStateCapturePublisher::inflightBytes() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return reserved_bytes_;
}

void HiddenStateCapturePublisher::discardReserved(int64_t request_id) {
    cancelReserved(request_id, std::nullopt);
}

void HiddenStateCapturePublisher::cancelReserved(int64_t request_id, std::optional<uint64_t> generation) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto                        entry = reservations_.find(request_id);
    if (entry == reservations_.end() || entry->second.submitted
        || (generation && entry->second.generation != *generation)) {
        return;
    }
    reserved_bytes_ -= entry->second.bytes;
    reserved_tokens_ -= entry->second.tokens;
    reservations_.erase(entry);
    state_changed_.notify_all();
}

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
    if (!tensor.defined() || tensor.dim() != 2 || tensor.layout() != torch::kStrided) {
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
    return {failure_count_.load(std::memory_order_relaxed), broken_rejection_count_.load(std::memory_order_relaxed)};
}

HiddenStateCaptureFatalState HiddenStateCapturePublisher::fatalState() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    auto                        state = fatal_state_;
    state.failed                      = fatal_failed_.load(std::memory_order_acquire);
    return state;
}

std::string HiddenStateCapturePublisher::diagnostic() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return diagnostic_;
}

bool HiddenStateCapturePublisher::rejectRequest(std::string error, bool operational) {
    return rejectFailure(std::move(error),
                         FailureStage::PREPARE,
                         operational ? FailureDisposition::OPERATIONAL : FailureDisposition::REQUEST_ERROR);
}

bool HiddenStateCapturePublisher::rejectDuplicateRequest(int64_t request_id) {
    recordDuplicateRequestId(request_id, std::to_string(request_id), "capture admission metadata");
    return rejectRequest("duplicate capture request id " + std::to_string(request_id));
}

void HiddenStateCapturePublisher::latchFatal(const char* error_message) noexcept {
    // Latch health before any metric/log/string allocation, including when the
    // original failure is an allocator failure. Diagnostics are best-effort.
    capture_enabled_.store(false, std::memory_order_release);
    fatal_failed_.store(true, std::memory_order_release);
    try {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!fatal_state_.failed) {
            fatal_state_.message = error_message;
            fatal_state_.failed  = true;
        }
    } catch (...) {}
}

void HiddenStateCapturePublisher::reportMetrics(RtpLLMHiddenStateCaptureMetricsCollector& collector) {
    if (owner_ && metrics_reporter_) {
        metrics_reporter_->report<RtpLLMHiddenStateCaptureMetrics, RtpLLMHiddenStateCaptureMetricsCollector>(
            nullptr, &collector);
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
    collector.hard_contract_failure_qps  = failure_disposition == FailureDisposition::HARD_CONTRACT;
    collector.request_error_failure_qps  = failure_disposition == FailureDisposition::REQUEST_ERROR;
    collector.operational_failure_qps    = failure_disposition == FailureDisposition::OPERATIONAL;
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
        broken = broken_reason_.has_value() || fatal_failed_.load(std::memory_order_acquire);
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
    if (fatal_failed_.load(std::memory_order_acquire)) {
        return false;
    }
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
        success           = false;
        const auto detail = py::str(error).cast<std::string>();
        // Completion errors belong to their original store batch. A true
        // accelerator/allocator failure still latches owner health; it must not
        // become a serving error on whichever batch happens to run next.
        if (isFatalCapturePythonException(error) || isFatalCaptureMessage(detail)) {
            completeFailure(detail, failure_stage, FailureDisposition::FATAL, "asynchronous store completion");
            return false;
        } else {
            recordFailure(failure_stage, FailureDisposition::OPERATIONAL);
        }
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
    if (!capture_enabled_.load(std::memory_order_acquire)) {
        return false;
    }
    return rejectFailure(std::move(layout_error), FailureStage::LAYOUT, FailureDisposition::HARD_CONTRACT);
}

bool HiddenStateCapturePublisher::completeFailure(const std::string& error_message,
                                                  FailureStage       failure_stage,
                                                  FailureDisposition failure_disposition,
                                                  const char*        phase) {
    if (!owner_) {
        return false;
    }

    // Native runtime helpers can throw std::runtime_error rather than c10::Error.
    if (isFatalCaptureMessage(error_message)) {
        failure_disposition = FailureDisposition::FATAL;
    }
    // Latch before diagnostics/metrics allocate, especially on allocator failure.
    if (failure_disposition == FailureDisposition::FATAL) {
        latchFatal(error_message.empty() ? "TorchSpec hidden-state publisher failed" : error_message.c_str());
    }
    recordFailure(failure_stage, failure_disposition);
    const auto detail = error_message.empty() ? "TorchSpec hidden-state publisher failed" : error_message;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        diagnostic_ = detail;
    }

    const bool disable_capture = failure_stage == FailureStage::INITIALIZATION
                                 || failure_disposition == FailureDisposition::HARD_CONTRACT
                                 || failure_disposition == FailureDisposition::FATAL;
    // A worker failure is sticky owner health, not an error on whichever
    // serving batch happens to finish next.
    if (disable_capture && failure_disposition != FailureDisposition::FATAL) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        capture_enabled_.store(false, std::memory_order_release);
        if (!broken_reason_.has_value()) {
            broken_reason_ = detail;
        }
    }
    recordCaptureStatus();
    if (failure_disposition == FailureDisposition::FATAL) {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s encountered a sticky owner fatal failure; "
                          "capture is disabled, no later request is blamed: %s",
                          phase,
                          detail.c_str());
    } else if (disable_capture) {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s violated a capture-only contract; disabling capture "
                          "without changing the serving response: %s",
                          phase,
                          detail.c_str());
    } else {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s failed for the current capture batch; serving is "
                          "unaffected and capture remains enabled: %s",
                          phase,
                          detail.c_str());
    }

    return false;
}

void HiddenStateCapturePublisher::initialize(int local_rank) {
    if (!owner_) {
        return;
    }

    py::gil_scoped_acquire initialization_gil;
    try {
        py::object mooncake_module = perform(FailureStage::INITIALIZATION,
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
                         {"put_batch", "query_keys", "reclaim_abandoned_capture", "take_async_errors", "flush", "close"}) {
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
        perform(FailureStage::INITIALIZATION,
                FailureDisposition::HARD_CONTRACT,
                "invalid Mooncake batch PUT capacity",
                [&]() {
                    // Match Eagle's actual send branch when known. Stores without
                    // private backend details conservatively use configured capacity.
                    const bool gpu_send_available =
                        py::getattr(store_, "_gpu_direct_available", py::bool_(false)).cast<bool>()
                        && !py::getattr(store_, "_gpu_send_buffer", py::none()).is_none();
                    if (!gpu_send_available) {
                        const auto pool_size = py::getattr(mooncake_config_, "async_put_pool_size", py::none());
                        put_batch_capacity_  = pool_size.is_none() ? 1 : pool_size.cast<int64_t>();
                        if (put_batch_capacity_ <= 0) {
                            throw std::runtime_error("Mooncake host PUT is disabled (async_put_pool_size <= 0) "
                                                     "and no GPU Direct send buffer is available");
                        }
                    }
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
            } catch (const py::error_already_set& close_error) {
                if (isFatalCapturePythonError(close_error)) {
                    throw;
                }
                RTP_LLM_LOG_ERROR("failed to close partially initialized TorchSpec hidden-state store: %s",
                                  close_error.what());
            } catch (const c10::Error&) {
                throw;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& close_error) {
                RTP_LLM_LOG_ERROR("failed to close partially initialized TorchSpec hidden-state store: %s",
                                  close_error.what());
            }
        }
        mooncake_config_ = py::object();
        store_           = py::object();
        quantize_fn_     = py::object();
        completeFailure(
            error_message, failure.failureStage(), failure.failureDisposition(), "publisher initialization");
    }
}

void HiddenStateCapturePublisher::publishOnWorker(PublishTask&         task,
                                                  torch::Tensor        packed_hidden_states,
                                                  const torch::Tensor& input_ids,
                                                  const torch::Tensor& input_lengths,
                                                  const torch::Tensor& request_ids,
                                                  int64_t              expected_rows,
                                                  const std::string&   layout_context,
                                                  const c10::Device&   fallback_device) {
    RTP_LLM_PROFILE_SCOPE("hidden_state_capture.publish");
    const int64_t publish_start_us = owner_ ? autil::TimeUtility::currentTimeInMicroSeconds() : 0;
    // Admission may have raced an earlier task disabling this owner. Dropped
    // tasks still fence their producer event before releasing source tensors.
    if (owner_ && !capture_enabled_.load(std::memory_order_acquire)) {
        return;
    }

    std::string layout_error =
        validatePackedLayout(packed_hidden_states, expected_rows, fallback_device, layout_context);

    torch::Tensor fallback_last_hidden;
    torch::Tensor auxiliary_hidden_view;
    if (layout_error.empty()) {
        try {
            fallback_last_hidden  = packed_hidden_states.narrow(1, layer_count_ * hidden_size_, hidden_size_);
            auxiliary_hidden_view = packed_hidden_states.narrow(1, 0, layer_count_ * hidden_size_);
        } catch (const c10::Error&) {
            throw;
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& e) {
            layout_error = std::string("failed to create captured hidden-state views: ") + e.what();
        } catch (...) {
            layout_error = "failed to create captured hidden-state views with an unknown exception";
        }
    }

    const auto record_early_failure_metrics = [this, publish_start_us]() {
        if (owner_) {
            PublishMetrics metrics;
            metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
            recordPublish(/*success=*/false, metrics);
        }
    };

    if (!layout_error.empty()) {
        rejectLayout(layout_error);
        record_early_failure_metrics();
        return;
    }
    const auto& last_hidden_states = fallback_last_hidden;

    torch::Tensor  lengths_host;
    torch::Tensor  request_ids_host;
    int64_t        total_tokens = 0;
    PublishMetrics metrics;
    int64_t        quantize_start_us           = 0;
    int64_t        store_put_start_us          = 0;
    bool           quantize_latency_completed  = false;
    bool           store_put_latency_completed = false;
    int64_t        flushed_request_count       = 0;
    int64_t        flushed_token_count         = 0;

    const auto finish_failure =
        [&](const std::string& error_message, FailureStage failure_stage, FailureDisposition failure_disposition) {
            if (metrics.has_quantize_latency && !quantize_latency_completed) {
                metrics.quantize_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - quantize_start_us;
            }
            if (metrics.has_store_put_latency && !store_put_latency_completed) {
                metrics.store_put_latency_us += autil::TimeUtility::currentTimeInMicroSeconds() - store_put_start_us;
            }
            // Capture diagnostics/health are owner-local. Neither recoverable
            // failures nor async worker fatals are attributed to serving batches.
            const auto completion_detail =
                task.store_attempted ? " (flush-confirmed prefix: " + std::to_string(flushed_request_count) + "/"
                                           + std::to_string(lengths_host.numel()) + " requests, "
                                           + std::to_string(flushed_token_count) + "/" + std::to_string(total_tokens)
                                           + " tokens; remaining publication is unconfirmed)" :
                                       std::string();
            completeFailure(error_message + completion_detail, failure_stage, failure_disposition, "runtime publish");
            if (owner_) {
                metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
                metrics.request_count      = lengths_host.defined() ? lengths_host.numel() : 0;
                metrics.token_count        = total_tokens;
                recordPublish(/*success=*/false, metrics);
            }
        };

    if (owner_) {
        std::optional<ProfilingScope> gil_scope(std::in_place, "hidden_state_capture.gil_acquire");
        // The guard encloses the catch handlers too. Python exceptions and all
        // local Python handles are destroyed before it releases the GIL.
        py::gil_scoped_acquire gil;
        gil_scope.reset();
        try {

            perform(FailureStage::STORE,
                    FailureDisposition::OPERATIONAL,
                    "TorchSpec hidden-state store async-error polling failed",
                    [&]() { observeAsyncErrors(FailureStage::STORE, "publish start"); });
            if (!capture_enabled_.load(std::memory_order_acquire)) {
                return;
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
                            // This runs only on the worker after processTask() has
                            // synchronized the producer event. trySubmit() never
                            // performs this CUDA-to-host conversion.
                            py::gil_scoped_release release;
                            auto prepared_lengths     = input_lengths.is_cuda() ? input_lengths.cpu() : input_lengths;
                            auto prepared_request_ids = request_ids.is_cuda() ? request_ids.cpu() : request_ids;
                            return std::make_pair(prepared_lengths.contiguous(), prepared_request_ids.contiguous());
                        });
            lengths_host       = std::move(host_metadata.first);
            request_ids_host   = std::move(host_metadata.second);
            task.input_lengths = lengths_host;
            task.request_ids   = request_ids_host;
            perform(FailureStage::PREPARE,
                    FailureDisposition::REQUEST_ERROR,
                    "capture lengths differ from reservation",
                    [&] {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        for (int64_t i = 0; i < request_ids_host.numel(); ++i) {
                            const auto entry = reservations_.find(request_ids_host.data_ptr<int64_t>()[i]);
                            if (entry == reservations_.end()
                                || entry->second.tokens != lengths_host.data_ptr<int32_t>()[i]) {
                                throw std::runtime_error("capture request length must match its reserved token count");
                            }
                        }
                    });

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
            torch::Tensor stored_last_hidden = perform(FailureStage::PREPARE,
                                                       FailureDisposition::OPERATIONAL,
                                                       "failed to prepare last hidden states for storage",
                                                       [&]() {
                                                           py::gil_scoped_release release;
                                                           return last_hidden_states.to(torch::kBFloat16).contiguous();
                                                       });
            torch::Tensor hidden_states_scale;
            if (capture_dtype_ == HiddenStateCaptureDtype::FP8_E4M3) {
                auto quantizer_input = perform(FailureStage::QUANTIZE,
                                               FailureDisposition::OPERATIONAL,
                                               "failed to prepare TorchSpec FP8 quantizer input",
                                               [&]() {
                                                   py::gil_scoped_release release;
                                                   return auxiliary_hidden_view.contiguous();
                                               });
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

            torch::Tensor stored_input_ids;
            perform(FailureStage::PREPARE,
                    FailureDisposition::OPERATIONAL,
                    "failed to materialize contiguous hidden-state store payloads",
                    [&]() {
                        // Keep the blocking host-ID conversion/H2D transfer off
                        // the serving Python thread's GIL. Exceptions reacquire
                        // the GIL before perform() inspects them.
                        py::gil_scoped_release release;
                        stored_auxiliary = stored_auxiliary.contiguous();
                        stored_input_ids =
                            input_ids
                                .to(torch::TensorOptions().device(last_hidden_states.device()).dtype(torch::kInt64))
                                .contiguous();
                        if (hidden_states_scale.defined()) {
                            hidden_states_scale = hidden_states_scale.contiguous();
                        }
                        metrics.input_ids_bytes        = static_cast<int64_t>(stored_input_ids.nbytes());
                        metrics.auxiliary_hidden_bytes = static_cast<int64_t>(stored_auxiliary.nbytes());
                        metrics.last_hidden_bytes      = static_cast<int64_t>(stored_last_hidden.nbytes());
                        metrics.scale_bytes =
                            hidden_states_scale.defined() ? static_cast<int64_t>(hidden_states_scale.nbytes()) : 0;
                        metrics.payload_bytes = metrics.input_ids_bytes + metrics.auxiliary_hidden_bytes
                                                + metrics.last_hidden_bytes + metrics.scale_bytes;
                    });

            // Strong ownership outlives Python lists, partial PUTs and failed
            // flushes. Reservation quota is retained until local completion.
            task.payloads           = {stored_auxiliary, stored_input_ids, stored_last_hidden, hidden_states_scale};
            int64_t retained_bytes  = 0;
            bool    quota_available = retainedBytes(task, retained_bytes);
            if (quota_available && retained_bytes > task.quota_bytes) {
                py::gil_scoped_release      release;
                std::lock_guard<std::mutex> lock(state_mutex_);
                const auto                  extra = retained_bytes - task.quota_bytes;
                quota_available                   = extra <= limits_.max_inflight_bytes - reserved_bytes_;
                if (quota_available) {
                    reservations_.at(task.ids.front()).bytes += extra;
                    reserved_bytes_ += extra;
                    task.quota_bytes += extra;
                }
            }
            if (!quota_available) {
                throw CaptureFailure(FailureStage::PREPARE,
                                     FailureDisposition::OPERATIONAL,
                                     "retained capture tensors exceed reserved byte quota");
            }
            // Preflight the entire batch before the first chunk. Merely finding a
            // foreign key does not transfer its ownership to this reservation.
            std::vector<std::string> source_keys;
            for (const auto id : task.ids) {
                const auto               key = makeRequestKey(id);
                std::vector<std::string> keys{key + "_hs", key + "_ids", key + "_lhs", key + "_ready"};
                if (hidden_states_scale.defined()) {
                    keys.push_back(key + "_hss");
                }
                source_keys.insert(source_keys.end(), keys.begin(), keys.end());
                std::lock_guard<std::mutex> lock(state_mutex_);
                reservations_.at(id).keys = std::move(keys);
            }
            if (fatal_failed_.load(std::memory_order_acquire)) {
                return;
            }
            const auto existing = perform(FailureStage::STORE,
                                          FailureDisposition::OPERATIONAL,
                                          "source-key preflight query failed",
                                          [&]() { return querySourceKeys(source_keys); });
            for (const auto& key : source_keys) {
                if (existing[py::str(key)].cast<bool>()) {
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::REQUEST_ERROR,
                                         "capture source key already exists; refusing to overwrite it: " + key);
                }
            }
            py::list capture_layer_ids;
            for (const auto id : layer_ids_) {
                capture_layer_ids.append(id);
            }
            py::dict capture_metadata;
            capture_metadata["capture_layer_ids"] = capture_layer_ids;

            const auto request_count    = lengths_host.numel();
            const auto batch_sequence   = batch_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto first_request_id = request_ids_host.data_ptr<int64_t>()[0];
            const auto last_request_id  = request_ids_host.data_ptr<int64_t>()[request_count - 1];
            const auto batch_id         = makeBatchId(batch_sequence, first_request_id, last_request_id);
            const auto chunk_capacity   = put_batch_capacity_ > 0 ? std::min(put_batch_capacity_, request_count) :
                                                                   request_count;
            int64_t    token_offset     = 0;
            for (int64_t chunk_begin = 0; chunk_begin < request_count;) {
                if (!capture_enabled_.load(std::memory_order_acquire)) {
                    return;
                }
                const auto chunk_count        = std::min(chunk_capacity, request_count - chunk_begin);
                const auto chunk_first_id     = request_ids_host.data_ptr<int64_t>()[chunk_begin];
                const auto chunk_last_id      = request_ids_host.data_ptr<int64_t>()[chunk_begin + chunk_count - 1];
                const auto chunk_id           = chunk_count == request_count ? batch_id :
                                                     batch_id + "-chunk-" + std::to_string(chunk_begin / chunk_capacity + 1)
                                                         + "-requests-" + std::to_string(chunk_first_id) + "-"
                                                         + std::to_string(chunk_last_id);
                const auto chunk_token_offset = token_offset;
                py::list   batch_keys;
                py::list   batch_hidden_states;
                py::list   batch_input_ids;
                py::list   batch_last_hidden_states;
                py::list   batch_hidden_states_scale;
                for (int64_t i = chunk_begin; i < chunk_begin + chunk_count; ++i) {
                    const int64_t length     = lengths_host.data_ptr<int32_t>()[i];
                    const int64_t request_id = request_ids_host.data_ptr<int64_t>()[i];
                    batch_keys.append(makeRequestKey(request_id));
                    batch_hidden_states.append(stored_auxiliary.narrow(0, token_offset, length));
                    batch_input_ids.append(stored_input_ids.narrow(0, token_offset, length));
                    batch_last_hidden_states.append(stored_last_hidden.narrow(0, token_offset, length));
                    if (hidden_states_scale.defined()) {
                        batch_hidden_states_scale.append(hidden_states_scale.narrow(0, token_offset, length));
                    }
                    token_offset += length;
                }

                prepare_scope.reset();
                {
                    // Preserve the source batch ID and identify the actual store chunk.
                    ProfilingScope batch_marker(chunk_id.c_str());
                }
                std::optional<ProfilingScope> admit_scope(std::in_place, "hidden_state_capture.put_batch_admit");
                metrics.has_store_put_latency = true;
                store_put_start_us            = autil::TimeUtility::currentTimeInMicroSeconds();
                store_put_latency_completed   = false;
                py::object acceptance;
                // A throwing/partial PUT may already expose ready keys. Record
                // exactly this chunk before entering the store, not its suffix.
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    for (int64_t i = chunk_begin; i < chunk_begin + chunk_count; ++i) {
                        reservations_.at(task.ids[i]).put_attempted = true;
                    }
                }
                task.store_attempted = true;
                // A previous chunk's drain cannot cover a new, possibly partial PUT.
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
                    if (isFatalCapturePythonError(e)) {
                        throw;
                    }
                    const auto python_error = std::string(e.what());
                    if (e.matches(PyExc_FileExistsError)) {
                        const auto first_key = batch_keys[0].cast<std::string>();
                        recordDuplicateRequestId(chunk_first_id, first_key, "Mooncake batch admission");
                        throw CaptureFailure(FailureStage::STORE,
                                             FailureDisposition::REQUEST_ERROR,
                                             "TorchSpec hidden-state store rejected duplicate request keys for batch ["
                                                 + chunk_id + "]: " + python_error);
                    }
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::OPERATIONAL,
                                         "failed to admit TorchSpec hidden-state batch [" + chunk_id
                                             + "]: " + python_error);
                } catch (const c10::Error&) {
                    throw;
                } catch (const std::bad_alloc&) {
                    throw;
                } catch (const std::exception& e) {
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::OPERATIONAL,
                                         "failed to admit TorchSpec hidden-state batch [" + chunk_id + "]: " + e.what());
                } catch (...) {
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::OPERATIONAL,
                                         "failed to admit TorchSpec hidden-state batch [" + chunk_id
                                             + "] with an unknown exception");
                }

                if (fatal_failed_.load(std::memory_order_acquire)) {
                    return;
                }
                perform(FailureStage::STORE,
                        FailureDisposition::HARD_CONTRACT,
                        ("invalid EagleMooncakeStore.put_batch() metadata for batch [" + chunk_id + "]").c_str(),
                        [&]() {
                            if (!py::isinstance<py::list>(acceptance)
                                || py::len(acceptance) != static_cast<py::ssize_t>(chunk_count)) {
                                throw std::runtime_error("expected one accepted metadata item per chunk request");
                            }
                            const auto metadata = acceptance.cast<py::list>();
                            for (int64_t i = 0; i < chunk_count; ++i) {
                                const auto key = batch_keys[i].cast<std::string>();
                                const auto id  = request_ids_host.data_ptr<int64_t>()[chunk_begin + i];
                                const auto context = "key=" + key + ", request_id=" + std::to_string(id);
                                if (!py::isinstance<py::dict>(metadata[i])) {
                                    throw std::runtime_error("accepted metadata must be a dict for " + context);
                                }
                                const auto item = metadata[i].cast<py::dict>();
                                if (!item.contains("shapes") || !item.contains("dtypes")
                                    || !py::isinstance<py::dict>(item["shapes"])
                                    || !py::isinstance<py::dict>(item["dtypes"])) {
                                    throw std::runtime_error("accepted metadata needs shapes/dtypes dicts for " + context);
                                }
                                const auto shapes = item["shapes"].cast<py::dict>();
                                const auto dtypes = item["dtypes"].cast<py::dict>();
                                // Eagle returns positional metadata, not echoed keys/IDs.
                                // Validate each item against this chunk's exact input tensors.
                                const auto validate_tensor = [&](const char* name, py::handle tensor) {
                                    if (!shapes.contains(name) || !dtypes.contains(name)
                                        || !py::object(shapes[name]).equal(py::object(tensor.attr("shape")))
                                        || !py::object(dtypes[name]).equal(py::object(tensor.attr("dtype")))) {
                                        throw std::runtime_error("accepted metadata mismatch for " + context + ", " + name);
                                    }
                                };
                                validate_tensor("hidden_states", batch_hidden_states[i]);
                                validate_tensor("input_ids", batch_input_ids[i]);
                                validate_tensor("last_hidden_states", batch_last_hidden_states[i]);
                                if (hidden_states_scale.defined()) {
                                    validate_tensor("hidden_states_scale", batch_hidden_states_scale[i]);
                                }
                            }
                        });
                metrics.store_put_latency_us += autil::TimeUtility::currentTimeInMicroSeconds() - store_put_start_us;
                store_put_latency_completed = true;
                admit_scope.reset();

                const bool flush_succeeded = perform(FailureStage::STORE,
                                                     FailureDisposition::OPERATIONAL,
                                                     ("TorchSpec hidden-state store flush failed for batch [" + chunk_id
                                                      + "]").c_str(),
                                                     [&]() { return flushStore(); });
                if (fatal_failed_.load(std::memory_order_acquire)) {
                    return;
                }
                task.store_flushed = true;
                if (!flush_succeeded) {
                    finish_failure("TorchSpec hidden-state store reported an asynchronous publish failure for batch ["
                                       + chunk_id + "]",
                                   FailureStage::STORE,
                                   FailureDisposition::OPERATIONAL);
                    return;
                }
                flushed_request_count += chunk_count;
                flushed_token_count += token_offset - chunk_token_offset;
                // No host lease is reused until this chunk has completely drained.
                // Keep all source/payload ownership and quota until processTask fences.
                chunk_begin += chunk_count;
            }
        } catch (const CaptureFailure& failure) {
            finish_failure(failure.what(), failure.failureStage(), failure.failureDisposition());
            return;
        } catch (const py::error_already_set& error) {
            finish_failure(error.what(),
                           FailureStage::STORE,
                           isFatalCapturePythonError(error) ? FailureDisposition::FATAL :
                                                              FailureDisposition::OPERATIONAL);
            return;
        } catch (const c10::Error&) {
            throw;
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& e) {
            finish_failure(std::string("unexpected hidden-state publisher failure: ") + e.what(),
                           FailureStage::PREPARE,
                           FailureDisposition::OPERATIONAL);
            return;
        } catch (...) {
            finish_failure("unexpected hidden-state publisher failure with an unknown exception",
                           FailureStage::PREPARE,
                           FailureDisposition::OPERATIONAL);
            return;
        }
    }

    if (owner_) {
        metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
        metrics.request_count      = lengths_host.numel();
        metrics.token_count        = total_tokens;
        recordPublish(/*success=*/true, metrics);
    }
    task.succeeded = true;
    return;
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
        if (!request_ids.defined() || !request_ids.device().is_cpu() || request_ids.scalar_type() != torch::kInt64
            || request_ids.dim() != 1 || request_ids.numel() == 0 || request_ids.layout() != torch::kStrided) {
            return rejectRequest("invalid capture request metadata: CPU int64 strided ids required", true);
        }
        // Admission never performs a device-to-host wait. The gatherer keeps
        // these metadata tensors on CPU; if a caller violates that contract we
        // drop only capture and preserve the canonical response path.
        if (!input_ids.defined() || input_ids.dim() != 1 || input_ids.numel() != expected_rows
            || (input_ids.scalar_type() != torch::kInt32 && input_ids.scalar_type() != torch::kInt64)
            || input_ids.layout() != torch::kStrided || !input_lengths.defined() || input_lengths.dim() != 1
            || !input_lengths.device().is_cpu() || input_lengths.layout() != torch::kStrided
            || input_lengths.scalar_type() != torch::kInt32 || !request_ids.defined() || request_ids.dim() != 1
            || request_ids.layout() != torch::kStrided || !request_ids.device().is_cpu()
            || request_ids.scalar_type() != torch::kInt64 || input_lengths.numel() != request_ids.numel()
            || request_ids.numel() == 0) {
            return rejectFailure(
                "capture submit requires CPU int32 lengths, CPU int64 request ids and aligned input ids",
                FailureStage::PREPARE,
                FailureDisposition::REQUEST_ERROR);
        }

        auto task       = std::make_shared<PublishTask>();
        task->packed    = std::move(packed_hidden_states);
        task->input_ids = input_ids.device().is_cpu() ? input_ids.clone().contiguous() : input_ids;
        // Pinned lengths may be the destination of an outstanding producer D2H
        // copy. Do not clone/read them on CPU before the ready event is fenced.
        task->input_lengths   = input_lengths.is_pinned() ? input_lengths : input_lengths.clone().contiguous();
        task->request_ids     = request_ids.clone().contiguous();
        task->expected_rows   = expected_rows;
        task->layout_context  = layout_context;
        task->fallback_device = task->packed.device();

        task->ids.reserve(task->request_ids.numel());
        std::unordered_set<int64_t> seen_request_ids;
        for (int64_t i = 0; i < task->request_ids.numel(); ++i) {
            const int64_t request_id = task->request_ids.data_ptr<int64_t>()[i];
            if (request_id < 0) {
                return rejectRequest("capture request id must be non-negative, got " + std::to_string(request_id));
            }
            if (!seen_request_ids.insert(request_id).second) {
                return rejectDuplicateRequest(request_id);
            }
            task->ids.push_back(request_id);
        }
        if (!task->input_ids.device().is_cpu() && task->input_ids.device() != task->packed.device()) {
            return rejectFailure("capture submit metadata does not cover the packed hidden rows",
                                 FailureStage::PREPARE,
                                 FailureDisposition::REQUEST_ERROR);
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        if (stopping_ || !capture_enabled_.load(std::memory_order_acquire)) {
            return false;
        }
        int64_t rows = 0;
        for (const auto id : task->ids) {
            const auto entry = reservations_.find(id);
            if (entry == reservations_.end() || entry->second.submitted
                || entry->second.tokens > expected_rows - rows) {
                return false;
            }
            rows += entry->second.tokens;
            task->quota_bytes += entry->second.bytes;
        }
        if (rows != expected_rows) {
            return false;
        }
        int64_t estimated_bytes = 0;
        int64_t store_bytes     = 0;
        int64_t source_bytes    = 0;
        // Storage sizes are host metadata: no device wait. A view into an
        // oversized backing allocation must not silently bypass the byte cap.
        if (!payloadBytes(rows, estimated_bytes, &store_bytes) || !retainedBytes(*task, source_bytes)
            || source_bytes > limits_.max_inflight_bytes - store_bytes) {
            return false;
        }
        const auto required_bytes = source_bytes + store_bytes;
        if (required_bytes > task->quota_bytes) {
            const auto extra = required_bytes - task->quota_bytes;
            if (extra > limits_.max_inflight_bytes - reserved_bytes_) {
                return false;
            }
            reservations_.at(task->ids.front()).bytes += extra;
            reserved_bytes_ += extra;
            task->quota_bytes += extra;
        }
#if USING_CUDA || USING_ROCM
        if (!task->packed.device().is_cpu()) {
            c10::DeviceGuard            guard(task->packed.device());
            c10::impl::VirtualGuardImpl impl(task->packed.device().type());
            task->producer_stream = impl.getStream(task->packed.device());
            task->ready           = runtimeCreateEvent();
        }
#else
        if (!task->packed.device().is_cpu()) {
            return false;
        }
#endif
        task->dropped = pending_.size() >= static_cast<size_t>(limits_.max_pending_batches);
        if (task->dropped) {
            // The bounded ledger retains producer storage until a worker-only
            // fence, even when no slot in the publication queue is available.
            cleanup_requested_ = true;
        } else {
            pending_.push_back(task);
        }
        for (const auto request_id : task->ids) {
            auto& reservation     = reservations_.at(request_id);
            reservation.submitted = true;
            reservation.task      = task;
        }
        state_changed_.notify_all();
        return !task->dropped;
    } catch (const c10::Error&) {
        // Producer-side accelerator/OOM failures remain fatal to the existing
        // model error path; recoverable capture failures do not.
        throw;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        return rejectFailure(error.what(), FailureStage::PREPARE, FailureDisposition::OPERATIONAL);
    } catch (...) {
        return rejectFailure(
            "unknown capture submission failure", FailureStage::PREPARE, FailureDisposition::OPERATIONAL);
    }
}

bool HiddenStateCapturePublisher::flushStore() {
    if (fatal_failed_.load(std::memory_order_acquire)) {
        return false;
    }
    if (!owner_ || !store_ || store_.is_none()) {
        return true;
    }
    RTP_LLM_PROFILE_SCOPE("hidden_state_capture.flush");
    {
        RTP_LLM_PROFILE_SCOPE("hidden_state_capture.flush_python");
        store_.attr("flush")();
    }
    return observeAsyncErrors(FailureStage::STORE, "after publish flush");
}

void HiddenStateCapturePublisher::releaseLocalPayloads(PublishTask& task) {
    // Local completion permits freeing tensors, not remote source reservations.
    task.payloads.clear();
    task.packed        = torch::Tensor();
    task.input_ids     = torch::Tensor();
    task.input_lengths = torch::Tensor();
    task.request_ids   = torch::Tensor();
    task.ready.reset();
    std::vector<int64_t>().swap(task.ids);
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (auto entry = reservations_.begin(); entry != reservations_.end();) {
        if (entry->second.task.get() == &task && !entry->second.put_attempted) {
            auto released = entry++;
            releaseReservation(released);
        } else {
            ++entry;
        }
    }
}

void HiddenStateCapturePublisher::releaseReservation(std::unordered_map<int64_t, Reservation>::iterator entry) {
    reserved_bytes_ -= entry->second.bytes;
    reserved_tokens_ -= entry->second.tokens;
    reservations_.erase(entry);
}

py::dict HiddenStateCapturePublisher::querySourceKeys(const std::vector<std::string>& keys) {
    py::list requested;
    for (const auto& key : keys) {
        requested.append(key);
    }
    auto result = store_.attr("query_keys")(requested);
    if (!py::isinstance<py::dict>(result) || py::len(result) != requested.size()) {
        throw CaptureFailure(FailureStage::STORE,
                             FailureDisposition::HARD_CONTRACT,
                             "query_keys returned an incomplete source manifest");
    }
    auto states = result.cast<py::dict>();
    // Validate the whole answer before changing any ledger entry. Error codes,
    // missing entries and malformed answers are never proof of absence.
    for (const auto& key : requested) {
        if (!states.contains(key) || !py::isinstance<py::bool_>(states[key])) {
            throw CaptureFailure(FailureStage::STORE,
                                 FailureDisposition::HARD_CONTRACT,
                                 "query_keys must return strict booleans for every source key");
        }
    }
    return states;
}

void HiddenStateCapturePublisher::pollSourceKeys() {
    if (fatal_failed_.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<std::shared_ptr<PublishTask>> dropped;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (const auto& entry : reservations_) {
            const auto& task = entry.second.task;
            if (task && task->dropped && !task->cleanup_started) {
                dropped.push_back(task);
                task->cleanup_started = true;
            }
        }
    }
    for (const auto& task : dropped) {
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
        processTask(task);
    }
    if (fatal_failed_.load(std::memory_order_acquire)) {
        return;
    }

    struct Candidate {
        int64_t                  id;
        uint64_t                 generation;
        std::vector<std::string> keys;
        std::string              request_key;
    };
    std::vector<Candidate>   candidates;
    std::vector<std::string> keys;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (const auto& entry : reservations_) {
            const auto& reservation = entry.second;
            if (reservation.put_attempted && reservation.task && reservation.task->local_complete
                && reservation.task->store_flushed) {
                candidates.push_back({entry.first, reservation.generation, reservation.keys, {}});
                keys.insert(keys.end(), reservation.keys.begin(), reservation.keys.end());
            }
        }
    }
    if (candidates.empty()) {
        return;
    }
    py::gil_scoped_acquire gil;
    try {
        for (auto& candidate : candidates) {
            candidate.request_key = makeRequestKey(candidate.id);
            keys.push_back(candidate.request_key + "_abandoned");
        }
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
        const auto states = querySourceKeys(keys);
        for (const auto& candidate : candidates) {
            if (fatal_failed_.load(std::memory_order_acquire)) {
                break;
            }
            try {
                bool releasable = false;
                if (states[py::str(candidate.request_key + "_abandoned")].cast<bool>()) {
                    // Only an explicit delivery waiver authorizes producer deletion,
                    // after local fences and native PUT drain. Clear marker-only
                    // remnants too; missing payload alone must not skip reclamation.
                    auto reclaimed = store_.attr("reclaim_abandoned_capture")(candidate.request_key);
                    if (!py::isinstance<py::bool_>(reclaimed)) {
                        throw std::runtime_error("reclaim_abandoned_capture must return a strict boolean");
                    }
                    releasable = reclaimed.cast<bool>();
                } else {
                    releasable = !candidate.keys.empty()
                                 && std::all_of(candidate.keys.begin(), candidate.keys.end(), [&](const auto& key) {
                                        return !states[py::str(key)].cast<bool>();
                                    });
                }
                if (fatal_failed_.load(std::memory_order_acquire)) {
                    return;
                }
                if (releasable) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    const auto                  entry = reservations_.find(candidate.id);
                    if (entry != reservations_.end() && entry->second.generation == candidate.generation) {
                        releaseReservation(entry);
                    }
                }
            } catch (const py::error_already_set& error) {
                if (isFatalCapturePythonError(error)) {
                    throw;
                }
                completeFailure(error.what(), FailureStage::STORE, FailureDisposition::OPERATIONAL,
                                "abandoned capture cleanup; request quota retained");
            } catch (const c10::Error&) {
                throw;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& error) {
                if (isFatalCaptureMessage(error.what())) {
                    throw;
                }
                completeFailure(error.what(), FailureStage::STORE, FailureDisposition::OPERATIONAL,
                                "abandoned capture cleanup; request quota retained");
            }
        }
    } catch (const py::error_already_set& error) {
        completeFailure(error.what(),
                        FailureStage::STORE,
                        isFatalCapturePythonError(error) ? FailureDisposition::FATAL : FailureDisposition::OPERATIONAL,
                        "source cleanup query; quota retained");
    } catch (const c10::Error& error) {
        completeFailure(error.what(), FailureStage::STORE, FailureDisposition::FATAL, "source cleanup query");
    } catch (const std::bad_alloc& error) {
        completeFailure(error.what(), FailureStage::STORE, FailureDisposition::FATAL, "source cleanup query");
    } catch (const std::exception& error) {
        // Even a malformed query response is unknown cleanup state. Keep all
        // reservations and retry; never delete source keys on the producer.
        completeFailure(error.what(),
                        FailureStage::STORE,
                        FailureDisposition::OPERATIONAL,
                        "source cleanup query; quota retained");
    }
    state_changed_.notify_all();
}

void HiddenStateCapturePublisher::quarantineResources() noexcept {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (quarantine_ && !reservations_.empty()) {
            quarantine_->reservations                          = std::move(reservations_);
            [[maybe_unused]] auto* retained_until_process_exit = quarantine_.release();
            reserved_bytes_                                    = 0;
            reserved_tokens_                                   = 0;
        }
    }
    // release() transfers the live refcount without touching Python, so this
    // emergency path is also safe outside the GIL.
    (void)store_.release();
    (void)mooncake_config_.release();
    (void)quantize_fn_.release();
}

void HiddenStateCapturePublisher::processTask(const std::shared_ptr<PublishTask>& task) {
    // Fatal owners retain unknown work without any further runtime/store cleanup.
    // The worker still consumes queue references so poll/stop can reach idle.
    if (task->local_complete || fatal_failed_.load(std::memory_order_acquire)) {
        return;
    }
    at::ThreadLocalStateGuard thread_state(task->thread_state);
    // Preserve dispatch TLS, but never let worker scopes retain Kineto events
    // beyond the producer's profiler stop, which can free their backing buffer.
    at::DisableRecordFunctionGuard record_function_guard;
    bool                      local_safe = false;
    try {
        c10::DeviceGuard                device_guard(task->packed.device());
        std::optional<c10::Stream>      stream;
        std::optional<c10::StreamGuard> stream_guard;
        if (!task->packed.device().is_cpu()) {
            c10::impl::VirtualGuardImpl impl(task->packed.device().type());
            stream = impl.getStreamFromGlobalPool(task->packed.device(), false);
            // Pool streams can wrap around while a producer still uses one.
            // Never serialize exporter work back onto the serving stream.
            if (task->producer_stream && *stream == *task->producer_stream) {
                stream = impl.getStreamFromGlobalPool(task->packed.device(), false);
            }
            stream_guard.emplace(*stream);
        }
        if (task->ready) {
            task->ready->synchronize();
        }
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
        if (!task->dropped && capture_enabled_.load(std::memory_order_acquire)) {
            try {
                publishOnWorker(*task,
                                task->packed,
                                task->input_ids,
                                task->input_lengths,
                                task->request_ids,
                                task->expected_rows,
                                task->layout_context,
                                task->fallback_device);
            } catch (const c10::Error& error) {
                completeFailure(error.what(), FailureStage::STORE, FailureDisposition::FATAL, "capture worker");
            } catch (const std::bad_alloc& error) {
                completeFailure(error.what(), FailureStage::STORE, FailureDisposition::FATAL, "capture worker");
            } catch (const std::exception& error) {
                completeFailure(error.what(), FailureStage::STORE, FailureDisposition::OPERATIONAL, "capture worker");
            } catch (...) {
                completeFailure("unknown capture worker failure",
                                FailureStage::STORE,
                                FailureDisposition::OPERATIONAL,
                                "capture worker");
            }
        }
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
        if (task->store_attempted && !task->store_flushed) {
            py::gil_scoped_acquire gil;
            try {
                (void)flushStore();
                if (fatal_failed_.load(std::memory_order_acquire)) {
                    return;
                }
                task->store_flushed = true;
            } catch (const py::error_already_set& error) {
                completeFailure(error.what(),
                                FailureStage::STORE,
                                isFatalCapturePythonError(error) ? FailureDisposition::FATAL :
                                                                   FailureDisposition::HARD_CONTRACT,
                                "failed PUT drain; retaining payload and quota");
            } catch (const c10::Error&) {
                throw;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& error) {
                completeFailure(error.what(),
                                FailureStage::STORE,
                                FailureDisposition::HARD_CONTRACT,
                                "failed PUT drain; retaining payload and quota");
            }
        }
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
        if (stream) {
            // All copies/quantization kernels on this worker stream must finish
            // before their strongly held inputs and staging tensors are freed.
            stream->synchronize();
        }
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
#if USING_CUDA || USING_ROCM
        if (task->store_attempted && !task->succeeded && !task->packed.device().is_cpu()) {
            // check_cuda_value throws std::runtime_error; a failed device fence
            // must remain fatal and keep this task unfenced.
            perform(FailureStage::STORE, FailureDisposition::FATAL, "capture worker device-wide fence failed", [] {
                runtimeSyncAndCheck();
            });
        }
#endif
        if (fatal_failed_.load(std::memory_order_acquire)) {
            return;
        }
        task->local_fenced = true;
        local_safe         = !task->store_attempted || task->store_flushed;
    } catch (const CaptureFailure& failure) {
        completeFailure(
            failure.what(), failure.failureStage(), failure.failureDisposition(), "capture worker event wait");
    } catch (const c10::Error& error) {
        completeFailure(error.what(), FailureStage::STORE, FailureDisposition::FATAL, "capture worker event wait");
    } catch (const std::bad_alloc& error) {
        completeFailure(error.what(), FailureStage::STORE, FailureDisposition::FATAL, "capture worker event wait");
    } catch (const std::exception& error) {
        completeFailure(
            error.what(), FailureStage::STORE, FailureDisposition::OPERATIONAL, "capture worker event wait");
    } catch (...) {
        completeFailure("unknown capture worker event failure",
                        FailureStage::STORE,
                        FailureDisposition::OPERATIONAL,
                        "capture worker event wait");
    }

    if (fatal_failed_.load(std::memory_order_acquire)) {
        return;
    }
    task->local_complete = local_safe;
    if (local_safe) {
        releaseLocalPayloads(*task);
    }
    state_changed_.notify_all();
}

void HiddenStateCapturePublisher::workerLoop() noexcept {
    const auto report_failure = [this](const char* message) noexcept {
        latchFatal(message);
        try {
            completeFailure(message, FailureStage::SHUTDOWN, FailureDisposition::FATAL, "capture worker loop");
        } catch (...) {
            RTP_LLM_LOG_ERROR("capture worker failed while reporting sticky owner failure");
        }
    };
    try {
        {
            // Python exception formatting/destruction must also stay under the GIL.
            py::gil_scoped_acquire gil;
            try {
                initialize(local_rank_);
            } catch (const py::error_already_set& error) {
                completeFailure(error.what(),
                                FailureStage::INITIALIZATION,
                                isFatalCapturePythonError(error) ? FailureDisposition::FATAL :
                                                                  FailureDisposition::OPERATIONAL,
                                "publisher worker initialization");
            }
        }
        for (;;) {
            std::shared_ptr<PublishTask> task;
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
            }
            pollSourceKeys();
        }
        pollSourceKeys();
    } catch (const std::exception& error) {
        report_failure(error.what());
    } catch (...) {
        report_failure("unknown capture worker loop failure");
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        worker_active_   = false;
        worker_finished_ = true;
    }
    state_changed_.notify_all();
}

void HiddenStateCapturePublisher::flushAndClose() noexcept {
    // A concurrent closer must not wait on close_mutex_ holding the GIL while
    // the first closer is joining a Python worker.
    std::optional<py::gil_scoped_release> release;
    if (Py_IsInitialized() && PyGILState_Check()) {
        release.emplace();
    }
    std::lock_guard<std::mutex> close_lock(close_mutex_);
    const bool fatal_at_shutdown = fatal_failed_.load(std::memory_order_acquire);
    try {
        auto stop_worker = [this] {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                stopping_ = true;
            }
            state_changed_.notify_all();
            if (worker_.joinable()) {
                worker_.join();
            }
        };

        if (Py_IsInitialized() && PyGILState_Check()) {
            py::gil_scoped_release release;
            stop_worker();
        } else {
            stop_worker();
        }

        capture_enabled_.store(false, std::memory_order_release);
        if (fatal_at_shutdown || fatal_failed_.load(std::memory_order_acquire)) {
            quarantineResources();
            return;
        }
        if (Py_IsInitialized()) {
            py::gil_scoped_acquire gil;
            bool                   drained = false;
            if (store_ && !store_.is_none()) {
                try {
                    (void)flushStore();
                    // Operational completion errors do not undo native drain.
                    // Fatal observation, including a returned error list, does.
                    drained = !fatal_failed_.load(std::memory_order_acquire);
                } catch (const py::error_already_set& error) {
                    drained = false;
                    completeFailure(error.what(),
                                    FailureStage::SHUTDOWN,
                                    isFatalCapturePythonError(error) ? FailureDisposition::FATAL :
                                                                       FailureDisposition::OPERATIONAL,
                                    "publisher shutdown flush");
                } catch (const c10::Error& error) {
                    drained = false;
                    completeFailure(
                        error.what(), FailureStage::SHUTDOWN, FailureDisposition::FATAL, "publisher shutdown flush");
                } catch (const std::bad_alloc& error) {
                    drained = false;
                    completeFailure(
                        error.what(), FailureStage::SHUTDOWN, FailureDisposition::FATAL, "publisher shutdown flush");
                } catch (const std::exception& error) {
                    drained = false;
                    completeFailure(error.what(),
                                    FailureStage::SHUTDOWN,
                                    FailureDisposition::OPERATIONAL,
                                    "publisher shutdown flush");
                }
                if (fatal_failed_.load(std::memory_order_acquire)) {
                    quarantineResources();
                    return;
                }
                if (drained) {
                    // Recover a prior failed flush only when the worker already
                    // fenced local device work. Remote ownership is checked below.
                    std::vector<std::shared_ptr<PublishTask>> completed;
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        for (const auto& entry : reservations_) {
                            const auto& task = entry.second.task;
                            if (task && task->local_fenced && !task->local_complete
                                && std::find(completed.begin(), completed.end(), task) == completed.end()) {
                                completed.push_back(task);
                            }
                        }
                    }
                    for (const auto& task : completed) {
                        if (fatal_failed_.load(std::memory_order_acquire)) {
                            quarantineResources();
                            return;
                        }
                        task->store_flushed  = true;
                        task->local_complete = true;
                        releaseLocalPayloads(*task);
                    }
                }
                // A successful local flush is not evidence that the consumer has
                // removed its source keys. Check before close can tear down store
                // ownership, and retain even locally complete published requests.
                pollSourceKeys();
                if (fatal_failed_.load(std::memory_order_acquire)) {
                    quarantineResources();
                    return;
                }
                bool unresolved_source = false;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    for (const auto& entry : reservations_) {
                        const auto& reservation = entry.second;
                        unresolved_source = unresolved_source || reservation.put_attempted
                                            || (reservation.task && !reservation.task->local_complete);
                    }
                }
                if (unresolved_source) {
                    quarantineResources();
                    RTP_LLM_LOG_WARNING("capture shutdown retains unconsumed source keys or unresolved local work");
                    return;
                }
                if (fatal_failed_.load(std::memory_order_acquire)) {
                    quarantineResources();
                    return;
                }
                try {
                    store_.attr("close")();
                } catch (const py::error_already_set& error) {
                    drained = false;
                    completeFailure(error.what(),
                                    FailureStage::SHUTDOWN,
                                    isFatalCapturePythonError(error) ? FailureDisposition::FATAL :
                                                                       FailureDisposition::OPERATIONAL,
                                    "publisher shutdown close");
                } catch (const c10::Error& error) {
                    drained = false;
                    completeFailure(
                        error.what(), FailureStage::SHUTDOWN, FailureDisposition::FATAL, "publisher shutdown close");
                } catch (const std::bad_alloc& error) {
                    drained = false;
                    completeFailure(
                        error.what(), FailureStage::SHUTDOWN, FailureDisposition::FATAL, "publisher shutdown close");
                } catch (const std::exception& error) {
                    drained = false;
                    completeFailure(error.what(),
                                    FailureStage::SHUTDOWN,
                                    FailureDisposition::OPERATIONAL,
                                    "publisher shutdown close");
                }
            }
            if (fatal_failed_.load(std::memory_order_acquire)) {
                quarantineResources();
                return;
            }
            bool unresolved = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (const auto& entry : reservations_) {
                    const auto& task = entry.second.task;
                    unresolved = unresolved || (task && !task->local_complete);
                }
            }
            if (unresolved) {
                // A failed fence/flush gives no safe lifetime endpoint. This
                // bounded quarantine intentionally survives the publisher.
                // Admission is already stopped, so it cannot grow unboundedly.
                quarantineResources();
                RTP_LLM_LOG_ERROR("capture shutdown quarantined unresolved payloads and store ownership; "
                                  "drained=%d",
                                  drained);
            } else {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    reservations_.clear();
                    reserved_bytes_  = 0;
                    reserved_tokens_ = 0;
                }
                store_           = py::object();
                quantize_fn_     = py::object();
                mooncake_config_ = py::object();
            }
        }

    } catch (const std::exception& error) {
        latchFatal(error.what());
        quarantineResources();
        RTP_LLM_LOG_WARNING("hidden-state capture shutdown failed: %s", error.what());
    } catch (...) {
        latchFatal("unknown capture shutdown failure");
        quarantineResources();
        RTP_LLM_LOG_WARNING("hidden-state capture shutdown failed with an unknown exception");
    }
}

}  // namespace rtp_llm
