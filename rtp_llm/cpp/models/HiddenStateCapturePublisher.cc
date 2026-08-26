#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

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

HiddenStateCapturePublisher::HiddenStateCapturePublisher(int64_t                      layer_count,
                                                         int64_t                      hidden_size,
                                                         HiddenStateCaptureDtype      capture_dtype,
                                                         c10::ScalarType              model_dtype,
                                                         bool                         owner,
                                                         int                          local_rank,
                                                         bool                         fail_open,
                                                         kmonitor::MetricsReporterPtr metrics_reporter):
    layer_count_(layer_count),
    hidden_size_(hidden_size),
    capture_dtype_(capture_dtype),
    model_dtype_(model_dtype),
    owner_(owner),
    metrics_reporter_(owner ? std::move(metrics_reporter) : nullptr),
    fail_open_(owner && fail_open) {
    if (owner_) {
        RTP_LLM_LOG_INFO("[TorchSpec hidden-state publisher] configured failure_policy=%s",
                         fail_open ? "fail-open" : "fail-closed");
    }
    initialize(local_rank);
}

HiddenStateCapturePublisher::~HiddenStateCapturePublisher() {
    // Dropping a py::object normally decrefs through the Python C API. Shutdown
    // is owned by PyWrappedModel under the GIL; if that contract is ever missed,
    // leak the references rather than invoking Python from this destructor.
    (void)mooncake_config_.release();
    (void)store_.release();
    (void)quantize_fn_.release();
}

void HiddenStateCapturePublisher::beginForward() {
    std::optional<std::string> stale_error;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        stale_error = std::exchange(deferred_error_, std::nullopt);
    }
    if (owner_ && stale_error.has_value()) {
        RTP_LLM_LOG_WARNING(
            "[TorchSpec hidden-state publisher] discarding an unconsumed deferred error at the start of a new model "
            "forward so it is not attributed to the new batch: %s",
            stale_error->c_str());
    }
}

bool HiddenStateCapturePublisher::shouldPublish() {
    recordBatch();
    if (capture_enabled_.load(std::memory_order_acquire)) {
        return true;
    }

    // A fail-closed hard failure permanently breaks capture for this publisher.
    // Re-queue its reason only when a later request actually asks to publish.
    bool broken_rejection = false;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (broken_reason_.has_value()) {
            broken_rejection = true;
            if (!deferred_error_.has_value()) {
                deferred_error_ = broken_reason_;
            }
        }
    }
    if (broken_rejection) {
        recordBrokenRejection();
    } else {
        recordDisabledSkip();
    }
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
    return {failure_count_.load(std::memory_order_relaxed), broken_rejection_count_.load(std::memory_order_relaxed)};
}

std::optional<std::string> HiddenStateCapturePublisher::takeDeferredError() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return std::exchange(deferred_error_, std::nullopt);
}

void HiddenStateCapturePublisher::deferError(std::string error_message, bool disable_capture) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (disable_capture && !broken_reason_.has_value()) {
        broken_reason_ = error_message;
    }
    if (!deferred_error_.has_value()) {
        deferred_error_ = std::move(error_message);
    }
    if (disable_capture) {
        capture_enabled_.store(false, std::memory_order_release);
    }
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

void HiddenStateCapturePublisher::recordFailure(FailureStage       failure_stage,
                                                FailureDisposition failure_disposition,
                                                bool               fail_open) {
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
    collector.fail_open_disable_qps      = fail_open;
    reportMetrics(collector);
}

void HiddenStateCapturePublisher::recordDisabledSkip() {
    RtpLLMHiddenStateCaptureMetricsCollector collector;
    collector.disabled_skip_qps = true;
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
    collector.fail_open_enabled  = fail_open_.load(std::memory_order_acquire) ? 1 : 0;
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

void HiddenStateCapturePublisher::observeAsyncErrors(FailureStage failure_stage, const char* phase) {
    py::object result = store_.attr("take_async_errors")();
    if (!py::isinstance<py::list>(result)) {
        throw std::runtime_error("EagleMooncakeStore.take_async_errors() must return a list");
    }

    const auto errors = result.cast<py::list>();
    for (const auto error : errors) {
        const auto detail = py::str(error).cast<std::string>();
        // These failures belong to batches accepted by earlier forwards. They
        // are observable failures, but must never change the current stream's
        // deferred error or the publisher's enabled state.
        recordFailure(failure_stage, FailureDisposition::OPERATIONAL, /*fail_open=*/false);
        RTP_LLM_LOG_ERROR(
            "[TorchSpec hidden-state publisher] %s observed an asynchronous Mooncake failure from a previously "
            "accepted batch; the original batch/request IDs are preserved in the error and the current stream "
            "continues: %s",
            phase,
            detail.c_str());
    }
}

bool HiddenStateCapturePublisher::rejectFailure(std::string        error_message,
                                                FailureStage       failure_stage,
                                                FailureDisposition failure_disposition) {
    return completeFailure(error_message,
                           failure_stage,
                           failure_disposition,
                           "capture contract validation",
                           /*defer_error=*/true);
}

bool HiddenStateCapturePublisher::rejectLayout(std::string layout_error) {
    return rejectFailure(std::move(layout_error), FailureStage::LAYOUT, FailureDisposition::HARD_CONTRACT);
}

bool HiddenStateCapturePublisher::completeFailure(const std::string& error_message,
                                                  FailureStage       failure_stage,
                                                  FailureDisposition failure_disposition,
                                                  const char*        phase,
                                                  bool               defer_error) {
    if (!owner_) {
        return false;
    }

    const bool failed_open =
        failure_disposition == FailureDisposition::OPERATIONAL && fail_open_.load(std::memory_order_acquire);
    recordFailure(failure_stage, failure_disposition, failed_open);
    const auto detail = error_message.empty() ? "TorchSpec hidden-state publisher failed" : error_message;
    if (failed_open) {
        capture_enabled_.store(false, std::memory_order_release);
        recordCaptureStatus();
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s failed operationally; fail-open is enabled, disabling "
                          "store publishing for this model instance: %s",
                          phase,
                          detail.c_str());
        return false;
    }

    const bool hard_contract = failure_disposition == FailureDisposition::HARD_CONTRACT;
    deferError(detail, hard_contract);
    recordCaptureStatus();
    if (hard_contract) {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s violated a hard contract; disabling store publishing "
                          "and rejecting this and future capture batches after distributed alignment: %s",
                          phase,
                          detail.c_str());
    } else {
        RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] %s failed for the current batch; capture remains enabled "
                          "for subsequent batches: %s",
                          phase,
                          detail.c_str());
    }
    if (!defer_error) {
        throw std::runtime_error(detail);
    }
    return false;
}

void HiddenStateCapturePublisher::initialize(int local_rank) {
    if (!owner_) {
        return;
    }

    const auto perform = [](FailureStage       failure_stage,
                            FailureDisposition failure_disposition,
                            const char*        context,
                            auto&&             operation) -> decltype(auto) {
        try {
            return operation();
        } catch (const CaptureFailure&) {
            throw;
        } catch (const py::error_already_set& e) {
            throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + ": " + e.what());
        } catch (const std::exception& e) {
            throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + ": " + e.what());
        } catch (...) {
            throw CaptureFailure(
                failure_stage, failure_disposition, std::string(context) + " with an unknown exception");
        }
    };

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
                        RTP_LLM_FAIL("Mooncake hidden_dim %ld does not match RTP hidden size %ld",
                                     configured_hidden_dim,
                                     hidden_size_);
                    }
                    const auto configured_layer_count = mooncake_config_.attr("num_aux_layers").cast<int64_t>();
                    if (configured_layer_count != layer_count_) {
                        RTP_LLM_FAIL("Mooncake num_aux_layers %ld does not match RTP capture layer count %ld",
                                     configured_layer_count,
                                     layer_count_);
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
                    for (const char* method_name : {"put_batch", "take_async_errors"}) {
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
                        failure.failureDisposition(),
                        "publisher initialization",
                        /*defer_error=*/false);
    }
}

torch::Tensor HiddenStateCapturePublisher::makeFallback(const torch::Tensor& hidden_states,
                                                        int64_t              token_count,
                                                        const c10::Device&   fallback_device) const {
    const auto fallback_options = torch::TensorOptions(model_dtype_).device(fallback_device);
    if (hidden_states.defined() && hidden_states.dim() == 2 && hidden_states.size(0) == token_count) {
        if (hidden_states.size(1) == hidden_size_) {
            return hidden_states.to(fallback_options).contiguous();
        }
        if (hidden_states.size(1) == packedWidth()) {
            return hidden_states.narrow(1, layer_count_ * hidden_size_, hidden_size_).to(fallback_options).contiguous();
        }
    }
    return torch::zeros({token_count, hidden_size_}, fallback_options);
}

torch::Tensor HiddenStateCapturePublisher::publish(torch::Tensor        packed_hidden_states,
                                                   const torch::Tensor& input_ids,
                                                   const torch::Tensor& input_lengths,
                                                   const torch::Tensor& request_ids,
                                                   int64_t              expected_rows,
                                                   const std::string&   layout_context,
                                                   const c10::Device&   fallback_device) {
    const int64_t publish_start_us = owner_ ? autil::TimeUtility::currentTimeInMicroSeconds() : 0;
    // `shouldPublish()` records the disabled/broken request state before the
    // Python forward. Keep producing packed hidden states on every TP rank so
    // the existing model/FFN communication shape remains identical, but stop
    // owner-local validation/store work once publishing has been disabled.
    if (owner_ && !capture_enabled_.load(std::memory_order_acquire)) {
        return makeFallback(packed_hidden_states, expected_rows, fallback_device);
    }

    std::string layout_error =
        validatePackedLayout(packed_hidden_states, expected_rows, fallback_device, layout_context);
    const auto set_layout_error = [&layout_error](std::string message) {
        if (layout_error.empty()) {
            layout_error = std::move(message);
        }
    };

    torch::Tensor fallback_last_hidden;
    torch::Tensor auxiliary_hidden_view;
    if (layout_error.empty()) {
        try {
            fallback_last_hidden  = packed_hidden_states.narrow(1, layer_count_ * hidden_size_, hidden_size_);
            auxiliary_hidden_view = packed_hidden_states.narrow(1, 0, layer_count_ * hidden_size_);
        } catch (const std::exception& e) {
            set_layout_error(std::string("failed to create captured hidden-state views: ") + e.what());
        } catch (...) {
            set_layout_error("failed to create captured hidden-state views with an unknown exception");
        }
    }

    torch::Tensor last_hidden_states;
    if (layout_error.empty()) {
        try {
            last_hidden_states = fallback_last_hidden.contiguous();
        } catch (const std::exception& e) {
            rejectFailure(std::string("failed to prepare captured hidden-state tensors: ") + e.what(),
                          FailureStage::PREPARE,
                          FailureDisposition::OPERATIONAL);
            if (owner_) {
                PublishMetrics metrics;
                metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
                recordPublish(/*success=*/false, metrics);
            }
            return fallback_last_hidden;
        } catch (...) {
            rejectFailure("failed to prepare captured hidden-state tensors with an unknown exception",
                          FailureStage::PREPARE,
                          FailureDisposition::OPERATIONAL);
            if (owner_) {
                PublishMetrics metrics;
                metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
                recordPublish(/*success=*/false, metrics);
            }
            return fallback_last_hidden;
        }
    }

    if (!layout_error.empty()) {
        rejectLayout(layout_error);
        if (owner_) {
            PublishMetrics metrics;
            metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
            recordPublish(/*success=*/false, metrics);
        }
        if (last_hidden_states.defined()) {
            return last_hidden_states;
        }
        if (fallback_last_hidden.defined()) {
            return fallback_last_hidden;
        }
        const int64_t fallback_tokens =
            input_ids.defined() ?
                input_ids.numel() :
                (packed_hidden_states.defined() && packed_hidden_states.dim() > 0 ? packed_hidden_states.size(0) : 0);
        return makeFallback(packed_hidden_states, fallback_tokens, fallback_device);
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
                metrics.store_put_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - store_put_start_us;
            }
            // Runtime failures must not escape forward before all ranks finish their
            // existing Normal/MTP alignment. Executors consume the owner-local
            // deferred error later; no capture-specific collective is introduced here.
            completeFailure(error_message,
                            failure_stage,
                            failure_disposition,
                            "runtime publish",
                            /*defer_error=*/true);
            if (owner_) {
                metrics.publish_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - publish_start_us;
                metrics.request_count      = lengths_host.defined() ? lengths_host.numel() : 0;
                metrics.token_count        = total_tokens;
                recordPublish(/*success=*/false, metrics);
            }
        };

    if (owner_) {
        try {
            py::gil_scoped_acquire gil;
            const auto             perform = [](FailureStage       failure_stage,
                                    FailureDisposition failure_disposition,
                                    const char*        context,
                                    auto&&             operation) -> decltype(auto) {
                try {
                    return operation();
                } catch (const CaptureFailure&) {
                    throw;
                } catch (const py::error_already_set& e) {
                    throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + ": " + e.what());
                } catch (const std::exception& e) {
                    throw CaptureFailure(failure_stage, failure_disposition, std::string(context) + ": " + e.what());
                } catch (...) {
                    throw CaptureFailure(
                        failure_stage, failure_disposition, std::string(context) + " with an unknown exception");
                }
            };

            perform(FailureStage::STORE,
                    FailureDisposition::OPERATIONAL,
                    "TorchSpec hidden-state store async-error polling failed",
                    [&]() { observeAsyncErrors(FailureStage::STORE, "publish start"); });

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
                        [&]() { return last_hidden_states.to(torch::kBFloat16).contiguous(); });
            torch::Tensor hidden_states_scale;
            if (capture_dtype_ == HiddenStateCaptureDtype::FP8_E4M3) {
                auto quantizer_input         = perform(FailureStage::QUANTIZE,
                                               FailureDisposition::OPERATIONAL,
                                               "failed to prepare TorchSpec FP8 quantizer input",
                                               [&]() { return auxiliary_hidden_view.contiguous(); });
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
                                           [&]() { return auxiliary_hidden_view.to(torch::kBFloat16).contiguous(); });
            }

            torch::Tensor stored_input_ids;
            perform(FailureStage::PREPARE,
                    FailureDisposition::OPERATIONAL,
                    "failed to materialize contiguous hidden-state store payloads",
                    [&]() {
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

            py::list batch_keys;
            py::list batch_hidden_states;
            py::list batch_input_ids;
            py::list batch_last_hidden_states;
            py::list batch_hidden_states_scale;
            int64_t  token_offset = 0;
            for (int64_t i = 0; i < lengths_host.numel(); ++i) {
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

            const auto batch_sequence   = batch_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto first_request_id = request_ids_host.data_ptr<int64_t>()[0];
            const auto last_request_id  = request_ids_host.data_ptr<int64_t>()[request_ids_host.numel() - 1];
            const auto batch_id         = makeBatchId(batch_sequence, first_request_id, last_request_id);

            metrics.has_store_put_latency = true;
            store_put_start_us            = autil::TimeUtility::currentTimeInMicroSeconds();
            py::object acceptance;
            try {
                if (hidden_states_scale.defined()) {
                    acceptance = store_.attr("put_batch")(py::arg("batch_id")            = batch_id,
                                                          py::arg("request_ids")         = batch_keys,
                                                          py::arg("hidden_states")       = batch_hidden_states,
                                                          py::arg("input_ids")           = batch_input_ids,
                                                          py::arg("last_hidden_states")  = batch_last_hidden_states,
                                                          py::arg("hidden_states_scale") = batch_hidden_states_scale);
                } else {
                    acceptance = store_.attr("put_batch")(py::arg("batch_id")           = batch_id,
                                                          py::arg("request_ids")        = batch_keys,
                                                          py::arg("hidden_states")      = batch_hidden_states,
                                                          py::arg("input_ids")          = batch_input_ids,
                                                          py::arg("last_hidden_states") = batch_last_hidden_states);
                }
            } catch (const py::error_already_set& e) {
                const auto python_error = std::string(e.what());
                if (e.matches(PyExc_FileExistsError)) {
                    const auto first_key = batch_keys[0].cast<std::string>();
                    recordDuplicateRequestId(first_request_id, first_key, "Mooncake batch admission");
                    throw CaptureFailure(FailureStage::STORE,
                                         FailureDisposition::REQUEST_ERROR,
                                         "TorchSpec hidden-state store rejected duplicate request keys for batch ["
                                             + batch_id + "]: " + python_error);
                }
                throw CaptureFailure(FailureStage::STORE,
                                     FailureDisposition::OPERATIONAL,
                                     "failed to admit TorchSpec hidden-state batch [" + batch_id
                                         + "]: " + python_error);
            } catch (const std::exception& e) {
                throw CaptureFailure(FailureStage::STORE,
                                     FailureDisposition::OPERATIONAL,
                                     "failed to admit TorchSpec hidden-state batch [" + batch_id + "]: " + e.what());
            } catch (...) {
                throw CaptureFailure(FailureStage::STORE,
                                     FailureDisposition::OPERATIONAL,
                                     "failed to admit TorchSpec hidden-state batch [" + batch_id
                                         + "] with an unknown exception");
            }

            if (!py::isinstance<py::list>(acceptance)
                || py::len(acceptance) != static_cast<py::ssize_t>(lengths_host.numel())) {
                throw CaptureFailure(
                    FailureStage::STORE,
                    FailureDisposition::HARD_CONTRACT,
                    "EagleMooncakeStore.put_batch() must return one accepted metadata item per request for batch ["
                        + batch_id + "]");
            }
            metrics.store_put_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - store_put_start_us;
            store_put_latency_completed  = true;
        } catch (const CaptureFailure& failure) {
            finish_failure(failure.what(), failure.failureStage(), failure.failureDisposition());
            return last_hidden_states;
        } catch (const std::exception& e) {
            finish_failure(std::string("unexpected hidden-state publisher failure: ") + e.what(),
                           FailureStage::PREPARE,
                           FailureDisposition::OPERATIONAL);
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
    return last_hidden_states;
}

void HiddenStateCapturePublisher::flushAndClose() noexcept {
    const auto record_shutdown_failure = [this](const char* operation, const char* detail) noexcept {
        try {
            recordFailure(FailureStage::SHUTDOWN, FailureDisposition::OPERATIONAL, /*fail_open=*/false);
        } catch (...) {
            RTP_LLM_LOG_ERROR("[TorchSpec hidden-state publisher] failed to record shutdown failure metrics");
        }
        RTP_LLM_LOG_ERROR(
            "[TorchSpec hidden-state publisher] failed to %s during shutdown; asynchronous errors retain their "
            "original batch/request IDs: %s",
            operation,
            detail);
    };

    if (store_ && !store_.is_none()) {
        try {
            store_.attr("flush")();
        } catch (const std::exception& e) {
            record_shutdown_failure("flush the hidden-state store", e.what());
        } catch (...) {
            record_shutdown_failure("flush the hidden-state store", "unknown exception");
        }

        // Poll even after a failed flush, then always close. Some store
        // implementations report completed failures separately from flush().
        try {
            observeAsyncErrors(FailureStage::SHUTDOWN, "shutdown after flush");
        } catch (const std::exception& e) {
            record_shutdown_failure("take hidden-state store asynchronous errors", e.what());
        } catch (...) {
            record_shutdown_failure("take hidden-state store asynchronous errors", "unknown exception");
        }

        try {
            store_.attr("close")();
        } catch (const std::exception& e) {
            record_shutdown_failure("close the hidden-state store", e.what());
        } catch (...) {
            record_shutdown_failure("close the hidden-state store", "unknown exception");
        }
        try {
            store_ = py::object();
        } catch (...) {
            // Destruction is best-effort and must never throw.
            (void)store_.release();
        }
    }
    try {
        quantize_fn_ = py::object();
    } catch (...) {
        (void)quantize_fn_.release();
    }
    try {
        mooncake_config_ = py::object();
    } catch (...) {
        (void)mooncake_config_.release();
    }
}

}  // namespace rtp_llm
