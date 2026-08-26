#pragma once

#include <atomic>
#include <cstdint>

#include <mutex>
#include <optional>
#include <string>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm {

struct HiddenStateCaptureStats {
    uint64_t failure_count{0};
    uint64_t broken_rejection_count{0};
};

// TODO: Once the current contract is published as an immutable wheel, pin its
// URL and SHA using RTP's Bazel requirements/lock workflow.
class HiddenStateCapturePublisher final {
public:
    HiddenStateCapturePublisher(int64_t                      layer_count,
                                int64_t                      hidden_size,
                                HiddenStateCaptureDtype      capture_dtype,
                                c10::ScalarType              model_dtype,
                                bool                         owner,
                                int                          local_rank,
                                bool                         fail_open,
                                kmonitor::MetricsReporterPtr metrics_reporter);
    ~HiddenStateCapturePublisher();

    HiddenStateCapturePublisher(const HiddenStateCapturePublisher&)            = delete;
    HiddenStateCapturePublisher& operator=(const HiddenStateCapturePublisher&) = delete;

    void                       beginForward();
    bool                       shouldPublish();
    int64_t                    packedWidth() const;
    std::string                validatePackedLayout(const torch::Tensor& tensor,
                                                    int64_t              expected_rows,
                                                    const c10::Device&   expected_device,
                                                    const std::string&   context) const;
    HiddenStateCaptureStats    stats() const;
    std::optional<std::string> takeDeferredError();

    bool rejectLayout(std::string layout_error);

    torch::Tensor
    makeFallback(const torch::Tensor& hidden_states, int64_t token_count, const c10::Device& fallback_device) const;
    torch::Tensor publish(torch::Tensor        packed_hidden_states,
                          const torch::Tensor& input_ids,
                          const torch::Tensor& input_lengths,
                          const torch::Tensor& request_ids,
                          int64_t              expected_rows,
                          const std::string&   layout_context,
                          const c10::Device&   fallback_device);

    // PyWrappedModel calls this explicitly while holding the GIL. The publisher
    // destructor intentionally performs no Python calls.
    void flushAndClose() noexcept;

private:
    enum class FailureStage {
        INITIALIZATION,
        LAYOUT,
        PREPARE,
        QUANTIZE,
        STORE,
        SHUTDOWN,
    };

    enum class FailureDisposition {
        HARD_CONTRACT,
        REQUEST_ERROR,
        OPERATIONAL,
    };

    class CaptureFailure;

    struct PublishMetrics {
        int64_t publish_latency_us{0};
        int64_t quantize_latency_us{0};
        int64_t store_put_latency_us{0};
        int64_t request_count{0};
        int64_t token_count{0};
        int64_t payload_bytes{0};
        int64_t input_ids_bytes{0};
        int64_t auxiliary_hidden_bytes{0};
        int64_t last_hidden_bytes{0};
        int64_t scale_bytes{0};
        bool    has_quantize_latency{false};
        bool    has_store_put_latency{false};
    };

    bool rejectFailure(std::string error_message, FailureStage failure_stage, FailureDisposition failure_disposition);
    void initialize(int local_rank);
    std::string makeRequestKey(int64_t request_id) const;
    std::string makeBatchId(uint64_t batch_sequence, int64_t first_request_id, int64_t last_request_id) const;
    bool        completeFailure(const std::string& error_message,
                                FailureStage       failure_stage,
                                FailureDisposition failure_disposition,
                                const char*        phase,
                                bool               defer_error);
    void        deferError(std::string error_message, bool disable_capture);
    void        reportMetrics(RtpLLMHiddenStateCaptureMetricsCollector& collector);
    void        recordBatch();
    void        recordFailure(FailureStage failure_stage, FailureDisposition failure_disposition, bool fail_open);
    void        recordDisabledSkip();
    void        recordBrokenRejection();
    void        recordDuplicateRequestId(int64_t request_id, const std::string& key, const char* source);
    void        recordCaptureStatus();
    void        recordPublish(bool success, const PublishMetrics& metrics);
    void        observeAsyncErrors(FailureStage failure_stage, const char* phase);

private:
    const int64_t                 layer_count_;
    const int64_t                 hidden_size_;
    const HiddenStateCaptureDtype capture_dtype_;
    const c10::ScalarType         model_dtype_;
    const bool                    owner_;
    std::string                   store_key_namespace_;
    kmonitor::MetricsReporterPtr  metrics_reporter_;

    std::atomic<bool>     fail_open_{false};
    std::atomic<bool>     capture_enabled_{true};
    std::atomic<uint64_t> failure_count_{0};
    std::atomic<uint64_t> broken_rejection_count_{0};
    std::atomic<uint64_t> batch_sequence_{0};

    std::mutex                 error_mutex_;
    std::optional<std::string> deferred_error_;
    std::optional<std::string> broken_reason_;

    pybind11::object mooncake_config_;
    pybind11::object store_;
    pybind11::object quantize_fn_;
};

}  // namespace rtp_llm
