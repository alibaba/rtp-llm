#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm {

enum class HiddenStateCaptureFatalKind {
    NONE,
    TENSOR,
    OUT_OF_MEMORY,
    ACCELERATOR,
    ALLOCATION,
    RUNTIME
};
enum class HiddenStateCaptureFailureStage {
    INITIALIZATION,
    LAYOUT,
    PREPARE,
    QUANTIZE,
    STORE,
    SHUTDOWN
};

struct HiddenStateCaptureFatalState {
    HiddenStateCaptureFatalKind    kind{HiddenStateCaptureFatalKind::NONE};
    HiddenStateCaptureFailureStage stage{HiddenStateCaptureFailureStage::INITIALIZATION};
    std::string                    message;
};

struct HiddenStateCaptureStats {
    uint64_t                    failure_count{0};
    uint64_t                    broken_rejection_count{0};
    HiddenStateCaptureFatalKind fatal_kind{HiddenStateCaptureFatalKind::NONE};
};

// Per publishing owner, not per TP rank. Payload limits exclude temporary
// capture/quantization/staging allocations; deployments need additional headroom.
struct HiddenStateCaptureLimits {
    int64_t max_inflight_samples{64};
    int64_t max_inflight_bytes{1073741824};
    int64_t max_inflight_tokens{131072};
    int64_t max_pending_batches{8};
    int64_t poll_interval_ms{1000};
};

struct HiddenStateCaptureQuotaSnapshot {
    int64_t samples{0};
    int64_t bytes{0};
    int64_t tokens{0};
    int64_t reserved_samples{0};
    int64_t pending_batches{0};
    int64_t published_samples{0};
    int64_t cleanup_pending_samples{0};
};

class HiddenStateCapturePublisher final {
public:
    HiddenStateCapturePublisher(int64_t                      layer_count,
                                int64_t                      hidden_size,
                                HiddenStateCaptureDtype      capture_dtype,
                                c10::ScalarType              model_dtype,
                                bool                         owner,
                                int                          local_rank,
                                kmonitor::MetricsReporterPtr metrics_reporter,
                                HiddenStateCaptureLimits     limits    = {},
                                std::vector<int64_t>         layer_ids = {});
    ~HiddenStateCapturePublisher();

    HiddenStateCapturePublisher(const HiddenStateCapturePublisher&)            = delete;
    HiddenStateCapturePublisher& operator=(const HiddenStateCapturePublisher&) = delete;

    // Admission must happen before capture allocation. IDs are unique for the
    // run namespace. Non-owners consume the synchronized decision, never reserve.
    bool tryReserve(int64_t request_id, int64_t valid_tokens);
    // Serving cancellation only releases RESERVED entries whose local capture
    // work has finished/been discarded. Once submitted (including internally
    // dropped exports), this is a no-op: publication and source cleanup belong
    // exclusively to the publisher, independent of the serving stream lifetime.
    void                            cancelReserved(int64_t request_id);
    HiddenStateCaptureQuotaSnapshot quotaSnapshot() const;

    void                    beginForward();
    bool                    shouldPublish();
    int64_t                 packedWidth() const;
    std::string             validatePackedLayout(const torch::Tensor& tensor,
                                                 int64_t              expected_rows,
                                                 const c10::Device&   expected_device,
                                                 const std::string&   context) const;
    HiddenStateCaptureStats stats() const;
    // Sticky owner-level worker failure, never attributed to a later stream.
    HiddenStateCaptureFatalState fatalState() const;
    // Capture diagnostics are telemetry-only; never return a serving deferred error.
    std::optional<std::string> takeDeferredError();
    bool                       rejectLayout(std::string layout_error);

    // Compatibility helper: only returns a real supplied final or packed tail;
    // malformed capture never manufactures a zero-valued normal model result.
    torch::Tensor
    makeFallback(const torch::Tensor& hidden_states, int64_t token_count, const c10::Device& fallback_device) const;

    // No Python, network, device wait, or flush on this path. Request IDs must
    // be CPU int64 metadata. CPU metadata/IDs are snapshotted; device payloads
    // are strongly owned and fenced on the producer stream, not cloned. The
    // caller must not mutate/reuse their storage while export is pending.
    // Recoverable submission failures return false. c10::Error (including
    // accelerator/OOM errors) and std::bad_alloc propagate to serving unchanged;
    // neither this method nor publish() may be noexcept.
    bool          trySubmit(torch::Tensor        packed_hidden_states,
                            const torch::Tensor& input_ids,
                            const torch::Tensor& input_lengths,
                            const torch::Tensor& request_ids,
                            int64_t              expected_rows,
                            const std::string&   layout_context);
    torch::Tensor publish(torch::Tensor        packed_hidden_states,
                          const torch::Tensor& input_ids,
                          const torch::Tensor& input_lengths,
                          const torch::Tensor& request_ids,
                          int64_t              expected_rows,
                          const std::string&   layout_context,
                          const c10::Device&   fallback_device,
                          const torch::Tensor& canonical_hidden_states = torch::Tensor());

    // Explicit maintenance barrier, never called by forward. Releases a held
    // GIL while waiting for the worker. Published source quota is not released.
    bool flushPending();
    // Interpreter must remain alive. Stops admission and joins without holding
    // the GIL; the worker alone owns Python store calls and reference cleanup.
    void flushAndClose() noexcept;

private:
    using FailureStage = HiddenStateCaptureFailureStage;
    enum class FailureDisposition {
        HARD_CONTRACT,
        REQUEST_ERROR,
        OPERATIONAL,
        FATAL
    };
    enum class ReservationState {
        RESERVED,
        SUBMITTED,
        PUBLISHED,
        CLEANUP_PENDING
    };
    class CaptureFailure;
    struct PublishTask;
    struct Reservation {
        int64_t                      tokens{0};
        int64_t                      bytes{0};
        ReservationState             state{ReservationState::RESERVED};
        std::shared_ptr<PublishTask> task;
        std::vector<std::string>     keys;
    };

    template<typename Operation>
    decltype(auto) perform(FailureStage       failure_stage,
                           FailureDisposition failure_disposition,
                           const char*        context,
                           Operation&&        operation);
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

    static HiddenStateCaptureFatalKind nativeFatalKind(const std::exception& error) noexcept;
    static HiddenStateCaptureFatalKind pythonFatalKind(pybind11::handle error);
    FailureDisposition
         classifyWorkerFailure(const std::exception& error, FailureStage stage, FailureDisposition fallback) noexcept;
    void latchFatal(FailureStage stage, HiddenStateCaptureFatalKind kind, const char* message) noexcept;
    bool payloadBytes(int64_t tokens, int64_t& bytes) const;
    void workerLoop() noexcept;
    void processTask(const std::shared_ptr<PublishTask>& task);
    torch::Tensor publishOnWorker(PublishTask& task);
    bool          flushStore();
    void          pollSourceKeys();
    void          closeStore() noexcept;
    void          releaseReservation(std::unordered_map<int64_t, Reservation>::iterator entry);
    void          recordQuota();
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
    void        recordFailure(FailureStage failure_stage, FailureDisposition failure_disposition);
    void        recordBrokenRejection();
    void        recordDuplicateRequestId(int64_t request_id, const std::string& key, const char* source);
    void        recordCaptureStatus();
    void        recordPublish(bool success, const PublishMetrics& metrics);
    bool        observeAsyncErrors(FailureStage failure_stage, const char* phase);

    const int64_t layer_count_;
    // Real configured order, never inferred from the layer count. Missing IDs
    // are tolerated for source compatibility but prevent worker publication.
    const std::vector<int64_t>               layer_ids_;
    const int64_t                            hidden_size_;
    const HiddenStateCaptureDtype            capture_dtype_;
    const c10::ScalarType                    model_dtype_;
    const bool                               owner_;
    const int                                local_rank_;
    const HiddenStateCaptureLimits           limits_;
    std::string                              store_key_namespace_;
    kmonitor::MetricsReporterPtr             metrics_reporter_;
    std::atomic<bool>                        capture_enabled_{true};
    std::atomic<uint64_t>                    failure_count_{0};
    std::atomic<uint64_t>                    broken_rejection_count_{0};
    std::atomic<uint64_t>                    batch_sequence_{0};
    mutable std::mutex                       error_mutex_;
    std::optional<std::string>               broken_reason_;
    std::atomic<HiddenStateCaptureFatalKind> fatal_kind_{HiddenStateCaptureFatalKind::NONE};
    FailureStage                             fatal_stage_{FailureStage::INITIALIZATION};
    std::string                              fatal_message_;

    mutable std::mutex                       state_mutex_;
    std::mutex                               close_mutex_;
    std::condition_variable                  state_changed_;
    std::unordered_map<int64_t, Reservation> reservations_;
    std::deque<std::shared_ptr<PublishTask>> pending_;
    int64_t                                  reserved_bytes_{0};
    int64_t                                  reserved_tokens_{0};
    bool                                     stopping_{false};
    bool                                     cleanup_requested_{false};
    bool                                     worker_active_{false};
    bool                                     worker_finished_{false};
    std::thread                              worker_;

    pybind11::object mooncake_config_;
    pybind11::object store_;
    pybind11::object quantize_fn_;
};

}  // namespace rtp_llm
