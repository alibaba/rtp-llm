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

namespace test {
struct PyWrappedModelTestPeer;
}

struct HiddenStateCaptureStats {
    uint64_t failure_count{0};
    uint64_t broken_rejection_count{0};
};

// Per-owner quota lasts until local completion AND confirmed source-key removal.
// Bytes include retained packed source, IDs, request metadata and store tensors.
// Allocator/container overhead, transient
// quantization workspace and store-internal staging require extra headroom.
struct HiddenStateCaptureLimits {
    int64_t max_inflight_samples{64};
    int64_t max_inflight_bytes{1073741824};
    int64_t max_inflight_tokens{131072};
    int64_t max_pending_batches{8};
    int64_t poll_interval_ms{1000};
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
                                kmonitor::MetricsReporterPtr metrics_reporter,
                                HiddenStateCaptureLimits     limits               = {},
                                std::vector<int64_t>         layer_ids            = {},
                                std::string                  initialization_error = {});
    ~HiddenStateCapturePublisher();

    HiddenStateCapturePublisher(const HiddenStateCapturePublisher&)            = delete;
    HiddenStateCapturePublisher& operator=(const HiddenStateCapturePublisher&) = delete;

    HiddenStateCaptureReservationPtr tryReserve(int64_t request_id, int64_t valid_tokens);
    bool                             hasReservation(int64_t request_id) const;
    int64_t                          inflightSamples() const;
    int64_t                          inflightBytes() const;
    // Only the current synchronous model forward may discard by request ID.
    // Stream cancellation must use its generation-bound reservation token.
    void discardReserved(int64_t request_id);

    bool                    shouldPublish();
    int64_t                 packedWidth() const;
    std::string             validatePackedLayout(const torch::Tensor& tensor,
                                                 int64_t              expected_rows,
                                                 const c10::Device&   expected_device,
                                                 const std::string&   context) const;
    HiddenStateCaptureStats stats() const;
    // Worker failures belong to this owner, never to a later serving batch.
    HiddenStateCaptureFatalState fatalState() const;
    std::string                  diagnostic() const;
    bool                         rejectRequest(std::string error, bool operational = false);
    bool                         rejectDuplicateRequest(int64_t request_id);
    bool                         rejectLayout(std::string layout_error);

    // Every request needs a live, unsubmitted reservation, including TP1/direct
    // callers. A stale reservation drops submission, not the synchronized model
    // capture bit or canonical response.
    // CPU IDs are snapshotted. Device tensors and pinned lengths are strongly
    // retained, not cloned; callers must not mutate/reuse them while pending.
    // Pinned lengths may be an outstanding D2H destination: only the worker
    // reads them, after fencing the producer event. No Python/store wait here.
    bool trySubmit(torch::Tensor        packed_hidden_states,
                   const torch::Tensor& input_ids,
                   const torch::Tensor& input_lengths,
                   const torch::Tensor& request_ids,
                   int64_t              expected_rows,
                   const std::string&   layout_context);

    // Only a delivery-owner <key>_abandoned marker permits producer reclamation,
    // after local completion and native PUT drain. Normal live data is never deleted.
    // Interpreter must remain alive until this completes. Stops admission,
    // drains local work; unconsumed source keys retain store ownership.
    void flushAndClose() noexcept;

private:
    friend struct test::PyWrappedModelTestPeer;

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
        FATAL,
    };

    class CaptureFailure;
    struct PublishTask;
    struct ReservationLifetime {
        std::mutex                   mutex;
        HiddenStateCapturePublisher* publisher{nullptr};
    };
    struct Reservation {
        uint64_t                     generation{0};
        int64_t                      tokens{0};
        int64_t                      bytes{0};
        bool                         submitted{false};
        std::shared_ptr<PublishTask> task;
        std::vector<std::string>      keys;
        bool                         put_attempted{false};
    };

    struct LifetimeQuarantine {
        std::unordered_map<int64_t, Reservation> reservations;
    };

    template<typename Operation>
    static decltype(auto) perform(FailureStage       failure_stage,
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

    void cancelReserved(int64_t request_id, std::optional<uint64_t> generation);
    bool rejectFailure(std::string error_message, FailureStage failure_stage, FailureDisposition failure_disposition);
    void initialize(int local_rank);
    void workerLoop() noexcept;
    void processTask(const std::shared_ptr<PublishTask>& task);
    bool payloadBytes(int64_t tokens, int64_t& bytes, int64_t* store_bytes = nullptr) const;
    bool retainedBytes(const PublishTask& task, int64_t& bytes) const;
    bool flushStore();
    void          releaseLocalPayloads(PublishTask& task);
    void          releaseReservation(std::unordered_map<int64_t, Reservation>::iterator entry);
    pybind11::dict querySourceKeys(const std::vector<std::string>& keys);
    void          pollSourceKeys();
    void quarantineResources() noexcept;
    void publishOnWorker(PublishTask&         task,
                         torch::Tensor        packed_hidden_states,
                         const torch::Tensor& input_ids,
                         const torch::Tensor& input_lengths,
                         const torch::Tensor& request_ids,
                         int64_t              expected_rows,
                         const std::string&   layout_context,
                         const c10::Device&   fallback_device);
    std::string makeRequestKey(int64_t request_id) const;
    std::string makeBatchId(uint64_t batch_sequence, int64_t first_request_id, int64_t last_request_id) const;
    bool        completeFailure(const std::string& error_message,
                                FailureStage       failure_stage,
                                FailureDisposition failure_disposition,
                                const char*        phase);
    void        latchFatal(const char* error_message) noexcept;
    void        reportMetrics(RtpLLMHiddenStateCaptureMetricsCollector& collector);
    void        recordBatch();
    void        recordFailure(FailureStage failure_stage, FailureDisposition failure_disposition);
    void        recordBrokenRejection();
    void        recordDuplicateRequestId(int64_t request_id, const std::string& key, const char* source);
    void        recordCaptureStatus();
    void        recordPublish(bool success, const PublishMetrics& metrics);
    bool        observeAsyncErrors(FailureStage failure_stage, const char* phase);

private:
    const int64_t                  layer_count_;
    const std::vector<int64_t>     layer_ids_;
    const int64_t                  hidden_size_;
    const int                      local_rank_;
    const HiddenStateCaptureLimits limits_;
    const HiddenStateCaptureDtype  capture_dtype_;
    const c10::ScalarType          model_dtype_;
    const bool                     owner_;
    std::string                    store_key_namespace_;
    kmonitor::MetricsReporterPtr   metrics_reporter_;

    std::atomic<bool>     capture_enabled_{true};
    std::atomic<uint64_t> failure_count_{0};
    std::atomic<uint64_t> broken_rejection_count_{0};
    std::atomic<uint64_t> batch_sequence_{0};

    mutable std::mutex           error_mutex_;
    HiddenStateCaptureFatalState fatal_state_;
    // Sticky hard stop: unknown tasks/store stay owned until post-join quarantine;
    // fatal is never evidence of a completed PUT, drain, or local fence.
    std::atomic<bool>            fatal_failed_{false};
    std::string                  diagnostic_;
    std::optional<std::string>   broken_reason_;

    // Tokens retain only this cancellation gate, not the publisher or Executor.
    const std::shared_ptr<ReservationLifetime> reservation_lifetime_ = std::make_shared<ReservationLifetime>();
    mutable std::mutex                         state_mutex_;
    std::mutex                                 close_mutex_;
    std::condition_variable                    state_changed_;
    std::unordered_map<int64_t, Reservation>   reservations_;
    std::deque<std::shared_ptr<PublishTask>>   pending_;
    int64_t                                    reserved_bytes_{0};
    int64_t                                    reserved_tokens_{0};
    uint64_t                                   reservation_generation_{0};
    bool                                       stopping_{false};
    bool                                       worker_active_{false};
    bool                                       worker_finished_{false};
    bool                                       cleanup_requested_{false};

    std::thread                         worker_;
    std::unique_ptr<LifetimeQuarantine> quarantine_;

    // Setup creates exactly async_put_pool_size host leases. Zero is used only
    // for the synchronous GPU Direct send path, which needs no host lease limit.
    int64_t          put_batch_capacity_{0};
    pybind11::object mooncake_config_;
    pybind11::object store_;
    pybind11::object quantize_fn_;
};

}  // namespace rtp_llm
