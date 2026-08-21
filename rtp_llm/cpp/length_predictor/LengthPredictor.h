#pragma once

#include <torch/all.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/length_predictor/LengthPredictorState.h"

namespace rtp_llm {

// One stream row inside a submitted step batch.
struct LengthPredictorEntry {
    // Keeps the owning GenerateStream alive until the worker finishes writing
    // back, without making this module depend on the stream library.
    std::shared_ptr<void>  keepalive;
    LengthPredictorState*  state = nullptr;
    int32_t                row   = -1;  // row index in the submitted hidden batch
    int64_t                decode_step = -1;  // generated token count before this step's new token
};

class LengthPredictorConfig {
public:
    int64_t hidden_dim   = 7168;
    int64_t feature_dim  = 128;
    int64_t state_dim    = 32;
    int64_t adapter_dim  = 32;
    int64_t time_dim     = 16;
    int64_t num_bins     = 64;
    double  t_cap        = 73581.0;
    double  scale_limit  = 0.5;
    double  shift_limit  = 0.5;
    double  layernorm_eps  = 1e-5;
    int64_t history_stride = 4;
    int64_t predict_stride = 20;
    // Validation-locked transition curve.
    double hard_anchor_until  = 100.0;
    double half_life_tokens   = 40.0;
    double max_history_weight = 1.0;
};

// Validation-Calibrated remaining-length predictor, async pipeline form.
//
//   dispatch thread, per step (submitStep):
//     validation + bounded-queue push only. No CUDA API is touched on the
//     critical path (~1us). Precondition: the caller invokes submitStep after
//     its own step D2H synchronization, so the kernels that produced `hidden`
//     have already completed and the worker may read it from another stream
//     without cross-stream events.
//
//   single worker thread (owns a dedicated pool CUDA stream):
//     hand-written fused kernel: standardize + GEMM(folded W') + bias' + GELU
//     over the WHOLE batch [B,H] -> [B,F] on its own stream; async D2H into a
//     pinned ring slot; synchronizes only its own stream; then per entry
//     routes by decode step t with hand-written fp32 math:
//       t == 0            -> Prefill anchor + first GRU write
//       (t-1) % 4 == 0    -> GRU observation update
//       t % 20 == 0       -> FiLM + frozen head + alpha(t) log1p fusion
//     and atomically publishes predicted_total. Values become visible with a
//     1-2 step delay by design.
//
// LayerNorm gamma/beta are folded into the encoder weights at load time; the
// encoder weight is stored transposed [H,F] for coalesced kernel access. The
// queue is bounded: when the worker falls behind, whole packets are dropped
// (observe-only feature must never backpressure the engine).
class LengthPredictor {
public:
    // Process-wide instance controlled by RTP_LLM_LENGTH_PREDICTOR_CHECKPOINT.
    // Returns nullptr when the env is unset or loading failed (logged once).
    static LengthPredictor* instance();

    explicit LengthPredictor(const std::string& checkpoint_path);
    // Testing constructor: takes flat-named weights directly.
    LengthPredictor(const LengthPredictorConfig& config, std::unordered_map<std::string, torch::Tensor> weights);
    ~LengthPredictor();

    LengthPredictor(const LengthPredictor&)            = delete;
    LengthPredictor& operator=(const LengthPredictor&) = delete;

    // Submit one engine step. `hidden` is model_output.hidden_states [B, H]
    // (any float dtype, GPU or CPU); `entries` lists the eligible streams with
    // their batch rows and decode steps. Asynchronous and non-blocking.
    void submitStep(const torch::Tensor& hidden, std::vector<LengthPredictorEntry> entries);

    const LengthPredictorConfig& config() const {
        return config_;
    }
    // True when decode step t is on the anchor/observation/prediction grid.
    // Lets the caller skip entry construction for rows the worker would drop.
    bool wantsStep(int64_t decode_step) const {
        return decode_step == 0 || (decode_step - 1) % config_.history_stride == 0
               || decode_step % config_.predict_stride == 0;
    }
    double  alphaAt(int64_t decode_step) const;
    double  fuseTotals(double anchor_total, double history_total, double alpha) const;
    int64_t droppedPackets() const {
        return dropped_packets_.load(std::memory_order_relaxed);
    }
    // Blocks until every packet submitted so far has been consumed (tests).
    void drainForTest();

    // Test hooks. Single-threaded use only; they share the worker scratch.
    void encodeRowForTest(const float* hidden, float* feature) const;
    void processRowForTest(LengthPredictorState& state, const float* feature, int64_t decode_step) const;

private:
    struct Slot {
        torch::Tensor pinned;  // [capacity, F] fp32, pinned when CUDA is available
        torch::Tensor device;  // [capacity, F] fp32 CUDA scratch (undefined on CPU path)
        int64_t       capacity = 0;
    };
    struct StepPacket {
        torch::Tensor                     hidden;  // producing kernels already complete
        std::vector<LengthPredictorEntry> entries;
    };

    void init(const LengthPredictorConfig& config, std::unordered_map<std::string, torch::Tensor> weights);
    void validateConfig() const;
    static std::vector<float> toVector(const torch::Tensor& tensor);

    bool ensureDeviceWeights(const torch::Device& device);
    int  acquireSlot(int64_t batch);
    void releaseSlot(int slot);
    void workerLoop();
    // Worker thread: GPU encode (own stream) or CPU encode, then row routing.
    void processPacket(StepPacket& packet);

    // Hand-written fp32 math (worker thread only).
    void  encodeRowCpu(const float* hidden, float* feature) const;
    void  processRow(LengthPredictorState& state, const float* feature, int64_t decode_step) const;
    void  gruUpdate(std::vector<float>& state, const float* feature, double delta_step) const;
    // Writes num_bins logits into scratch and returns expm1(expected log1p).
    double headExpectedLength(const float* feature, double decode_step) const;
    void   modulate(const float* feature, const float* state, float* modulated) const;

    LengthPredictorConfig config_;

    // Encoder (GPU side): gamma/beta folded, transposed [H, F], fp32.
    torch::Tensor encoder_weight_t_cpu_;
    torch::Tensor encoder_bias_cpu_;
    torch::Tensor encoder_weight_t_device_;
    torch::Tensor encoder_bias_device_;
    bool          device_weights_ready_ = false;
    std::mutex    device_mutex_;

    // CPU-side weights, row-major [out, in].
    std::vector<float> adapter_w_, adapter_b_;
    std::vector<float> gru_w_ih_, gru_w_hh_, gru_b_ih_, gru_b_hh_;
    std::vector<float> mod1_w_, mod1_b_, mod2_w_, mod2_b_;
    int64_t            modulator_hidden_dim_ = 0;
    std::vector<float> time_w_, time_b_;
    std::vector<float> fusion1_w_, fusion1_b_;
    std::vector<float> fusion2_w_, fusion2_b_;
    std::vector<float> bin_centers_;  // log1p space

    // Worker scratch, sized at init (worker thread only).
    struct Scratch {
        std::vector<float> adapted;    // adapter_dim + 2
        std::vector<float> gates_i;    // 3 * state_dim
        std::vector<float> gates_h;    // 3 * state_dim
        std::vector<float> modulated;  // feature_dim
        std::vector<float> mod_inner;  // modulator_hidden_dim
        std::vector<float> mod_raw;    // 2 * feature_dim
        std::vector<float> fused;      // feature_dim + time_dim
        std::vector<float> inner;      // feature_dim
        std::vector<float> logits;     // num_bins
        std::vector<float> new_state;  // state_dim
    };
    mutable Scratch scratch_;

    // Pinned ring + queue.
    std::vector<Slot>        slots_;
    std::vector<int>         free_slots_;
    std::mutex               slot_mutex_;
    // Worker-owned pool CUDA stream (type-erased CudaStreamHolder; null until
    // the first GPU packet and on CPU-only builds).
    std::shared_ptr<void>    worker_stream_;
    std::deque<StepPacket>   queue_;
    std::mutex               queue_mutex_;
    std::condition_variable  queue_cv_;
    std::condition_variable  drained_cv_;
    size_t                   in_flight_ = 0;
    bool                     stopping_  = false;
    std::thread              worker_;
    std::atomic<int64_t>     dropped_packets_{0};
    std::atomic<bool>        runtime_disabled_{false};
};

}  // namespace rtp_llm
