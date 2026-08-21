#include "rtp_llm/cpp/length_predictor/LengthPredictor.h"

#include <torch/script.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAGuard.h"
#include "rtp_llm/cpp/length_predictor/kernels/length_encoder_kernel.h"
#endif

namespace rtp_llm {

namespace {

constexpr char   kCheckpointEnv[]  = "RTP_LLM_LENGTH_PREDICTOR_CHECKPOINT";
constexpr int    kNumSlots         = 4;
constexpr size_t kMaxQueuedPackets = kNumSlots;

constexpr int64_t kDropLogInterval = 1024;

#if USING_CUDA
struct CudaStreamHolder {
    at::cuda::CUDAStream stream;
    explicit CudaStreamHolder(c10::DeviceIndex device):
        stream(at::cuda::getStreamFromPool(/*isHighPriority=*/false, device)) {}
};
#endif

double
readScalar(const std::unordered_map<std::string, torch::Tensor>& weights, const std::string& name, double fallback) {
    auto it = weights.find(name);
    if (it == weights.end()) {
        return fallback;
    }
    return it->second.to(torch::kFloat64).reshape({-1})[0].item<double>();
}

torch::Tensor takeWeight(std::unordered_map<std::string, torch::Tensor>& weights,
                         const std::string&                              name,
                         std::initializer_list<int64_t>                  shape) {
    auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("length predictor weight pack is missing tensor: " + name);
    }
    auto tensor = it->second.to(torch::kFloat32).contiguous();
    if (static_cast<size_t>(tensor.dim()) != shape.size()) {
        throw std::runtime_error("length predictor tensor " + name + " has wrong rank");
    }
    int64_t dim = 0;
    for (const int64_t expected : shape) {
        if (expected >= 0 && tensor.size(dim) != expected) {
            throw std::runtime_error("length predictor tensor " + name + " has wrong shape at dim "
                                     + std::to_string(dim) + ": expected " + std::to_string(expected) + ", got "
                                     + std::to_string(tensor.size(dim)));
        }
        ++dim;
    }
    if (!tensor.isfinite().all().item<bool>()) {
        throw std::runtime_error("length predictor tensor " + name + " contains NaN/Inf");
    }
    return tensor;
}

// y = W x + b with row-major W [out, in].
void gemv(const std::vector<float>& w,
          const std::vector<float>& b,
          const float* __restrict__ x,
          float* __restrict__ y,
          int out,
          int in) {
    for (int i = 0; i < out; ++i) {
        const float* __restrict__ row = w.data() + static_cast<size_t>(i) * in;
        float acc                     = b[i];
        for (int k = 0; k < in; ++k) {
            acc += row[k] * x[k];
        }
        y[i] = acc;
    }
}

// erf-form GELU, matching torch.nn.GELU(approximate='none').
inline float geluErf(float value) {
    return 0.5f * value * (1.0f + std::erf(value * 0.70710678118654752440f));
}

inline float sigmoidf(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

LengthPredictor* LengthPredictor::instance() {
    static std::unique_ptr<LengthPredictor> predictor = []() -> std::unique_ptr<LengthPredictor> {
        const char* path = std::getenv(kCheckpointEnv);
        if (path == nullptr || path[0] == '\0') {
            return nullptr;
        }
        try {
            auto result = std::make_unique<LengthPredictor>(std::string(path));
            RTP_LLM_LOG_INFO("length predictor enabled from %s (hidden_dim=%ld, t_cap=%.0f, "
                             "history_stride=%ld, predict_stride=%ld, hard_anchor=%.0f, half_life=%.0f, max_w=%.2f)",
                             path,
                             result->config().hidden_dim,
                             result->config().t_cap,
                             result->config().history_stride,
                             result->config().predict_stride,
                             result->config().hard_anchor_until,
                             result->config().half_life_tokens,
                             result->config().max_history_weight);
            return result;
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("length predictor disabled: failed to load %s: %s", path, e.what());
            return nullptr;
        }
    }();
    return predictor.get();
}

LengthPredictor::LengthPredictor(const std::string& checkpoint_path) {
    auto container = torch::jit::load(checkpoint_path, torch::kCPU);

    std::unordered_map<std::string, torch::Tensor> weights;
    for (const auto& buffer : container.named_buffers(/*recurse=*/true)) {
        weights[buffer.name] = buffer.value;
    }
    for (const auto& parameter : container.named_parameters(/*recurse=*/true)) {
        weights[parameter.name] = parameter.value;
    }
    if (weights.empty()) {
        throw std::runtime_error(checkpoint_path + ": weight pack contains no tensors");
    }

    LengthPredictorConfig config;
    config.hidden_dim     = static_cast<int64_t>(readScalar(weights, "config_hidden_dim", config.hidden_dim));
    config.feature_dim    = static_cast<int64_t>(readScalar(weights, "config_feature_dim", config.feature_dim));
    config.state_dim      = static_cast<int64_t>(readScalar(weights, "config_state_dim", config.state_dim));
    config.adapter_dim    = static_cast<int64_t>(readScalar(weights, "config_adapter_dim", config.adapter_dim));
    config.time_dim       = static_cast<int64_t>(readScalar(weights, "config_time_dim", config.time_dim));
    config.num_bins       = static_cast<int64_t>(readScalar(weights, "config_num_bins", config.num_bins));
    config.t_cap          = readScalar(weights, "config_t_cap", config.t_cap);
    config.scale_limit    = readScalar(weights, "config_scale_limit", config.scale_limit);
    config.shift_limit    = readScalar(weights, "config_shift_limit", config.shift_limit);
    config.layernorm_eps  = readScalar(weights, "config_layernorm_eps", config.layernorm_eps);
    config.history_stride = static_cast<int64_t>(readScalar(weights, "config_history_stride", config.history_stride));
    config.predict_stride = static_cast<int64_t>(readScalar(weights, "config_predict_stride", config.predict_stride));
    config.hard_anchor_until  = readScalar(weights, "config_hard_anchor_until", config.hard_anchor_until);
    config.half_life_tokens   = readScalar(weights, "config_half_life_tokens", config.half_life_tokens);
    config.max_history_weight = readScalar(weights, "config_max_history_weight", config.max_history_weight);

    init(config, std::move(weights));
}

LengthPredictor::LengthPredictor(const LengthPredictorConfig&                   config,
                                 std::unordered_map<std::string, torch::Tensor> weights) {
    init(config, std::move(weights));
}

LengthPredictor::~LengthPredictor() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stopping_ = true;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

std::vector<float> LengthPredictor::toVector(const torch::Tensor& tensor) {
    auto         contiguous = tensor.to(torch::kFloat32).contiguous();
    const float* data       = contiguous.data_ptr<float>();
    return std::vector<float>(data, data + contiguous.numel());
}

void LengthPredictor::init(const LengthPredictorConfig&                   config,
                           std::unordered_map<std::string, torch::Tensor> weights) {
    config_ = config;
    validateConfig();
    const int64_t hidden  = config_.hidden_dim;
    const int64_t feature = config_.feature_dim;
    const int64_t state   = config_.state_dim;
    const int64_t adapter = config_.adapter_dim;
    const int64_t time    = config_.time_dim;
    const int64_t bins    = config_.num_bins;

    // Fold LayerNorm gamma/beta into the encoder linear:
    //   W' = W * diag(gamma),  b' = b + W * beta
    // and transpose to [H, F] for the coalesced kernel layout. The runtime
    // LayerNorm then degenerates to a pure standardization.
    auto ln_gamma         = takeWeight(weights, "encoder_ln_weight", {hidden});
    auto ln_beta          = takeWeight(weights, "encoder_ln_bias", {hidden});
    auto enc_w            = takeWeight(weights, "encoder_linear_weight", {feature, hidden});
    auto enc_b            = takeWeight(weights, "encoder_linear_bias", {feature});
    auto folded_w         = enc_w * ln_gamma.unsqueeze(0);
    auto folded_b         = enc_b + enc_w.matmul(ln_beta);
    encoder_weight_t_cpu_ = folded_w.t().contiguous();
    encoder_bias_cpu_     = folded_b.contiguous();

    time_w_    = toVector(takeWeight(weights, "time_linear_weight", {time, 2}));
    time_b_    = toVector(takeWeight(weights, "time_linear_bias", {time}));
    fusion1_w_ = toVector(takeWeight(weights, "fusion1_weight", {feature, feature + time}));
    fusion1_b_ = toVector(takeWeight(weights, "fusion1_bias", {feature}));
    fusion2_w_ = toVector(takeWeight(weights, "fusion2_weight", {bins, feature}));
    fusion2_b_ = toVector(takeWeight(weights, "fusion2_bias", {bins}));
    adapter_w_ = toVector(takeWeight(weights, "adapter_weight", {adapter, feature}));
    adapter_b_ = toVector(takeWeight(weights, "adapter_bias", {adapter}));
    gru_w_ih_  = toVector(takeWeight(weights, "gru_weight_ih", {3 * state, adapter + 2}));
    gru_w_hh_  = toVector(takeWeight(weights, "gru_weight_hh", {3 * state, state}));
    gru_b_ih_  = toVector(takeWeight(weights, "gru_bias_ih", {3 * state}));
    gru_b_hh_  = toVector(takeWeight(weights, "gru_bias_hh", {3 * state}));

    auto mod1_w           = takeWeight(weights, "modulator1_weight", {-1, state});
    modulator_hidden_dim_ = mod1_w.size(0);
    mod1_w_               = toVector(mod1_w);
    mod1_b_               = toVector(takeWeight(weights, "modulator1_bias", {modulator_hidden_dim_}));
    mod2_w_               = toVector(takeWeight(weights, "modulator2_weight", {2 * feature, modulator_hidden_dim_}));
    mod2_b_               = toVector(takeWeight(weights, "modulator2_bias", {2 * feature}));

    auto centers = takeWeight(weights, "bin_centers", {bins});
    if (bins > 1 && !(centers.slice(0, 1) > centers.slice(0, 0, bins - 1)).all().item<bool>()) {
        throw std::runtime_error("length predictor bin centers must be strictly increasing");
    }
    bin_centers_ = toVector(centers);

    scratch_.adapted.resize(adapter + 2);
    scratch_.gates_i.resize(3 * state);
    scratch_.gates_h.resize(3 * state);
    scratch_.modulated.resize(feature);
    scratch_.mod_inner.resize(modulator_hidden_dim_);
    scratch_.mod_raw.resize(2 * feature);
    scratch_.fused.resize(feature + time);
    scratch_.inner.resize(feature);
    scratch_.logits.resize(bins);
    scratch_.new_state.resize(state);

    slots_.resize(kNumSlots);
    free_slots_.reserve(kNumSlots);
    for (int i = 0; i < kNumSlots; ++i) {
        free_slots_.push_back(i);
    }

    worker_ = std::thread([this] { workerLoop(); });
}

void LengthPredictor::validateConfig() const {
    if (config_.hidden_dim <= 0 || config_.feature_dim <= 0 || config_.state_dim <= 0 || config_.adapter_dim <= 0
        || config_.time_dim <= 0 || config_.num_bins <= 0) {
        throw std::runtime_error("length predictor config dimensions must be positive");
    }
    if (config_.t_cap <= 0 || config_.history_stride <= 0 || config_.predict_stride <= 0
        || config_.layernorm_eps <= 0) {
        throw std::runtime_error("length predictor t_cap, strides, and layernorm eps must be positive");
    }
    if (config_.half_life_tokens <= 0 || config_.max_history_weight < 0 || config_.max_history_weight > 1) {
        throw std::runtime_error("length predictor transition curve parameters are invalid");
    }
}

double LengthPredictor::alphaAt(int64_t decode_step) const {
    const double step = static_cast<double>(decode_step);
    if (step <= config_.hard_anchor_until) {
        return 0.0;
    }
    const double elapsed = step - config_.hard_anchor_until;
    return config_.max_history_weight * (1.0 - std::exp2(-elapsed / config_.half_life_tokens));
}

double LengthPredictor::fuseTotals(double anchor_total, double history_total, double alpha) const {
    if (alpha <= 0.0) {
        return anchor_total;
    }
    const double log_anchor  = std::log1p(std::max(anchor_total, 0.0));
    const double log_history = std::log1p(std::max(history_total, 0.0));
    return std::expm1((1.0 - alpha) * log_anchor + alpha * log_history);
}

bool LengthPredictor::ensureDeviceWeights(const torch::Device& device) {
    std::lock_guard<std::mutex> lock(device_mutex_);
    if (device_weights_ready_) {
        return encoder_weight_t_device_.device() == device;
    }
    encoder_weight_t_device_ = encoder_weight_t_cpu_.to(device);
    encoder_bias_device_     = encoder_bias_cpu_.to(device);
    device_weights_ready_    = true;
    return true;
}

int LengthPredictor::acquireSlot(int64_t batch) {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    if (free_slots_.empty()) {
        return -1;
    }
    const int index = free_slots_.back();
    free_slots_.pop_back();
    (void)batch;  // buffers are (re)allocated by the caller that knows the device
    return index;
}

void LengthPredictor::releaseSlot(int slot) {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    free_slots_.push_back(slot);
}

void LengthPredictor::submitStep(const torch::Tensor& hidden, std::vector<LengthPredictorEntry> entries) {
    if (runtime_disabled_.load(std::memory_order_relaxed)) {
        return;
    }
    if (!hidden.defined() || hidden.dim() != 2 || entries.empty()) {
        return;
    }
    const int64_t batch = hidden.size(0);
    if (hidden.size(1) != config_.hidden_dim) {
        RTP_LLM_LOG_ERROR("length predictor disabled: hidden dim %ld does not match weight pack dim %ld",
                          hidden.size(1),
                          config_.hidden_dim);
        runtime_disabled_.store(true, std::memory_order_relaxed);
        return;
    }
#if !USING_CUDA
    if (hidden.is_cuda()) {
        return;
    }
#endif
    // Keep only well-formed rows; grid routing itself stays on the worker.
    entries.erase(std::remove_if(entries.begin(),
                                 entries.end(),
                                 [batch](const LengthPredictorEntry& entry) {
                                     return entry.state == nullptr || entry.row < 0 || entry.row >= batch
                                            || entry.decode_step < 0;
                                 }),
                  entries.end());
    if (entries.empty()) {
        return;
    }

    // Critical path ends here: hand the raw hidden reference to the worker.
    // The producing kernels are already complete (the dispatcher synchronized
    // its token D2H before calling us), so the worker can read the tensor from
    // its own CUDA stream, and the reference held by the packet keeps the
    // memory from being recycled by the caching allocator.
    StepPacket packet;
    packet.hidden  = hidden;
    packet.entries = std::move(entries);
    StepPacket victim;  // destroyed outside the lock
    int64_t    dropped = 0;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.size() >= kMaxQueuedPackets) {
            // Prefer fresh observations over a stale backlog: evict the oldest
            // queued packet. in_flight_ stays balanced (-1 evicted, +1 pushed).
            victim = std::move(queue_.front());
            queue_.pop_front();
            dropped = dropped_packets_.fetch_add(1, std::memory_order_relaxed) + 1;
        } else {
            ++in_flight_;
        }
        queue_.push_back(std::move(packet));
    }
    queue_cv_.notify_one();
    if (dropped > 0 && dropped % kDropLogInterval == 1) {
        RTP_LLM_LOG_WARNING("length predictor dropped %ld packets: worker is behind", dropped);
    }
}

void LengthPredictor::processPacket(StepPacket& packet) {
    const torch::Tensor& hidden     = packet.hidden;
    const int64_t        batch      = hidden.size(0);
    const int            slot_index = acquireSlot(batch);
    if (slot_index < 0) {  // cannot happen while kMaxQueuedPackets <= kNumSlots
        dropped_packets_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Slot& slot = slots_[slot_index];

    if (hidden.is_cuda()) {
#if USING_CUDA
        if (!ensureDeviceWeights(hidden.device())) {
            RTP_LLM_LOG_ERROR("length predictor disabled: hidden device changed after weight residency");
            runtime_disabled_.store(true, std::memory_order_relaxed);
            releaseSlot(slot_index);
            return;
        }
        if (!worker_stream_) {
            worker_stream_ = std::make_shared<CudaStreamHolder>(hidden.device().index());
        }
        auto& stream = static_cast<CudaStreamHolder*>(worker_stream_.get())->stream;
        // Guard so slot allocations and the launches below all live on the
        // worker's own stream; the main stream never sees this work.
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        if (slot.capacity < batch) {
            slot.pinned =
                torch::empty({batch, config_.feature_dim},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
            slot.device   = torch::empty({batch, config_.feature_dim},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(hidden.device()));
            slot.capacity = batch;
        }
        const float* weight_t = encoder_weight_t_device_.data_ptr<float>();
        const float* bias     = encoder_bias_device_.data_ptr<float>();
        float*       out      = slot.device.data_ptr<float>();
        const int    b        = static_cast<int>(batch);
        const int    h        = static_cast<int>(config_.hidden_dim);
        const int    f        = static_cast<int>(config_.feature_dim);
        switch (hidden.scalar_type()) {
            case torch::kFloat32:
                invokeLengthEncoderForward<float>(hidden.data_ptr<float>(),
                                                  weight_t,
                                                  bias,
                                                  out,
                                                  b,
                                                  h,
                                                  f,
                                                  static_cast<float>(config_.layernorm_eps),
                                                  stream.stream());
                break;
            case torch::kHalf:
                invokeLengthEncoderForward<__half>(reinterpret_cast<const __half*>(hidden.data_ptr()),
                                                   weight_t,
                                                   bias,
                                                   out,
                                                   b,
                                                   h,
                                                   f,
                                                   static_cast<float>(config_.layernorm_eps),
                                                   stream.stream());
                break;
            case torch::kBFloat16:
                invokeLengthEncoderForward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()),
                                                          weight_t,
                                                          bias,
                                                          out,
                                                          b,
                                                          h,
                                                          f,
                                                          static_cast<float>(config_.layernorm_eps),
                                                          stream.stream());
                break;
            default:
                RTP_LLM_LOG_ERROR("length predictor disabled: unsupported hidden dtype");
                runtime_disabled_.store(true, std::memory_order_relaxed);
                releaseSlot(slot_index);
                return;
        }
        cudaMemcpyAsync(slot.pinned.data_ptr<float>(),
                        out,
                        static_cast<size_t>(batch) * config_.feature_dim * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream.stream());
        // Blocks only this worker thread on its own stream.
        cudaStreamSynchronize(stream.stream());
#else
        releaseSlot(slot_index);
        return;
#endif
    } else {
        // CPU hidden (CPU-only builds, tests): encode entry rows directly.
        if (slot.capacity < batch) {
            slot.pinned   = torch::empty({batch, config_.feature_dim},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            slot.capacity = batch;
        }
        auto         source = hidden.to(torch::kFloat32).contiguous();
        const float* src    = source.data_ptr<float>();
        float*       dst    = slot.pinned.data_ptr<float>();
        for (const auto& entry : packet.entries) {
            encodeRowCpu(src + static_cast<size_t>(entry.row) * config_.hidden_dim,
                         dst + static_cast<size_t>(entry.row) * config_.feature_dim);
        }
    }

    const float* features = slot.pinned.data_ptr<float>();
    for (const auto& entry : packet.entries) {
        processRow(*entry.state, features + static_cast<size_t>(entry.row) * config_.feature_dim, entry.decode_step);
    }
    releaseSlot(slot_index);
}

void LengthPredictor::workerLoop() {
    for (;;) {
        StepPacket packet;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stopping_) {
                    return;
                }
                continue;
            }
            packet = std::move(queue_.front());
            queue_.pop_front();
        }
        processPacket(packet);
        packet = StepPacket();  // drop hidden/stream keepalives before signaling drain
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            --in_flight_;
        }
        drained_cv_.notify_all();
    }
}

void LengthPredictor::drainForTest() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    drained_cv_.wait(lock, [this] { return in_flight_ == 0; });
}

void LengthPredictor::encodeRowCpu(const float* hidden, float* feature) const {
    const int h = static_cast<int>(config_.hidden_dim);
    const int f = static_cast<int>(config_.feature_dim);

    double sum = 0.0, sumsq = 0.0;
    for (int k = 0; k < h; ++k) {
        sum += hidden[k];
        sumsq += static_cast<double>(hidden[k]) * hidden[k];
    }
    const double mean     = sum / h;
    const double variance = std::max(sumsq / h - mean * mean, 0.0);
    const float  rstd     = static_cast<float>(1.0 / std::sqrt(variance + config_.layernorm_eps));
    const float  mean_f   = static_cast<float>(mean);

    const float* weight_t = encoder_weight_t_cpu_.data_ptr<float>();  // [H, F]
    const float* bias     = encoder_bias_cpu_.data_ptr<float>();
    for (int j = 0; j < f; ++j) {
        feature[j] = bias[j];
    }
    for (int k = 0; k < h; ++k) {
        const float normalized        = (hidden[k] - mean_f) * rstd;
        const float* __restrict__ row = weight_t + static_cast<size_t>(k) * f;
        for (int j = 0; j < f; ++j) {
            feature[j] += normalized * row[j];
        }
    }
    for (int j = 0; j < f; ++j) {
        feature[j] = geluErf(feature[j]);
    }
}

double LengthPredictor::headExpectedLength(const float* feature, double decode_step) const {
    const int f    = static_cast<int>(config_.feature_dim);
    const int time = static_cast<int>(config_.time_dim);
    const int bins = static_cast<int>(config_.num_bins);

    const float time_features[2] = {
        static_cast<float>(decode_step / config_.t_cap),
        static_cast<float>(std::log1p(decode_step) / std::log1p(config_.t_cap)),
    };
    float* fused = scratch_.fused.data();
    for (int j = 0; j < f; ++j) {
        fused[j] = feature[j];
    }
    gemv(time_w_, time_b_, time_features, fused + f, time, 2);
    for (int j = 0; j < time; ++j) {
        fused[f + j] = geluErf(fused[f + j]);
    }
    gemv(fusion1_w_, fusion1_b_, fused, scratch_.inner.data(), f, f + time);
    for (int j = 0; j < f; ++j) {
        scratch_.inner[j] = geluErf(scratch_.inner[j]);
    }
    gemv(fusion2_w_, fusion2_b_, scratch_.inner.data(), scratch_.logits.data(), bins, f);

    float max_logit = scratch_.logits[0];
    for (int j = 1; j < bins; ++j) {
        max_logit = std::max(max_logit, scratch_.logits[j]);
    }
    double normalizer = 0.0, expected = 0.0;
    for (int j = 0; j < bins; ++j) {
        const double p = std::exp(static_cast<double>(scratch_.logits[j]) - max_logit);
        normalizer += p;
        expected += p * bin_centers_[j];
    }
    return std::max(std::expm1(expected / normalizer), 0.0);
}

void LengthPredictor::modulate(const float* feature, const float* state, float* modulated) const {
    const int f = static_cast<int>(config_.feature_dim);
    const int m = static_cast<int>(modulator_hidden_dim_);
    gemv(mod1_w_, mod1_b_, state, scratch_.mod_inner.data(), m, static_cast<int>(config_.state_dim));
    for (int j = 0; j < m; ++j) {
        scratch_.mod_inner[j] = geluErf(scratch_.mod_inner[j]);
    }
    gemv(mod2_w_, mod2_b_, scratch_.mod_inner.data(), scratch_.mod_raw.data(), 2 * f, m);
    const float scale_limit = static_cast<float>(config_.scale_limit);
    const float shift_limit = static_cast<float>(config_.shift_limit);
    for (int j = 0; j < f; ++j) {
        const float scale = 1.0f + scale_limit * std::tanh(scratch_.mod_raw[j]);
        const float shift = shift_limit * std::tanh(scratch_.mod_raw[f + j]);
        modulated[j]      = feature[j] * scale + shift;
    }
}

void LengthPredictor::gruUpdate(std::vector<float>& state, const float* feature, double delta_step) const {
    const int adapter = static_cast<int>(config_.adapter_dim);
    const int s       = static_cast<int>(config_.state_dim);

    float* input = scratch_.adapted.data();
    gemv(adapter_w_, adapter_b_, feature, input, adapter, static_cast<int>(config_.feature_dim));
    for (int j = 0; j < adapter; ++j) {
        input[j] = geluErf(input[j]);
    }
    const double stride = static_cast<double>(config_.history_stride);
    input[adapter]      = static_cast<float>(delta_step / stride);
    input[adapter + 1]  = static_cast<float>(std::log1p(delta_step) / std::log1p(stride));

    gemv(gru_w_ih_, gru_b_ih_, input, scratch_.gates_i.data(), 3 * s, adapter + 2);
    gemv(gru_w_hh_, gru_b_hh_, state.data(), scratch_.gates_h.data(), 3 * s, s);
    const float* gi = scratch_.gates_i.data();
    const float* gh = scratch_.gates_h.data();
    for (int j = 0; j < s; ++j) {
        const float reset     = sigmoidf(gi[j] + gh[j]);
        const float update    = sigmoidf(gi[s + j] + gh[s + j]);
        const float candidate = std::tanh(gi[2 * s + j] + reset * gh[2 * s + j]);
        scratch_.new_state[j] = (1.0f - update) * candidate + update * state[j];
    }
    state.assign(scratch_.new_state.begin(), scratch_.new_state.end());
}

void LengthPredictor::processRow(LengthPredictorState& state, const float* feature, int64_t decode_step) const {
    if (decode_step == 0) {
        if (state.anchor_ready) {
            return;
        }
        // Prefill Once: remaining == total at t=0. Predict-then-consume: the
        // prefill feature enters the GRU only after the anchor is computed.
        state.anchor_total = headExpectedLength(feature, 0.0);
        state.predicted_total.store(state.anchor_total, std::memory_order_relaxed);
        state.gru_state.assign(config_.state_dim, 0.0f);
        gruUpdate(state.gru_state, feature, 0.0);
        state.last_obs_step = 0;
        state.anchor_ready  = true;
        return;
    }
    if (!state.anchor_ready) {
        return;
    }
    // Offline contract: hidden observed every history_stride tokens starting
    // from the first decode token (t = 1, 1+s, ...); formal predictions every
    // predict_stride tokens. Prediction always precedes the same-step write.
    const bool is_prediction  = decode_step % config_.predict_stride == 0;
    const bool is_observation = (decode_step - 1) % config_.history_stride == 0;

    if (is_prediction) {
        const double alpha = alphaAt(decode_step);
        if (alpha <= 0.0) {
            state.predicted_total.store(state.anchor_total, std::memory_order_relaxed);
        } else {
            modulate(feature, state.gru_state.data(), scratch_.modulated.data());
            const double history_remaining =
                headExpectedLength(scratch_.modulated.data(), static_cast<double>(decode_step));
            const double history_total = history_remaining + static_cast<double>(decode_step);
            state.predicted_total.store(fuseTotals(state.anchor_total, history_total, alpha),
                                        std::memory_order_relaxed);
        }
    }
    if (is_observation && decode_step > state.last_obs_step) {
        gruUpdate(state.gru_state, feature, static_cast<double>(decode_step - state.last_obs_step));
        state.last_obs_step = decode_step;
    }
}

void LengthPredictor::encodeRowForTest(const float* hidden, float* feature) const {
    encodeRowCpu(hidden, feature);
}

void LengthPredictor::processRowForTest(LengthPredictorState& state, const float* feature, int64_t decode_step) const {
    processRow(state, feature, decode_step);
}

}  // namespace rtp_llm
