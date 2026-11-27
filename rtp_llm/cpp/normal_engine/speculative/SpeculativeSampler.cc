#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeKernels.cuh"
#endif

namespace rtp_llm {
namespace speculative {

#if USING_CUDA
namespace {

constexpr size_t kMaxDSparkCachedGraphs = 32;

inline void hashCombine(size_t& seed, size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

int64_t padDSparkBatchSize(int64_t batch_size) {
    constexpr std::array<int64_t, 7> kBuckets{1, 2, 4, 8, 16, 24, 32};
    for (int64_t bucket : kBuckets) {
        if (batch_size <= bucket) {
            return bucket;
        }
    }
    int64_t padded = 64;
    while (padded < batch_size && padded <= std::numeric_limits<int64_t>::max() / 2) {
        padded <<= 1;
    }
    return padded >= batch_size ? padded : batch_size;
}

torch::Tensor allocateAlignedCudaTensor(c10::IntArrayRef sizes, const torch::TensorOptions& options) {
    auto tensor = torch::zeros(sizes, options);
    if (tensor.numel() == 0 || (reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 511ULL) == 0) {
        return tensor;
    }
    const auto elem_size    = static_cast<int64_t>(tensor.element_size());
    const auto pad_elems    = (512 + elem_size - 1) / elem_size;
    auto       backing      = torch::zeros({tensor.numel() + pad_elems}, options);
    const auto raw_addr     = reinterpret_cast<uintptr_t>(backing.data_ptr());
    const auto aligned_addr = (raw_addr + 511ULL) & ~511ULL;
    const auto byte_offset  = static_cast<int64_t>(aligned_addr - raw_addr);
    RTP_LLM_CHECK_WITH_INFO(byte_offset % elem_size == 0, "512B alignment offset must align to element size");
    return backing.narrow(0, byte_offset / elem_size, tensor.numel()).view(sizes);
}

class DSparkCaptureModeGuard {
public:
    DSparkCaptureModeGuard(): prev_(CaptureCheck::in_cuda_graph_capture) {
        CaptureCheck::in_cuda_graph_capture = true;
    }

    ~DSparkCaptureModeGuard() {
        CaptureCheck::in_cuda_graph_capture = prev_;
    }

    DSparkCaptureModeGuard(const DSparkCaptureModeGuard&)            = delete;
    DSparkCaptureModeGuard& operator=(const DSparkCaptureModeGuard&) = delete;

private:
    bool prev_;
};

}  // namespace

enum class DSparkSampleMode : uint8_t {
    DENSE_PROBS = 0,
};

struct DSparkGraphKey {
    int64_t          padded_bsz        = 0;
    int64_t          gamma             = 0;
    int64_t          draft_vocab_size  = 0;
    c10::ScalarType  base_logits_dtype = c10::ScalarType::Undefined;
    bool             has_markov        = false;
    c10::ScalarType  markov_dtype      = c10::ScalarType::Undefined;
    int64_t          markov_rank       = 0;
    bool             has_d2t           = false;
    c10::ScalarType  d2t_dtype         = c10::ScalarType::Undefined;
    c10::DeviceIndex device_index      = 0;
    DSparkSampleMode sample_mode       = DSparkSampleMode::DENSE_PROBS;

    bool operator==(const DSparkGraphKey& other) const {
        return padded_bsz == other.padded_bsz && gamma == other.gamma && draft_vocab_size == other.draft_vocab_size
               && base_logits_dtype == other.base_logits_dtype && has_markov == other.has_markov
               && markov_dtype == other.markov_dtype && markov_rank == other.markov_rank && has_d2t == other.has_d2t
               && d2t_dtype == other.d2t_dtype && device_index == other.device_index && sample_mode == other.sample_mode;
    }
};

struct DSparkGraphKeyHash {
    size_t operator()(const DSparkGraphKey& key) const {
        size_t seed = 0;
        hashCombine(seed, std::hash<int64_t>{}(key.padded_bsz));
        hashCombine(seed, std::hash<int64_t>{}(key.gamma));
        hashCombine(seed, std::hash<int64_t>{}(key.draft_vocab_size));
        hashCombine(seed, std::hash<int>{}(static_cast<int>(key.base_logits_dtype)));
        hashCombine(seed, std::hash<bool>{}(key.has_markov));
        hashCombine(seed, std::hash<int>{}(static_cast<int>(key.markov_dtype)));
        hashCombine(seed, std::hash<int64_t>{}(key.markov_rank));
        hashCombine(seed, std::hash<bool>{}(key.has_d2t));
        hashCombine(seed, std::hash<int>{}(static_cast<int>(key.d2t_dtype)));
        hashCombine(seed, std::hash<int>{}(static_cast<int>(key.device_index)));
        hashCombine(seed, std::hash<uint8_t>{}(static_cast<uint8_t>(key.sample_mode)));
        return seed;
    }
};

struct DSparkBoundWeights {
    const void* w1_ptr     = nullptr;
    int64_t     w1_rows    = 0;
    int64_t     w1_stride0 = 0;
    const void* w2_ptr     = nullptr;
    int64_t     w2_stride0 = 0;
    int64_t     w2_stride1 = 0;
    const void* d2t_ptr    = nullptr;
    int64_t     d2t_numel  = 0;

    bool operator==(const DSparkBoundWeights& other) const {
        return w1_ptr == other.w1_ptr && w1_rows == other.w1_rows && w1_stride0 == other.w1_stride0
               && w2_ptr == other.w2_ptr && w2_stride0 == other.w2_stride0 && w2_stride1 == other.w2_stride1
               && d2t_ptr == other.d2t_ptr && d2t_numel == other.d2t_numel;
    }

    bool operator!=(const DSparkBoundWeights& other) const {
        return !(*this == other);
    }
};

struct DSparkGraphEntry {
    at::cuda::CUDAGraph                 graph;
    DSparkBoundWeights                  bound_weights;
    torch::Tensor                       base_logits_slot;
    torch::Tensor                       anchors_slot;
    torch::Tensor                       temperature_slot;
    torch::Tensor                       seed_arr;
    torch::Tensor                       offset_arr;
    torch::Tensor                       markov_emb_buf;
    torch::Tensor                       markov_bias_buf;
    torch::Tensor                       step_logits_buf;
    torch::Tensor                       sampled_draft_buf;
    torch::Tensor                       sample_valid_buf;
    torch::Tensor                       token_ids_out;
    torch::Tensor                       all_probs_out;
    torch::Event                        replay_done_event = cuda_graph::makeGraphEvent();
    int64_t                             last_replay_stream_id = 0;
    bool                                has_replayed          = false;
    int64_t                             padded_bsz            = 0;
    int64_t                             gamma                 = 0;
    int64_t                             draft_v               = 0;
    bool                                has_markov            = false;
    bool                                has_d2t               = false;
    std::list<DSparkGraphKey>::iterator lru_it;
};

struct DSparkGraphState {
    std::mutex                                                                             mutex;
    std::unordered_map<DSparkGraphKey, std::unique_ptr<DSparkGraphEntry>, DSparkGraphKeyHash> cache;
    std::list<DSparkGraphKey>                                                              lru_order;
    std::unordered_map<c10::DeviceIndex, cuda_graph::GraphStream>                          capture_streams;
    bool                                                                                   capture_dirty = false;
    size_t                                                                                 capture_count = 0;
    size_t                                                                                 replay_count  = 0;
};

namespace {

void waitEntryReplayCompletion(DSparkGraphEntry& entry, c10::DeviceIndex device_index) {
    if (!entry.has_replayed) {
        return;
    }
    at::cuda::CUDAGuard device_guard(device_index);
    entry.replay_done_event.synchronize();
}

void stageDSparkGraphInputs(DSparkGraphEntry&    entry,
                            const torch::Tensor& base_logits,
                            const torch::Tensor& anchors_i32,
                            const torch::Tensor& temperature,
                            int64_t              live_bsz) {
    entry.base_logits_slot.narrow(0, 0, live_bsz * entry.gamma)
        .copy_(base_logits.narrow(1, 0, entry.draft_v), /*non_blocking=*/true);
    entry.anchors_slot.narrow(0, 0, live_bsz).copy_(anchors_i32, /*non_blocking=*/true);
    entry.temperature_slot.narrow(0, 0, live_bsz).copy_(temperature, /*non_blocking=*/true);
    const auto [seed, base_offset] = execGetSeedAndOffset(static_cast<int>(entry.gamma * entry.padded_bsz * 4));
    launchPrepareDSparkDraftGraphInputs(entry.anchors_slot,
                                        entry.temperature_slot,
                                        entry.seed_arr,
                                        entry.offset_arr,
                                        live_bsz,
                                        entry.padded_bsz,
                                        entry.gamma,
                                        seed,
                                        base_offset);
}

void runDSparkDraftGraphBody(DSparkGraphEntry&    entry,
                             const torch::Tensor& markov_w1,
                             const torch::Tensor& markov_w2_t,
                             const torch::Tensor& d2t_map) {
    for (int64_t step = 0; step < entry.gamma; ++step) {
        if (entry.has_markov) {
            auto previous_tokens = step == 0 ? entry.anchors_slot : entry.token_ids_out.select(1, step - 1);
            launchGatherMarkovEmbedding(markov_w1, previous_tokens, entry.markov_emb_buf);
            torch::mm_out(entry.markov_bias_buf, entry.markov_emb_buf, markov_w2_t);
        }

        torch::Tensor step_target =
            entry.padded_bsz == 1 ? entry.all_probs_out.select(1, step) : entry.step_logits_buf;
        launchDSparkAddMarkovDivTemp(entry.base_logits_slot,
                                     entry.markov_bias_buf,
                                     entry.temperature_slot,
                                     step_target,
                                     step,
                                     entry.gamma,
                                     entry.draft_v);

        // Draft q applies request temperature only. Materialize that exact
        // dense distribution once, sample from it with FlashInfer, and pass
        // the same q to rejection sampling. Request top-k/top-p stay target-side.
        auto sampling_probabilities = execSamplerSoftmaxInplace(step_target);
        execSampleFromProbsInto({
            sampling_probabilities,
            entry.sampled_draft_buf,
            entry.sample_valid_buf,
            entry.seed_arr.select(0, step),
            entry.offset_arr.select(0, step),
        });
        auto step_token_out = entry.token_ids_out.select(1, step);
        if (entry.has_d2t) {
            launchMapDraftToTargetInt32(entry.sampled_draft_buf, d2t_map, step_token_out);
        } else {
            step_token_out.copy_(entry.sampled_draft_buf, /*non_blocking=*/true);
        }
        if (entry.padded_bsz != 1 || sampling_probabilities.data_ptr() != step_target.data_ptr()) {
            entry.all_probs_out.select(1, step).copy_(sampling_probabilities, /*non_blocking=*/true);
        }
    }
}

SamplerOutput cloneDSparkGraphOutputs(const DSparkGraphEntry& entry, int64_t live_bsz) {
    SamplerOutput output;
    output.token_ids                = entry.token_ids_out.narrow(0, 0, live_bsz).clone();
    output.all_probs                = entry.all_probs_out.narrow(0, 0, live_bsz).clone();
    output.token_ids_are_point_mass = false;
    return output;
}

std::unique_ptr<DSparkGraphEntry> allocateDSparkGraphEntry(const DSparkGraphKey&     key,
                                                           const DSparkBoundWeights& bound_weights,
                                                           const torch::Device&      device) {
    auto       entry    = std::make_unique<DSparkGraphEntry>();
    const auto cuda_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    const auto cuda_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto cuda_b8  = torch::TensorOptions().dtype(torch::kBool).device(device);
    const auto base_opt = torch::TensorOptions().dtype(key.base_logits_dtype).device(device);

    entry->bound_weights    = bound_weights;
    entry->padded_bsz       = key.padded_bsz;
    entry->gamma            = key.gamma;
    entry->draft_v          = key.draft_vocab_size;
    entry->has_markov       = key.has_markov;
    entry->has_d2t          = key.has_d2t;
    entry->base_logits_slot = allocateAlignedCudaTensor({key.padded_bsz * key.gamma, key.draft_vocab_size}, base_opt);
    entry->anchors_slot     = allocateAlignedCudaTensor({key.padded_bsz}, cuda_i32);
    entry->temperature_slot = allocateAlignedCudaTensor({key.padded_bsz}, cuda_f32);
    entry->seed_arr         = allocateAlignedCudaTensor({key.gamma, key.padded_bsz}, cuda_i64);
    entry->offset_arr       = allocateAlignedCudaTensor({key.gamma, key.padded_bsz}, cuda_i64);
    if (key.has_markov) {
        const auto markov_opt  = torch::TensorOptions().dtype(key.markov_dtype).device(device);
        entry->markov_emb_buf  = allocateAlignedCudaTensor({key.padded_bsz, key.markov_rank}, markov_opt);
        entry->markov_bias_buf = allocateAlignedCudaTensor({key.padded_bsz, key.draft_vocab_size}, markov_opt);
    }
    if (key.padded_bsz > 1) {
        entry->step_logits_buf = allocateAlignedCudaTensor({key.padded_bsz, key.draft_vocab_size}, cuda_f32);
    }
    entry->sampled_draft_buf = allocateAlignedCudaTensor({key.padded_bsz}, cuda_i32);
    entry->sample_valid_buf  = allocateAlignedCudaTensor({key.padded_bsz}, cuda_b8);
    entry->token_ids_out     = allocateAlignedCudaTensor({key.padded_bsz, key.gamma}, cuda_i32);
    entry->all_probs_out     = allocateAlignedCudaTensor({key.padded_bsz, key.gamma, key.draft_vocab_size}, cuda_f32);
    return entry;
}

}  // namespace
#else
struct DSparkGraphState {};
#endif

SpeculativeSampler::SpeculativeSampler(torch::Tensor d2t_map, size_t propose_step):
    d2t_map_(std::move(d2t_map)), propose_step_(propose_step) {}

SpeculativeSampler::~SpeculativeSampler() {
#if USING_CUDA
    if (dspark_graph_state_) {
        std::lock_guard<std::mutex> lock(dspark_graph_state_->mutex);
        if (dspark_graph_state_->capture_dirty) {
            RTP_LLM_LOG_ERROR("SpeculativeSampler destroyed with dirty DSpARK CUDA graph capture state; "
                              "skipping event drain and releasing dirty graph cache");
            (void)dspark_graph_state_.release();
            return;
        }
        for (auto& [key, entry] : dspark_graph_state_->cache) {
            if (entry && entry->has_replayed) {
                try {
                    waitEntryReplayCompletion(*entry, key.device_index);
                } catch (const std::exception& e) {
                    RTP_LLM_LOG_WARNING("Failed to synchronize DSpARK replay_done_event during teardown: %s",
                                        e.what());
                } catch (...) {
                    RTP_LLM_LOG_WARNING("Failed to synchronize DSpARK replay_done_event during teardown");
                }
            }
        }
    }
#endif
}

void SpeculativeSampler::setDSparkCudaGraphEnabled(bool enabled) {
    dspark_cuda_graph_enabled_ = enabled;
}

bool SpeculativeSampler::dsparkCudaGraphEnabled() const {
    return dspark_cuda_graph_enabled_;
}

size_t SpeculativeSampler::dsparkGraphCacheSize() const {
#if USING_CUDA
    if (!dspark_graph_state_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(dspark_graph_state_->mutex);
    return dspark_graph_state_->cache.size();
#else
    return 0;
#endif
}

size_t SpeculativeSampler::dsparkGraphCaptureCount() const {
#if USING_CUDA
    if (!dspark_graph_state_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(dspark_graph_state_->mutex);
    return dspark_graph_state_->capture_count;
#else
    return 0;
#endif
}

size_t SpeculativeSampler::dsparkGraphReplayCount() const {
#if USING_CUDA
    if (!dspark_graph_state_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(dspark_graph_state_->mutex);
    return dspark_graph_state_->replay_count;
#else
    return 0;
#endif
}

bool SpeculativeSampler::dsparkCaptureDirty() const {
#if USING_CUDA
    if (!dspark_graph_state_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(dspark_graph_state_->mutex);
    return dspark_graph_state_->capture_dirty;
#else
    return false;
#endif
}

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;

    if (top_k == 1) {
        output.token_ids = torch::argmax(logits, -1, true);
        output.all_probs = torch::zeros_like(logits).scatter_(-1, output.token_ids, 1.0);
    } else {
        auto draft_probs = torch::softmax(logits, -1);
        output.token_ids = std::get<1>(torch::topk(draft_probs, top_k, -1));
        output.all_probs = std::move(draft_probs);
    }

    int batch_size = output.token_ids.size(0);
    execMappingDraft2Target({output.token_ids, d2t_map_, batch_size, 0, 1});
    output.token_ids = output.token_ids.to(torch::kInt32);

    return output;
}

SamplerOutput SpeculativeSampler::sampleDSparkDraft(const torch::Tensor& base_logits,
                                                    const torch::Tensor& anchors,
                                                    const torch::Tensor& temperature,
                                                    const torch::Tensor& markov_w1,
                                                    const torch::Tensor& markov_w2,
                                                    size_t               draft_vocab_size) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dspark_draft");
    RTP_LLM_CHECK_WITH_INFO(temperature.defined() && temperature.is_cuda() && temperature.is_contiguous()
                                && temperature.scalar_type() == torch::kFloat32 && temperature.dim() == 1,
                            "DSpARK draft temperatures must be contiguous CUDA FP32 [B]");
    const auto batch_size = temperature.numel();
    RTP_LLM_CHECK_WITH_INFO(base_logits.defined() && base_logits.is_cuda()
                                && base_logits.device() == temperature.device() && base_logits.stride(1) == 1
                                && (base_logits.scalar_type() == torch::kFloat32
                                    || base_logits.scalar_type() == torch::kBFloat16)
                                && base_logits.dim() == 2
                                && base_logits.size(0) == batch_size * static_cast<int64_t>(propose_step_)
                                && base_logits.size(1) >= static_cast<int64_t>(draft_vocab_size),
                            "DSpARK C++ lm_head must emit row-contiguous CUDA FP32/BF16 [B*gamma,vocab_padded] logits "
                            "with vocab_padded >= draft vocab size");
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.is_cuda() && anchors.device() == base_logits.device()
                                && anchors.numel() == batch_size,
                            "DSpARK anchors must be a CUDA tensor with one token per request");

    const auto gamma   = static_cast<int64_t>(propose_step_);
    const auto draft_v = static_cast<int64_t>(draft_vocab_size);
    RTP_LLM_CHECK_WITH_INFO(markov_w1.defined() == markov_w2.defined(),
                            "DSpARK markov_w1 and markov_w2 must both be defined or both be undefined");
    const bool has_markov = markov_w1.defined() && markov_w2.defined() && markov_w1.numel() > 0
                            && markov_w2.numel() > 0 && markov_w1.size(1) > 0;
    if (has_markov) {
        RTP_LLM_CHECK_WITH_INFO(markov_w1.device() == base_logits.device()
                                    && markov_w2.device() == base_logits.device(),
                                "DSpARK Markov weights must reside on the same device as base_logits");
        RTP_LLM_CHECK_WITH_INFO(markov_w1.dim() == 2 && markov_w2.dim() == 2 && markov_w1.size(1) == markov_w2.size(1)
                                    && markov_w2.size(0) >= draft_v && markov_w1.stride(1) == 1
                                    && markov_w1.scalar_type() == markov_w2.scalar_type()
                                    && (markov_w1.scalar_type() == torch::kBFloat16
                                        || markov_w1.scalar_type() == torch::kFloat32),
                                "DSpARK Markov weights must be 2D [target_vocab,rank] and [draft_vocab,rank] "
                                "tensors with matching rank and BF16/FP32 dtype");
    }

    const auto cuda_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device());
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(base_logits.device());
    if (batch_size == 0 || gamma == 0 || draft_v == 0) {
        SamplerOutput output;
        output.token_ids                = torch::empty({batch_size, gamma}, cuda_i32);
        output.all_probs                = torch::empty({batch_size, gamma, draft_v}, cuda_f32);
        output.token_ids_are_point_mass = false;
        return output;
    }

#if USING_CUDA
    at::cuda::CUDAGuard device_guard(base_logits.device());
    auto                anchors_i32 = anchors.scalar_type() == torch::kInt32 ?
                                          anchors.reshape({batch_size}) :
                                          anchors.reshape({batch_size}).to(torch::kInt32);
    torch::Tensor       markov_w2_slice;
    torch::Tensor       markov_w2_t;
    if (has_markov) {
        markov_w2_slice = markov_w2.narrow(0, 0, draft_v);
        markov_w2_t     = markov_w2_slice.transpose(0, 1);
    }

    const bool d2t_defined_non_empty = d2t_map_.defined() && d2t_map_.numel() > 0;
    const bool d2t_graph_compatible  = !d2t_defined_non_empty
                                      || (d2t_map_.is_cuda() && d2t_map_.device() == base_logits.device()
                                          && d2t_map_.dim() == 1 && d2t_map_.is_contiguous()
                                          && (d2t_map_.scalar_type() == torch::kInt64
                                              || d2t_map_.scalar_type() == torch::kInt32));
    const int64_t padded_bsz = padDSparkBatchSize(batch_size);
    const bool    within_rng_limit =
        gamma <= static_cast<int64_t>(std::numeric_limits<int>::max()) / (4 * std::max<int64_t>(padded_bsz, 1));

    if (dspark_cuda_graph_enabled_ && d2t_graph_compatible && within_rng_limit) {
        if (!dspark_graph_state_) {
            dspark_graph_state_ = std::make_unique<DSparkGraphState>();
        }
        std::lock_guard<std::mutex> lock(dspark_graph_state_->mutex);
        RTP_LLM_CHECK_WITH_INFO(!dspark_graph_state_->capture_dirty,
                                "DSpARK CUDA graph capture state is dirty after a prior capture failure");

        const bool         has_d2t      = d2t_defined_non_empty;
        const auto         device_index = base_logits.device().index();
        DSparkGraphKey     key;
        key.padded_bsz        = padded_bsz;
        key.gamma             = gamma;
        key.draft_vocab_size  = draft_v;
        key.base_logits_dtype = base_logits.scalar_type();
        key.has_markov        = has_markov;
        key.markov_dtype      = has_markov ? markov_w1.scalar_type() : c10::ScalarType::Undefined;
        key.markov_rank       = has_markov ? markov_w1.size(1) : 0;
        key.has_d2t           = has_d2t;
        key.d2t_dtype         = has_d2t ? d2t_map_.scalar_type() : c10::ScalarType::Undefined;
        key.device_index      = device_index;
        key.sample_mode       = DSparkSampleMode::DENSE_PROBS;

        DSparkBoundWeights bound_weights;
        if (has_markov) {
            bound_weights.w1_ptr     = markov_w1.data_ptr();
            bound_weights.w1_rows    = markov_w1.size(0);
            bound_weights.w1_stride0 = markov_w1.stride(0);
            bound_weights.w2_ptr     = markov_w2_slice.data_ptr();
            bound_weights.w2_stride0 = markov_w2_slice.stride(0);
            bound_weights.w2_stride1 = markov_w2_slice.stride(1);
        }
        if (has_d2t) {
            bound_weights.d2t_ptr   = d2t_map_.data_ptr();
            bound_weights.d2t_numel = d2t_map_.numel();
        }

        const cuda_graph::GraphStream caller_stream = cuda_graph::graphGetCurrentStream();
        auto                          it            = dspark_graph_state_->cache.find(key);
        if (it != dspark_graph_state_->cache.end()) {
            DSparkGraphEntry& entry = *it->second;
            if (entry.bound_weights == bound_weights) {
                dspark_graph_state_->lru_order.splice(
                    dspark_graph_state_->lru_order.begin(), dspark_graph_state_->lru_order, entry.lru_it);
                if (entry.has_replayed && entry.last_replay_stream_id != static_cast<int64_t>(caller_stream.id())) {
                    entry.replay_done_event.block(caller_stream);
                }
                stageDSparkGraphInputs(entry, base_logits, anchors_i32, temperature, batch_size);
                entry.graph.replay();
                auto output = cloneDSparkGraphOutputs(entry, batch_size);
                entry.replay_done_event.record(caller_stream);
                entry.has_replayed          = true;
                entry.last_replay_stream_id = static_cast<int64_t>(caller_stream.id());
                ++dspark_graph_state_->replay_count;
                return output;
            }
            waitEntryReplayCompletion(entry, key.device_index);
            dspark_graph_state_->lru_order.erase(entry.lru_it);
            dspark_graph_state_->cache.erase(it);
        }

        if (dspark_graph_state_->cache.size() >= kMaxDSparkCachedGraphs) {
            const DSparkGraphKey evict_key = dspark_graph_state_->lru_order.back();
            auto                 evict_it  = dspark_graph_state_->cache.find(evict_key);
            if (evict_it != dspark_graph_state_->cache.end()) {
                waitEntryReplayCompletion(*evict_it->second, evict_key.device_index);
                dspark_graph_state_->cache.erase(evict_it);
            }
            dspark_graph_state_->lru_order.pop_back();
        }

        auto stream_it = dspark_graph_state_->capture_streams.find(device_index);
        if (stream_it == dspark_graph_state_->capture_streams.end()) {
            stream_it =
                dspark_graph_state_->capture_streams.emplace(device_index, cuda_graph::graphGetStreamFromPool(true))
                    .first;
        }
        const cuda_graph::GraphStream& capture_stream = stream_it->second;
        // Allocate a dedicated mempool per cached entry so LRU or weight-rebind eviction
        // never leaves a shared pool handle with use_count == 0 for future captures.
        const cuda_graph::GraphPoolHandle graph_pool = cuda_graph::graphPoolHandle();

        auto         entry              = allocateDSparkGraphEntry(key, bound_weights, base_logits.device());
        stageDSparkGraphInputs(*entry, base_logits, anchors_i32, temperature, batch_size);
        torch::Event caller_ready_event = cuda_graph::makeGraphEvent();
        caller_ready_event.record(caller_stream);

        {
            cuda_graph::GraphStreamGuard capture_stream_guard(capture_stream);
            caller_ready_event.block(capture_stream);

            // Warm up allocator/cuBLAS/kernels twice on capture_stream before capture_begin.
            runDSparkDraftGraphBody(*entry, markov_w1, markov_w2_t, d2t_map_);
            runDSparkDraftGraphBody(*entry, markov_w1, markov_w2_t, d2t_map_);
            capture_stream.synchronize();

            dspark_graph_state_->capture_dirty = true;
            DSparkCaptureModeGuard capture_mode_guard;
            bool                   capturing = false;
            try {
                cuda_graph::graphCaptureBegin(entry->graph, graph_pool);
                capturing = true;
                runDSparkDraftGraphBody(*entry, markov_w1, markov_w2_t, d2t_map_);
                entry->graph.capture_end();
                capturing = false;
            } catch (...) {
                if (capturing) {
                    cudaGraph_t aborted_graph = nullptr;
                    (void)cudaStreamEndCapture(capture_stream.stream(), &aborted_graph);
                    if (aborted_graph != nullptr) {
                        (void)cudaGraphDestroy(aborted_graph);
                    }
                }
                throw;
            }

            // Verification replay on capture_stream executes the exact staged inputs once.
            entry->graph.replay();
            capture_stream.synchronize();
            dspark_graph_state_->capture_dirty = false;
            entry->replay_done_event.record(capture_stream);
        }

        entry->replay_done_event.block(caller_stream);
        auto output = cloneDSparkGraphOutputs(*entry, batch_size);
        entry->replay_done_event.record(caller_stream);
        entry->has_replayed          = true;
        entry->last_replay_stream_id = static_cast<int64_t>(caller_stream.id());

        dspark_graph_state_->lru_order.push_front(key);
        entry->lru_it = dspark_graph_state_->lru_order.begin();
        dspark_graph_state_->cache.emplace(key, std::move(entry));
        ++dspark_graph_state_->capture_count;
        return output;
    }

    auto all_probabilities = torch::empty({batch_size, gamma, draft_v}, cuda_f32);
    auto token_ids         = torch::empty({batch_size, gamma}, cuda_i32);
    torch::Tensor markov_emb_buf;
    torch::Tensor markov_bias_buf;
    if (has_markov) {
        markov_emb_buf  = torch::empty({batch_size, markov_w1.size(1)}, markov_w1.options());
        markov_bias_buf = torch::empty({batch_size, draft_v}, markov_w1.options());
    }
    auto step_logits_buf = batch_size == 1 ? torch::Tensor() : torch::empty({batch_size, draft_v}, cuda_f32);

    for (int64_t step = 0; step < gamma; ++step) {
        if (has_markov) {
            auto previous_tokens = step == 0 ? anchors_i32 : token_ids.select(1, step - 1);
            launchGatherMarkovEmbedding(markov_w1, previous_tokens, markov_emb_buf);
            torch::mm_out(markov_bias_buf, markov_emb_buf, markov_w2_t);
        }

        torch::Tensor step_target = batch_size == 1 ? all_probabilities.select(1, step) : step_logits_buf;
        launchDSparkAddMarkovDivTemp(base_logits, markov_bias_buf, temperature, step_target, step, gamma, draft_v);

        // Draft q applies request temperature only. Materialize that exact
        // dense distribution once, sample from it with FlashInfer, and pass
        // the same q to rejection sampling. Request top-k/top-p stay target-side.
        auto sampling_probabilities = execSamplerSoftmaxInplace(step_target);
        auto sampled_draft_tokens   = execSampleFromProbs(sampling_probabilities);
        if (sampled_draft_tokens.scalar_type() != torch::kInt32) {
            sampled_draft_tokens = sampled_draft_tokens.to(torch::kInt32);
        }
        auto step_token_out = token_ids.select(1, step);
        if (d2t_defined_non_empty) {
            if (d2t_map_.is_cuda()) {
                launchMapDraftToTargetInt32(sampled_draft_tokens, d2t_map_, step_token_out);
            } else {
                step_token_out.copy_(
                    d2t_map_.index_select(0, sampled_draft_tokens.cpu().to(torch::kLong))
                        .to(torch::kInt32)
                        .to(base_logits.device(), /*non_blocking=*/true));
            }
        } else {
            step_token_out.copy_(sampled_draft_tokens, /*non_blocking=*/true);
        }
        if (batch_size != 1 || sampling_probabilities.data_ptr() != step_target.data_ptr()) {
            all_probabilities.select(1, step).copy_(sampling_probabilities, /*non_blocking=*/true);
        }
    }

    SamplerOutput output;
    output.token_ids                = std::move(token_ids);
    output.all_probs                = std::move(all_probabilities);
    output.token_ids_are_point_mass = false;
    return output;
#else
    auto all_probabilities = torch::empty({batch_size, gamma, draft_v}, cuda_f32);
    auto previous_tokens   = anchors.reshape({batch_size}).to(torch::kLong);
    std::vector<torch::Tensor> token_columns;
    token_columns.reserve(propose_step_);
    auto proposal_logits = base_logits.narrow(1, 0, draft_vocab_size).view({batch_size, gamma, draft_v});
    if (proposal_logits.scalar_type() != torch::kFloat32) {
        proposal_logits = proposal_logits.to(torch::kFloat32);
    }
    auto          temperature_column = temperature.unsqueeze(1);
    torch::Tensor markov_w2_t;
    if (has_markov) {
        markov_w2_t = markov_w2.narrow(0, 0, draft_v).transpose(0, 1);
    }

    for (int64_t step = 0; step < gamma; ++step) {
        auto logits = proposal_logits.select(1, step).clone();
        if (has_markov) {
            auto markov_embedding = markov_w1.index_select(0, previous_tokens);
            auto markov_bias      = torch::mm(markov_embedding, markov_w2_t).to(torch::kFloat32);
            logits.add_(markov_bias);
        }

        logits.div_(temperature_column);
        auto sampling_probabilities = execSamplerSoftmaxInplace(logits);
        auto sampled_draft_tokens   = execSampleFromProbs(sampling_probabilities).to(torch::kInt32);
        auto sampled_target_tokens  = sampled_draft_tokens;
        if (d2t_map_.defined() && d2t_map_.numel() > 0) {
            sampled_target_tokens = d2t_map_.index_select(0, sampled_draft_tokens.to(torch::kLong)).to(torch::kInt32);
        }
        all_probabilities.select(1, step).copy_(sampling_probabilities);
        token_columns.push_back(sampled_target_tokens);
        previous_tokens = sampled_target_tokens.to(torch::kLong);
    }

    SamplerOutput output;
    output.token_ids                = torch::stack(token_columns, 1).contiguous();
    output.all_probs                = std::move(all_probabilities);
    output.token_ids_are_point_mass = false;
    return output;
#endif
}

torch::Tensor dflashDraftProbabilities(const torch::Tensor& logits, const GenerateConfig& config) {
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) > 0, "DFlash logits must be [tokens,vocab]");
    auto scores = logits.to(torch::kFloat32);
    // Greedy requests never read the sampling parameters, so settle them first:
    // the repo's own smoke load sends {temperature: 0, top_p: 0} and admission
    // accepts it (see case_runner.py).
    if (!config.stochastic() || config.temperature == 0.0f) {
        return torch::zeros_like(scores).scatter_(-1, scores.argmax(-1, true), 1.0);
    }
    // Admission lets unusable sampling parameters through: the OpenAI endpoint
    // passes top_p = 0 / > 1 through unchanged and is_positive_number accepts
    // inf temperatures. This function runs inside the engine loop thread, so
    // instead of failing the whole engine, normalize admission-tolerant values
    // the same way MtpExecutor::decodeStep sanitizes DSpARK temperatures: an
    // invalid temperature or top_p falls back to 1.0 (no temperature skew, no
    // nucleus cut) and a non-positive top_k means "no top-k filtering".
    constexpr float kMinDraftTemperature = 1.0e-6f;
    auto temperature =
        std::isfinite(config.temperature) && config.temperature > 0.0f ? config.temperature : 1.0f;
    // Tiny positive temperatures (e.g. denormals) survive the normalization
    // above but overflow scores/temperature to inf, which turns softmax into
    // NaN; multinomial on NaN then fails in the engine loop thread or samples
    // a garbage token. Apply the same floor as the DSpARK sanitize.
    temperature = std::max(temperature, kMinDraftTemperature);
    const auto top_k      = config.top_k > 0 ? config.top_k : 0;
    const auto top_p      = std::isfinite(config.top_p) && config.top_p > 0.0f && config.top_p <= 1.0f ? config.top_p : 1.0f;
    scores                = scores / temperature;
    torch::Tensor indices;
    if (top_k > 0 && top_k < scores.size(1)) {
        auto top = scores.topk(top_k, -1);
        scores   = std::get<0>(top);
        indices  = std::get<1>(top);
    }
    auto probabilities = torch::softmax(scores, -1);
    if (top_p < 1.0f) {
        auto sorted               = probabilities.sort(-1, /*descending=*/true);
        auto sorted_probabilities = std::get<0>(sorted);
        // Keep the token crossing the nucleus threshold, as in the reference.
        auto keep            = sorted_probabilities.cumsum(-1) - sorted_probabilities < config.top_p;
        sorted_probabilities = sorted_probabilities * keep;
        probabilities        = torch::zeros_like(probabilities).scatter_(-1, std::get<1>(sorted), sorted_probabilities);
        probabilities        = probabilities / probabilities.sum(-1, true);
    }
    if (indices.defined()) {
        probabilities = torch::zeros_like(logits, probabilities.options()).scatter_(-1, indices, probabilities);
    }
    return probabilities;
}

SamplerOutput SpeculativeSampler::sampleDFlashDraft(const torch::Tensor&                base_logits,
                                                    const std::list<GenerateStreamPtr>& streams,
                                                    size_t                              draft_vocab_size) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dflash_draft");
    const auto batch_size = static_cast<int64_t>(streams.size());
    TORCH_CHECK(!d2t_map_.defined(), "DFlash samples in the full target vocabulary");
    TORCH_CHECK(batch_size > 0 && propose_step_ > 0 && base_logits.dim() == 2
                    && base_logits.size(0) == batch_size * static_cast<int64_t>(propose_step_)
                    && base_logits.size(1) >= static_cast<int64_t>(draft_vocab_size),
                "DFlash logits must contain only mask rows: [B*gamma,vocab_padded]");
    auto logits =
        base_logits.narrow(1, 0, draft_vocab_size)
            .reshape({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)});
    std::vector<torch::Tensor> probability_rows;
    std::vector<torch::Tensor> token_rows;
    probability_rows.reserve(batch_size);
    token_rows.reserve(batch_size);
    bool    all_greedy = true;
    int64_t row        = 0;
    for (const auto& stream : streams) {
        TORCH_CHECK(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                    "DFlash does not support tiled or beam sampling");
        const auto config = stream->generateConfig();
        // Greedy requests short-circuit inside dflashDraftProbabilities and
        // never read the sampling parameters, so only stochastic requests need
        // the admission-tolerant sanitize; mirror MtpExecutor::decodeStep's
        // DSpARK handling and log the substitution so a misconfigured client
        // stays visible to operators instead of failing silently.
        if (config->stochastic() && config->temperature != 0.0f) {
            if (!std::isfinite(config->temperature) || config->temperature < 0.0f) {
                RTP_LLM_LOG_WARNING("DFlash received invalid sampling temperature=%g for stream=%ld; using 1.0",
                                    config->temperature,
                                    stream->streamId());
                config->temperature = 1.0f;
            }
            if (config->top_k < 0) {
                RTP_LLM_LOG_WARNING("DFlash received invalid top_k=%d for stream=%ld; using 0",
                                    config->top_k,
                                    stream->streamId());
                config->top_k = 0;
            }
            if (!std::isfinite(config->top_p) || config->top_p <= 0.0f || config->top_p > 1.0f) {
                RTP_LLM_LOG_WARNING("DFlash received invalid top_p=%g for stream=%ld; using 1.0",
                                    config->top_p,
                                    stream->streamId());
                config->top_p = 1.0f;
            }
        }
        auto       probabilities = dflashDraftProbabilities(logits[row++], *config);
        const bool greedy       = !config->stochastic() || config->temperature == 0.0f;
        all_greedy                = all_greedy && greedy;
        torch::Tensor tokens;
        if (greedy) {
            tokens = probabilities.argmax(-1);
        } else {
            auto generator = stream->getGenerator();
            tokens         = torch::multinomial(probabilities,
                                        1,
                                        false,
                                        generator.defined() ? std::optional<at::Generator>(generator) : std::nullopt)
                         .squeeze(-1);
        }
        token_rows.push_back(tokens.to(torch::kInt32));
        probability_rows.push_back(std::move(probabilities));
    }
    SamplerOutput output;
    output.token_ids                = torch::stack(token_rows).contiguous();
    output.token_ids_are_point_mass = all_greedy;
    if (!all_greedy) {
        // Mixed batches retain explicit one-hot q for their greedy rows.
        output.all_probs = torch::stack(probability_rows).contiguous();
    }
    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const SpeculativeSamplingParams& params,
                                                     SamplerOutput&                   draft_sampler_output,
                                                     SamplerOutput&                   target_sampler_output) {
    // TensorHolder release point (SpeculativeSampler): advances host tensors
    // staged for rejection sampling H2D in the previous forward.
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, params, draft_sampler_output, target_sampler_output);

    return sample_output;
}

SpeculativeSamplerOutput SpeculativeSampler::forwardWithSpecCap(const SpeculativeSamplingParams& params,
                                                                SamplerOutput&                   draft_sampler_output,
                                                                SamplerOutput&                   target_sampler_output,
                                                                const PrepareSpecCapFn&          prepare_spec_cap,
                                                                bool&                            fused_cap_applied) {
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    fused_cap_applied = false;
    batchSample(
        sample_output, params, draft_sampler_output, target_sampler_output, &prepare_spec_cap, &fused_cap_applied);
    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&        sample_output,
                                     const SpeculativeSamplingParams& params,
                                     SamplerOutput&                   draft_sampler_output,
                                     SamplerOutput&                   target_sampler_output,
                                     const PrepareSpecCapFn*          prepare_spec_cap,
                                     bool*                            fused_cap_applied) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample");
    torch::Device target_device = getTorchCudaDevice();

    const int batch_size = params.do_sample.size(0);

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto       draft_token_probs      = draft_sampler_output.all_probs;
    auto       target_token_probs     = target_sampler_output.all_probs;
    const bool draft_probs_point_mass = draft_sampler_output.token_ids_are_point_mass;

    buffer_holder_.hold_host(draft_token_ids);
    auto draft_token_ids_d_t = draft_token_ids.to(target_device, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        buffer_holder_.hold_host(target_token_ids_d_t);
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    const auto& do_sample = params.do_sample;
    buffer_holder_.hold_host(do_sample);
    auto do_sample_d = do_sample.to(target_device, true);

    auto          rand_options      = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

    // Override per-stream uniform samples with seeded generator when random_seed is set,
    // ensuring deterministic acceptance for reproducible iter_count.
    {
        int idx = 0;
        for (const auto& gen : params.generators) {
            if (gen.defined()) {
                uniform_samples_d[idx] = torch::rand({(long)propose_step_ + 1}, gen, std::nullopt, rand_options);
            }
            idx++;
        }
    }

    auto          draft_token_probs_d_t  = draft_token_probs;
    auto          target_token_probs_d_t = target_token_probs;
    torch::Tensor output_token_ids_d =
        torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                     torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));
    torch::Tensor output_accepted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

    RTP_LLM_CHECK_WITH_INFO(draft_probs_point_mass
                                || (draft_token_probs_d_t.defined() && draft_token_probs_d_t.dim() == 3),
                            "draft probabilities must be [B, steps, vocab] unless token ids define a point mass");
    if (!draft_probs_point_mass && draft_token_probs_d_t.size(2) != target_token_probs_d_t.size(2)) {
        const int64_t target_vocab_size = target_token_probs_d_t.size(2);
        const int64_t num_spec          = draft_token_probs_d_t.size(1);

        // Reuse pre-allocated padding buffer to avoid per-forward GPU allocation.
        // Grow-only along batch / num_spec dims; vocab dim must match exactly.
        const bool need_realloc = !draft_probs_padding_buffer_.defined()
                                  || draft_probs_padding_buffer_.size(0) < (int64_t)batch_size
                                  || draft_probs_padding_buffer_.size(1) < num_spec
                                  || draft_probs_padding_buffer_.size(2) != target_vocab_size
                                  || draft_probs_padding_buffer_.dtype() != draft_token_probs_d_t.dtype()
                                  || draft_probs_padding_buffer_.device() != draft_token_probs_d_t.device();
        if (need_realloc) {
            const int64_t cap_b =
                std::max((int64_t)batch_size,
                         draft_probs_padding_buffer_.defined() ? draft_probs_padding_buffer_.size(0) : (int64_t)0);
            const int64_t cap_s = std::max(
                num_spec, draft_probs_padding_buffer_.defined() ? draft_probs_padding_buffer_.size(1) : (int64_t)0);
            draft_probs_padding_buffer_ =
                torch::zeros({cap_b, cap_s, target_vocab_size}, draft_token_probs_d_t.options());
        }

        auto draft_probs_padding = draft_probs_padding_buffer_.narrow(0, 0, (int64_t)batch_size).narrow(1, 0, num_spec);
        draft_probs_padding.zero_();
        if (d2t_map_.defined()) {
            draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map_},
                                           draft_token_probs_d_t);
        } else {
            // Full-vocabulary DFlash omits synthetic lm-head padding from q.
            // The target sampler retains padded columns, each with zero mass.
            RTP_LLM_CHECK_WITH_INFO(draft_token_probs_d_t.size(2) <= target_vocab_size,
                                    "draft vocabulary exceeds target vocabulary without a d2t map");
            draft_probs_padding.narrow(2, 0, draft_token_probs_d_t.size(2)).copy_(draft_token_probs_d_t);
        }
        draft_token_probs_d_t = draft_probs_padding;
    }

    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.execRejectionSampling");
        execRejectionSampling({
            draft_token_probs_d_t,
            draft_token_ids_d_t,
            uniform_samples_d,
            target_token_probs_d_t,
            target_token_ids_d_t,
            output_token_ids_d,
            output_accepted_token_num_d,
            do_sample_d,
            draft_probs_point_mass,
        });
    }

    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling");

    bool          has_force    = false;
    torch::Tensor force_mask_d;
    if (params.force_accept.defined() && batch_size > 0) {
        auto        force_src = params.force_accept.contiguous();
        const auto* force_ptr = force_src.data_ptr<bool>();
        for (int idx = 0; idx < batch_size; ++idx) {
            if (force_ptr[idx]) {
                has_force = true;
                break;
            }
        }
        if (has_force) {
            auto force_mask_h = torch::empty(
                {(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
            std::memcpy(force_mask_h.data_ptr<bool>(), force_ptr, static_cast<size_t>(batch_size) * sizeof(bool));
            buffer_holder_.hold_host(force_mask_h);
            force_mask_d = force_mask_h.to(target_device, /*non_blocking=*/true);
        }
    }

    torch::Tensor cap_gpu;
    if (prepare_spec_cap != nullptr && *prepare_spec_cap) {
        cap_gpu = (*prepare_spec_cap)(target_device);
    }

#if USING_CUDA
    if (output_token_ids_d.is_cuda() && output_accepted_token_num_d.is_cuda() && output_token_ids_d.is_contiguous()
        && output_accepted_token_num_d.is_contiguous()
        && (!cap_gpu.defined()
            || (cap_gpu.is_cuda() && cap_gpu.is_contiguous()
                && (cap_gpu.scalar_type() == torch::kInt32 || cap_gpu.scalar_type() == torch::kInt64)))) {
        auto draft_for_force = has_force ? draft_token_ids_d_t.contiguous() : torch::Tensor();
        auto target_for_post = (has_force || cap_gpu.defined()) ?
                                   (target_token_ids_d_t.scalar_type() == torch::kInt32 ?
                                        target_token_ids_d_t :
                                        target_token_ids_d_t.to(torch::kInt32)) :
                                   torch::Tensor();
        launchPostRejectionAndCap(output_token_ids_d,
                                  output_accepted_token_num_d,
                                  output_token_ids_d,
                                  output_accepted_token_num_d,
                                  force_mask_d,
                                  draft_for_force,
                                  target_for_post,
                                  cap_gpu,
                                  /*clamp_minus_one=*/true);
        if (fused_cap_applied != nullptr) {
            *fused_cap_applied = cap_gpu.defined();
        }
        sample_output.accept_tokens = output_token_ids_d;
        sample_output.accept_len    = output_accepted_token_num_d;
        return;
    }
#endif

    // forceSpAccept: override rejection sampling results for streams that requested
    // forced acceptance — accept all draft tokens plus the target bonus token.
    if (has_force) {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling.forceSpAccept");
        // target_token_ids_d_t layout: [batch_size * (propose_step+1), token_stride]
        // Extract the bonus token at position propose_step for each batch item.
        int64_t token_stride = target_token_ids_d_t.size(1);
        auto    target_bonus_t =
            target_token_ids_d_t.reshape({(long)batch_size, (long)(propose_step_ + 1), token_stride});
        auto target_bonus = target_bonus_t.select(1, propose_step_).select(1, token_stride - 1).unsqueeze(1);
        // forced_tokens: draft tokens [0..propose_step-1] + target bonus
        auto forced_tokens          = torch::cat({draft_token_ids_d_t, target_bonus}, 1);
        auto force_mask_2d          = force_mask_d.unsqueeze(1).expand_as(output_token_ids_d);
        output_token_ids_d          = torch::where(force_mask_2d, forced_tokens, output_token_ids_d);
        output_accepted_token_num_d = torch::where(
            force_mask_d,
            torch::full_like(output_accepted_token_num_d, (int32_t)(propose_step_ + 1)),
            output_accepted_token_num_d);
    }

    // use async sample here, we assume accept all tokens
    // so we need to reset -1 to 0 in output_token_ids_d
    output_token_ids_d.index_put_({output_token_ids_d == -1}, 0);
    sample_output.accept_tokens = output_token_ids_d;
    sample_output.accept_len    = output_accepted_token_num_d;
}

}  // namespace speculative
}  // namespace rtp_llm
