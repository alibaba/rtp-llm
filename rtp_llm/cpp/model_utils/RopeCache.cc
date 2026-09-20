#include "rtp_llm/cpp/model_utils/RopeCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/Exception.h"

#if USING_ROCM
#include <ATen/detail/HIPHooksInterface.h>
#else
#include <ATen/detail/CUDAHooksInterface.h>
#endif
#include <map>
#include <mutex>
#include <tuple>

namespace rtp_llm {

namespace {

using RopeCacheKey = std::tuple<int, int, int, float, float, float, int, float, float, int, bool, bool, int>;

int currentDevice() {
#if USING_ROCM
    return at::detail::getHIPHooks().current_device();
#else
    return at::detail::getCUDAHooks().current_device();
#endif
}

RopeCacheKey makeRopeCacheKey(const RopeConfig& rope_config,
                              const int         max_position_embeddings,
                              const bool        is_cuda,
                              const bool        interleave) {
    const int cache_length = rope_config.style == RopeStyle::Yarn ?
                                 static_cast<int>(rope_config.max_pos * rope_config.scale) :
                                 max_position_embeddings;
    return {static_cast<int>(rope_config.style),
            rope_config.dim,
            rope_config.base,
            rope_config.scale,
            rope_config.factor1,
            rope_config.factor2,
            rope_config.max_pos,
            rope_config.extrapolation_factor,
            rope_config.mscale,
            cache_length,
            is_cuda,
            interleave,
            currentDevice()};
}

}  // namespace

float yarnFindCorrectionDim(const int num_rotations,
                            const int rope_dim,
                            const int rope_theta,
                            const int max_position_embeddings) {
    return static_cast<float>(rope_dim
                              * std::log(static_cast<float>(max_position_embeddings / (num_rotations * 2.f * M_PI))))
           / (2.f * std::log(static_cast<float>(rope_theta)));
}

torch::Tensor genBaseCache(const int   rope_dim,
                           const int   rope_theta,
                           const float rope_scale,
                           const int   max_position_embeddings,
                           const bool  interleave) {
    // Compute on GPU to avoid CPU multi-threading non-determinism
    // (CPU torch.cos/sin can produce different float32 results across processes
    //  due to MKL/SIMD thread scheduling, causing ~2% cross-rank divergence)
    auto gpu_opts = torch::TensorOptions(torch::kInt64).device(torch::kCUDA);
    auto inv_freq =
        1.f / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, gpu_opts).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings * rope_scale, gpu_opts).to(torch::kFloat32);
    t.div_(rope_scale);
    auto freqs = torch::outer(t, inv_freq);
    auto cos   = freqs.cos();
    auto sin   = freqs.sin();

    torch::Tensor cos_sin;
    if (interleave) {
        cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();
    } else {
        cos_sin = torch::cat({cos, sin}, 1).contiguous();
    }

    return cos_sin;
}

torch::Tensor genYarnCache(const int   rope_dim,
                           const int   rope_theta,
                           const float rope_scale,
                           const int   max_position_embeddings,
                           const int   beta_slow,
                           const int   beta_fast,
                           const float extrapolation_factor,
                           const float mscale,
                           const bool  interleave) {
    auto gpu_opts  = torch::TensorOptions(torch::kInt64).device(torch::kCUDA);
    auto pos_freqs = torch::pow(rope_theta, torch::arange(0, rope_dim, 2, gpu_opts).to(torch::kFloat32) / rope_dim);
    auto inv_freq_extrapolation = 1.f / pos_freqs;
    auto inv_freq_interpolation = 1.f / (rope_scale * pos_freqs);
    // Match YarnRope's dynamic CUDA path: beta_fast defines the low
    // correction dimension and beta_slow defines the high dimension.
    float low  = static_cast<float>(std::max(
        0,
        static_cast<int>(std::floor(yarnFindCorrectionDim(beta_fast, rope_dim, rope_theta, max_position_embeddings)))));
    float high = static_cast<float>(std::min(
        rope_dim - 1,
        static_cast<int>(std::ceil(yarnFindCorrectionDim(beta_slow, rope_dim, rope_theta, max_position_embeddings)))));
    if (std::fabs(low - high) < 1e-6) {
        high += 0.001f;
    }
    auto linear        = (torch::arange(rope_dim / 2, gpu_opts).to(torch::kFloat32) - low) / (high - low);
    auto ramp          = torch::clamp(linear, 0, 1);
    auto inv_freq_mask = (1.f - ramp) * extrapolation_factor;
    auto inv_freq      = inv_freq_interpolation * (1.f - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;
    auto t             = torch::arange(max_position_embeddings * rope_scale, gpu_opts).to(torch::kFloat32);
    auto freqs         = torch::outer(t, inv_freq);
    auto cos           = freqs.cos() * mscale;
    auto sin           = freqs.sin() * mscale;

    torch::Tensor cos_sin;
    if (interleave) {
        cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();
    } else {
        cos_sin = torch::cat({cos, sin}, 1).contiguous();
    }

    return cos_sin;
}

torch::Tensor getRopeCache(const RopeConfig& rope_config, const int max_position_embeddings, const bool interleave) {
    RTP_LLM_LOG_INFO("%s  max_position_embeddings: %d, interleave: %d",
                     rope_config.DebugRopeConfigStr().c_str(),
                     max_position_embeddings,
                     interleave);
    torch::Tensor rope_cache;

    switch (rope_config.style) {
        case RopeStyle::Base:
            rope_cache =
                genBaseCache(rope_config.dim, rope_config.base, rope_config.scale, max_position_embeddings, interleave);
            break;

        case RopeStyle::Yarn:
            rope_cache = genYarnCache(rope_config.dim,
                                      rope_config.base,
                                      rope_config.scale,
                                      rope_config.max_pos,
                                      static_cast<int>(rope_config.factor1),
                                      static_cast<int>(rope_config.factor2),
                                      rope_config.extrapolation_factor,
                                      rope_config.mscale,
                                      interleave);
            break;

        default:
            RTP_LLM_LOG_ERROR("unsupported rope_style = %d", rope_config.style);
            throw RTP_EXCEPTION("unsupported rope_style: %d", rope_config.style);
    }

    return rope_cache;
}

// cos/sin cache format: true=interleaved [cos,sin,cos,sin,...], false=non-interleaved
// [cos,cos,...,sin,sin,...]
RopeCache getRopeCacheOnce(const RopeConfig& rope_config,
                           const int         max_position_embeddings,
                           const bool        is_cuda,
                           const bool        interleave) {
    if (max_position_embeddings <= 0) {
        RTP_LLM_LOG_WARNING("getRopeCacheOnce called with max_position_embeddings=%d, returning unused cache",
                            max_position_embeddings);
        RopeCache empty;
        empty.used = false;
        return empty;
    }

    const bool supported = is_cuda ? rope_config.style == RopeStyle::Base || rope_config.style == RopeStyle::Yarn :
                                     rope_config.style == RopeStyle::Base;
    if (!supported) {
        return RopeCache();
    }

    static std::mutex                        cache_mutex;
    static std::map<RopeCacheKey, RopeCache> caches;
    const auto key = makeRopeCacheKey(rope_config, max_position_embeddings, is_cuda, interleave);

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        found = caches.find(key);
    if (found != caches.end()) {
        return found->second;
    }

    RopeCache cache;
    cache.used = true;
    cache.base = rope_config.base;
    cache.dim  = rope_config.dim;
    cache.data = getRopeCache(rope_config, max_position_embeddings, interleave);
    return caches.emplace(key, std::move(cache)).first->second;
}

bool checkRopeCache(const RopeConfig& rope_config, const RopeCache& rope_cache) {
    return rope_cache.used && rope_cache.dim == rope_config.dim && rope_cache.base == rope_config.base
           && rope_cache.data.defined();
}

}  // namespace rtp_llm
