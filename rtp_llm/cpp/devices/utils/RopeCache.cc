#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/Exception.h"

namespace rtp_llm {

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
    auto inv_freq =
        1.f / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings * rope_scale, torch::kInt64).to(torch::kFloat32);
    t.div_(rope_scale);
    auto freqs = torch::outer(t, inv_freq);
    auto cos   = freqs.cos().to(torch::kFloat32);
    auto sin   = freqs.sin().to(torch::kFloat32);

    torch::Tensor cos_sin;
    if (interleave) {
        // Interleaved format: [cos[0], sin[0], cos[1], sin[1], cos[2], sin[2], ...]
        cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();
    } else {
        // Non-interleaved format (flashinfer compatible): [cos[0], cos[1], ..., sin[0], sin[1], ...]
        // Cosine is the first half and Sine is the second half on rotary_dim
        cos_sin = torch::cat({cos, sin}, 1).contiguous();
    }

    return cos_sin.cuda();
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
    auto pos_freqs =
        torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto  inv_freq_extrapolation = 1.f / pos_freqs;
    auto  inv_freq_interpolation = 1.f / (rope_scale * pos_freqs);
    float low                    = static_cast<float>(std::max(
        0,
        static_cast<int>(std::floor(yarnFindCorrectionDim(beta_slow, rope_dim, rope_theta, max_position_embeddings)))));
    float high                   = static_cast<float>(std::min(
        rope_dim - 1,
        static_cast<int>(std::ceil(yarnFindCorrectionDim(beta_fast, rope_dim, rope_theta, max_position_embeddings)))));
    if (std::fabs(low - high) < 1e-6) {
        high += 0.001f;
    }
    auto linear        = (torch::arange(rope_dim / 2, torch::kInt64).to(torch::kFloat32) - low) / (high - low);
    auto ramp          = torch::clamp(linear, 0, 1);
    auto inv_freq_mask = (1.f - ramp) * extrapolation_factor;
    auto inv_freq      = inv_freq_interpolation * (1.f - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;
    auto t             = torch::arange(max_position_embeddings * rope_scale, torch::kInt64).to(torch::kFloat32);
    auto freqs         = torch::outer(t, inv_freq);
    auto cos           = freqs.cos().to(torch::kFloat32) * mscale;
    auto sin           = freqs.sin().to(torch::kFloat32) * mscale;

    torch::Tensor cos_sin;
    if (interleave) {
        // Interleaved format: [cos[0], sin[0], cos[1], sin[1], cos[2], sin[2], ...]
        cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();
    } else {
        // Non-interleaved format (flashinfer compatible): [cos[0], cos[1], ..., sin[0], sin[1], ...]
        // Cosine is the first half and Sine is the second half on rotary_dim
        cos_sin = torch::cat({cos, sin}, 1).contiguous();
    }

    return cos_sin.cuda();
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
    // Maintain two separate caches: one for interleaved format, one for non-interleaved format
    static std::once_flag rope_cache_flag_interleaved;
    static RopeCache      rope_cache_interleaved;

    static std::once_flag rope_cache_flag_non_interleaved;
    static RopeCache      rope_cache_non_interleaved;

    if (interleave) {
        // Use interleaved cache
        std::call_once(rope_cache_flag_interleaved, [&]() {
            rope_cache_interleaved.used =
                is_cuda ? rope_config.style == RopeStyle::Base || rope_config.style == RopeStyle::Yarn :
                          rope_config.style == RopeStyle::Base;
            if (rope_cache_interleaved.used) {
                rope_cache_interleaved.base = rope_config.base;
                rope_cache_interleaved.dim  = rope_config.dim;
                rope_cache_interleaved.data = getRopeCache(rope_config, max_position_embeddings, interleave);
            }
        });
        return rope_cache_interleaved;
    } else {
        // Use non-interleaved cache
        std::call_once(rope_cache_flag_non_interleaved, [&]() {
            rope_cache_non_interleaved.used =
                is_cuda ? rope_config.style == RopeStyle::Base || rope_config.style == RopeStyle::Yarn :
                          rope_config.style == RopeStyle::Base;
            if (rope_cache_non_interleaved.used) {
                rope_cache_non_interleaved.base = rope_config.base;
                rope_cache_non_interleaved.dim  = rope_config.dim;
                rope_cache_non_interleaved.data = getRopeCache(rope_config, max_position_embeddings, interleave);
            }
        });
        return rope_cache_non_interleaved;
    }
}

bool checkRopeCache(const RopeConfig& rope_config, const RopeCache& rope_cache) {
    return rope_cache.used && rope_cache.dim == rope_config.dim && rope_cache.base == rope_config.base
           && rope_cache.data.defined();
}

}  // namespace rtp_llm
