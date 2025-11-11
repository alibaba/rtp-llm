#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
float yarnFindCorrectionDim(int num_rotations, int rope_dim, int rope_theta, int max_position_embeddings) {
    return static_cast<float>(rope_dim
                              * std::log(static_cast<float>(max_position_embeddings / (num_rotations * 2.f * M_PI))))
           / (2.f * std::log(static_cast<float>(rope_theta)));
}

torch::Tensor genBaseCache(int rope_dim, int rope_theta, float rope_scale, int max_position_embeddings) {
    auto inv_freq =
        1.f / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings * rope_scale, torch::kInt64).to(torch::kFloat32);
    t.div_(rope_scale);
    auto freqs   = torch::outer(t, inv_freq);
    auto cos     = freqs.cos().to(torch::kFloat32);
    auto sin     = freqs.sin().to(torch::kFloat32);
    auto cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();

    return cos_sin.cuda();
}

torch::Tensor genYarnCache(int   rope_dim,
                           int   rope_theta,
                           float rope_scale,
                           int   max_position_embeddings,
                           int   beta_slow,
                           int   beta_fast,
                           float extrapolation_factor,
                           float mscale) {
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
    auto cos_sin       = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();

    return cos_sin.cuda();
}

/**
 * @brief Get the Rope Cos Sin object, TODO: move to python
 *
 * @param rope_config
 * @param max_position_embeddings
 * @return torch::Tensor
 */
torch::Tensor getRopeCache(const RopeConfig& rope_config, int max_position_embeddings) {
    RTP_LLM_LOG_INFO(
        "%s  max_position_embeddings: %d", rope_config.DebugRopeConfigStr().c_str(), max_position_embeddings);
    torch::Tensor rope_cache;

    switch (rope_config.style) {
        case RopeStyle::Base:
            rope_cache = genBaseCache(rope_config.dim, rope_config.base, rope_config.scale, max_position_embeddings);
            break;

        case RopeStyle::Yarn:
            rope_cache = genYarnCache(rope_config.dim,
                                      rope_config.base,
                                      rope_config.scale,
                                      rope_config.max_pos,
                                      static_cast<int>(rope_config.factor1),
                                      static_cast<int>(rope_config.factor2),
                                      rope_config.extrapolation_factor,
                                      rope_config.mscale);
            break;

        default:
            RTP_LLM_LOG_ERROR("unsupported rope_style = %d", rope_config.style);
            throw OpException({OpErrorType::ERROR_UNIMPLEMENTED, "unsupported rope_style"});
    }

    return rope_cache;
}

} // namespace rtm_llm
