#include "rtp_llm/cpp/utils/RopeCosSin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
torch::Tensor genNormalCosSin(int rope_dim, int rope_theta, float rope_scale, int max_position_embeddings) {
    auto inv_freq =
        1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings, torch::kInt64).to(torch::kFloat32);
    t.div_(rope_scale);
    auto freqs   = torch::outer(t, inv_freq);
    auto cos     = freqs.cos().to(torch::kFloat32);
    auto sin     = freqs.sin().to(torch::kFloat32);
    auto cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();
    return cos_sin.cuda();
}

/**
 * @brief Get the Rope Cos Sin object, TODO: move to python
 *
 * @param device
 * @param rope_style
 * @param rope_dim
 * @param rope_theta
 * @param rope_scale
 * @param max_position_embeddings
 * @return BufferPtr
 */
torch::Tensor
getRopeCosSin(RopeStyle rope_style, int rope_dim, int rope_theta, float rope_scale, int max_position_embeddings) {
    RTP_LLM_LOG_INFO("rope: style = %d, dim = %d, theta = %d, scale = %f, max_position_embeddings = %d",
                     rope_style,
                     rope_dim,
                     rope_theta,
                     rope_scale,
                     max_position_embeddings);
    torch::Tensor cos_sin;

    switch (rope_style) {
        case RopeStyle::No:
            break;

        case RopeStyle::Base:
            cos_sin = genNormalCosSin(rope_dim, rope_theta, rope_scale, max_position_embeddings);
            break;

        default:
            RTP_LLM_LOG_WARNING("unsupported rope_style = %d, not use rope_cache", rope_style);
            break;
    }

    return cos_sin;
}

} // namespace rtm_llm
