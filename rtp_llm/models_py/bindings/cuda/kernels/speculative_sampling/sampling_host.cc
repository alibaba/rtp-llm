#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_host_utils.h"

namespace rtp_llm {
namespace {
bool canUseSplitRejectionSampling(const RejectionSamplingLaunchConfig& config) {
    return sampler::optimizationEnabled("RTP_LLM_SPLIT_REJECTION") && config.batch_size > 0 && config.batch_size <= 8
           && config.num_speculative_tokens == 3 && config.target_vocab_size == 248320 && sampler::isSm120();
}
}  // namespace

cudaError_t dispatchRejectionSampling(const RejectionSamplingParams&       params,
                                      const RejectionSamplingLaunchConfig& config,
                                      cudaStream_t                         stream) {
    float* draft_probs = params.draft_probs_point_mass ? nullptr : params.draft_probs_d.data_ptr<float>();
    if (canUseSplitRejectionSampling(config)) {
        const int tiles     = (config.target_vocab_size + 4095) / 4096;
        auto      sums      = torch::empty({config.batch_size, tiles}, params.target_probs_d.options());
        auto      positions = torch::empty({config.batch_size}, params.draft_token_ids_d.options());
        return invokeSplitRejectionSampling(draft_probs,
                                            params.draft_token_ids_d.data_ptr<int32_t>(),
                                            params.uniform_samples_d.data_ptr<float>(),
                                            params.target_probs_d.data_ptr<float>(),
                                            params.target_token_ids_d.data_ptr<int32_t>(),
                                            config.target_token_stride,
                                            params.output_token_ids_d.data_ptr<int32_t>(),
                                            params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                            params.do_sample_d.data_ptr<bool>(),
                                            config.batch_size,
                                            config.num_speculative_tokens,
                                            config.target_vocab_size,
                                            params.draft_probs_point_mass,
                                            sums.data_ptr<float>(),
                                            positions.data_ptr<int32_t>(),
                                            stream);
    }
    return invokeRejectionSampling(draft_probs,
                                   params.draft_token_ids_d.data_ptr<int32_t>(),
                                   params.uniform_samples_d.data_ptr<float>(),
                                   params.target_probs_d.data_ptr<float>(),
                                   params.target_token_ids_d.data_ptr<int32_t>(),
                                   config.target_token_stride,
                                   params.output_token_ids_d.data_ptr<int32_t>(),
                                   params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                   params.do_sample_d.data_ptr<bool>(),
                                   config.batch_size,
                                   config.num_speculative_tokens,
                                   config.target_vocab_size,
                                   stream,
                                   params.draft_probs_point_mass);
}
}  // namespace rtp_llm
