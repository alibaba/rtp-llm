#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "3rdparty/trt_beam_search/beamSearch.h"
#include "3rdparty/trt_beam_search/beamSearchKernels.h"
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "3rdparty/trt_beam_search/beamSearch.h"
#include "3rdparty/trt_beam_search/beamSearchKernels.h"
#endif

using namespace std;
namespace rtp_llm {

#if USING_CUDA || USING_ROCM

BeamSearchOutput sampleBeamSearch(BeamSearchParams params) {
#if USING_CUDA
    auto cur_stream = at::cuda::getCurrentCUDAStream().stream();
#elif USING_ROCM
    auto cur_stream = at::hip::getCurrentHIPStream(at::hip::current_device()).stream();
#endif

    const int batch_size     = params.logits.size(0);
    const int beam_width_in  = params.logits.size(1);
    const int beam_width_out = params.num_beams_out != 0 ? params.num_beams_out : beam_width_in;
    const int vocab_size     = params.logits.size(2);
    const int max_seq_len    = params.token_ids.size(2);
    // TODO(zhangjianning.zjn): check the shape of params
    RTP_LLM_CHECK_WITH_INFO((vocab_size > 2 * beam_width_in),
                            "cuda beam search op need vocab_size[%d] > beam_width_in[%d] * 2",
                            vocab_size,
                            beam_width_in);
    RTP_LLM_CHECK_WITH_INFO((vocab_size > 2 * beam_width_out),
                            "cuda beam search op need vocab_size[%d] > beam_width_out[%d] * 2",
                            vocab_size,
                            beam_width_out);

#define DISPATCH_TYPE(T, T_EXPR, ...)                                                                                  \
    do {                                                                                                               \
        switch (T_EXPR) {                                                                                              \
            case DataType::TYPE_FP16: {                                                                                \
                using T = half;                                                                                        \
                (__VA_ARGS__)();                                                                                       \
            } break;                                                                                                   \
            case DataType::TYPE_FP32: {                                                                                \
                using T = float;                                                                                       \
                (__VA_ARGS__)();                                                                                       \
            } break;                                                                                                   \
            default:                                                                                                   \
                RTP_LLM_CHECK_WITH_INFO(false,                                                                         \
                                        "cuda beam search op does not support dtype[%d]",                              \
                                        (int)torchDTypeToDataType(params.logits.dtype()));                             \
        }                                                                                                              \
    } while (0)

#define DISPATCH_BOOL(BOOL, BOOL_EXPR, ...)                                                                            \
    do {                                                                                                               \
        if (BOOL_EXPR) {                                                                                               \
            constexpr bool BOOL = true;                                                                                \
            (__VA_ARGS__)();                                                                                           \
        } else {                                                                                                       \
            constexpr bool BOOL = false;                                                                               \
            (__VA_ARGS__)();                                                                                           \
        }                                                                                                              \
    } while (0)

    // compute log softmax for probability calculation
    at::Tensor logits_tsr = params.logits;

    // Stochastic Beam Search: Gumbel-Top-k Trick with per-batch noise scale.
    // Recommended noise_scale ranges:
    //   0.0    = deterministic (standard beam search)
    //   0.05-0.15 = light randomness (recommended for quality-first)
    //   0.2-0.4  = medium randomness (balance quality/diversity)
    //   0.5-1.0  = strong randomness (diversity-first, quality loss)
    //   1.0    = standard Stochastic Beam Search (unbiased sampling)
    if (params.temperature.defined() && params.temperature.numel() > 0) {
        // params.temperature is on host.
        auto         temperature_cpu = params.temperature.cpu().contiguous().view(-1);
        const float* temp_data       = temperature_cpu.data_ptr<float>();
        const int    temp_per_batch  = temperature_cpu.numel() / batch_size;

        bool   needs_noise     = false;
        auto   noise_scale_cpu = torch::zeros({batch_size, 1, 1}, torch::kFloat32);
        float* ns_data         = noise_scale_cpu.data_ptr<float>();
        for (int b = 0; b < batch_size; ++b) {
            float t    = temp_data[b * temp_per_batch];
            float ns   = (t > 0.0f && std::abs(t - 1.0f) > 1e-6f) ? t : 0.0f;
            ns_data[b] = ns;
            if (ns > 0.0f)
                needs_noise = true;
        }

        if (needs_noise) {
            auto noise_scale_device = noise_scale_cpu.to(logits_tsr.device());
            auto gumbel_noise       = torch::empty(logits_tsr.sizes(), logits_tsr.options().dtype(torch::kFloat32));
            if (!params.generators.empty()) {
                for (int b = 0; b < batch_size; ++b) {
                    auto gen_opt = (b < (int)params.generators.size() && params.generators[b].defined()) ?
                                       c10::make_optional(params.generators[b]) :
                                       c10::nullopt;
                    gumbel_noise[b].uniform_(1e-10, 1.0 - 1e-10, gen_opt);
                }
            } else {
                gumbel_noise.uniform_(1e-10, 1.0 - 1e-10);
            }
            gumbel_noise.log_().neg_().log_().neg_();
            logits_tsr = logits_tsr + (gumbel_noise * noise_scale_device).to(logits_tsr.dtype());
        }
    }

    // note the computation here is intentionally performed inplace to reduce memory usage.
    // When Gumbel noise is added above, logits_tsr is a freshly-allocated tensor, so the
    // in-place log_softmax is safe; otherwise it modifies params.logits in place as before.
    at::Tensor log_softmax_logits_tsr = logits_tsr;
    at::log_softmax_out(log_softmax_logits_tsr, logits_tsr, -1);

    // beam search heuristic
    auto                           logits_dtype = torchDTypeToDataType(params.logits.dtype());
    tensorrt_llm::BeamSearchConfig config;
    DISPATCH_TYPE(T, logits_dtype, [&]() {
        config = tensorrt_llm::configureBeamSearch<T>(batch_size, beam_width_in, beam_width_out, vocab_size);
    });

    // set beam search kernel workspace
    auto workspace =
        torch::empty({(int64_t)config.mWorkspaceSize}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    cudaMemsetAsync(workspace.data_ptr(), 0, workspace.nbytes(), cur_stream);

    // allocate output buffer
    auto opts_int          = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_float        = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto token_ids_out     = torch::empty({batch_size, beam_width_out, max_seq_len}, opts_int);
    auto beam_indices      = torch::empty({batch_size, beam_width_out}, opts_int);
    auto output_ids        = torch::empty({batch_size, beam_width_out}, opts_int);
    auto input_lengths_out = config.mVBWS ? torch::empty({batch_size, beam_width_out}, opts_int) : params.input_lengths;
    auto sequence_lengths_out = torch::empty({batch_size, beam_width_out}, opts_int);
    auto cum_log_probs_out =
        config.mVBWS ? torch::empty({batch_size, beam_width_out}, opts_float) : params.cum_log_probs;

    // set BeamHypotheses
    tensorrt_llm::kernels::BeamHypotheses BH;
    // basic scalar
    BH.bVBWS         = config.mVBWS;
    BH.nMaxBatchSize = batch_size;
    BH.nBatchSize    = batch_size;
    BH.nBeamWidthIn  = beam_width_in;
    BH.nBeamWidthOut = beam_width_out;
    BH.nMaxSeqLen    = max_seq_len;
    BH.nVocabSize    = vocab_size;
    BH.nVPart        = config.mVPart;
    // buffer size
    BH.nByteMaxSharedMemoryPerBlock = config.mByteMaxSharedMemoryPerBlock;
    BH.nByteSharedMemoryStage1      = config.mByteSharedMemoryStage1;
    BH.nByteSharedMemoryStage3      = config.mByteSharedMemoryStage3;
    // input and output ptr
    BH.inputLengthsIn     = params.input_lengths.data_ptr<int>();
    BH.inputLengthsOut    = input_lengths_out.data_ptr<int>();
    BH.sequenceLengthsIn  = params.sequence_lengths.data_ptr<int>();
    BH.sequenceLengthsOut = sequence_lengths_out.data_ptr<int>();
    BH.cumLogProbsIn      = params.cum_log_probs.data_ptr<float>();
    BH.cumLogProbsOut     = cum_log_probs_out.data_ptr<float>();
    BH.tokenIdsIn         = params.token_ids.data_ptr<int>();
    BH.tokenIdsOut        = token_ids_out.data_ptr<int>();
    BH.parentIdsPtr       = beam_indices.data_ptr<int>();
    BH.outputIdsPtr       = output_ids.data_ptr<int>();

    check_cuda_error();

    // invoke beam search kernel
    DISPATCH_TYPE(T, logits_dtype, [&]() {
        DISPATCH_BOOL(IS_V2, config.mV2, [&]() {
            tensorrt_llm::kernels::invokeTopkBeamSearch<T, IS_V2>(
                static_cast<T*>(log_softmax_logits_tsr.data_ptr()), nullptr, workspace.data_ptr(), BH, cur_stream);
        });
    });

    check_cuda_error();

    return BeamSearchOutput({std::move(token_ids_out),
                             std::move(input_lengths_out),
                             std::move(sequence_lengths_out),
                             std::move(cum_log_probs_out),
                             std::move(beam_indices)});

#undef DISPATCH_TYPE
#undef DISPATCH_BOOL
}

#else  // Any other devices

BeamSearchOutput sampleBeamSearch(BeamSearchParams params) {
    RTP_LLM_CHECK_WITH_INFO(false, "beam search is not supported on the device yet");
    return BeamSearchOutput({});
}

#endif  // USING_CUDA || USING_ROCM

}  // namespace rtp_llm
