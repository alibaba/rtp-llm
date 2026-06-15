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
    // note the computation here is intentionally performed inplace to reduce memory usage
    at::Tensor log_softmax_logits_tsr = params.logits;
    at::log_softmax_out(log_softmax_logits_tsr, params.logits, -1);

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

#elif USING_XPU  // XPU: PyTorch fallback (no TRT kernel)

// Basic beam search implementation using PyTorch ops.
// This is functionally correct but less optimized than the CUDA TensorRT-LLM kernel.
// It handles the core beam search algorithm:
//   1. Compute log-softmax probabilities
//   2. For each beam, find top-k candidates (k = beam_width_out)
//   3. Select the top beam_width_out beams globally across all candidates
//   4. Reconstruct token_ids, beam_indices, and cumulative log probs
BeamSearchOutput sampleBeamSearch(BeamSearchParams params) {
    const int batch_size     = params.logits.size(0);
    const int beam_width_in  = params.logits.size(1);
    const int beam_width_out = params.num_beams_out != 0
                                   ? static_cast<int>(params.num_beams_out)
                                   : beam_width_in;
    const int vocab_size     = params.logits.size(2);
    const int max_seq_len    = params.token_ids.size(2);

    // Device for output tensors — use the same device as logits
    auto device = params.logits.device();
    auto opts_int   = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // 1. Compute log-softmax probabilities: [batch, beam_in, vocab]
    auto log_probs = params.logits.to(torch::kFloat32).log_softmax(-1);

    // 2. Add cumulative log probs from previous steps: [batch, beam_in, 1]
    auto cum_log_probs_in = params.cum_log_probs.to(device).to(torch::kFloat32);
    log_probs = log_probs + cum_log_probs_in.unsqueeze(-1);

    // 3. Flatten beams and vocab: [batch, beam_in * vocab]
    auto flat_log_probs = log_probs.reshape({batch_size, beam_width_in * vocab_size});

    // 4. Select top beam_width_out candidates per batch: [batch, beam_width_out]
    auto topk_result = flat_log_probs.topk(beam_width_out, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
    auto topk_log_probs = std::get<0>(topk_result);  // [batch, beam_width_out]
    auto topk_indices   = std::get<1>(topk_result);   // [batch, beam_width_out]

    // 5. Decode beam indices and token ids from flattened indices
    auto beam_indices_out = (topk_indices / vocab_size).to(torch::kInt32);   // which beam
    auto output_ids       = (topk_indices % vocab_size).to(torch::kInt32);   // which token

    // 6. Build output token_ids by gathering from input token_ids
    //    For each (batch, out_beam), copy the history from the source beam
    auto token_ids_in  = params.token_ids.to(device);   // [batch, beam_in, max_seq_len]
    auto token_ids_out = torch::empty({batch_size, beam_width_out, max_seq_len}, opts_int);

    // Gather source beams: expand beam_indices to index into token_ids_in
    auto beam_idx_expanded = beam_indices_out.to(torch::kLong)
                                 .unsqueeze(-1)
                                 .expand({batch_size, beam_width_out, max_seq_len});
    token_ids_out = token_ids_in.gather(1, beam_idx_expanded);

    // 7. Build output sequence_lengths and input_lengths from source beams
    auto seq_lens_in  = params.sequence_lengths.to(device);  // [batch, beam_in]
    auto input_lens_in = params.input_lengths.to(device);     // [batch, beam_in]
    auto beam_idx_1d = beam_indices_out.to(torch::kLong);     // [batch, beam_width_out]
    auto sequence_lengths_out = seq_lens_in.gather(1, beam_idx_1d);
    auto input_lengths_out    = input_lens_in.gather(1, beam_idx_1d);

    // 8. Write newly selected tokens at the current step position
    //    (mirroring what the CUDA populateTokenIds kernel does)
    auto write_pos = sequence_lengths_out.to(torch::kLong).unsqueeze(-1);  // [batch, beam_out, 1]
    // Guard against scattering past the allocated sequence dimension.
    // Use device-side masking throughout to avoid per-step host sync.
    auto safe_mask = (write_pos < max_seq_len).squeeze(-1);  // [batch, beam_out]
    write_pos = write_pos.clamp(0, max_seq_len - 1);
    // Write tokens only for beams that have not overflowed (device-side where).
    auto safe_ids = output_ids.where(safe_mask, token_ids_out.gather(2, write_pos).squeeze(-1));
    token_ids_out.scatter_(2, write_pos, safe_ids.unsqueeze(-1));

    // 9. Increment sequence lengths only for beams that have not overflowed.
    sequence_lengths_out = sequence_lengths_out + safe_mask.to(torch::kInt32);

    // 10. Cumulative log probs
    auto cum_log_probs_out = topk_log_probs.to(torch::kFloat32);

    return BeamSearchOutput({std::move(token_ids_out),
                             std::move(input_lengths_out),
                             std::move(sequence_lengths_out),
                             std::move(cum_log_probs_out),
                             std::move(beam_indices_out)});
}

#else  // Any other devices

BeamSearchOutput sampleBeamSearch(BeamSearchParams params) {
    RTP_LLM_CHECK_WITH_INFO(false, "beam search is not supported on the device yet");
    return BeamSearchOutput({});
}

#endif  // USING_CUDA || USING_ROCM

}  // namespace rtp_llm
