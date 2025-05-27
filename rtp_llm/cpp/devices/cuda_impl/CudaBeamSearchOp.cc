#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

#include "3rdparty/trt_beam_search/beamSearch.h"
#include "3rdparty/trt_beam_search/beamSearchKernels.h"

using namespace std;
namespace rtp_llm {

BeamSearchOutput CudaDevice::sampleBeamSearch(const BeamSearchParams& params) {
    at::Tensor logits_tensor  = Buffer2torchTensor(params.logits, false);
    at::Tensor softmax_logits = torch::log_softmax(logits_tensor, -1);
    const int  batch_size     = params.logits.shape()[0];
    const int  beam_width_in  = params.logits.shape()[1];
    const int  beam_width_out = params.num_beams_out == 0 ? params.num_beams_out : beam_width_in;
    const int  vocab_size     = params.logits.shape()[2];
    const int  max_seq_len    = params.token_ids->shape()[2];
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
                RTP_LLM_CHECK_WITH_INFO(                                                                               \
                    false, "cuda beam search op dose not support dtype[%d]", params.logits.type());                    \
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

    tensorrt_llm::BeamSearchConfig config;
    DISPATCH_TYPE(T, params.logits.type(), [&]() {
        config = tensorrt_llm::configureBeamSearch<T>(batch_size, beam_width_in, beam_width_out, vocab_size);
    });

    // set trt kernel workspace
    BufferPtr workspace = allocateBuffer({DataType::TYPE_BYTES, {config.mWorkspaceSize}});
    cudaMemsetAsync(workspace->data(), 0, workspace->sizeBytes(), stream_);

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
    // essential ptr
    BH.inputLengthsIn     = params.input_lengths->data<int>();
    BH.inputLengthsOut    = params.input_lengths->data<int>();
    BH.sequenceLengthsIn  = params.sequence_lengths->data<int>();
    BH.sequenceLengthsOut = params.sequence_lengths->data<int>();
    auto input_lengths    = Buffer2torchTensor(params.input_lengths, false);
    auto sequence_lengths = Buffer2torchTensor(params.sequence_lengths, false);
    auto cum_log_probs_t  = Buffer2torchTensor(params.cum_log_probs, false);
    for (int i = 0; i < batch_size; i++) {
        if (input_lengths[i].equal(sequence_lengths[i])) {
            for (int j = 1; j < beam_width_in; j++) {
                cum_log_probs_t.index_put_({i, j}, -1e9);
            }
        }
    }
    BH.cumLogProbsIn  = params.cum_log_probs->data<float>();
    BH.cumLogProbsOut = params.cum_log_probs->data<float>();
    auto beam_indices = allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)batch_size, (size_t)beam_width_out}, AllocationType::DEVICE}, {});
    BH.parentIdsPtr = beam_indices->data<int>();
    auto output_ids = allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)batch_size, (size_t)beam_width_out}, AllocationType::DEVICE}, {});
    BH.outputIdsPtr = output_ids->data<int>();

    // invoke trt kernel
    DISPATCH_TYPE(T, params.logits.type(), [&]() {
        DISPATCH_BOOL(IS_V2, config.mV2, [&]() {
            tensorrt_llm::kernels::invokeTopkBeamSearch<T, IS_V2>(
                static_cast<T*>(softmax_logits.data_ptr()), nullptr, workspace->data(), BH, stream_);
        });
    });

    auto output_ids_tensor = Buffer2torchTensor(output_ids, false);
    auto token_ids         = Buffer2torchTensor(params.token_ids, false);
    auto beam_indices_in   = Buffer2torchTensor(beam_indices, false);

    // TODO(zhangjianning.zjn): use custom kernel for better performance
    auto new_token_ids   = clone({*params.token_ids});
    auto token_ids_clone = Buffer2torchTensor(new_token_ids, false);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < beam_width_out; j++) {
            auto beam_idx_in = beam_indices_in.index({i, j}).item<int>();
            int  seq_len     = sequence_lengths.index({i, j}).item<int>();

            auto old_tokens = token_ids.index({i, beam_idx_in});
            token_ids_clone.index_put_({i, j}, old_tokens);

            auto new_token = output_ids_tensor.index({i, j});
            token_ids_clone.index_put_({i, j, seq_len - 1}, new_token);
        }
    }

    return BeamSearchOutput({std::move(new_token_ids),
                             std::move(params.input_lengths),
                             std::move(params.sequence_lengths),
                             std::move(params.cum_log_probs),
                             std::move(beam_indices)});
}

}  // namespace rtp_llm