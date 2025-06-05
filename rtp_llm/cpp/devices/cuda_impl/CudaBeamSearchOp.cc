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
    const int batch_size     = params.logits.shape()[0];
    const int beam_width_in  = params.logits.shape()[1];
    const int beam_width_out = params.num_beams_out != 0 ? params.num_beams_out : beam_width_in;
    const int vocab_size     = params.logits.shape()[2];
    const int max_seq_len    = params.token_ids->shape()[2];
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
                RTP_LLM_CHECK_WITH_INFO(                                                                               \
                    false, "cuda beam search op does not support dtype[%d]", params.logits.type());                    \
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
    at::Tensor logits_tsr             = Buffer2torchTensor(params.logits, false);
    at::Tensor log_softmax_logits_tsr = logits_tsr.log_softmax(-1);

    // beam search heuristic
    tensorrt_llm::BeamSearchConfig config;
    DISPATCH_TYPE(T, params.logits.type(), [&]() {
        config = tensorrt_llm::configureBeamSearch<T>(batch_size, beam_width_in, beam_width_out, vocab_size);
    });

    // set beam search kernel workspace
    BufferPtr workspace = allocateBuffer({DataType::TYPE_BYTES, {config.mWorkspaceSize}});
    cudaMemsetAsync(workspace->data(), 0, workspace->sizeBytes(), stream_);

    // allocate output buffer
    BufferPtr input_lengths_out, sequence_lengths_out, cum_log_probs_out;
    auto      new_token_ids = allocateBuffer(
        {DataType::TYPE_INT32, {batch_size, beam_width_out, max_seq_len}, AllocationType::DEVICE}, {"new_token_ids"});
    auto beam_indices =
        allocateBuffer({DataType::TYPE_INT32, {batch_size, beam_width_out}, AllocationType::DEVICE}, {"beam_indices"});
    auto output_ids =
        allocateBuffer({DataType::TYPE_INT32, {batch_size, beam_width_out}, AllocationType::DEVICE}, {"output_ids"});
    if (config.mVBWS) {
        input_lengths_out = allocateBuffer({DataType::TYPE_INT32, {batch_size, beam_width_out}, AllocationType::DEVICE},
                                           {"input_length_out"});
        sequence_lengths_out = allocateBuffer(
            {DataType::TYPE_INT32, {batch_size, beam_width_out}, AllocationType::DEVICE}, {"sequence_lengths_out"});
        cum_log_probs_out = allocateBuffer({DataType::TYPE_FP32, {batch_size, beam_width_out}, AllocationType::DEVICE},
                                           {"cum_log_probs_out"});
    } else {
        input_lengths_out    = params.input_lengths;
        sequence_lengths_out = params.sequence_lengths;
        cum_log_probs_out    = params.cum_log_probs;
    }

    // if not variable beam width search, setup cum_log_probs for prefill
    if (!config.mVBWS) {
        // TODO(zhangjianning.zjn): would be better to use a single kernel to setup cum_log_probs
        auto input_lengths_tsr    = Buffer2torchTensor(params.input_lengths, false);
        auto sequence_lengths_tsr = Buffer2torchTensor(params.sequence_lengths, false);
        auto cum_log_probs_tsr    = Buffer2torchTensor(params.cum_log_probs, false);
        for (int i = 0; i < batch_size; i++) {
            if (input_lengths_tsr[i].equal(sequence_lengths_tsr[i])) {
                cum_log_probs_tsr.index_put_({i, torch::indexing::Slice(1, beam_width_in)}, -1e9);
            }
        }
    }

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
    // input and ouput ptr
    BH.inputLengthsIn     = params.input_lengths->data<int>();
    BH.inputLengthsOut    = input_lengths_out->data<int>();
    BH.sequenceLengthsIn  = params.sequence_lengths->data<int>();
    BH.sequenceLengthsOut = sequence_lengths_out->data<int>();
    BH.cumLogProbsIn      = params.cum_log_probs->data<float>();
    BH.cumLogProbsOut     = cum_log_probs_out->data<float>();
    BH.tokenIdsIn         = params.token_ids->data<int>();
    BH.tokenIdsOut        = new_token_ids->data<int>();
    BH.parentIdsPtr       = beam_indices->data<int>();
    BH.outputIdsPtr       = output_ids->data<int>();

    // invoke beam search kernel
    DISPATCH_TYPE(T, params.logits.type(), [&]() {
        DISPATCH_BOOL(IS_V2, config.mV2, [&]() {
            tensorrt_llm::kernels::invokeTopkBeamSearch<T, IS_V2>(
                static_cast<T*>(log_softmax_logits_tsr.data_ptr()), nullptr, workspace->data(), BH, stream_);
        });
    });

    return BeamSearchOutput({std::move(new_token_ids),
                             std::move(input_lengths_out),
                             std::move(sequence_lengths_out),
                             std::move(cum_log_probs_out),
                             std::move(beam_indices)});
}

}  // namespace rtp_llm