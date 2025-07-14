#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

#include "3rdparty/trt_beam_search/beamSearchKernels.h"

using namespace std;
namespace rtp_llm {

void CudaDevice::sampleBeamSearch(const BeamSearchParams& params) {
    const int batch_size  = params.logits.shape()[0];
    const int beam_width  = params.logits.shape()[1];
    const int vocab_size  = params.logits.shape()[2];
    const int max_seq_len = params.token_ids.shape()[2];
    RTP_LLM_CHECK_WITH_INFO((vocab_size > 2 * beam_width),
                            "cuda beam search op need vocab_size[%d] > beam_width[%d] * 2",
                            vocab_size,
                            beam_width);
    // set trt kernel workspace
    const int n_pad_beam_width = tensorrt_llm::kernels::padToNextPowerOfTwo(beam_width);
    const int n_topk           = batch_size * n_pad_beam_width * n_pad_beam_width * 2;
    const int n_temp_buffer    = batch_size * n_pad_beam_width * tensorrt_llm::kernels::nMaxVocabPartForStage1FastKernel
                              * (2 * (n_pad_beam_width * 2) + 2);
    const size_t workspace_size = roundUp(n_topk, 4) + roundUp(n_temp_buffer, 4);
    BufferPtr    workspace      = allocateBuffer({DataType::TYPE_FP32, {workspace_size}});
    cudaMemsetAsync(workspace->data(), 0, workspace->sizeBytes(), stream_);
    // set BeamHypotheses
    tensorrt_llm::kernels::BeamHypotheses BH;
    // basic scalar
    BH.nMaxBatchSize = batch_size;
    BH.nBatchSize    = batch_size;
    BH.nBeamWidth    = beam_width;
    BH.nMaxSeqLen    = max_seq_len;
    BH.nVocabSize    = vocab_size;
    // essential ptr
    BH.inputLengths       = params.input_lengths.data<int>();
    BH.sequenceLengths    = params.sequence_lengths.data<int>();
    auto input_lengths    = Buffer2torchTensor(params.input_lengths, false);
    auto sequence_lengths = Buffer2torchTensor(params.sequence_lengths, false);
    auto cum_log_probs_t  = Buffer2torchTensor(params.cum_log_probs, false);
    for (int i = 0; i < batch_size; i++) {
        if (input_lengths[i].equal(sequence_lengths[i])) {
            for (int j = 1; j < beam_width; j++) {
                cum_log_probs_t.index_put_({i, j}, -1e9);
            }
        }
    }
    BH.cumLogProbs = params.cum_log_probs.data<float>();
    BH.beamIdsPtr  = params.beam_index.data<int>();
    auto output_ids =
        allocateBuffer({DataType::TYPE_INT32, {(size_t)batch_size, (size_t)beam_width}, AllocationType::DEVICE}, {});
    BH.outputIdsPtr = output_ids->data<int>();

    // invoke trt kernel
    switch (params.logits.type()) {
        case DataType::TYPE_FP16:
            tensorrt_llm::kernels::invokeTopkSoftMax(params.logits.data<half>(), workspace->data(), BH, stream_);
            break;
        case DataType::TYPE_FP32:
            tensorrt_llm::kernels::invokeTopkSoftMax(params.logits.data<float>(), workspace->data(), BH, stream_);
            break;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "cuda beam search op dose not support dtype[%d]", params.logits.type());
    }

    auto output_ids_tensor = Buffer2torchTensor(output_ids, false);
    auto token_ids         = Buffer2torchTensor(params.token_ids, false);
    auto beam_index_output = Buffer2torchTensor(params.beam_index, false);

    auto token_ids_clone = token_ids.clone();
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < beam_width; j++) {
            int  sequence_index    = sequence_lengths[i][j].item<int>() + 1;
            auto select_beam_index = beam_index_output[i][j].item<int>();
            auto tmp               = token_ids[i][select_beam_index];
            token_ids_clone.index_put_({i, j}, tmp);
            token_ids_clone.index_put_({i, j, sequence_index - 1}, output_ids_tensor[i][j]);
        }
    }
    auto output_token_ids = torchTensor2Buffer(token_ids_clone);

    copy({params.token_ids, *output_token_ids});
    return;
}

}  // namespace rtp_llm