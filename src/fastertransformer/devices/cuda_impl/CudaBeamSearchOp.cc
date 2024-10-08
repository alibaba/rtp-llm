#include "src/fastertransformer/devices/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;
namespace fastertransformer {

void CudaDevice::sampleBeamSearch(const BeamSearchParams& params) {

    FT_LOG_DEBUG("sampleBeamSearch start");
    auto logits = Buffer2torchTensor(params.logits, false);
    auto token_ids = Buffer2torchTensor(params.token_ids, false);
    auto cum_log_probs = Buffer2torchTensor(params.cum_log_probs, false);
    auto input_lengths = Buffer2torchTensor(params.input_lengths, false);
    auto sequence_lengths = Buffer2torchTensor(params.sequence_lengths, false);
    auto beam_index_output = Buffer2torchTensor(params.beam_index, false);
    FT_LOG_DEBUG("sampleBeamSearch buffer to tensor done");

    int batch_size  = logits.size(0);
    // for beam history update cache.
    auto token_ids_clone = token_ids.clone();

    for (int i = 0; i < batch_size; i++) {
        // logits float[beam_width, vocab_size]
        auto beam_width = logits[i].size(0);
        // first topk from log softmax logits
        // probs: [beam_width, beam_width]
        auto [probs, index] = logits[i].log_softmax(-1).topk(beam_width, -1);
        // add cum_log_probs
        probs = probs + cum_log_probs[i].reshape({-1, 1});
        // second topk from probs
        // log_probs: [beam_width]
        auto [log_probs, beam_index] = probs.flatten(0, -1).topk(beam_width, -1);
        auto new_token_ids = torch::gather(index.flatten(0, -1), 0, beam_index);
        beam_index = torch::div(beam_index, beam_width, "floor").squeeze();
        // context first beam sampler
        if (input_lengths[i].equal(sequence_lengths[i])) {
            new_token_ids = index[0];
            log_probs = probs[0];
            beam_index = torch::zeros_like(new_token_ids);
        }
        // according beam index to update input token ids
        for (int j = 0; j < beam_width; j++) {
            int sequence_index = sequence_lengths[i][j].item<int>() + 1;
            auto select_beam_index = beam_index[j].item<int>();
            auto tmp = token_ids[i][select_beam_index];
            token_ids_clone.index_put_({i, j}, tmp);
            token_ids_clone.index_put_({i, j, sequence_index - 1}, new_token_ids[j]);
            cum_log_probs.index_put_({i, j}, log_probs[j]);
            beam_index_output.index_put_({i, j}, beam_index[j]);
        }
    }

    auto output_token_ids        = torchTensor2Buffer(token_ids_clone);
    auto output_cum_log_probs    = torchTensor2Buffer(cum_log_probs);
    auto output_beam_index       = torchTensor2Buffer(beam_index_output);

    copy({params.token_ids, *output_token_ids});
    copy({params.cum_log_probs, *output_cum_log_probs});
    copy({params.beam_index, *output_beam_index});
    return;
}

}