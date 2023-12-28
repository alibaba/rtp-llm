
#include "src/fastertransformer/kernels/online_softmax_beamsearch/online_softmax_beamsearch_kernels.hpp"

namespace fastertransformer {


template void invokeTopkSoftMax<float>(const float*    log_probs,
                                       const float*    bias,
                                       const bool*     finished,
                                       const int*      sequence_lengths,
                                       float*          cum_log_probs,
                                       float*          output_log_probs,
                                       int*            ids,
                                       void*           tmp_storage,
                                       const int       temp_storage_size,
                                       BeamHypotheses* beam_hyps,
                                       const int       batch_size,
                                       const int       beam_width,
                                       const int       vocab_size,
                                       const int*      end_ids,
                                       const float     diversity_rate,
                                       const float     length_penalty,
                                       cudaStream_t    stream);

}  // end of namespace fastertransformer
