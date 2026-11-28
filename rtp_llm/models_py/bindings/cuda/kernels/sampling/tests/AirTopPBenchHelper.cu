#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/tests/AirTopPBenchHelper.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/flashinfer/air_top_p.cuh"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/flashinfer/sampling.cuh"

namespace rtp_llm {

size_t airTopPWorkspaceBytes(uint32_t batch_size, uint32_t vocab_size) {
    namespace air                = flashinfer::sampling::air_top_p;
    auto         align256        = [](size_t x) { return ((x + 255) / 256) * 256; };
    const size_t counters_size   = align256(sizeof(air::Counter<float>) * batch_size);
    const size_t hist_size       = align256(sizeof(float) * air::NUM_BUCKETS * batch_size);
    const size_t count_hist_size = align256(sizeof(air::IdxT) * air::NUM_BUCKETS * batch_size);
    const size_t buf_elems       = static_cast<size_t>(air::calcBufLen<float>(static_cast<air::IdxT>(vocab_size)));
    const size_t buf_size        = align256(sizeof(float) * buf_elems * batch_size);
    return counters_size + hist_size + count_hist_size + buf_size * 2;
}

void invokeAirTopPRenormBench(float*       probs,
                              float*       renormed_prob,
                              float*       top_p_arr,
                              uint32_t     batch_size,
                              float        top_p_val,
                              uint32_t     vocab_size,
                              void*        workspace,
                              cudaStream_t stream) {
    flashinfer::sampling::air_top_p::AirTopPRenormProb<false, float>(
        probs, renormed_prob, top_p_arr, batch_size, top_p_val, vocab_size, workspace, stream);
}

void invokeFlashInferSamplingFromProbBench(const float* probs,
                                           int32_t*     output,
                                           bool*        valid,
                                           uint32_t     batch_size,
                                           uint32_t     vocab_size,
                                           uint64_t*    seed_arr,
                                           uint64_t*    offset_arr,
                                           cudaStream_t stream) {
    flashinfer::sampling::SamplingFromProb<float, int32_t>(const_cast<float*>(probs),
                                                           output,
                                                           valid,
                                                           nullptr,
                                                           batch_size,
                                                           vocab_size,
                                                           true,
                                                           seed_arr,
                                                           0,
                                                           offset_arr,
                                                           0,
                                                           stream);
}

void invokeFlashInferTopKTopPSamplingBench(const float*   probs,
                                           const int32_t* top_k_arr,
                                           const float*   top_p_arr,
                                           int32_t*       output,
                                           bool*          valid,
                                           uint32_t       batch_size,
                                           uint32_t       vocab_size,
                                           uint64_t*      seed_arr,
                                           uint64_t*      offset_arr,
                                           cudaStream_t   stream) {
    flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(const_cast<float*>(probs),
                                                                   const_cast<int32_t*>(top_k_arr),
                                                                   const_cast<float*>(top_p_arr),
                                                                   output,
                                                                   valid,
                                                                   nullptr,
                                                                   batch_size,
                                                                   static_cast<int32_t>(vocab_size),
                                                                   1.0f,
                                                                   vocab_size,
                                                                   true,
                                                                   seed_arr,
                                                                   0,
                                                                   offset_arr,
                                                                   0,
                                                                   stream);
}

}  // namespace rtp_llm
