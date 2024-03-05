#pragma once

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"

namespace fastertransformer
{
template<typename T>
class NormWrapper{
public:
    NormWrapper(LayerNormType layernorm_type, NormType norm_type, T alpha = 0.0f):
        layernorm_type_(layernorm_type), norm_type_(norm_type), alpha_(alpha) {}

    void initDecoderLayerNorm(T*           output, 
                              const T*     input, 
                              const T*     gamma, 
                              const T*     beta, 
                              const float  eps,
                              const size_t m,
                              const size_t n,
                              float*       scale,
                              float*       dynamic_scale,
                              const int    int8_mode,
                              cudaStream_t stream);
    
    void preAttentionLayerNorm(T*           output,
                               const T*     input,
                               const T*     gamma,
                               const T*     beta,
                               const float  eps,
                               const size_t m,
                               const size_t n,
                               float*       scale,
                               float*       dynamic_scale,
                               const int    int8_mode,
                               cudaStream_t stream);

    void attentionAddBiasResidualLayerNorm(T*           output,
                                           T*           norm_output,
                                           const T*     input,
                                           const T*     residual1,
                                           const T*     gamma,
                                           const T*     beta,
                                           const T*     bias,
                                           const float  eps,
                                           const size_t m,
                                           const size_t n,
                                           const float* scale_inter,
                                           const float* scale_out,
                                           float*       scale,
                                           float*       dynamic_scale,
                                           const int    int8_mode,
                                           cudaStream_t stream);

    void ffnAddBiasResidualLayerNorm(T*           output,
                                     const T*     input,
                                     const T*     residual1,
                                     const T*     residual2,
                                     const T*     bias,
                                     const T*     gamma,
                                     const T*     beta,
                                     const float  eps,
                                     const size_t m,
                                     const size_t n,
                                     const float* scale_inter,
                                     const float* scale_out,
                                     cudaStream_t stream);

private:
    LayerNormType layernorm_type_;
    NormType norm_type_;
    T alpha_;
    
};
} // namespace fastertransformer
