// Copyright 2026 Tencent

#ifndef RTP_LLM_MODELS_PY_BINDINGS_CUDA_KERNELS_HY4_IHC_H_
#define RTP_LLM_MODELS_PY_BINDINGS_CUDA_KERNELS_HY4_IHC_H_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {
namespace hy4_ihc {

void fuse_ihc_pre_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                        const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                        const float* hc_base_ptr, int num_batch, int hc_mult, int hidden_dim,
                        float norm_eps, float hc_eps, float magnitude, cudaStream_t stream,
                        const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f,
                        bool cast_bfloat_for_norm = false, bool use_pdl = true);

void fuse_ihc_post_pre_async(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                             float magnitude, float* scratch_ptr, cudaStream_t stream,
                             const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f,
                             bool cast_bfloat_for_norm = false, bool use_pdl = true);

size_t ihc_post_pre_scratch_floats(int num_batch, int hc_mult, int hidden_dim);

void fuse_ihc_post_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                         const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                         int hc_mult, int hidden_dim, cudaStream_t stream, bool use_pdl = true);

void fuse_ihc_head_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const float* w_ptr,
                         const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                         int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                         cudaStream_t stream, const __nv_bfloat16* rms_weight_ptr = nullptr,
                         float rms_eps = 0.f, bool cast_bfloat_for_norm = false,
                         bool use_pdl = true);

}  // namespace hy4_ihc
}  // namespace rtp_llm
#endif  // RTP_LLM_MODELS_PY_BINDINGS_CUDA_KERNELS_HY4_IHC_H_
