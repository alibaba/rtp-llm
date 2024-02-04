#include "src/fastertransformer/layers/GemmRunner.h"
#include <fstream>
namespace fastertransformer {



template<typename T>
void GemmRunner<T>::Gemm(int                      m,
                         int                      n,
                         int                      k,
                         const T*                 input,
                         const DenseWeight<T, T>* weight,
                         T*                       output,
                         int                      int8_mode,
                         bool                     use_sparse,
                         char*                    mixed_gemm_workspace,
                         size_t                   mixed_gemm_ws_bytes,
                         int                      m_padded) {
    // input: [m, k]
    // weight: [k, n]
    // output: [m, n]

    if (use_sparse) {
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, weight->sp_kernel, input, output);
#endif
    } else {
        if (int8_mode == 1) {
            if (m < SMALL_M_FAST_PATH && weight_only_cuda_kernel_enabled_ && k <= 20480 && n <= 25600) {
#if defined(USE_WEIGHT_ONLY) && USE_WEIGHT_ONLY == 1
                // PUSH_RANGE(stream_, fmtstr("weight only batched gemv: [%d,%d,%d]", m, n, k));
                fastertransformer::kernels::WeightOnlyActivationType weight_only_act_type;

                if (std::is_same<T, half>::value) {
                    weight_only_act_type = fastertransformer::kernels::WeightOnlyActivationType::FP16;
                }
#ifdef ENABLE_BF16
                else if (std::is_same<T, __nv_bfloat16>::value) {
                    weight_only_act_type = fastertransformer::kernels::WeightOnlyActivationType::BF16;
                }
#endif
                else {
                    FT_LOG_ERROR("weight only batched gemv only support half and bf16");
                }
                fastertransformer::kernels::WeightOnlyParams weight_only_batched_gemv_params{
                    reinterpret_cast<const uint8_t*>(weight->int8_kernel),
                    reinterpret_cast<const void*>(weight->weight_only_quant_scale),
                    nullptr,
                    reinterpret_cast<const void*>(input),
                    nullptr,
                    reinterpret_cast<void*>(output),
                    m,
                    n,
                    k,
                    0,
                    fastertransformer::kernels::WeightOnlyQuantType::Int8b,
                    fastertransformer::kernels::WeightOnlyType::PerChannel,
                    fastertransformer::kernels::WeightOnlyActivationFunctionType::Identity,
                    weight_only_act_type};
                fastertransformer::kernels::weight_only_batched_gemv_launcher(weight_only_batched_gemv_params, stream_);
                // POP_RANGE;
#endif
            } else {
                // Otherwise, let FT handle activation
                // PUSH_RANGE(stream_, fmtstr("weight_only_int8_fc gemm: [%d,%d,%d]", m, n, k));
                weight_only_int8_fc_runner_->gemm(input,
                                                  reinterpret_cast<const uint8_t*>(weight->int8_kernel),
                                                  weight->weight_only_quant_scale,
                                                  output,
                                                  m,
                                                  n,
                                                  k,
                                                  mixed_gemm_workspace,
                                                  mixed_gemm_ws_bytes,
                                                  stream_);
                // POP_RANGE;
            }
        } else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weight->kernel, n, input, k, output, n);
            sync_check_cuda_error();
        }
    }
}



template class GemmRunner<float>;
template class GemmRunner<half>;
#ifdef ENABLE_BF16
template class GemmRunner<__nv_bfloat16>;
#endif
}  // namespace fastertransformer
