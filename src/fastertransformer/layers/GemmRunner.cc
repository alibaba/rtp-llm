#include "src/fastertransformer/layers/GemmRunner.h"
#include <fstream>
namespace fastertransformer {

template<typename T>
void GemmRunner<T>::allocateWorkspace(size_t s) {
    workspace_ = (char*)allocator_->reMalloc(workspace_, sizeof(char)*s, true);
}
template<typename T>
void GemmRunner<T>::freeBuffer() {
    allocator_->free((void**)&workspace_);
}

template<typename T>
void GemmRunner<T>::Gemm(int m, int n, int k, const void* inputs, const DenseWeight<T, T>* weights, T* outputs, const float* scale_tokens) {
    // input: [m, k]
    // weight: [k, n]
    // output: [m, n]

    if (quant_algo_.int8Mode()) {
        FT_CHECK_WITH_INFO(weights->int8_kernel, "weights is needed in weight only int8 quant");
        FT_CHECK_WITH_INFO(weights->weight_only_quant_scale,
                           "weight_only_quant_scale is needed in weight only int8 quant");
        size_t ws_size = weight_only_matmul_plguin_->getWorkspaceSize(m, n, k);
        allocateWorkspace(ws_size);
        weight_only_matmul_plguin_->enqueue(reinterpret_cast<const void*>(inputs),
                                            reinterpret_cast<const void*>(weights->int8_kernel),
                                            reinterpret_cast<const void*>(weights->weight_only_quant_scale),
                                            reinterpret_cast<void*>(outputs),
                                            workspace_,
                                            m,
                                            n,
                                            k,
                                            stream_);
        sync_check_cuda_error();
    } else if (quant_algo_.smoothQuantInt8()) {
        FT_CHECK_WITH_INFO(weights->int8_kernel, "weights is needed in smooth quant");
        FT_CHECK_WITH_INFO(weights->scale, "weight scales is needed in smooth quant");
        FT_CHECK_WITH_INFO(scale_tokens, "act scales is needed in smooth quant");
        const size_t ws_size = smooth_quant_plugin_->getWorkspaceSize(m, n, k);
        allocateWorkspace(ws_size);
        smooth_quant_plugin_->enqueue(reinterpret_cast<const void*>(inputs),
                                      reinterpret_cast<const void*>(weights->int8_kernel),
                                      weights->scale,  // const float*
                                      scale_tokens,
                                      reinterpret_cast<void*>(outputs),
                                      workspace_,
                                      m,
                                      n,
                                      k,
                                      stream_);
        sync_check_cuda_error();

    } else if (quant_algo_.int4Mode() == true) {
        FT_CHECK_WITH_INFO(weights->int4_kernel, "weights is needed in weight only int4 quant");
        FT_CHECK_WITH_INFO(weights->weight_only_quant_scale,
                           "weight_only_quant_scale is needed in weight only int4 quant");
        size_t ws_size = weight_only_groupwise_matmul_plguin_->getWorkspaceSize(m, n, k);
        allocateWorkspace(ws_size);
        weight_only_groupwise_matmul_plguin_->enqueue(reinterpret_cast<const void*>(inputs),
                                                      reinterpret_cast<const void*>(weights->int4_kernel),
                                                      reinterpret_cast<const void*>(weights->weight_only_quant_scale),
                                                      reinterpret_cast<const void*>(weights->int4_zeros),
                                                      nullptr,
                                                      reinterpret_cast<void*>(outputs),
                                                      reinterpret_cast<void*>(workspace_),
                                                      m,
                                                      n,
                                                      k,
                                                      stream_);
        sync_check_cuda_error();
    } else {
        cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->kernel, n, reinterpret_cast<const T*>(inputs), k, outputs, n);
        sync_check_cuda_error();
    }
    freeBuffer();
}

template<typename T>
void GemmRunner<T>::GemmWithBias(int m, int n, int k, const void* inputs, const DenseWeight<T, T>* weights, T* outputs) {
    // input: [m, k]
    // weight: [k, n]
    // bias: [n]
    // output: [m, n]
    cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weights->kernel, n, reinterpret_cast<const T*>(inputs), k, weights->bias, outputs, n);
}

template class GemmRunner<float>;
template class GemmRunner<half>;
#ifdef ENABLE_BF16
template class GemmRunner<__nv_bfloat16>;
#endif
}  // namespace fastertransformer
