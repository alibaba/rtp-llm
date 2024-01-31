#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/cublas/cublas.h"

namespace fastertransformer {

class CudaDevice : public DeviceBase {
public:
    CudaDevice();
    ~CudaDevice();

public:
    int getDeviceId() const { return device_id_; }
    std::string type() const override { return "cuda"; }

public: // Ops method
    void layernorm(LayernormParams& params);
    void gemm(GemmParams& params);
    void contextAttention(AttentionModuleParams& params);
    void decoderSelfAttention(AttentionModuleParams& params);
    void attentionLayer(AttentionLayerParams& params);
    void ffnLayer(FfnLayerParams& params);
    void sampleTopP(SamplerParams& params);
    void sampleTopK(SamplerParams& params);
    void broadcast(BroadcastParams& params);
    void allReduceSum(AllReduceParams& params);

private:
    int device_id_;
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    cudaDeviceProp device_prop_;

    std::mutex cublas_wrapper_mutex_;
    std::unique_ptr<cublasAlgoMap> cublas_algo_map_;
    std::unique_ptr<cublasMMWrapper> cublas_mm_wrapper_;
};

} // namespace fastertransformer

