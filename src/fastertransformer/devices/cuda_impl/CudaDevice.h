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
    std::string type() const override { return "cuda"; }
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return host_allocator_.get(); }
    int getDeviceId() const { return device_id_; }

public:
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
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> host_allocator_;
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

