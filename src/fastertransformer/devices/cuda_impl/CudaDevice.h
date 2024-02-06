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
    OpStatus layernorm(LayernormParams& params);
    OpStatus gemm(GemmParams& params);
    OpStatus groupedGemm(GroupedGemmParams& params);
    OpStatus contextAttention(AttentionModuleParams& params);
    OpStatus decoderSelfAttention(AttentionModuleParams& params);
    OpStatus attentionLayer(AttentionLayerParams& params);
    OpStatus ffnLayer(FfnLayerParams& params);
    OpStatus sampleTopP(SamplerParams& params);
    OpStatus sampleTopK(SamplerParams& params);
    OpStatus broadcast(BroadcastParams& params);
    OpStatus allReduceSum(AllReduceParams& params);

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

