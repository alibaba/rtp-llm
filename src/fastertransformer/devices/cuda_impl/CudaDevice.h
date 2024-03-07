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
    void copy(const CopyParams& params);
    void layernorm(const LayernormParams& params);
    void gemm(const GemmParams& params);
    void groupedGemm(const GroupedGemmParams& params);
    void embeddingLookup(const EmbeddingLookupParams& params);
    void contextAttention(const AttentionModuleParams& params);
    void decoderSelfAttention(const AttentionModuleParams& params);
    void attentionLayer(const AttentionLayerParams& params);
    void ffnLayer(const FfnLayerParams& params);
    void sampleTopP(const SamplerParams& params);
    void sampleTopK(const SamplerParams& params);
    void broadcast(const BroadcastParams& params);
    void allReduceSum(const AllReduceParams& params);

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

