#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"
#include "src/fastertransformer/devices/DeviceData.h"
#include "src/fastertransformer/devices/BufferManager.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#if ENABLE_BF16
#include <hip/hip_bf16.h>
#endif

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/rocm/hip_utils.h"
#include "src/fastertransformer/rocm/hipblasMMWrapper.h"
#include "src/fastertransformer/rocm/rocmFmhaWrapper.h"

namespace fastertransformer {

class ROCmDevice: public DeviceBase {
public:
    ROCmDevice(const DeviceInitParams& params);
    ~ROCmDevice();

    void init() override;
    DeviceProperties getDeviceProperties() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return hostAllocator_.get(); }
    void copy(const CopyParams& params) override;
    TransposeOutput transpose(const TransposeParams& params) override;
    void syncAndCheck() override;
    BufferPtr gemm(const GemmParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    void activation(const ActivationParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    void sampleGreedy(const GreedyParams& params);
    DeviceStatus getDeviceStatus() override;

public:
    BufferPtr        testVecAdd(const BufferPtr a, const BufferPtr b);
    hipDeviceProp_t* getRocmDeviceProperties() {
        return &rocmDevProp;
    }
    
private:
    hipDeviceProp_t             rocmDevProp;
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> hostAllocator_;
    
    hipStream_t                 stream_ = nullptr;
    hipDeviceProp_t device_prop_;

    BufferPtr curandstate_buf_; // for sampler use.

    rocm::hipblasMMWrapper* hipblasMMWrapperPtr() const {
        return hipblas_mm_wrapper_.get();
    }

    std::mutex        hipblas_wrapper_mutex_;
    hipblasHandle_t   hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;

    std::unique_ptr<rocm::hipblasAlgoMap>   hipblas_algo_map_;
    std::unique_ptr<rocm::hipblasMMWrapper> hipblas_mm_wrapper_;

    // fmha
    std::unique_ptr<rocmFmhaWrapper>      fmha_runner_;
    bool use_openSource_fmha    = true;
};

}  // namespace fastertransformer
