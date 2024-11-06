#pragma once


#include "src/fastertransformer/devices/DeviceOps.h"
#include "src/fastertransformer/devices/DeviceData.h"
#include "src/fastertransformer/devices/BufferManager.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#if ENABLE_BF16
#include <hip/hip_bf16.h>
#endif

#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/rocm/hip_utils.h"
#include "src/fastertransformer/rocm/hipblasMMWrapper.h"
#include "src/fastertransformer/rocm/rocmFmhaWrapper.h"
#include "src/fastertransformer/rocm/quantizePreprocessors.h"
#include "src/fastertransformer/rocm/rocmMoeWrapper.h"
#include "src/fastertransformer/rocm/CKGemmWrapper.h"

#include "torch_hip_allocator.h"
#include "custom_ar_comm.h"

namespace fastertransformer {
    
class ROCmEvent : public DeviceEvent {
public:
    ROCmEvent(hipStream_t stream);
    ~ROCmEvent() override;

    void synchronize() const override;

private:
    hipEvent_t event_;
    hipStream_t stream_;
};

class ROCmDevice: public DeviceBase {
public:
    ROCmDevice(const DeviceInitParams& params);
    ~ROCmDevice();

    void init() override;
    DeviceProperties getDeviceProperties() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return hostAllocator_.get(); }
    void copy(const CopyParams& params) override;
    void noBlockCopy(const CopyParams& params) override;
    TransposeOutput transpose(const TransposeParams& params) override;
    void syncAndCheck() override;
    DevicePrepOutput prepareModelRun(const DevicePrepParams& params) override;
    BufferPtr gemm(const GemmParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    MultiplyOutput multiply(const MultiplyParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    BufferPtr activation(const ActivationParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    FfnLayerOutput moeFfnLayer(const FfnLayerParams& params) override;
    FfnLayerOutput ffnLayer(const FfnLayerParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    void sampleGreedy(const GreedyParams& params) override;
    MemoryStatus getDeviceMemoryStatus() override;
    BufferPtr loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    void syncCommunication(bool timeout = true) override;
    void broadcast(const BroadcastParams& params) override;
    AllReduceOutput allReduce(const AllReduceParams& params) override;
    PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params) override;
    void allGather(const AllGatherParams& params) override;
    void preRun() override { ROCM_CHECK(hipSetDevice(device_id_)); }
    DeviceEventPtr createEvent() override;

    BufferPtr quantize(const QuantizeParams& params) override;
    BufferPtr dequantize(const QuantizeParams& params);
    void      printBuffer(const BufferPtr buffer);

    static torch::Tensor packInt8TensorToPackedInt4(torch::Tensor weight);
    static torch::Tensor preprocessWeightsForMixedGemm(torch::Tensor row_major_quantized_weight, torch::ScalarType quant_type);

public:
    BufferPtr        testVecAdd(const BufferPtr a, const BufferPtr b);
    hipDeviceProp_t* getRocmDeviceProperties() {
        return &rocmDevProp;
    }

private:
    hipDeviceProp_t             rocmDevProp;
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> hostAllocator_;
    c10::hip::HIPCachingAllocator::HIPAllocator *origin_torch_hip_allocator_;

    hipStream_t     stream_ = nullptr;
    hipStream_t     no_block_copy_stream_;
    hipStream_t     assist_stream_  = nullptr;
    hipStream_t     current_stream_ = nullptr;
    hipDeviceProp_t device_prop_;

    BufferPtr curandstate_buf_; // for sampler use.

    rocm::hipblasMMWrapper* hipblasMMWrapperPtr() const {
        return hipblas_mm_wrapper_.get();
    }

    hipblasHandle_t   hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;

    std::unique_ptr<rocm::hipblasMMWrapper> hipblas_mm_wrapper_;

    // fmha
    std::unique_ptr<rocmFmhaWrapper>      fmha_runner_;
    bool use_openSource_fmha    = true;

    NcclParam nccl_param_;

    //moe
    std::unique_ptr<rocmMoeWrapper> moe_runner_;

    // for custom allreduce use
    std::unique_ptr<CustomAllReduceComm> custom_allreduce_comm_ = nullptr;

    //CK gemm
    std::unique_ptr<CKGemmWrapper> ck_gemm_runner_;


protected:
    bool use_multi_block_mode       = true;
};

}  // namespace fastertransformer
