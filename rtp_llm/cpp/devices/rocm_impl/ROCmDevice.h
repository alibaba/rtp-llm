#pragma once


#include "rtp_llm/cpp/devices/DeviceOps.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include "rtp_llm/cpp/devices/BufferManager.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#if ENABLE_BF16
#include <hip/hip_bf16.h>
#endif

#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/rocm/hip_utils.h"
#include "rtp_llm/cpp/rocm/hipblasMMWrapper.h"
#include "rtp_llm/cpp/rocm/rocmFmhaWrapper.h"
#include "rtp_llm/cpp/rocm/quantizePreprocessors.h"
#include "rtp_llm/cpp/rocm/rocmMoeWrapper.h"
#include "rtp_llm/cpp/rocm/rocmCKGemmWrapper.h"

#include "torch_hip_allocator.h"
#include "custom_ar_comm.h"

namespace rtp_llm {

    struct FlashInferAttnParams {
        BufferPtr float_workspace;
        BufferPtr int_workspace;
        BufferPtr int_host_workspace;
    
        BufferPtr batch_indice_host;
        BufferPtr positions_host;
        BufferPtr kvlen_host;
        BufferPtr paged_kv_last_page_len_host;
        BufferPtr paged_kv_last_page_len_1_host;
        BufferPtr page_indice_host;
    
        BufferPtr batch_indice;
        BufferPtr positions;
        BufferPtr paged_kv_last_page_len; // w/o current
        BufferPtr paged_kv_last_page_len_1; // w current
    
        BufferPtr qo_indptr;
        BufferPtr qo_indptr_host;
        BufferPtr page_indptr;
        BufferPtr page_indptr_host;
        BufferPtr page_indice;
    
        torch::Tensor float_workspace_t;
        torch::Tensor int_workspace_t;
        torch::Tensor int_host_workspace_t;
        torch::Tensor batch_indice_t;
        torch::Tensor positions_t;
        torch::Tensor paged_kv_last_page_len_t;
        torch::Tensor paged_kv_last_page_len_1_t;
    
        torch::Tensor qo_indptr_t;
        torch::Tensor qo_indptr_host_t;
        torch::Tensor page_indptr_t;
        torch::Tensor page_indptr_host_t;
        torch::Tensor kvlen_host_t;
        torch::Tensor page_indice_t;
        // for flashmla only
        BufferPtr kv_cache_block_id;
        BufferPtr kvlen;
    
        torch::Tensor kv_cache_block_id_t;
        torch::Tensor kvlen_t;
    
        std::vector<torch::Tensor> flash_mla_plan;
    
        bool decode = true;
        torch::Tensor plan;
    
        static ParamsPtr prepareDecodeFlashInferAttnParams(
            rtp_llm::DeviceBase *device,
            const rtp_llm::AttentionConfigs &attn_configs,
            const BufferPtr &sequence_lengths_host,
            const BufferPtr &input_lengths_host,
            const BufferPtr &kv_cache_block_id_host,
            DataType dtype);
    
        static ParamsPtr preparePrefillFlashInferAttnParams(
            rtp_llm::DeviceBase *device,
            const rtp_llm::AttentionConfigs &attn_configs,
            const BufferPtr &prefix_lengths_host,
            const BufferPtr &sequence_lengths_host,
            const BufferPtr &input_lengths_host,
            const BufferPtr &kv_cache_block_id_host,
            DataType dtype);
    };

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
    void bufMemset(Buffer& buf, int val, DeviceStream stream = DeviceStream::DEFAULT) override;
    TransposeOutput transpose(const TransposeParams& params) override;
    void checkError() override;
    void syncAndCheck() override;
    DevicePrepOutput prepareModelRun(const DevicePrepParams& params) override;
    BufferPtr gemm(const GemmParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    MultiplyOutput multiply(const MultiplyParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    LayernormOutput layernormWithStride(const LayernormWithStrideParams& params) override;
    BufferPtr activation(const ActivationParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput mlaContextAttention(const MlaAttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    FfnLayerOutput moeFfnLayer(const FfnLayerParams& params) override;
    FfnLayerOutput ffnLayer(const FfnLayerParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    GreedyOutput sampleGreedy(const GreedyParams& params) override;
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
    static torch::Tensor preprocessWeightsForMixedGemm(torch::Tensor row_major_quantized_weight, torch::ScalarType quant_type, const std::string &arch);
    void QInputBatchMatmulWrapper(torch::Tensor& fused_q_input_t, const MlaAttentionModuleParams& params);
    void DecoderOutputGemmWrapper(torch::Tensor& qkv_output_t, const torch::Tensor& mla_out_t, const MlaAttentionModuleParams& params);

    void mlaAbsorbAttention(const MlaAttentionModuleParams& params);
    void mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) override;
    SliceOutput slice(const SliceParams& params) override;

protected:
    void InvokeROCmDeepGemm(const GemmParams& params,
                            BufferPtr         output);

public:
    hipStream_t getStream() {return stream_;}
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
    std::unique_ptr<rocmCKGemmWrapper> ck_gemm_runner_;


protected:
    bool use_multi_block_mode       = true;
};

}  // namespace rtp_llm
