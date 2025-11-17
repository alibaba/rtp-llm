#pragma once

#include "rtp_llm/cpp/devices/DeviceOps.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include "rtp_llm/cpp/devices/BufferManager.h"
#include "rtp_llm/cpp/devices/rocm_impl/NativeHipGraphRunner.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#if ENABLE_BF16
#include <hip/hip_bf16.h>
#endif

#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/rocm/hipblasMMWrapper.h"
#include "rtp_llm/cpp/rocm/rocmFmhaWrapper.h"
#include "rtp_llm/cpp/rocm/quantizePreprocessors.h"
// #include "rtp_llm/cpp/rocm/rocmMoeWrapper.h"
#include "rtp_llm/cpp/rocm/rocmCKGemmWrapper.h"
#include "rtp_llm/cpp/rocm/rocmCKW8A8GeluGemmWrapper.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/rocm/custom_ar/custom_ar_comm.h"
#ifdef ENABLE_DEEP_EP
#include "rtp_llm/cpp/devices/rocm_impl/DeepEPBuffer.h"
#endif

#include "torch_hip_allocator.h"

namespace rtp_llm {

struct AiterAttnParams {
    BufferPtr     sequence_lengths;
    BufferPtr     sequence_lengths_host;
    torch::Tensor sequence_lengths_t;

    KVBlockArray kv_block_array;
    BufferPtr    kv_cache_offset;

    static ParamsPtr prepareDecodeAiterAttnParams(rtp_llm::DeviceBase* device,
                                              const BufferPtr& sequence_lengths_host,
                                              const AttentionConfigs& configs,
                                              const int kv_cache_offset,
                                              const BufferPtr& kv_cache_block_id);
};

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
    BufferPtr paged_kv_last_page_len;    // w/o current
    BufferPtr paged_kv_last_page_len_1;  // w current

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

    bool          decode = true;
    torch::Tensor plan;

    static ParamsPtr prepareDecodeFlashInferAttnParams(rtp_llm::DeviceBase*             device,
                                                       const rtp_llm::AttentionConfigs& attn_configs,
                                                       const BufferPtr&                 sequence_lengths_host,
                                                       const BufferPtr&                 input_lengths_host,
                                                       const BufferPtr&                 kv_cache_block_id_host,
                                                       DataType                         dtype);

    static ParamsPtr preparePrefillFlashInferAttnParams(rtp_llm::DeviceBase*             device,
                                                        const rtp_llm::AttentionConfigs& attn_configs,
                                                        const BufferPtr&                 prefix_lengths_host,
                                                        const BufferPtr&                 sequence_lengths_host,
                                                        const BufferPtr&                 input_lengths_host,
                                                        const BufferPtr&                 kv_cache_block_id_host,
                                                        DataType                         dtype);
};

class ROCmEvent: public DeviceEvent {
public:
    ROCmEvent(hipStream_t stream);
    ~ROCmEvent() override;

    void synchronize() const override;
    bool checkReadiness() const override;

private:
    hipEvent_t  event_;
    hipStream_t stream_;
};

class ROCmCommHook: public DeviceHook {
public:
    ROCmCommHook(hipStream_t main_stream, hipStream_t comm_stream);
    ~ROCmCommHook() override;

    void hook_sync() const override;

private:
    hipEvent_t  hook_event_;
    hipStream_t main_stream_;
    hipStream_t comm_stream_;
};

struct CKAttn {
    KVBlockArray kv_block_array;
    BufferPtr    kv_cache_offset;
    BufferPtr    kv_cache_offset_h;

    torch::Tensor kv_cache_block_id_device;

    torch::Tensor prefix_lengths;
    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor input_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor padding_offset;
    int           max_seq_len;
    bool          decode_plan;

    DataType attn_type;

    static void setKvCache(KVBlockArray& kv_block_array, const KvCacheInfo& kv_cache) {
        kv_block_array.mPrimaryPoolPtr = kv_cache.k_cache_buffer->data();
        if (kv_cache.k_scale_buffer) {
            kv_block_array.scale = kv_cache.k_scale_buffer->data();
        }
    }
};

using CKAttnPtr = std::shared_ptr<CKAttn>;

class ROCmDevice: public DeviceBase {
public:
    ROCmDevice(const DeviceInitParams& params);
    ~ROCmDevice();

    void             init() override;
    DeviceProperties getDeviceProperties() override;
    IAllocator*      getAllocator() override {
        return allocator_.get();
    }
    IAllocator* getHostAllocator() override {
        return hostAllocator_.get();
    }
    void                   copy(const CopyParams& params) override;
    void                   noBlockCopy(const CopyParams& params) override;
    void                   bufMemset(Buffer& buf, int val, DeviceStream stream = DeviceStream::DEFAULT) override;
    TransposeOutput        transpose(const TransposeParams& params) override;
    void                   checkError() override;
    void                   syncAndCheck() override;
    void                   overlappedCommBarrier() override;
    DeviceHookPtr          createCommHook() override;
    void                   overlappedComputeBarrier() override;
    DevicePrepOutput       prepareModelRun(const DevicePrepParams& params) override;
    BufferPtr              gemm(const GemmParams& params) override;
    SelectOutput           select(const SelectParams& params) override;
    MultiplyOutput         multiply(const MultiplyParams& params) override;
    BufferPtr              embeddingLookup(const EmbeddingLookupParams& params) override;
    LayernormOutput        layernorm(const LayernormParams& params) override;
    LayernormOutput        layernormWithStride(const LayernormWithStrideParams& params) override;
    QkRmsNormOutput        qkRmsNorm(const QkRmsNormParams& params) override;
    BufferPtr              activation(const ActivationParams& params) override;
    AttentionModuleOutput  contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput  mlaContextAttention(const MlaAttentionModuleParams& params) override;
    AttentionModuleOutput  decoderSelfAttention(const AttentionModuleParams& params) override;
    MoeGateSelectOutput    moeGateSelect(const FfnLayerParams& params) override;
    FfnLayerOutput         moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    FfnLayerOutput         ffnLayer(const FfnLayerParams& params) override;
    MoeDispatchOutput      epDispatch(const MoeDispatchParams& params) override;
    MoeCombineOutput       epCombine(const MoeCombineParams& params) override;
    FfnLayerOutput         gatherCombineOutput(const MoeCombineOutput& params) override;
    MoeDispatchOutput deepEpDispatch(const MoeDispatchParams& params);
    MoeCombineOutput deepEpCombine(const MoeCombineParams& params);
    MoeDispatchOutput deepEpLLDispatch(const MoeDispatchParams& params);
    MoeCombineOutput deepEpLLCombine(const MoeCombineParams& params);
    FfnLayerOutput deepEpLLMoeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    BufferPtr              softmax(const SoftmaxParams& params) override;
    GreedyOutput           sampleGreedy(const GreedyParams& params) override;
    MemoryStatus           getDeviceMemoryStatus() override;
    BufferPtr              loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    BufferPtr              mhaQKVGemm(const AttentionLayerParams& params) override;
    void                   syncCommunication(bool timeout = true) override;
    void                   broadcast(const BroadcastParams& params) override;
    AllReduceOutput        allReduce(const AllReduceParams& params) override;
    PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params) override;
    void                   allGather(const AllGatherParams& params) override;
    SplitOutput            split(const SplitParams& params) override;
    AllToAllOutput         allToAll(const AllToAllParams& params) override;

    void preRun() override {
        ROCM_CHECK(hipSetDevice(device_id_));
    }
    DeviceEventPtr createEvent() override;

    BufferPtr quantize(const QuantizeParams& params) override;
    BufferPtr dequantize(const QuantizeParams& params);
    void      printBuffer(const BufferPtr buffer);

    void QInputBatchMatmulWrapper(torch::Tensor& fused_q_input_t, const MlaAttentionModuleParams& params);
    void DecoderOutputGemmWrapper(torch::Tensor&                  qkv_output_t,
                                  const torch::Tensor&            mla_out_t,
                                  const MlaAttentionModuleParams& params);

    void         mlaAbsorbAttention(const MlaAttentionModuleParams& params) override;
    void         mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) override;
    SliceOutput  slice(const SliceParams& params) override;
    KVBlockArray getKVBlockArray(const AttentionModuleParams& params,
                                 const Buffer&                kv_cache_offset_pointers,
                                 int                          batch_size,
                                 bool                         use_fp8_fmha,
                                 bool                         use_offset_array = false);

    std::shared_ptr<NativeGraphRunner> getNativeGraphRunner() override {
        return std::make_shared<NativeHipGraphRunner<GptModelInputs, GptModelOutputs>>(this);
    }
    void registerARGraphBuffers() {
        if (custom_allreduce_comm_)
            custom_allreduce_comm_->registerGraphBuffers();
    }

protected:
    void InvokeROCmDeepGemm(const GemmParams& params, BufferPtr output);
    void InvokeROCmPTPCGemm(const GemmParams& params, BufferPtr output);
    void HipblasltPTPCGemm(const GemmParams& params, BufferPtr output);
    void InvokeROCmDeepGemmWi8Ai8(const GemmParams& params, BufferPtr output);
    // void prepareCommBuffer(const PrepareCommBufferParams& params) override;

    void updateExpertGpuLoads(const MoeConfigs&          moe_conf,
                              const OptionalExpertStats& expert_stats,
                              BufferPtr                  expert_ids) override;

    void balanceExperts(BufferPtr                  expert_ids,
                        const OptionalExpertStats& expert_stats,
                        const MoeConfigs&          moe_conf,
                        const FfnLayerWeights&     weights);

public:
    void setStream(hipStream_t stream) {
        current_stream_ = stream;
        stream_         = stream;
        hipblas_mm_wrapper_->setStream(stream);
    }
    hipStream_t getStream(DeviceStream stream);
    hipStream_t getStream() {
        return stream_;
    }
    hipDeviceProp_t* getRocmDeviceProperties() {
        return &rocmDevProp;
    }
    ParamsPtr PrepareCKAttn(const AttentionConfigs& configs,
                            int                     kv_block_offset,
                            const BufferPtr&        kv_cache_block_id,
                            int                     batch_size);
    void      maskLogits(Buffer& logits, const Buffer& mask) override;

private:
    hipDeviceProp_t                              rocmDevProp;
    std::unique_ptr<IAllocator>                  allocator_;
    std::unique_ptr<IAllocator>                  hostAllocator_;
    c10::hip::HIPCachingAllocator::HIPAllocator* origin_torch_hip_allocator_;

    std::unique_ptr<at::hip::HIPStreamMasqueradingAsCUDA> torch_default_stream_;
    std::unique_ptr<at::hip::HIPStreamMasqueradingAsCUDA> torch_comm_stream_;
    hipStream_t     stream_ = nullptr;
    hipStream_t     no_block_copy_stream_;
    hipStream_t     communication_stream_;
    hipStream_t     assist_stream_  = nullptr;
    hipStream_t     current_stream_ = nullptr;
    hipDeviceProp_t device_prop_;

    BufferPtr curandstate_buf_;  // for sampler use.

    rocm::hipblasMMWrapper* hipblasMMWrapperPtr() const {
        return hipblas_mm_wrapper_.get();
    }

    hipblasHandle_t   hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;

    std::unique_ptr<rocm::hipblasMMWrapper> hipblas_mm_wrapper_;

    // fmha
    std::unique_ptr<rocmFmhaWrapper> fmha_runner_;
    bool                             use_openSource_fmha = true;

    NcclParam tp_nccl_param_;
    NcclParam dp_nccl_param_;
    NcclParam dp_tp_nccl_param_;
    NcclParam ffn_tp_nccl_param_;

    void      initNcclParam(size_t             rank,
                            size_t             world_size,
                            const std::string& ip,
                            size_t             port,
                            const std::string& tp_group_name,
                            NcclParam&         nccl_param);
    NcclParam getNcclParam(ParallelMode mode);
    // moe
    // std::unique_ptr<rocmMoeWrapper> moe_runner_;
    bool initDeepEPBuffer();
#ifdef ENABLE_DEEP_EP
    std::unique_ptr<DeepEPBuffer> deepep_buffer_ = nullptr;  // for deep_ep use
#endif
    uint32_t ll_num_max_token_per_rank = 0;

    // for custom allreduce use
    std::unique_ptr<CustomAllReduceComm> custom_allreduce_comm_ = nullptr;

    // BufferPtr will be error when multi stream, tmp hold
    // std::vector<BufferPtr> overlap_hold_buffers_;
    // std::unique_ptr<CommBuffer> attn_ag_comm_buffer_ = nullptr;
    // std::unique_ptr<CommBuffer> attn_ag_scale_comm_buffer_ = nullptr;
    // std::unique_ptr<CommBuffer> attn_rs_comm_buffer_ = nullptr;
    // std::unique_ptr<CommBuffer> ffn_ag_comm_buffer_ = nullptr;
    // std::unique_ptr<CommBuffer> ffn_ag_scale_comm_buffer_ = nullptr;
    // std::unique_ptr<CommBuffer> ffn_rs_comm_buffer_ = nullptr;

    // CK gemm
    std::unique_ptr<rocmCKGemmWrapper> ck_gemm_runner_;

    // CK W8A8 Gelu gemm
    std::unique_ptr<rocmCKW8A8GeluGemmWrapper> ck_w8a8_gelu_gemm_runner_;

protected:
    bool use_multi_block_mode = false;
};

}  // namespace rtp_llm
