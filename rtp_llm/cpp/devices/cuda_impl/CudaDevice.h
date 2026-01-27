#pragma once

#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/cuda/cufmha/TRTAttn.h"
#include "rtp_llm/cpp/cuda/cufmha/cufmha.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/cublas/cublas.h"
#include "rtp_llm/cpp/cuda/cuggemm/cuggemm.h"
#include "rtp_llm/cpp/cuda/custom_ar/custom_ar_comm.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/cuda/comm_buffer/comm_buffer.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#ifdef ENABLE_DEEP_EP
#include "rtp_llm/cpp/devices/cuda_impl/DeepEPBuffer.h"
#endif
#include "trt_plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"
#include "trt_plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "trt_plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"

#include <memory>

namespace trt_plugins = tensorrt_llm::plugins;

namespace rtp_llm {

nvinfer1::DataType nvinfer1DtypeConvert(rtp_llm::DataType dtype);

class CudaGemmArguments;

class CudaEvent: public DeviceEvent {
public:
    CudaEvent(cudaStream_t stream);
    ~CudaEvent() override;

    void        synchronize() const override;
    bool        checkReadiness() const override;
    cudaEvent_t getEvent() const {
        return event_;
    }

private:
    cudaEvent_t  event_;
    cudaStream_t stream_;
};

class CudaCommHook: public DeviceHook {
public:
    CudaCommHook(cudaStream_t main_stream, cudaStream_t comm_stream);
    ~CudaCommHook() override;

    void hook_sync() const override;

private:
    cudaEvent_t  hook_event_;
    cudaStream_t main_stream_;
    cudaStream_t comm_stream_;
};

class CudaDevice: public DeviceBase {
public:
    CudaDevice(const DeviceInitParams& params);
    ~CudaDevice();

public:
    void             init() override;
    DeviceProperties getDeviceProperties() override;
    MemoryStatus     getDeviceMemoryStatus() override;
    IAllocator*      getAllocator() override {
        return allocator_.get();
    }
    IAllocator* getHostAllocator() override {
        return host_allocator_.get();
    }

    void             checkError() override;
    void             syncAndCheck() override;
    void             printDebugInfo() override;
    void             syncDeviceStream(DeviceStream stream) override;
    void             syncCommunication(bool timeout = true) override;
    void             syncCommunication(ParallelMode mode, bool timeout = true) override;
    void             overlappedCommBarrier() override;
    DeviceHookPtr    createCommHook() override;
    void             overlappedComputeBarrier() override;
    DevicePrepOutput prepareModelRun(const DevicePrepParams& params) override;
    DeviceEventPtr   createEvent() override;
    DeviceEventPtr   createTorchEvent() override;
    bool             useGroupGemm() const;

private:
    void         checkUseOpenSourceFMHA();
    void         checkUseTrtV1FMHA();
    void         checkUseTrtV2FMHA();
    void         checkUseMultiBlockMode();
    void         checkUseXQA();
    void         checkSupportTrtFp8FMHA();
    bool         useFp8Fmha(const DevicePrepParams& params) const;
    void         initMoeRunner(const DataType compute_type, const DataType weights_type);
    void         initNcclParam(size_t             rank,
                               size_t             world_size,
                               const std::string& ip,
                               size_t             port,
                               const std::string& tp_group_name,
                               NcclParam&         nccl_param);
    void         commBarrier(const NcclParam& nccl_param);
    void         printDeviceMemoryUsage(std::string stage);
    bool         initDeepEPBuffer();
    void         updateCurrentTorchStream() override;
    void         checkUseGroupGemm();
    NcclParam    getNcclParam(ParallelMode mode);
    cudaStream_t getCommStream(ParallelMode mode, bool overlap);
    template<typename QuantType>
    LayernormOutput _layernorm(const LayernormParams& params);
    GreedyOutput    flashinferSampleGreedy(const GreedyParams& params, const BufferPtr& transposed_tokens);
    void processLogits(const GreedyParams& params, const BufferPtr& device_tokens, const BufferPtr& transposed_tokens);

public:
    void setStream(cudaStream_t stream) {
        stream_ = stream;
    }
    cudaStream_t getStream() {
        return stream_;
    }
    cudaStream_t  getStream(DeviceStream stream);
    torch::Device getTorchDevice() override {
        return torch::Device(torch::kCUDA);
    };
    // different from getDeviceProperties
    cudaDeviceProp getDeviceProp() const {
        return device_prop_;
    }
    void profileStart() override;
    void profileStop() override;

public:
    // TODO(zhangjianning.zjn): unify batchCopy and multiCopy
    void                          copy(const CopyParams& params);
    void                          multiMergeCopy(const MultiMergeCopyParams& params) override;
    void                          multiCopy(const MultiCopyParams& params) override;
    void                          batchCopy(const BatchCopyParams& params) override;
    void                          noBlockCopy(const CopyParams& params) override;
    void                          noBlockCopy(const MultiCopyParams& params) override;
    void                          bufMemset(Buffer& buf, int val, DeviceStream stream = DeviceStream::DEFAULT) override;
    TransposeOutput               transpose(const TransposeParams& params) override;
    AddBiasOutput                 addbias(const AddBiasParams& params) override;
    ConvertOutput                 convert(const ConvertParams& params) override;
    SelectOutput                  select(const SelectParams& params) override;
    SplitOutput                   split(const SplitParams& params) override;
    SliceOutput                   slice(const SliceParams& params) override;
    LayernormOutput               layernorm(const LayernormParams& params) override;
    LayernormOutput               layernormWithStride(const LayernormWithStrideParams& params) override;
    QkRmsNormOutput               qkRmsNorm(const QkRmsNormParams& params) override;
    BufferPtr                     gemm(const GemmParams& params) override;
    GroupedGemmOutput             groupedGemm(const GroupedGemmParams& params) override;
    MultiplyOutput                multiply(const MultiplyParams& params) override;
    BufferPtr                     embeddingLookup(const EmbeddingLookupParams& params) override;
    BufferPtr                     activation(const ActivationParams& params) override;
    BufferPtr                     loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    ReduceScatterLoraLinearOutput loraLinearReduceScatter(const LoraLinearReduceScatterParams& params) override;
    void                          chainSpeculativeSampling(const SpeculativeSamplingParams& params) override;
    AllGatherLoraLinearOutput     allGatherloraLinear(const AllGatherLoraLinearParams& params) override;
    BufferPtr                     softmax(const SoftmaxParams& params) override;
    AttentionModuleOutput         contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput         decoderSelfAttention(const AttentionModuleParams& params) override;
    MoeGateSelectOutput           moeGateSelect(const FfnLayerParams& params) override;
    FfnLayerOutput  moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    FfnLayerOutput  moeFfnFp8(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    FfnLayerOutput  moeFfnFp8Contiguous(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    FfnLayerOutput  moeFfnFp8Masked(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    GreedyOutput    sampleGreedy(const GreedyParams& params) override;
    void            broadcast(const BroadcastParams& params) override;
    AllReduceOutput allReduce(const AllReduceParams& params) override;
    void            allGather(const AllGatherParams& params) override;
    AllToAllOutput  allToAll(const AllToAllParams& params) override;
    void            batchSendRecv(const BatchSendRecvParams& params, const ParallelMode& mode) override;
    void            reduceScatter(const ReduceScatterParams& params) override;
    PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params) override;
    BufferPtr              mlaQKVGemm(const AttentionLayerParams& params) override;
    void                   mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) override;
    BeamSearchOutput       sampleBeamSearch(const BeamSearchParams& params) override;
    BufferPtr              quantize(const QuantizeParams& params) override;
    void                   preRun() override;
    bool                   checkNAN(const Buffer& input) override;
    void                   moeGateSelectWithBias(const FfnLayerParams& params,
                                                 BufferPtr             gate,
                                                 BufferPtr             gate_with_bias,
                                                 BufferPtr             expert_scales,
                                                 BufferPtr             expert_for_source_row,
                                                 int                   normalization_mode);
    void                   prepareMoEGate(const FfnLayerParams& params, BufferPtr gate);
    void                   mlaAbsorbAttention(const MlaAttentionModuleParams& params) override;
    void                   mlaContextAttention(const MlaAttentionModuleParams& params) override;
    MoeDispatchOutput      epDispatch(const MoeDispatchParams& params) override;
    MoeCombineOutput       epCombine(const MoeCombineParams& params) override;
    FfnLayerOutput         gatherCombineOutput(const MoeCombineOutput& params) override;

    void QInputBatchMatmulWrapper(torch::Tensor& fused_q_input_t, const MlaAttentionModuleParams& params);
    void DecoderOutputGemmWrapper(torch::Tensor&                  qkv_output_t,
                                  const torch::Tensor&            mla_out_t,
                                  const MlaAttentionModuleParams& params);

    MoeDispatchOutput deepEpDispatch(const MoeDispatchParams& params);
    MoeCombineOutput  deepEpCombine(const MoeCombineParams& params);
    MoeDispatchOutput deepEpLLDispatch(const MoeDispatchParams& params);
    MoeCombineOutput  deepEpLLCombine(const MoeCombineParams& params);
    FfnLayerOutput    deepEpLLMoeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);

    void prepareCommBuffer(const PrepareCommBufferParams& params) override;
    void maskLogits(Buffer& logits, const Buffer& mask) override;
    void sparseMaskLogits(Buffer& logits, const Buffer& batch_idx, const Buffer& mask) override;
    void weightLogits(WeightMaskLogitsParams& params) override;

    void perfRangePush(const std::string& name) const override;
    void perfRangePop() const override;

public:
    ParamsPtr prepareTrtAttn(const AttentionConfigs& configs, const BufferPtr& kv_cache_block_id, int batch_size);

    ParamsPtr prepareTrtAttn(const AttentionConfigs& configs,
                             const BufferPtr&        kv_cache,
                             const BufferPtr&        kv_cache_block_id,
                             int                     batch_size);

    std::shared_ptr<cufmha>
    selectCuFMHARunner(const AttentionConfigs& configs, DataType attn_dtype, bool has_alibi_slopes);

protected:
    DevicePrepOutput prepareModelRunCommon(const DevicePrepParams& params);
    bool             checkSpecDecode(const DevicePrepParams& params, bool skip_no_prefix = true);

    void InvokeSmoothQaunt(const GemmParams& params, const CudaGemmArguments arguments, BufferPtr output);
    void InvokeWeightOnlyGemm(const GemmParams& params, const CudaGemmArguments arguments, BufferPtr output);
    void InvokeGeneralGemm(const GemmParams& params, const CudaGemmArguments arguments, BufferPtr output);
    void InvokeDeepGemm(const GemmParams& params, const CudaGemmArguments arguments, BufferPtr& output);

    void prefillAttention(const AttentionModuleParams& params,
                          KVBlockArray                 kv_block_array,
                          const BufferPtr&             q_no_transpose_output,  // for flashinfer
                          const BufferPtr&             q_output,
                          const BufferPtr&             k_output,
                          const BufferPtr&             v_output,
                          const BufferPtr&             qkv_buf_fp8);

    MoeEpPlanOutput equalEpPlan(const MoeEpPlanParams& params);

    void updateExpertGpuLoads(const MoeConfigs&          moe_conf,
                              const OptionalExpertStats& expert_stats,
                              BufferPtr                  expert_ids) override;

    void balanceExperts(BufferPtr                  expert_ids,
                        const OptionalExpertStats& expert_stats,
                        const MoeConfigs&          moe_conf,
                        const FfnLayerWeights&     weights);

#ifdef ENABLE_DEEP_EP
    size_t initDeepEPLLMaxTokenPerRank(const DeviceInitParams& params);
#endif

protected:
    std::unique_ptr<at::cuda::CUDAStream> torch_default_stream_;
    std::unique_ptr<at::cuda::CUDAStream> torch_comm_stream_;
    cudaStream_t                          stream_;
    cudaStream_t                          eplb_stream_;
    cudaStream_t                          no_block_copy_stream_;
    cudaStream_t                          communication_stream_;

    std::unique_ptr<cublasMMWrapper> cublas_mm_wrapper_;

    std::unique_ptr<trt_plugins::WeightOnlyQuantMatmulPlugin> weight_only_matmul_plugin_;
    std::unique_ptr<trt_plugins::SmoothQuantGemmPlugin>       smooth_quant_plugin_;

    std::unique_ptr<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin> weight_only_groupwise_matmul_plugin_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin>               moe_plugin_;

    FMHAType fmha_type_ = FMHAType::NONE;
    // for speculative decoding, draft model and score model has different config such as kv_head_num
    // here we need separate cufmha_runner for them to avoid frequently setup cufmha runner with different config
    std::vector<std::shared_ptr<cufmha>> cufmha_runner_pool_;
    std::shared_ptr<cufmha>              cufmha_runner_;
    std::unique_ptr<cuggemm>             cuggemm_runner_;
    bool                                 use_multi_block_mode = false;

private:
    std::unique_ptr<IAllocator>                                     allocator_;
    std::unique_ptr<IAllocator>                                     host_allocator_;
    c10::cuda::CUDACachingAllocator::CUDAAllocator*                 origin_torch_cuda_allocator_;
    std::unique_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> managed_torch_cuda_allocator_;

    cublasHandle_t   cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    cudaDeviceProp   device_prop_;

    std::mutex                     cublas_wrapper_mutex_;
    std::unique_ptr<cublasAlgoMap> cublas_algo_map_;

    NcclParam tp_nccl_param_;
    NcclParam dp_tp_nccl_param_;
    NcclParam ffn_tp_nccl_param_;

    std::unique_ptr<CustomAllReduceComm> custom_allreduce_comm_ = nullptr;  // for custom allreduce use

    // BufferPtr will be error when multi stream, tmp hold
    std::unique_ptr<CommBuffer> attn_ag_comm_buffer_       = nullptr;
    std::unique_ptr<CommBuffer> attn_ag_scale_comm_buffer_ = nullptr;
    std::unique_ptr<CommBuffer> attn_rs_comm_buffer_       = nullptr;
    std::unique_ptr<CommBuffer> ffn_ag_comm_buffer_        = nullptr;
    std::unique_ptr<CommBuffer> ffn_ag_scale_comm_buffer_  = nullptr;
    std::unique_ptr<CommBuffer> ffn_rs_comm_buffer_        = nullptr;

#ifdef ENABLE_DEEP_EP
    std::unique_ptr<DeepEPBuffer> deepep_buffer_ = nullptr;  // for deep_ep use
#endif

    std::vector<BufferPtr> moe_hold_host_buffers_;

protected:
    bool use_trtv1_fmha             = false;
    bool use_trtv2_fmha             = false;
    bool use_trtv2_fmha_paged       = false;
    bool use_open_source_fmha       = false;
    bool use_open_source_fmha_paged = false;
    bool use_xqa                    = false;
    bool use_group_gemm             = false;
    bool support_trt_fp8_fmha       = false;
    bool use_fp8_fmha_              = false;

    bool use_stable_scatter_add = false;

    uint32_t ll_num_max_token_per_rank = 0;
    // for local perf
    bool                                        hack_moe_expert_ = false;
    std::shared_ptr<c10::cuda::CUDAStreamGuard> guard_;
};

}  // namespace rtp_llm
