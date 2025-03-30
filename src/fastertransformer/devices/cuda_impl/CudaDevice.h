#pragma once

#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/cuda/cufmha/cufmha.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/cuggemm/cuggemm.h"
#include "src/fastertransformer/cuda/custom_ar/custom_ar_comm.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "trt_plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"
#include "trt_plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "trt_plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include <memory>

namespace trt_plugins = tensorrt_llm::plugins;

namespace fastertransformer {

enum class FMHAType {
    NONE,
    PAGED_TRT_V2,
    TRT_V2,
    PAGED_OPEN_SOURCE,
    OPEN_SOURCE,
    TRT_V1
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

    // including both decode and prefill, for mla
    BufferPtr total_batch_indices_host;
    BufferPtr total_positions_host;
    BufferPtr total_page_indices_host;
    BufferPtr total_page_indptr_host;
    BufferPtr total_kv_last_page_len_1_host;

    BufferPtr total_batch_indices;
    BufferPtr total_positions;
    BufferPtr total_page_indices;
    BufferPtr total_page_indptr;
    BufferPtr total_kv_last_page_len_1;

    torch::Tensor total_batch_indices_t;
    torch::Tensor total_positions_t;
    torch::Tensor total_page_indices_t;
    torch::Tensor total_page_indptr_t;
    torch::Tensor total_kv_last_page_len_1_t;

    // for flashmla only
    BufferPtr kv_cache_block_id;
    BufferPtr kvlen;

    torch::Tensor kv_cache_block_id_t;
    torch::Tensor kvlen_t;

    std::vector<torch::Tensor> flash_mla_plan;

    bool decode = true;
    torch::Tensor plan;

    static FlashInferAttnParamsPtr prepareFlashInferAttnParams(
            fastertransformer::DeviceBase *device,
            const fastertransformer::AttentionConfigs &attn_configs,
            const BufferPtr &sequence_lengths_host,
            const BufferPtr &input_lengths_host,
            const BufferPtr &kv_cache_block_id_host,
            DataType dtype);

};

nvinfer1::DataType nvinfer1DtypeConvert(fastertransformer::DataType dtype);

class CudaGemmArguments;

class CudaEvent : public DeviceEvent {
public:
    CudaEvent(cudaStream_t stream);
    ~CudaEvent() override;

    void synchronize() const override;

private:
    cudaEvent_t event_;
    cudaStream_t stream_;
};

class CudaCommHook : public DeviceHook {
public:
    CudaCommHook(cudaStream_t main_stream, cudaStream_t comm_stream);
    ~CudaCommHook() override;

    void hook_sync() const override;

private:
    cudaEvent_t hook_event_;
    cudaStream_t main_stream_;
    cudaStream_t comm_stream_;
};

class CudaDevice : public DeviceBase {
public:
    CudaDevice(const DeviceInitParams& params);
    ~CudaDevice();

public:
    void init() override;
    DeviceProperties getDeviceProperties() override;
    MemoryStatus getDeviceMemoryStatus() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return host_allocator_.get(); }

    void syncAndCheck() override;
    void syncCommunication(bool timeout = true) override;
    void overlappedCommBarrier() override;
    DeviceHookPtr createCommHook() override;
    DevicePrepOutput prepareModelRun(const DevicePrepParams& params) override;
    DeviceEventPtr createEvent() override;
    bool useGroupGemm() const;

private:
    void checkUseOpenSourceFMHA();
    void checkUseTrtV1FMHA();
    void checkUseTrtV2FMHA();
    void checkUseMultiBlockMode();
    void checkSupportTrtFp8FMHA();
    void checkUseFlashinferSampleKernel();
    bool useFp8Fmha(const DevicePrepParams& params) const;
    void initMoeRunner(const DataType compute_type, const DataType weights_type);
    void initNcclParam(size_t rank, size_t world_size, const std::string& ip, size_t port,
                       const std::string& tp_group_name, NcclParam& nccl_param);
    void checkUseGroupGemm();
    NcclParam getNcclParam(ParallelMode mode);
    template<typename QuantType>
    LayernormOutput _layernorm(const LayernormParams& params);
    bool checkUseFlashinferSampleGreedy(const GreedyParams& params);
    GreedyOutput flashinferSampleGreedy(const GreedyParams& params, const BufferPtr &transposed_tokens);
    void processLogits(const GreedyParams& params, const BufferPtr &device_tokens, const BufferPtr &transposed_tokens);
    void completeSampleGreedy(const GreedyParams& params, const BufferPtr &transposed_tokens);

public:
    cudaStream_t getStream() {return stream_;}
    torch::Device getTorchDevice() override { return torch::Device(torch::kCUDA);};

public:
    void copy(const CopyParams& params) override;
    void noBlockCopy(const CopyParams& params) override;
    void bufMemset(Buffer& buf, int val) override;
    TransposeOutput transpose(const TransposeParams& params) override;
    AddBiasOutput addbias(const AddBiasParams& params) override;
    ConvertOutput convert(const ConvertParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    SplitOutput split(const SplitParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    BufferPtr gemm(const GemmParams& params) override;
    GroupedGemmOutput groupedGemm(const GroupedGemmParams& params) override;
    MultiplyOutput multiply(const MultiplyParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    BufferPtr activation(const ActivationParams& params) override;
    BufferPtr loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    MoeGateSelectOutput moeGateSelect(const FfnLayerParams& params) override;
    FfnLayerOutput moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    FfnLayerOutput moeFfnFp8(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) override;
    GreedyOutput sampleGreedy(const GreedyParams& params) override;
    void broadcast(const BroadcastParams& params) override;
    AllReduceOutput allReduce(const AllReduceParams& params) override;
    void allGather(const AllGatherParams& params) override;
    AllToAllOutput allToAll(const AllToAllParams& params) override;
    PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params) override;
    BufferPtr mlaQKVGemm(const AttentionLayerParams& params) override;
    void mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) override;
    void sampleBeamSearch(const BeamSearchParams& params) override;
    BufferPtr quantize(const QuantizeParams& params) override;
    void preRun() override { check_cuda_error(cudaSetDevice(device_id_)); }

    void prepareMoEGate(const FfnLayerParams& params,
                        BufferPtr             gate,
                        torch::Tensor&        gate_with_bias_tensor,
                        BufferPtr&            gate_with_bias);
    void mlaDecoderSelfAttention(const MlaDecoderAttentionParams& params) override;
    void mlaContextAttention(const MlaAttentionModuleParams& params) override;
    MoeDispatchOutput epDispatch(const MoeDispatchParams& params) override;
    FfnLayerOutput epCombine(const MoeCombineParams& params) override;

    torch::Tensor QInputBatchMatmulWrapper(const MlaDecoderAttentionParams& params);
    torch::Tensor DecoderOutputGemmWrapper(const torch::Tensor& attn_out_t, const MlaDecoderAttentionParams& params);

    static torch::Tensor packInt8TensorToPackedInt4(torch::Tensor weight);
    static torch::Tensor preprocessWeightsForMixedGemm(torch::Tensor row_major_quantized_weight, torch::ScalarType quant_type, const std::string &arch);
    static std::vector<torch::Tensor> symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor weight, torch::ScalarType quant_type, const std::string &arch);

    void perfRangePush(const std::string& name) const override;
    void perfRangePop() const override;
protected:

    void InvokeSmoothQaunt(const GemmParams&       params,
                           const CudaGemmArguments arguments,
                           BufferPtr               output);
    void InvokeWeightOnlyGemm(const GemmParams&       params,
                              const CudaGemmArguments arguments,
                              BufferPtr               output);
    void InvokeGeneralGemm(const GemmParams&       params,
                           const CudaGemmArguments arguments,
                           BufferPtr               output);
    void InvokeDeepGemm(const GemmParams&       params,
                        const CudaGemmArguments arguments,
                        BufferPtr               output);
    void selectCuFMHARunner(const DevicePrepParams& params);

    KVBlockArray getKVBlockArray(const AttentionModuleParams& params,
                                 const Buffer&                kv_cache_offset_pointers,
                                 int                          batch_size,
                                 bool                         use_fp8_fmha);

    void prefillAttention(const AttentionModuleParams& params,
                          KVBlockArray                 kv_block_array,
                          const BufferPtr&             q_output,
                          const BufferPtr&             k_output,
                          const BufferPtr&             v_output,
                          const BufferPtr&             qkv_buf_fp8);

protected:
    cudaStream_t stream_;
    cudaStream_t no_block_copy_stream_;
    cudaStream_t communication_stream_;
    std::unique_ptr<cublasMMWrapper> cublas_mm_wrapper_;

    std::unique_ptr<trt_plugins::WeightOnlyQuantMatmulPlugin> weight_only_matmul_plugin_;
    std::unique_ptr<trt_plugins::SmoothQuantGemmPlugin> smooth_quant_plugin_;

    std::unique_ptr<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin> weight_only_groupwise_matmul_plugin_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;

    FMHAType fmha_type_ = FMHAType::NONE;
    // for speculative decoding, draft model and score model has different config such as kv_head_num
    // here we need separate cufmha_runner for them to avoid frequently setup cufmha runner with different config
    std::vector<std::shared_ptr<cufmha>> cufmha_runner_pool_;
    std::shared_ptr<cufmha> cufmha_runner_;
    std::unique_ptr<cuggemm> cuggemm_runner_;
    bool use_multi_block_mode       = false;

private:
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> host_allocator_;
    c10::cuda::CUDACachingAllocator::CUDAAllocator *origin_torch_cuda_allocator_;
    std::unique_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> managed_torch_cuda_allocator_;

    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    cudaDeviceProp device_prop_;

    std::mutex cublas_wrapper_mutex_;
    std::unique_ptr<cublasAlgoMap> cublas_algo_map_;

    NcclParam tp_nccl_param_;
    NcclParam dp_nccl_param_;
    NcclParam dp_tp_nccl_param_;

    BufferPtr curandstate_buf_; // for sampler use.

    std::unique_ptr<CustomAllReduceComm> custom_allreduce_comm_ = nullptr; // for custom allreduce use

    // BufferPtr will be error when multi stream, tmp hold
    std::vector<BufferPtr> overlap_hold_buffers_;
protected:
    bool use_trtv1_fmha             = false;
    bool use_trtv2_fmha             = false;
    bool use_trtv2_fmha_paged       = false;
    bool use_open_source_fmha       = false;
    bool use_open_source_fmha_paged = false;
    bool use_group_gemm             = false;
    bool support_trt_fp8_fmha       = false;
    bool use_fp8_fmha_              = false;
    bool use_flashinfer_sample_kernel      = false;
};

} // namespace fastertransformer
