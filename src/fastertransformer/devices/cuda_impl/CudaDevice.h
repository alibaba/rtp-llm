#pragma once

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

nvinfer1::DataType nvinfer1DtypeConvert(fastertransformer::DataType dtype);

class CudaGemmArguments;

class CudaDevice : public DeviceBase {
public:
    CudaDevice(const DeviceInitParams& params);
    ~CudaDevice();

public:
    void init() override;
    DeviceProperties getDeviceProperties() override;
    DeviceStatus getDeviceStatus() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return host_allocator_.get(); }

    void syncAndCheck() override;
    void syncCommunication(bool timeout = true) override;
    DevicePrepOutput prepareModelRun(const DevicePrepParams& params) override;
    bool useGroupGemm() const override;

private:
    void checkUseOpenSourceFMHA(size_t tokens_per_block);
    void checkUseTrtV1FMHA();
    void checkUseTrtV2FMHA();
    void checkUseMultiBlockMode();
    void initMoeRunner(const DataType compute_type, const DataType weights_type);
    void checkUseGroupGemm();

public:
    cudaStream_t getStream() {return stream_;}
    NcclParam getNcclParam() {return nccl_param_;}
    torch::Device getTorchDevice() override { return torch::Device(torch::kCUDA);};

public:
    void copy(const CopyParams& params) override;
    void bufMemset(Buffer& buf, int val) override;
    TransposeOutput transpose(const TransposeParams& params) override;
    AddBiasOutput addbias(const AddBiasParams& params) override;
    ConvertOutput convert(const ConvertParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    BufferPtr gemm(const GemmParams& params) override;
    GroupedGemmOutput groupedGemm(const GroupedGemmParams& params) override;
    MultiplyOutput multiply(const MultiplyParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    BufferPtr multimodalEmbedding(const MultimodalEmbeddingParams& params) override;
    BufferPtr activation(const ActivationParams& params) override;
    BufferPtr loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    FfnLayerOutput moeFfnLayer(const FfnLayerParams& params) override;
    void sampleGreedy(const GreedyParams& params) override;
    void broadcast(const BroadcastParams& params) override;
    AllReduceOutput allReduce(const AllReduceParams& params) override;
    void allGather(const AllGatherParams& params) override;
    PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params) override;
    BufferPtr mlaQKVGemm(const AttentionLayerParams& params) override;

    BufferPtr quantize(const QuantizeParams& params) override;
    void preRun() override { check_cuda_error(cudaSetDevice(device_id_)); }

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

private:
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> host_allocator_;
    c10::cuda::CUDACachingAllocator::CUDAAllocator *origin_torch_cuda_allocator_;

    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    cudaDeviceProp device_prop_;

    std::mutex cublas_wrapper_mutex_;
    std::unique_ptr<cublasAlgoMap> cublas_algo_map_;
    std::unique_ptr<cublasMMWrapper> cublas_mm_wrapper_;

    std::unique_ptr<trt_plugins::WeightOnlyQuantMatmulPlugin> weight_only_matmul_plugin_;
    std::unique_ptr<trt_plugins::SmoothQuantGemmPlugin> smooth_quant_plugin_;

    std::unique_ptr<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin> weight_only_groupwise_matmul_plugin_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;

    NcclParam nccl_param_;

    BufferPtr curandstate_buf_; // for sampler use.

    std::unique_ptr<CustomAllReduceComm> custom_allreduce_comm_ = nullptr; // for custom allreduce use

    FMHAType fmha_type_ = FMHAType::NONE;
    std::unique_ptr<cufmha> cufmha_runner_;
    std::unique_ptr<cuggemm> cuggemm_runner_;
    bool use_trtv1_fmha             = false;
    bool use_trtv2_fmha             = false;
    bool use_trtv2_fmha_paged       = false;
    bool use_open_source_fmha       = false;
    bool use_open_source_fmha_paged = false;
    bool use_multi_block_mode       = false;
    bool use_group_gemm             = false;
};

} // namespace fastertransformer
