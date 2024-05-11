#pragma once

#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/cublas/cublasMMWrapper.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/trt_plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "src/fastertransformer/trt_plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "src/fastertransformer/trt_plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"
#include "src/fastertransformer/utils/LoRAWeight.h"
#include <string>

namespace trt_plugins = tensorrt_llm::plugins;
namespace tc = tensorrt_llm::common;
namespace fastertransformer {

template<typename T>
class GemmRunner {
private:
    cudaStream_t                               stream_;
    IAllocator*                                allocator_;
    cublasMMWrapper*                           cublas_wrapper_;

    tc::QuantAlgo quant_algo_;
    std::shared_ptr<trt_plugins::WeightOnlyQuantMatmulPlugin>          weight_only_matmul_plguin_;
    std::shared_ptr<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin> weight_only_groupwise_matmul_plguin_;
    std::shared_ptr<trt_plugins::SmoothQuantGemmPlugin> smooth_quant_plugin_;

    char* workspace_ = nullptr;

public:
    GemmRunner(
        cudaStream_t stream, IAllocator* allocator, cublasMMWrapper* cublas_wrapper, tc::QuantAlgo quant_algo):
        stream_(stream),
        allocator_(allocator),
        cublas_wrapper_(cublas_wrapper),
        quant_algo_(quant_algo)
    {
        nvinfer1::DataType datatype;
        if (std::is_same<T, half>::value) {
            datatype = nvinfer1::DataType::kHALF;
        } else if (std::is_same<T, __nv_bfloat16>::value) {
            datatype = nvinfer1::DataType::kBF16;
        } else {
            FT_LOG_ERROR("not supported yet");
        }
        if (quant_algo_.smoothQuantInt8()) {
            auto quant_mode = tc::QuantMode::fromDescription(true, true, true, true, false, false, false, false);
            smooth_quant_plugin_ = std::make_shared<trt_plugins::SmoothQuantGemmPlugin>(quant_mode, datatype);
        } else if (quant_algo_.weightOnly()) {
            if (quant_algo_.getGroupSize() > 0) {
                weight_only_groupwise_matmul_plguin_ =
                    std::make_shared<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin>(
                            datatype, true, quant_algo_.getGroupSize(), quant_algo_.getWeightBits());
            } else {
                weight_only_matmul_plguin_ = std::make_shared<trt_plugins::WeightOnlyQuantMatmulPlugin>(
                        datatype, trt_plugins::WeightTypeId::INT8);
            }
        }
    }
    ~GemmRunner() {
        freeBuffer();
    }
    void freeBuffer();

    void Gemm(int m, int n, int k, const void* inputs, const DenseWeight<T, T>* weights, T* outputs, const float* scale_tokens=nullptr);
    void GemmWithBias(int m, int n, int k, const void* inputs, const DenseWeight<T, T>* weights, T* outputs);

private:
    void allocateWorkspace(size_t s);

};

}  // namespace fastertransformer
