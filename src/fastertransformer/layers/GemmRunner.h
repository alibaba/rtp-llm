#pragma once

#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/cublas/cublasMMWrapper.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/trt_plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
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

    char* workspace_ = nullptr;

public:
    GemmRunner(
        cudaStream_t stream, IAllocator* allocator, cublasMMWrapper* cublas_wrapper, tc::QuantAlgo quant_algo):
        stream_(stream),
        allocator_(allocator),
        cublas_wrapper_(cublas_wrapper),
        quant_algo_(quant_algo){
        int                sm = getSMVersion();
        nvinfer1::DataType datatype;
        if (std::is_same<T, half>::value) {
            datatype = nvinfer1::DataType::kHALF;
        } else if (std::is_same<T, __nv_bfloat16>::value) {
            datatype = nvinfer1::DataType::kBF16;
        } else {
            FT_LOG_ERROR("not supported yet");
        }

        if (quant_algo_.int8Mode() == 1) {

            weight_only_matmul_plguin_ = std::make_shared<trt_plugins::WeightOnlyQuantMatmulPlugin>(
                datatype, trt_plugins::WeightTypeId::INT8);
        }
        else if (quant_algo_.int4Mode() == true) {
            if (sm < 80) {
                FT_LOG_ERROR("int4 mode not supported yet");
            }
            weight_only_groupwise_matmul_plguin_ =
                std::make_shared<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin>(
                    datatype, quant_algo_.useZeros(), quant_algo_.getGroupSize());
        }
    }

    GemmRunner(
        cudaStream_t stream, IAllocator* allocator, cublasMMWrapper* cublas_wrapper, int int8_mode):
        stream_(stream),
        allocator_(allocator),
        cublas_wrapper_(cublas_wrapper){
        int                sm = getSMVersion();
        nvinfer1::DataType datatype;
        quant_algo_ = tc::QuantAlgo(int8_mode);
        if (std::is_same<T, half>::value) {
            datatype = nvinfer1::DataType::kHALF;
        } else if (std::is_same<T, __nv_bfloat16>::value) {
            datatype = nvinfer1::DataType::kBF16;
        } else {
            FT_LOG_ERROR("not supported yet");
        }

        if (quant_algo_.int8Mode() == 1) {
            weight_only_matmul_plguin_ = std::make_shared<trt_plugins::WeightOnlyQuantMatmulPlugin>(
                datatype, trt_plugins::WeightTypeId::INT8);
        }
    }


    ~GemmRunner() {
        freeBuffer();
    }
    void freeBuffer();

    void Gemm(int m, int n, int k, const T* inputs, const DenseWeight<T, T>* weights, T* outputs);

private:
    void allocateWorkspace(size_t s);

};

}  // namespace fastertransformer