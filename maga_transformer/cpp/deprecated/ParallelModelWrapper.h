#pragma once

#include "src/fastertransformer/th_op/multi_gpu_gpt/Base.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/deprecated/GptWeights.h"
#include "maga_transformer/cpp/deprecated/ParallelLogitsWrapper.h"
#include "maga_transformer/cpp/deprecated/ParallelWordEmbeddingWrapper.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include <memory>

namespace ft = fastertransformer;
namespace th = torch;

namespace rtp_llm {

class IModelWrapper {
public:
    virtual ~IModelWrapper() {}
    virtual std::unique_ptr<GptModelOutputs> forward(const ModelRequest& model_request) = 0;
};

template<typename T>
class ParallelModelWrapperImpl: public IModelWrapper {
public:
    ParallelModelWrapperImpl(const GptInitParameter&                                                 gpt_init_parameter,
                             const int                                                               tensor_para_size,
                             const std::string&                                                      master_ip,
                             const int                                                               master_port,
                             const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
                             const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights);
    ~ParallelModelWrapperImpl();
    void                             allocateBuffer(size_t total_batch_size, size_t h_token_num);
    void                             freeBuffer();
    void                             initialize();
    std::unique_ptr<GptModelOutputs> forward(const ModelRequest& model_request) override;

private:
    void setPaddingOffsetAndCuSeqLens(ft::Tensor& padding_offset,
                                      ft::Tensor& cu_seqlens,
                                      const uint  context_h_token_num,
                                      const uint  context_batch_size,
                                      const uint  max_context_seq_length,
                                      const int*  input_lengths);

private:
    const GptInitParameter&                            params_;
    const DataType                                     data_type_;
    ft::NcclParam                                      tensor_para_;
    ft::NcclParam                                      pipeline_para_;
    cudaStream_t                                       stream_;
    ft::IAllocator*                                    allocator_;
    ft::CudaDevice*                                    device_;
    std::shared_ptr<GptGlobalWeights<T>>               global_weights_;
    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights_;

    T*   all_hidden_states_  = nullptr;
    T*   last_hidden_states_ = nullptr;
    int* combo_tokens_       = nullptr;
    int* padding_offset_     = nullptr;
    int* cu_seqlens_         = nullptr;
    int* input_lengths_      = nullptr;
    int* sequence_lengths_   = nullptr;
    int* prefix_lengths_     = nullptr;

    // gpu copy is async, so need cpu mem hold
    std::vector<int> padding_offset_cpu_;
    std::vector<int> cu_seqlens_cpu_;
    std::vector<int> position_ids_;

    ft::cublasMMWrapper*                             cublas_wrapper_;
    std::unique_ptr<ParallelWordEmbeddingWrapper<T>> parallel_word_embedding_wrapper_;
    std::unique_ptr<ft::ParallelGpt<T>>              parallel_gpt_decoder_;
    std::unique_ptr<ParallelLogitsWrapper<T>>        parallel_logits_wrapper_;
};

class ParallelModelWrapper {
public:
    ParallelModelWrapper(const GptInitParameter&                                                 gpt_init_parameter,
                         const int                                                               tensor_para_size,
                         const std::string&                                                      master_ip,
                         const int                                                               master_port,
                         const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights);

    ~ParallelModelWrapper(){};

    std::unique_ptr<GptModelOutputs> forward(const ModelRequest& model_request);

private:
    IModelWrapper* model_wrapper_;
};

}  // namespace rtp_llm
