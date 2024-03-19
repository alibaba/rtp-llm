#pragma once

#include "maga_transformer/cpp/components/parallel_gpt/GptWeights.h"
#include "maga_transformer/cpp/components/parallel_gpt/ParallelGpt.h"
#include "maga_transformer/cpp/components/parallel_gpt/ParallelLogitsWrapper.h"
#include "maga_transformer/cpp/components/parallel_gpt/ParallelWordEmbeddingWrapper.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"

#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include <memory>

namespace rtp_llm {
namespace ft = fastertransformer;

class IModelWrapper {
public:
    virtual ~IModelWrapper() {}
    virtual std::shared_ptr<ModelOutput> forward(const ModelRequest& model_request) = 0;
};

template<typename T>
class ParallelModelWrapperImpl: public IModelWrapper {
public:
    ParallelModelWrapperImpl(const GptInitParameter&                                         gpt_init_parameter,
                             const int                                                       tensor_para_size,
                             const std::string&                                              master_ip,
                             const int                                                       master_port,
                             const std::unordered_map<std::string, ft::Tensor>&              global_weights,
                             const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_weights,
                             const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_weights,
                             const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_scales);
    ~ParallelModelWrapperImpl();
    void                         allocateBuffer(size_t total_batch_size, size_t h_token_num);
    void                         freeBuffer();
    void                         initialize();
    std::shared_ptr<ModelOutput> forward(const ModelRequest& model_request) override;

private:
    // ft::KVBlockArray convert_to_kv_block_array(const th::Tensor& kv_cache_blocks, const th::Tensor& kv_cache_scales);

    ft::Tensor
    genAttentionMask(const ft::Tensor& input_lengths, const uint context_batch_size, const uint max_context_seq_length);

    ft::Tensor getPositionsId(const size_t      h_token_num,
                              const uint        generate_batch_size,
                              const uint        context_batch_size,
                              const th::Tensor& input_lengths,
                              const th::Tensor& sequence_lengths);

    ft::Tensor calculate_loss(const ft::Tensor& all_hidden_states);

    void setPaddingOffsetAndCuSeqLens(GptCommonInputs* inputs,
                                      const uint       context_h_token_num,
                                      const uint       context_batch_size,
                                      const uint       max_context_seq_length,
                                      const int*       input_lengths);

    std::unique_ptr<GptCommonInputs> prepareGptCommonInputs(const ModelRequest& model_request);

private:
    const GptInitParameter&        params_;
    const DataType                 data_type_;
    ft::NcclParam                  tensor_para_;
    ft::NcclParam                  pipeline_para_;
    cudaStream_t                   stream_;
    ft::IAllocator*                allocator_;
    std::shared_ptr<GptWeights<T>> weights_;
    T*                             linear_bias_slopes_ = nullptr;
    // int*     position_ids_         = nullptr;
    // int64_t* block_pointers_       = nullptr;
    // int64_t* block_scale_pointers_ = nullptr;
    // gpu copy is async, so need cpu mem hold
    std::vector<int> padding_offset_;
    std::vector<int> cu_seqlens_;
    std::vector<int> position_ids_;

    std::unique_ptr<ft::cublasMMWrapper>             cublas_wrapper_;
    std::unique_ptr<ParallelWordEmbeddingWrapper<T>> parallel_word_embedding_wrapper_;
    std::unique_ptr<ParallelGpt<T>>                  parallel_gpt_decoder_;
    std::unique_ptr<ParallelLogitsWrapper<T>>        parallel_logits_wrapper_;
};

class ParallelModelWrapper {
public:
    ParallelModelWrapper(const GptInitParameter&                                         gpt_init_parameter,
                         const int                                                       tensor_para_size,
                         const std::string&                                              master_ip,
                         const int                                                       master_port,
                         const std::unordered_map<std::string, ft::Tensor>&              global_weights,
                         const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_weights,
                         const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_weights,
                         const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_scales);

    ~ParallelModelWrapper(){};

    std::shared_ptr<ModelOutput> forward(const ModelRequest& model_request);

private:
    IModelWrapper* model_wrapper_;
};

}  // namespace rtp_llm
