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
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLoRALayerWeight.h"
#include <memory>

namespace rtp_llm {

class IModelWrapper {
public:
    virtual ~IModelWrapper() {}
    virtual bool                             useFMHA() = 0;
    virtual GptModelOutputs forward(const ModelRequest& model_request) = 0;
};

template<typename T>
class ParallelModelWrapperImpl: public IModelWrapper {
public:
    ParallelModelWrapperImpl(
            const ft::GptInitParameter&                                             gpt_init_parameter,
            ft::NcclParam                                                           tensor_para,
            ft::NcclParam                                                           pipeline_para,
            const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
            const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights);
    ~ParallelModelWrapperImpl();
    void                             allocateBuffer(size_t total_batch_size, size_t h_token_num, GptModelOutputs& model_output);
    void                             createAttentionMask(size_t context_batch_size, size_t max_context_seq_length, int* input_lengths_host);
    void                             freeBuffer();
    void                             initialize();
    bool                             useFMHA() override;
    GptModelOutputs forward(const ModelRequest& model_request) override;
private:
    void setPaddingOffsetAndCuSeqLens(ft::Tensor& padding_offset,
                                      ft::Tensor& cu_seqlens,
                                      const uint  context_h_token_num,
                                      const uint  context_batch_size,
                                      const uint  max_context_seq_length,
                                      const int*  input_lengths);

private:
    const ft::GptInitParameter                         params_;
    const ft::DataType                                 data_type_;
    ft::NcclParam                                      tensor_para_;
    ft::NcclParam                                      pipeline_para_;
    cudaStream_t                                       stream_;
    ft::IAllocator*                                    allocator_;
    ft::CudaDevice*                                    device_;
    std::shared_ptr<GptGlobalWeights<T>>               global_weights_;
    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights_;
    std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*> gpt_lora_layer_weights_;

    T*   all_hidden_states_  = nullptr;
    T*   last_hidden_states_  = nullptr;
    T*   attention_mask_     = nullptr;
    int* combo_tokens_       = nullptr;
    int* combo_token_types_  = nullptr;
    int* combo_position_ids_ = nullptr;
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
    std::unique_ptr<ft::NormWrapper<T>> norm_wrapper_;
};

class ParallelModelWrapper {
public:
    ParallelModelWrapper(
            const ft::GptInitParameter&                                             gpt_init_parameter,
            const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
            const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights);

    ~ParallelModelWrapper(){};
    void addLoRA(const int64_t                                                           lora_id,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {}
    void removeLoRA(const int64_t lora_id) {}

    bool useFMHA();

    GptModelOutputs forward(const ModelRequest& model_request);

private:
    IModelWrapper* model_wrapper_;
};

}  // namespace rtp_llm
