#include "maga_transformer/cpp/deprecated/ParallelWordEmbeddingWrapper.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace rtp_llm {

template<typename T>
ParallelWordEmbeddingWrapper<T>::ParallelWordEmbeddingWrapper(const GptInitParameter&   gpt_init_parameter,
                                                              ft::NcclParam             tensor_para,
                                                              cudaStream_t              stream,
                                                              ft::cublasMMWrapper*      cublas_wrapper,
                                                              ft::IAllocator*           allocator,
                                                              bool                      is_free_buffer_after_forward,
                                                              const ft::DenseWeight<T>* embedding_table,
                                                              const ft::DenseWeight<T>* postition_table):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    params_(gpt_init_parameter),
    tensor_para_(tensor_para),
    local_head_num_(gpt_init_parameter.head_num_ / tensor_para.world_size_),
    embedding_table_(embedding_table),
    postition_table_(postition_table) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void ParallelWordEmbeddingWrapper<T>::forward(ft::Tensor&      embeddings,
                                              const ft::Tensor tokens,
                                              ft::Tensor       position_ids) {
    PUSH_RANGE(stream_, "input id embedding lookup");
    const ft::DataType data_type = ft::getTensorType<T>();

    const int*   input_ids = reinterpret_cast<int*>(tokens.getPtr<int>());
    const size_t token_num = tokens.shape()[0];

    ft::Tensor nccl_embeddings;
    // if (tensor_para_.world_size_ > 1) {
    //     nccl_embeddings = ft::Tensor(ft::MEMORY_GPU, data_type, embeddings.shape(), allocator_, true);
    // }
    // else {
    // save tmp memory
    nccl_embeddings = ft::Tensor(ft::MEMORY_GPU, data_type, embeddings.shape(), embeddings.getPtr<T>());
    // }
    const int    data_size       = embeddings.size() / tensor_para_.world_size_;
    T*           word_embeddings = reinterpret_cast<T*>(nccl_embeddings.getPtr<T>()) + data_size * tensor_para_.rank_;
    const size_t local_hidden_units = local_head_num_ * params_.size_per_head_;
    ft::invokeEmebeddingLookup<T>(word_embeddings,
                                  embedding_table_->kernel,
                                  params_.has_positional_encoding_ ? postition_table_->kernel : nullptr,
                                  (T*)nullptr,
                                  input_ids,
                                  params_.has_positional_encoding_ ? position_ids.getPtr<int>() : nullptr,
                                  (int*)nullptr,
                                  token_num,
                                  local_hidden_units,
                                  stream_);
    if (tensor_para_.world_size_ > 1) {
        PUSH_RANGE(stream_, "all gather");
        ftNcclAllGather(nccl_embeddings.getPtr<T>(),
                        nccl_embeddings.getPtr<T>(),
                        data_size,
                        tensor_para_.rank_,
                        tensor_para_,
                        stream_);
        ft::invokeTransposeAxis01(embeddings.getPtr<T>(),
                                  nccl_embeddings.getPtr<T>(),
                                  token_num,
                                  local_hidden_units,
                                  stream_);

        POP_RANGE;
    }

    sync_check_cuda_error();
    POP_RANGE;
}
template class ParallelWordEmbeddingWrapper<float>;
template class ParallelWordEmbeddingWrapper<half>;
#ifdef ENABLE_BF16
template class ParallelWordEmbeddingWrapper<__nv_bfloat16>;
#endif

}  // namespace rtp_llm
