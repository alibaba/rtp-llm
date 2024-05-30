#include "maga_transformer/cpp/deprecated/ParallelWordEmbeddingWrapper.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace rtp_llm {

template<typename T>
ParallelWordEmbeddingWrapper<T>::ParallelWordEmbeddingWrapper(const ft::GptInitParameter&   gpt_init_parameter,
                                                              ft::NcclParam             tensor_para,
                                                              cudaStream_t              stream,
                                                              ft::cublasMMWrapper*      cublas_wrapper,
                                                              ft::IAllocator*           allocator,
                                                              bool                      is_free_buffer_after_forward,
                                                              const ft::DenseWeight<T>* embedding_table,
                                                              const ft::DenseWeight<T>* postition_table,
                                                              const ft::DenseWeight<T>* token_type_table):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    params_(gpt_init_parameter),
    tensor_para_(tensor_para),
    local_head_num_(gpt_init_parameter.head_num_ / tensor_para.world_size_),
    embedding_table_(embedding_table),
    postition_table_(postition_table),
    token_type_table_(token_type_table),
    device_(dynamic_cast<ft::CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda)))
{
    allocator_ = device_->getAllocator();
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ParallelWordEmbeddingWrapper<T>::~ParallelWordEmbeddingWrapper() {
    freeBuffer();
}

template<typename T>
void ParallelWordEmbeddingWrapper<T>::allocateBuffer(size_t h_token_num) {
    size_t hidden_units = params_.hidden_size_;
    nccl_embedding_ = (T*)allocator_->reMalloc(nccl_embedding_, sizeof(T) * h_token_num * hidden_units);
}

template<typename T>
void ParallelWordEmbeddingWrapper<T>::freeBuffer() {
    allocator_->free((void**)nccl_embedding_);
}

template<typename T>
void ParallelWordEmbeddingWrapper<T>::forward(ft::Tensor&      embeddings,
                                              const ft::Tensor tokens,
                                              const ft::Tensor token_types,
                                              const ft::Tensor position_ids) {
    PUSH_RANGE(stream_, "input id embedding lookup");
    const ft::DataType data_type = ft::getTensorType<T>();

    const int*   input_ids = reinterpret_cast<int*>(tokens.getPtr<int>());
    const size_t token_num = tokens.shape()[0];

    ft::Tensor nccl_embeddings;
    if (tensor_para_.world_size_ > 1) {
        allocateBuffer(embeddings.shape()[0]);
        nccl_embeddings = ft::Tensor(ft::MEMORY_GPU, data_type, embeddings.shape(), nccl_embedding_);
    }
    else {
        // save tmp memory
        nccl_embeddings = ft::Tensor(ft::MEMORY_GPU, data_type, embeddings.shape(), embeddings.getPtr<T>());
    }
    const int    data_size       = embeddings.size() / tensor_para_.world_size_;
    T*           word_embeddings = reinterpret_cast<T*>(nccl_embeddings.getPtr<T>()) + data_size * tensor_para_.rank_;
    const size_t local_hidden_units = local_head_num_ * params_.size_per_head_;
    ft::invokeEmebeddingLookup<T>(word_embeddings,
                                  embedding_table_->kernel,
                                  postition_table_->kernel,
                                  token_type_table_->kernel,
                                  input_ids,
                                  postition_table_->kernel ? position_ids.getPtr<int>() : nullptr,
                                  token_type_table_->kernel ? token_types.getPtr<int>() : nullptr,
                                  token_num,
                                  local_hidden_units,
                                  stream_);
    if (tensor_para_.world_size_ > 1) {
        PUSH_RANGE(stream_, "all gather");
        auto embedding_buf = std::make_shared<ft::Buffer>(
            ft::MemoryType::MEMORY_GPU, ft::getTensorType<T>(),
            nccl_embeddings.shape(), nccl_embeddings.getPtr<T>());
        device_->allGather({{embedding_buf}});
        ft::invokeTransposeAxis012(embeddings.getPtr<T>(),
                                  nccl_embeddings.getPtr<T>(),
                                  tensor_para_.world_size_,
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
