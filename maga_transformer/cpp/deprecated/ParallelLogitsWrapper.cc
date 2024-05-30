#include "maga_transformer/cpp/deprecated/ParallelLogitsWrapper.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace rtp_llm {
template<typename T>
ParallelLogitsWrapper<T>::ParallelLogitsWrapper(const ft::GptInitParameter&   gpt_init_parameter,
                                                ft::NcclParam             tensor_para,
                                                cudaStream_t              stream,
                                                ft::cublasMMWrapper*      cublas_wrapper,
                                                ft::IAllocator*           allocator,
                                                bool                      is_free_buffer_after_forward,
                                                const ft::DenseWeight<T>* embedding_table):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    params_(gpt_init_parameter),
    tensor_para_(tensor_para),
    local_head_num_(gpt_init_parameter.head_num_ / tensor_para.world_size_),
    embedding_table_(embedding_table),
    device_(dynamic_cast<ft::CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda))) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_ = device_->getAllocator();
}

template<typename T>
ParallelLogitsWrapper<T>::~ParallelLogitsWrapper()
{
    freeBuffer();
}

template<typename T>
void ParallelLogitsWrapper<T>::allocateBuffer(size_t h_token_num) {
    size_t hidden_units = params_.hidden_size_;
    nccl_logits_ = (T*)allocator_->reMalloc(nccl_logits_, sizeof(T) * h_token_num * params_.vocab_size_);
}

template<typename T>
void ParallelLogitsWrapper<T>::freeBuffer() {
    allocator_->free((void**)nccl_logits_);
}

template<typename T>
void ParallelLogitsWrapper<T>::forward(ft::Tensor& logits, const ft::Tensor hidden_states) {
    PUSH_RANGE(stream_, "input id embedding lookup");
    const ft::DataType data_type = ft::getTensorType<T>();

    const size_t token_num = hidden_states.shape()[0];

    ft::Tensor nccl_logits;
    if (tensor_para_.world_size_ > 1) {
        allocateBuffer(logits.shape()[0]);
        nccl_logits = ft::Tensor(ft::MEMORY_GPU, ft::DataType::TYPE_FP32, logits.shape(), nccl_logits_);
    }
    else {
        // save tmp memory
        nccl_logits = ft::Tensor(ft::MEMORY_GPU, ft::DataType::TYPE_FP32, logits.shape(), logits.getPtr<float>());
    }

    assert(params_.vocab_size_ % tensor_para_.world_size_ == 0);
    const int local_vocab_size = params_.vocab_size_ / tensor_para_.world_size_;
    float     alpha            = 1.0f;
    float     beta             = 0.0f;
    const int data_size        = logits.size() / tensor_para_.world_size_;
    // T*           word_embeddings  = reinterpret_cast<T*>(nccl_logits.getPtr<T>()) + data_size * tensor_para_.rank_;
    const size_t hidden_units = params_.head_num_ * params_.size_per_head_;
    const cudaDataType_t gemm_data_type = ft::getCudaDataType<T>();
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          local_vocab_size,  // n
                          token_num,
                          hidden_units,  // k
                          &alpha,
                          embedding_table_->kernel,
                          gemm_data_type,
                          hidden_units,               // k
                          hidden_states.getPtr<T>(),  // OPT: no final layer norm
                          gemm_data_type,
                          hidden_units,  // k
                          &beta,
                          nccl_logits.getPtr<float>() + tensor_para_.rank_ * data_size,
                          CUDA_R_32F,
                          local_vocab_size, /* n */
                          CUDA_R_32F,
                          cublasGemmAlgo_t(-1));

    if (tensor_para_.world_size_ > 1) {
        PUSH_RANGE(stream_, "all gather");
        auto logits_buf = std::make_shared<ft::Buffer>(
            ft::MemoryType::MEMORY_GPU, ft::getTensorType<T>(),
            nccl_logits.shape(), nccl_logits.getPtr<T>());
        device_->allGather({{logits_buf}});
        ft::invokeTransposeAxis012(logits.getPtr<float>(),
                                  nccl_logits.getPtr<float>(),
                                  tensor_para_.world_size_,
                                  token_num,
                                  local_vocab_size,
                                  stream_);

        POP_RANGE;
    }

    sync_check_cuda_error();
    POP_RANGE;
}

template class ParallelLogitsWrapper<float>;
template class ParallelLogitsWrapper<half>;
#ifdef ENABLE_BF16
template class ParallelLogitsWrapper<__nv_bfloat16>;
#endif

}  // namespace rtp_llm
