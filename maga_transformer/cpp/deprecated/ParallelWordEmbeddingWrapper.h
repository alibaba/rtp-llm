#pragma once

#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/utils/DenseWeight.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

namespace rtp_llm {
namespace ft = fastertransformer;

template<typename T>
class ParallelWordEmbeddingWrapper: public ft::BaseLayer {
private:
    const GptInitParameter&   params_;
    ft::NcclParam             tensor_para_;
    const size_t              local_head_num_;
    const ft::DenseWeight<T>* embedding_table_;
    const ft::DenseWeight<T>* postition_table_;
    T*                        nccl_embedding_;
    ft::IAllocator*           allocator_;
    ft::CudaDevice*           device_;

public:
    ParallelWordEmbeddingWrapper(const GptInitParameter&   gpt_init_parameter,
                                 ft::NcclParam             tensor_para,
                                 cudaStream_t              stream,
                                 ft::cublasMMWrapper*      cublas_wrapper,
                                 ft::IAllocator*           allocator,
                                 bool                      is_free_buffer_after_forward,
                                 const ft::DenseWeight<T>* embedding_table,
                                 const ft::DenseWeight<T>* postition_table);

    ~ParallelWordEmbeddingWrapper();
    void allocateBuffer(size_t h_token_num);
    void allocateBuffer() override{};
    void freeBuffer() override;
    void forward(ft::Tensor& embeddings, const ft::Tensor tokens, ft::Tensor position_ids);
};

}  // namespace rtp_llm
