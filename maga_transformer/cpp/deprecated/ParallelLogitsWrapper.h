#pragma once

#include "src/fastertransformer/cuda/cublas/cublasMMWrapper.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/utils/DenseWeight.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

namespace ft = fastertransformer;
namespace rtp_llm {

template<typename T>
class ParallelLogitsWrapper: public ft::BaseLayer {
private:
    const GptInitParameter&   params_;
    ft::NcclParam             tensor_para_;
    const size_t              local_head_num_;
    const ft::DenseWeight<T>* embedding_table_;
    T*                        nccl_logits_;
    ft::IAllocator*           allocator_;
    ft::CudaDevice*           device_;

public:
    ParallelLogitsWrapper(const GptInitParameter&   gpt_init_parameter,
                          ft::NcclParam             tensor_para,
                          cudaStream_t              stream,
                          ft::cublasMMWrapper*      cublas_wrapper,
                          ft::IAllocator*           allocator,
                          bool                      is_free_buffer_after_forward,
                          const ft::DenseWeight<T>* embedding_table);

    ~ParallelLogitsWrapper();
    void allocateBuffer(size_t h_token_num);
    void allocateBuffer() override{};
    void freeBuffer() override;
    void forward(ft::Tensor& embeddings, const ft::Tensor last_hidden_states);
};

}  // namespace rtp_llm
