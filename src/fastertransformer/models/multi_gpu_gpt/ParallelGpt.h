#pragma once

#include <vector>

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/NormWrapper.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class ParallelGpt: public BaseLayer {
private:
    // meta data
    const GptInitParameter& params_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    float* attention_query_dynamic_scale_  = nullptr;
    float* ffn_intermediate_dynamic_scale_ = nullptr;

    bool is_qk_buf_float_;

    BaseAttentionLayer<T>* parallel_attention_wrapper_;

    FfnLayer<T>*                    ffn_layer_;
    std::unique_ptr<NormWrapper<T>> norm_wrapper_;

    void allocateBuffer() override;
    void allocateBuffer(size_t total_batch_size, size_t max_seq_len, bool reuse_buf, bool pre_attn_ln);
    void freeBuffer() override;
    bool isValidLayerParallelId(uint l);
    void initialize();
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

    T*      decoder_normed_input_    = nullptr;
    T*      attn_normed_input_       = nullptr;
    T*      self_attn_output_        = nullptr;
    T*      normed_self_attn_output_ = nullptr;
    T*      decoder_layer_output_    = nullptr;
    size_t* h_pinned_token_num_ptr_  = nullptr;
    int*    padding_offset_          = nullptr;
    int*    cu_seqlens_              = nullptr;
    int*    context_lengths_         = nullptr;
    int*    sequence_lengths_        = nullptr;
    int*    prefix_lengths_          = nullptr;
    int64_t* block_pointers_         = nullptr;
    int64_t* block_scale_pointers_   = nullptr;

    std::vector<int64_t> block_pointers_vector_;
    std::vector<int64_t> block_scale_pointers_vector_;
protected:
public:
    ParallelGpt(const GptInitParameter&             gpt_init_parameter,
                NcclParam                           tensor_para,
                NcclParam                           pipeline_para,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                bool                                is_qk_buf_float,
                bool                                sparse                   = false,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                int                                 enable_custom_all_reduce = 0);

    ParallelGpt(ParallelGpt<T> const& decoder);

    ~ParallelGpt();
    void preAllocate();

    void convert_to_block_pointers(TensorMap*                                            output_tensors,
                                   const TensorMap*                                      input_tensors,
                                   int total_batch_size);
    void forward(TensorMap*                                            output_tensors,
                 const TensorMap*                                      input_tensors,
                 const std::vector<ParallelGptDecoderLayerWeight<T>*>* decoder_layer_weights);
};

}  // namespace fastertransformer
