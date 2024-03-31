#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLoRALayerWeight.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/utils/compiler_config.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {
class IFtGpt {
public:
    virtual ~IFtGpt() {}
    virtual void forward(th::Tensor&              decoder_output,
                         th::optional<th::Tensor> key_cache,
                         th::optional<th::Tensor> value_cache,
                         th::Tensor&              decoder_input,
                         th::Tensor&              input_lengths,
                         th::Tensor&              sequence_lengths,
                         th::Tensor&              block_index_map,
                         th::optional<th::Tensor> lora_ids,
                         th::optional<th::Tensor> attention_mask,
                         th::optional<th::Tensor> position_ids,                
                         th::optional<th::Tensor> linear_bias_slopes,
                         th::optional<th::Tensor> prefix_prompt_lengths,
                         th::optional<th::Tensor> count_prefix_length,
                         th::optional<th::Tensor> max_prefix_length,
                         th::optional<th::Tensor> key_cache_scale,
                         th::optional<th::Tensor> value_cache_scale) = 0;
    
    virtual void addLoRA(const int                                                       lora_id,
                        const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                        const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)=0;

    virtual void removeLoRA(const int lora_id)=0;
    virtual bool UseFMHA()=0;
};

template<typename T>
class FtGpt: public IFtGpt {
public:
    FtGpt(const GptInitParameter&   gpt_init_parameter,
          const int                     tensor_para_size,
          const int                     pipeline_para_size,
          const std::string&            master_ip,
          const int                     master_port,
          const std::vector<std::unordered_map<std::string, th::Tensor>> &weights);

    ~FtGpt() override;

    // bool initMem(ft::IAllocator* allocator);
    void forward(th::Tensor&              decoder_output,
                 th::optional<th::Tensor> key_cache,
                 th::optional<th::Tensor> value_cache,
                 th::Tensor&              decoder_input,
                 th::Tensor&              input_lengths,
                 th::Tensor&              sequence_lengths,
                 th::Tensor&              block_index_map,
                 th::optional<th::Tensor> lora_ids,
                 th::optional<th::Tensor> attention_mask,
                 th::optional<th::Tensor> position_ids,
                 th::optional<th::Tensor> linear_bias_slopes,
                 th::optional<th::Tensor> prefix_prompt_lengths,
                 th::optional<th::Tensor> count_prefix_length,
                 th::optional<th::Tensor> max_prefix_length,
                 th::optional<th::Tensor> key_cache_scale,
                 th::optional<th::Tensor> value_cache_scale) override;
    
    virtual void addLoRA(const int                                                       lora_id,
                        const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                        const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights) override;

    virtual void removeLoRA(const int lora_id) override;

    virtual bool UseFMHA() override;

private:
    const GptInitParameter& gpt_init_parameter_;

    const std::vector<std::unordered_map<std::string, th::Tensor>> weights_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    cublasLtHandle_t      cublaslt_handle_;
    std::mutex*           cublas_wrapper_mutex_;
    ft::cublasAlgoMap*    cublas_algo_map_;
    ft::ParallelGpt<T>*   parallel_gpt_;
    ft::Allocator<ft::AllocatorType::TH>* allocator_;
    ft::cublasMMWrapper* cublas_wrapper_;
    struct cudaDeviceProp prop_;

    std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*> gpt_lora_layer_weights_;
    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights_;
};

class ParallelGptOp: public th::jit::CustomClassHolder {
public:
    ParallelGptOp(c10::intrusive_ptr<GptInitParameter>                            gpt_init_parameter,
                  const int64_t                                                   tensor_para_size,
                  const int64_t                                                   pipeline_para_size,
                  const std::string                                               master_ip,
                  const int64_t                                                   master_port,
                  const std::vector<std::unordered_map<std::string, th::Tensor>>& weights);

    ~ParallelGptOp();

    th::Tensor forward(th::Tensor               decoder_input,
                       th::optional<th::Tensor> key_cache,
                       th::optional<th::Tensor> value_cache,
                       th::Tensor               input_lengths,
                       th::Tensor               sequence_lengths,
                       th::Tensor               block_index_map,
                       th::optional<th::Tensor> lora_ids,
                       th::optional<th::Tensor> attention_mask,
                       th::optional<th::Tensor> position_ids,
                       th::optional<th::Tensor> linear_bias_slopes,
                       th::optional<th::Tensor> prefix_prompt_lengths,
                       th::optional<th::Tensor> count_prefix_length,
                       th::optional<th::Tensor> max_prefix_length,
                       th::optional<th::Tensor> key_cache_scale,
                       th::optional<th::Tensor> value_cache_scale);

    
    void addLoRA(const int64_t                                                   lora_id,
                         const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                         const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights);
    void removeLoRA(const int64_t lora_id);

    bool UseFMHA();

private:
    GptInitParameter        gpt_init_parameter_;
    size_t                  tensor_para_size_;
    size_t                  pipeline_para_size_;

    at::ScalarType          scalar_type_;
    IFtGpt*   gpt_;
    std::vector<th::Tensor> weights;

    // The chunk size (16 / sizeof(T)) for key cache in fmha.
    size_t chunk_size_;
};

}  // namespace torch_ext
