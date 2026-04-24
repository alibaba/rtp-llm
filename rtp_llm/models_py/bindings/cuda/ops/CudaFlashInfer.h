#pragma once

#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"

namespace rtp_llm {

class AttentionConfigs;

struct FlashInferAttnParams: ParamsBase {
private:
    torch::Tensor float_workspace_;
    torch::Tensor int_workspace_;
    torch::Tensor int_host_workspace_;

    torch::Tensor buf_h_;
    torch::Tensor buf_d_;

public:
    int batch_size      = 0;
    int input_token_num = 0;

private:
    int page_num   = 0;
    int max_kv_len = 0;
    int max_q_len  = 0;
    int accu_q_len = 0;

public:
    torch::Tensor float_workspace_d;
    torch::Tensor int_workspace_d;
    torch::Tensor int_workspace_h;

    torch::Tensor page_indptr_h;
    torch::Tensor qo_indptr_h;
    torch::Tensor batch_indice_h;
    torch::Tensor positions_h;
    torch::Tensor kvlen_h;
    torch::Tensor paged_kv_last_page_len_h;
    torch::Tensor page_indice_h;

    torch::Tensor page_indptr_d;
    torch::Tensor qo_indptr_d;
    torch::Tensor batch_indice_d;
    torch::Tensor positions_d;
    torch::Tensor kvlen_d;
    torch::Tensor paged_kv_last_page_len_d;
    torch::Tensor page_indice_d;

    torch::Tensor kv_cache_block_id_d;

    std::vector<torch::Tensor> flash_mla_plan;
    MlaOpsType                 mla_ops_type = MlaOpsType::AUTO;

    bool                      decode_plan = true;
    torch::Tensor             plan;
    DataType                  dtype = DataType::TYPE_INVALID;
    rtp_llm::AttentionConfigs attn_configs;
    bool                      is_prefill;
    bool                      enable_cuda_graph = false;

    static bool check(const rtp_llm::AttentionConfigs& attn_configs,
                      DataType                         dtype,
                      bool                             is_prefill,
                      MlaOpsType                       mla_ops_type        = MlaOpsType::AUTO,
                      bool                             disable_flash_infer = false);

    static bool checkPrefill(const rtp_llm::AttentionConfigs& attn_configs,
                             const torch::Tensor&             prefix_lengths_host,
                             const torch::Tensor&             input_lengths_host,
                             DataType                         dtype,
                             bool                             skip_no_prefix,
                             MlaOpsType                       mla_ops_type        = MlaOpsType::AUTO,
                             bool                             disable_flash_infer = false);

    static bool      checkDecode(const rtp_llm::AttentionConfigs& attn_configs,
                                 DataType                         dtype,
                                 MlaOpsType                       mla_ops_type        = MlaOpsType::AUTO,
                                 bool                             disable_flash_infer = false);
    bool             check_recycle() override;
    static ParamsPtr prepare(const rtp_llm::AttentionConfigs& attn_configs,
                             const torch::Tensor&             prefix_lengths_host,
                             const torch::Tensor&             sequence_lengths_host,
                             const torch::Tensor&             input_lengths_host,
                             const torch::Tensor&             kv_cache_block_id_host,
                             const torch::Tensor&             kv_cache_block_id_device,
                             DataType                         dtype,
                             MlaOpsType                       mla_ops_type,
                             bool                             enable_cuda_graph,
                             bool                             skip_no_prefix = true);
    void             run(const AttentionModuleParams& params,
                         const torch::Tensor&         input_q,
                         const torch::Tensor&         fp16_out,
                         int64_t                      stream);

private:
    static std::tuple<torch::Tensor, std::vector<torch::Tensor>>
    allocateManyBuffer(const std::vector<std::vector<int64_t>>& shapes, AllocationType atype);

    static FlashInferAttnParams* create(int batch_size, int token_num, int page_num);

    void genPlan(int     batch_size,
                 int     q_length,
                 int     local_head_num,
                 int     local_head_num_kv,
                 int     size_per_head,
                 int     tokens_per_block,
                 int     kv_lora_rank,
                 bool    use_mla,
                 int64_t stream,
                 bool    enable_cuda_graph);

    static bool sameQLength(const torch::Tensor& input_lengths_host, int context_batch_size, int& q_length);

public:
    static bool                  isDecode(int input_token_num);
    static void                  recycle(void* p);
    void                         fillParams(torch::Tensor sequence_lengths,
                                            torch::Tensor input_lengths,
                                            torch::Tensor kv_cache_block_id_host,
                                            int           batch_size,
                                            int           seq_size_per_block,
                                            torch::Tensor prefix_lengths = torch::Tensor()) override;
    void                         fillFlashInfer(const torch::Tensor& prefix_lengths_host,
                                                const torch::Tensor& sequence_lengths_host,
                                                const torch::Tensor& input_lengths_host,
                                                const torch::Tensor& kv_cache_block_id_host,
                                                const int            batch_size,
                                                const int            tokens_per_block);
    void                         refreshFlashInferBuf(int batch_size, int token_num);
    static FlashInferAttnParams* get(int batch_size, int input_token_num);
};

using FlashInferAttnParamsPtr = std::shared_ptr<FlashInferAttnParams>;

struct ParamsCache {
    // use inline to make sure the static params unique globally.
    static inline std::deque<FlashInferAttnParams*> DECODE_PARAMS_CACHE;
    static inline std::deque<FlashInferAttnParams*> PREFILL_PARAMS_CACHE;
};
}  // namespace rtp_llm
