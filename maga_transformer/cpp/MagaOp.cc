#include "maga_transformer/cpp/MagaOp.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"

using namespace std;
using namespace torch;

namespace rtp_llm {

MagaOp::MagaOp(const MagaInitParams& maga_init_params) {
}

MagaOp::~MagaOp() {
}

intrusive_ptr<MagaQuery> MagaOp::forward(intrusive_ptr<QueryRequest> query) {
    return th::make_intrusive<MagaQuery>(query);
};

void MagaOp::model_loop() {
    while (true) {
        const auto requests = query_manager_->get_requests();
        if (!requests) {
            sleep(0.1);
            continue;
        }

        const auto merged_request = query_assembler_->assemble_requests(requests);
    }
}

} // namespace rtp_llm

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(MasterInfo)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, ip)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, th_nccl_port)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, context_decoder_nccl_port)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, decoder_nccl_port)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, gpt_nccl_port)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, dynamic_decoder_nccl_port)
    ADD_TORCH_JIT_PROPERTY(MasterInfo, nccl_op_port);

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(DistributedConfig)
    ADD_TORCH_JIT_PROPERTY(DistributedConfig, master_info)
    ADD_TORCH_JIT_PROPERTY(DistributedConfig, tp_size)
    ADD_TORCH_JIT_PROPERTY(DistributedConfig, pp_size)
    ADD_TORCH_JIT_PROPERTY(DistributedConfig, world_size)
    ADD_TORCH_JIT_PROPERTY(DistributedConfig, world_rank)
    ADD_TORCH_JIT_PROPERTY(DistributedConfig, local_world_size);

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(MagaInitParams)
    ADD_TORCH_JIT_PROPERTY(MagaInitParams, distributed_config)
    ADD_TORCH_JIT_PROPERTY(MagaInitParams, gpt_init_parameter);

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(GenerateConfig)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, max_seq_len)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, max_new_tokens)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, num_validate_token)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, beam_size)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, top_k)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, top_p)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, temperature)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, repetition_penalty)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, presence_penalty)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, min_length)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, length_penalty)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, beam_search_diversity_rate)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, random_seed)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, top_p_decay)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, top_p_min)
    ADD_TORCH_JIT_PROPERTY(GenerateConfig, top_p_reset_ids);

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(ErrorInfo)
    ADD_TORCH_JIT_PROPERTY(ErrorInfo, has_error)
    ADD_TORCH_JIT_PROPERTY(ErrorInfo, error_message);

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(QueryRequest)
    ADD_TORCH_JIT_PROPERTY(QueryRequest, generate_config)
    ADD_TORCH_JIT_PROPERTY(QueryRequest, input_ids)
    ADD_TORCH_JIT_PROPERTY(QueryRequest, input_embeddings);

DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(GenerateResponse)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, output_token_ids)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, finished)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, all_finished)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, error_info)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, log_probs)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, hidden_states)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, attentions)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, logits)
    ADD_TORCH_JIT_PROPERTY(GenerateResponse, loss);

DECLARE_TORCH_JIT_CLASS(MagaQuery)
    ADD_TORCH_JIT_METHOD(MagaQuery, next_response)
    ADD_TORCH_JIT_METHOD(MagaQuery, cancel);

DECLARE_TORCH_JIT_CLASS(MagaOp)
    ADD_TORCH_JIT_METHOD(MagaOp, forward);

