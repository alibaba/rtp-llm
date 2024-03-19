#pragma once
#include "maga_transformer/cpp/common/torch_bind.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace th = torch;

namespace rtp_llm {

class MasterInfo : public th::jit::CustomClassHolder {
public:
    std::string ip;
    int64_t th_nccl_port;
    int64_t context_decoder_nccl_port;
    int64_t decoder_nccl_port;
    int64_t gpt_nccl_port;
    int64_t dynamic_decoder_nccl_port;
    int64_t nccl_op_port;
};

class DistributedConfig : public th::jit::CustomClassHolder {
public:
    th::intrusive_ptr<MasterInfo> master_info;
    int64_t tp_size;
    int64_t pp_size;
    int64_t world_size;
    int64_t world_rank;
    int64_t local_world_size;
};

class PyModelWeights : public th::jit::CustomClassHolder {
public:
    std::unordered_map<std::string, th::Tensor> model_global_weights_;
    std::vector<std::unordered_map<std::string, th::Tensor>> layer_weights_;
    std::vector<std::unordered_map<std::string, th::Tensor>> layer_int8_weights_;
    std::vector<std::unordered_map<std::string, th::Tensor>> layer_int8_scales_;
};

class MagaInitParams : public th::jit::CustomClassHolder {
public:
    th::intrusive_ptr<GptInitParameter>     gpt_init_parameter;
    th::intrusive_ptr<DistributedConfig>    distributed_config;
    th::intrusive_ptr<PyModelWeights>       model_weights;
};


} // namespace rtp_llm
