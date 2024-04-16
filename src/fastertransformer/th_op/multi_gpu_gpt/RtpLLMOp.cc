#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(const c10::intrusive_ptr<GptInitParameter>&                     gpt_init_params,
                    const std::vector<std::unordered_map<std::string, th::Tensor>>& layer_weights,
                    const c10::Dict<std::string, th::Tensor>&                       weights) {
    rtp_llm::MagaInitParams params;
    params.gpt_init_parameter = gpt_init_params;
    std::unordered_map<std::string, ft::ConstBufferPtr>              global_weights;
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights_;
    for (auto& it : weights) {
        global_weights.emplace(it.key(), ft::torchTensor2Buffer(it.value()));
    }
    for (auto& weights : layer_weights) {
        std::unordered_map<std::string, ft::ConstBufferPtr> __weights;
        for (auto& it : weights) {
            __weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        layer_weights_.emplace_back(std::move(__weights));
    }
    server_thread_ = std::thread(&RtpLLMOp::_init, this, params, std::move(layer_weights_), std::move(global_weights));
    server_thread_.detach();
    // _init(params, layer_weights_, global_weights);
}

void RtpLLMOp::_init(const rtp_llm::MagaInitParams                                          params,
                     const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights,
                     const std::unordered_map<std::string, ft::ConstBufferPtr>              weights) {

    std::string                  server_address("0.0.0.0:25333");
    rtp_llm::ModelRpcServiceImpl service(params, layer_weights, weights);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_ = builder.BuildAndStart();

    FT_LOG_INFO("Server listening on %s", server_address.c_str());
    server_->Wait();
}

void RtpLLMOp::stop() {
    server_->Shutdown();
}

RtpLLMOp::~RtpLLMOp() {
    server_->Shutdown();
}

// shared_ptr<rtp_llm::GenerateStream> RtpLLMOp::forward(shared_ptr<rtp_llm::GenerateInput> query) {
//     shared_ptr<rtp_llm::GenerateStream> stream = make_shared<rtp_llm::GenerateStream>(query);
//     // (void)engine_->enqueue(stream);
//     return stream;
// };

}  // namespace torch_ext

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

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::RtpLLMOp>("FasterTransformerRtpLLMOp")
#else
    torch::jit::class_<torch_ext::RtpLLMOp>("FasterTransformer", "RtpLLMOp")
#endif
        .def(torch::jit::init<>())  // quant_pre_scales
        .def("init", &torch_ext::RtpLLMOp::init)
        .def("stop", &torch_ext::RtpLLMOp::stop);
