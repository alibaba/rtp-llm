#include "maga_transformer/cpp/MagaOp.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"

using namespace std;
using namespace torch;

namespace rtp_llm {

MagaOp::MagaOp(const c10::intrusive_ptr<GptInitParameter>& maga_init_params,
               const std::vector<std::unordered_map<std::string, th::Tensor>>& layer_weights,
               const std::unordered_map<std::string, th::Tensor>&              weights) {
    // MagaInitParams params;
    // params.gpt_init_parameter = maga_init_params;
    // std::string server_address("0.0.0.0:25333");
    // ModelRpcServiceImpl service(params, layer_weights, weights);

    // grpc::ServerBuilder builder;
    // // 监听端口50051，不使用任何身份验证机制
    // builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // // 注册"StreamingOutputService"服务的实现
    // builder.RegisterService(&service);

    // // Assemble the server.
    // server_ = builder.BuildAndStart();
    // std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    // 等待服务器终止
    // server->Wait();
}

MagaOp::~MagaOp() {
    server_->Shutdown();
    server_->Wait();
}

shared_ptr<GenerateStream> MagaOp::forward(shared_ptr<GenerateInput> query) {
    shared_ptr<GenerateStream> stream = make_shared<GenerateStream>(query);
    // (void)engine_->enqueue(stream);
    return stream;
};

}  // namespace rtp_llm

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
