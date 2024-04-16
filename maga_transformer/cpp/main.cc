
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"

int main() {
    std::string                                                      server_address("0.0.0.0:25333");
    rtp_llm::MagaInitParams                                          maga_init_params;
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights;
    std::unordered_map<std::string, ft::ConstBufferPtr>              weights;
    rtp_llm::ModelRpcServiceImpl                                     service(maga_init_params, layer_weights, weights);

    grpc::ServerBuilder builder;
    // 监听端口50051，不使用任何身份验证机制
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // 注册"StreamingOutputService"服务的实现
    builder.RegisterService(&service);

    // Assemble the server.
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // 等待服务器终止
    server->Wait();
}
