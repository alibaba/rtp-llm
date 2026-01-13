#include "gtest/gtest.h"
#include <memory>
#include <unordered_map>
#include <thread>

#define private public
#include "rtp_llm/cpp/model_rpc/test/MockModelRpcServer.h"
#include "rtp_llm/cpp/model_rpc/ModelRpcServer.h"
#include "rtp_llm/cpp/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"

using namespace std;
namespace rtp_llm {

class DynamicLoraModelRpcServiceTest: public DeviceTestBase {
public:
    std::unique_ptr<ModelRpcService::Stub> MakeStub(const std::string& addr) {
        return ModelRpcService::NewStub(grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
    }
};

TEST_F(DynamicLoraModelRpcServiceTest, testSimple) {
    std::vector<int64_t> lora_ids(100);
    std::iota(lora_ids.begin(), lora_ids.end(), 0);
    auto                                          params = createMockEngineInitParams(device_);
    std::unique_ptr<rtp_llm::ModelRpcServiceImpl> model_rpc_server(new ModelRpcServiceImpl(params));
    grpc::ServerBuilder                           builder;
    int                                           port = 0;
    std::string                                   addr = "0.0.0.0:" + std::to_string(port);
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(model_rpc_server.get());
    std::unique_ptr<grpc::Server> rpc_server = builder.BuildAndStart();

    auto serverWorker = [addr, &model_rpc_server, &rpc_server]() {
        std::cout << "begin server worker" << std::endl;
        rpc_server->Wait();
    };

    auto addLoraWorker = [this, &model_rpc_server](std::vector<int64_t> lora_ids) {
        std::cout << "add lora server worker" << std::endl;
        auto lora_weights = createMockLoraWeights(this->device_);
        for (int i = 0; i < 100; i++) {
            size_t lora_id = lora_ids[std::rand() % lora_ids.size()];
            std::cout << "add lora " << lora_id << std::endl;
            model_rpc_server->addLoRA(lora_id, lora_weights[0], lora_weights[1]);
            std::this_thread::sleep_for(1ms);
        }
    };

    auto removeLoraWorker = [this, &model_rpc_server](std::vector<int64_t> lora_ids) {
        std::cout << "remove server worker" << std::endl;
        for (int i = 0; i < 100; i++) {
            auto lora_id = lora_ids[std::rand() % lora_ids.size()];
            std::cout << "remove lora " << lora_id << std::endl;
            model_rpc_server->removeLoRA(lora_id);
            std::this_thread::sleep_for(1ms);
        }
    };

    std::thread server(serverWorker);
    std::thread addLora(addLoraWorker, lora_ids);
    std::thread removeLora(removeLoraWorker, lora_ids);

    addLora.join();
    removeLora.join();
    rpc_server->Shutdown();
    server.join();

    for (const auto& [key, value] : model_rpc_server->lora_map_mutex_) {
        auto executor_ptr = dynamic_cast<NormalExecutor*>(model_rpc_server->engine_->executor_.get());
        auto lora_weight  = executor_ptr->model_->weights_.layers[0].self_attention_weights.qkv_lora_weights;
        ASSERT_EQ(lora_weight->hasLoraWeight(key), value->alive_);
    }
}

}  // namespace rtp_llm
