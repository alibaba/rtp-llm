#include "gtest/gtest.h"
#include <memory>
#include <unordered_map>

#include "rtp_llm/cpp/model_rpc/RpcServer.h"
#include "rtp_llm/cpp/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <grpc++/test/mock_stream.h>

using grpc::testing::MockClientReader;

using namespace std;
namespace rtp_llm {

class RpcServiceTest: public DeviceTestBase {
protected:
};

TEST_F(RpcServiceTest, testSimple) {
    GenerateInputPB input;
    // 设置测试数据

    // 设置流读取器的行为
    GenerateOutputPB output;
    // 设置测试期望的返回值
    // EXPECT_CALL(mock_reader_, Read(_))
    //     .WillOnce(DoAll(SetArgPointee<0>(output), Return(true))) // 第一次调用Read返回true
    //     .WillOnce(Return(false));  // 第二次调用Read返回false，表示流结束

    grpc::ClientContext context;

    MagaInitParams                                                        maga_init_params;
    std::vector<std::unordered_map<std::string, rtp_llm::ConstBufferPtr>> layer_weights;
    std::unordered_map<std::string, rtp_llm::ConstBufferPtr>              weights;

    RpcServiceImpl service(maga_init_params, layer_weights, weights);
    // auto stream = service.generate_stream(&context, input);

    // 读取并验证流数据
    GenerateOutputPB res;

    // // 清理
    // stream->Finish();
}

}  // namespace rtp_llm
