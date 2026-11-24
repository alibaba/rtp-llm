#include <gtest/gtest.h>

#include "autil/NetUtil.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {
namespace {

class LoopbackService: public KvCacheStoreService {
public:
    void load(google::protobuf::RpcController*,
              const CacheLoadRequest*    request,
              CacheLoadResponse*         response,
              google::protobuf::Closure* done) override {
        response->set_error_code(EC_SUCCESS);
        auto block = response->add_blocks();
        block->set_key(request->blocks(0).key());
        block->set_len(5);
        block->set_content("tcp\0!", 5);
        done->Run();
    }
    void transfer(google::protobuf::RpcController*,
                  const CacheTransferRequest*,
                  CacheTransferResponse*     response,
                  google::protobuf::Closure* done) override {
        response->set_error_code(EC_FAILED_INVALID_REQ);
        done->Run();
    }
    void blockRead(google::protobuf::RpcController*,
                   const BlockReadRequest*,
                   BlockReadResponse*         response,
                   google::protobuf::Closure* done) override {
        response->set_error_code(EC_FAILED_INVALID_REQ);
        done->Run();
    }
};

TEST(TcpTransportCpuTest, CpuTransportOpensRealLoopbackChannel) {
    LoopbackService service;
    TcpServer       server;
    ASSERT_TRUE(server.init(1, 1, false));
    ASSERT_TRUE(server.registerService(&service));
    auto port = autil::NetUtil::randomPort();
    ASSERT_TRUE(server.start(port));
    TcpClient client;
    ASSERT_TRUE(client.init(1));
    auto channel = client.getChannel("127.0.0.1", port);
    ASSERT_NE(nullptr, channel);
    EXPECT_FALSE(channel->ChannelBroken());
    EXPECT_EQ(channel, client.getChannel("127.0.0.1", port));
    KvCacheStoreService_Stub stub(channel.get(), google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    arpc::ANetRPCController  controller;
    controller.SetExpireTime(2000);
    CacheLoadRequest request;
    request.add_blocks()->set_key("loopback");
    CacheLoadResponse response;
    stub.load(&controller, &request, &response, nullptr);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    EXPECT_EQ(EC_SUCCESS, response.error_code());
    ASSERT_EQ(1, response.blocks_size());
    EXPECT_EQ("loopback", response.blocks(0).key());
    EXPECT_EQ(5, response.blocks(0).len());
    EXPECT_EQ(std::string("tcp\0!", 5),
              std::string(response.blocks(0).content().data(), response.blocks(0).content().size()));
}

}  // namespace
}  // namespace rtp_llm
