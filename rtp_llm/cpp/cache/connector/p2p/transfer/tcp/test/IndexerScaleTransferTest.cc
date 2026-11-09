#include <gtest/gtest.h>
#include <future>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"
#include <cstring>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTaskContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.pb.h"

namespace rtp_llm::transfer::tcp {
namespace {
class Done: public google::protobuf::Closure {
public:
    void Run() override {
        ++count;
    }
    int count = 0;
};
}  // namespace
TEST(IndexerScaleTransferTest, SenderConvertsOnlyPrivateLegacyPayload) {
    alignas(4) uint32_t source[132];
    std::fill(source, source + 128, 0x5a5a5a5a);
    std::fill(source + 128, source + 132, 0x7a7a7a7a);
    auto      key = std::make_shared<KeyBlockInfo>();
    BlockInfo block;
    block.addr           = source;
    block.size_bytes     = sizeof(source);
    block.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
    key->blocks.push_back(block);
    SendRequest request;
    request.block_info.emplace(42, key);
    TcpKVCacheSender sender;
    for (bool supported : {false, true}) {
        request.supports_indexer_layout = supported;
        auto wire = sender.makeTransferRequest(request, std::make_shared<TransferClientMetricsCollector>());
        ASSERT_TRUE(wire);
        const auto& sent = wire->blocks(0).blocks(0);
        uint32_t    scale;
        std::memcpy(&scale, sent.content().data() + 512, 4);
        EXPECT_EQ(scale, supported ? 0x7a7a7a7a : 0x3d000000);
        EXPECT_EQ(sent.indexer_scale_format(),
                  static_cast<uint32_t>(supported ? IndexerScaleFormat::UE8M0 : IndexerScaleFormat::FP32));
        EXPECT_EQ(source[128], 0x7a7a7a7a);
        EXPECT_EQ(std::memcmp(sent.content().data(), source, 512), 0);
    }
}
TEST(IndexerScaleTransferTest, LegacySenderWithoutMetadataConvertsBeforeReadiness) {
    alignas(4) uint32_t source[132], destination[132] = {};
    std::fill(source, source + 128, 0x5a5a5a5a);
    std::fill(source + 128, source + 132, 0x3d000000);
    auto      key = std::make_shared<KeyBlockInfo>();
    BlockInfo info;
    info.addr           = destination;
    info.size_bytes     = sizeof(destination);
    info.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
    key->blocks.push_back(info);
    auto task = std::make_shared<TransferTask>(KeyBlockInfoMap{{42, key}}, currentTimeMs() + 5000);
    tcp_transfer::TcpLayerBlockTransferRequest request;
    request.set_timeout_ms(5000);
    auto* offered = request.add_blocks();
    offered->set_key(42);
    auto* block = offered->add_blocks();
    block->set_len(sizeof(source));
    block->set_content(source, sizeof(source));
    tcp_transfer::TcpLayerBlockTransferResponse response;
    Done                                        done;
    TcpTaskContext                              context(nullptr, &request, &response, &done, nullptr);
    context.setTask(task);
    ASSERT_TRUE(context.startTransfer());
    CudaCopyUtil copy;
    ASSERT_TRUE(context.executeCopy(copy));
    context.run(true);
    EXPECT_TRUE(task->success());
    EXPECT_EQ(destination[128], 0x7a7a7a7a);
    EXPECT_EQ(std::memcmp(destination, source, 512), 0);
}
TEST(IndexerScaleTransferTest, DuplicateCopyWaitsThroughConversionAndReadiness) {
    alignas(4) uint32_t source[132], destination[132] = {};
    std::fill(source, source + 128, 0x5a5a5a5a);
    std::fill(source + 128, source + 132, 0x7a7a7a7a);
    auto      target = std::make_shared<KeyBlockInfo>();
    BlockInfo info;
    info.addr           = destination;
    info.size_bytes     = sizeof(destination);
    info.indexer_layout = {IndexerScaleFormat::FP32, 4, 528};
    target->blocks.push_back(info);
    auto task = std::make_shared<TransferTask>(KeyBlockInfoMap{{42, target}}, currentTimeMs() + 5000);
    tcp_transfer::TcpLayerBlockTransferRequest request;
    request.set_unique_key("duplicate-indexer");
    request.set_timeout_ms(5000);
    auto* key = request.add_blocks();
    key->set_key(42);
    auto* block = key->add_blocks();
    block->set_len(sizeof(source));
    block->set_content(source, sizeof(source));
    IndexerCacheLayout{IndexerScaleFormat::UE8M0, 4, 528}.toWire(block);
    tcp_transfer::TcpLayerBlockTransferResponse response_a, response_b;
    Done                                        done_a, done_b;
    TcpTaskContext                              a(nullptr, &request, &response_a, &done_a, nullptr);
    TcpTaskContext                              b(nullptr, &request, &response_b, &done_b, nullptr);
    a.setTask(task);
    b.setTask(task);
    ASSERT_TRUE(a.startTransfer());
    ASSERT_TRUE(b.startTransfer());
    CudaCopyUtil copy;
    ASSERT_TRUE(a.executeCopy(copy));
    std::promise<void> attempting;
    auto               second = std::async(std::launch::async, [&] {
        attempting.set_value();
        return b.executeCopy(copy);
    });
    attempting.get_future().wait();
    EXPECT_EQ(second.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    a.run(true);
    EXPECT_FALSE(task->done());
    ASSERT_TRUE(second.get());
    b.run(true);
    EXPECT_TRUE(task->success());
    for (int i = 0; i < 128; ++i)
        EXPECT_EQ(destination[i], source[i]);
    for (int i = 128; i < 132; ++i)
        EXPECT_EQ(destination[i], 0x3d000000);
    EXPECT_EQ(done_a.count, 1);
    EXPECT_EQ(done_b.count, 1);
}
}  // namespace rtp_llm::transfer::tcp

namespace rtp_llm::transfer::tcp {
TEST(IndexerScaleTransferTest, OrdinaryOpaqueCopyDoesNotEnterFormatConversionGuard) {
    const std::string source(17, 'x');
    char              destination[17] = {};
    auto              key             = std::make_shared<KeyBlockInfo>();
    BlockInfo         info;
    info.addr       = destination;
    info.size_bytes = sizeof(destination);
    key->blocks.push_back(info);
    auto task = std::make_shared<TransferTask>(KeyBlockInfoMap{{42, key}}, currentTimeMs() + 5000);
    tcp_transfer::TcpLayerBlockTransferRequest request;
    request.set_timeout_ms(1000);
    auto* offered = request.add_blocks();
    offered->set_key(42);
    auto* block = offered->add_blocks();
    block->set_len(source.size());
    block->set_content(source);
    tcp_transfer::TcpLayerBlockTransferResponse response;
    Done                                        done;
    TcpTaskContext                              context(nullptr, &request, &response, &done, nullptr);
    context.setTask(task);
    ASSERT_TRUE(context.startTransfer());
    ASSERT_TRUE(task->beginCacheWrite(true, currentTimeMs() + 1000));
    CudaCopyUtil copy;
    const bool   copied = context.executeCopy(copy);
    task->endCacheWrite(true);
    ASSERT_TRUE(copied);
    context.run(true);
    EXPECT_TRUE(task->success());
    EXPECT_EQ(std::memcmp(destination, source.data(), source.size()), 0);
    EXPECT_EQ(done.count, 1);
}
}  // namespace rtp_llm::transfer::tcp
