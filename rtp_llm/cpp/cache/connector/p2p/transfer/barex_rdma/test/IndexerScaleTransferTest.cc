#include <gtest/gtest.h>
#include <cstring>
#include <future>
#include <chrono>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTaskContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"

namespace rtp_llm::transfer::barex_rdma {
namespace {
class Done: public google::protobuf::Closure {
public:
    void Run() override {
        ++count;
    }
    int count = 0;
};
}  // namespace
TEST(IndexerScaleTransferTest, ConvertsBeforeReadyAndRetainsMetadataPastRpcDeadline) {
    for (bool legacy : {false, true})
        for (bool expired : {false, true}) {
            SCOPED_TRACE(legacy);
            SCOPED_TRACE(expired);
            alignas(4) uint32_t source[132];
            alignas(4) uint32_t destination[132] = {};
            std::fill(source, source + 128, 0x7e7e7e7e);  // FP8 448, source scale 3/448.
            const float arbitrary_scale = 3.0f / 448;
            uint32_t    arbitrary_bits;
            std::memcpy(&arbitrary_bits, &arbitrary_scale, 4);
            std::fill(source + 128, source + 132, arbitrary_bits);
            auto request = std::make_unique<rdma_transfer::RdmaLayerBlockTransferRequest>();
            request->set_unique_key("indexer");
            request->set_timeout_ms(5000);
            auto* key = request->add_blocks();
            key->set_key(42);
            auto* block = key->add_blocks();
            block->set_len(sizeof(source));
            block->mutable_rdma_info()->set_addr(reinterpret_cast<uintptr_t>(source));
            auto* rkey = block->mutable_rdma_info()->add_nic_rkeys();
            rkey->set_nicid(0);
            rkey->set_rkey(1);
            if (!legacy)
                IndexerCacheLayout{IndexerScaleFormat::FP32, 4, 528}.toWire(block);
            rdma_transfer::RdmaLayerBlockTransferRequest decoded;
            ASSERT_TRUE(decoded.ParseFromString(request->SerializeAsString()));
            request          = std::make_unique<decltype(decoded)>(std::move(decoded));
            auto      target = std::make_shared<KeyBlockInfo>();
            BlockInfo info;
            info.addr           = destination;
            info.size_bytes     = sizeof(destination);
            info.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
            target->blocks.push_back(info);
            KeyBlockInfoMap targets{{42, target}};
            bool            observed = false;
            auto            task =
                std::make_shared<TransferTask>(targets, currentTimeMs() + 5000, [&](const std::vector<int64_t>& keys) {
                    observed = true;
                    EXPECT_EQ(keys, std::vector<int64_t>{42});
                    for (int i = 0; i < 128; ++i)
                        EXPECT_EQ(destination[i], 0x7c7c7c7c);
                    for (int i = 128; i < 132; ++i)
                        EXPECT_EQ(destination[i], 0x78787878);
                    return true;
                });
            rdma_transfer::RdmaLayerBlockTransferResponse response;
            Done                                          done;
            RdmaTaskContext                               context(nullptr, request.get(), &response, &done, nullptr);
            context.setTask(task);
            auto built = context.buildRdmaBlockPairs();
            ASSERT_TRUE(built.success) << built.error_message;
            ASSERT_EQ(built.pairs.size(), 1);
            ASSERT_TRUE(context.startTransfer());
            // A watchdog may finish and destroy the RPC before the physical callback.
            if (expired)
                context.finishRpc(false, TransferErrorCode::TIMEOUT, "deadline expired");
            EXPECT_EQ(done.count, expired ? 1 : 0);
            request.reset();
            EXPECT_FALSE(observed);
            std::memcpy(built.pairs[0].first.data_ptr(), source, sizeof(source));
            if (expired)
                context.notifyTransferDone(true, TransferErrorCode::OK, "");
            else
                context.run(true);
            EXPECT_EQ(observed, !expired);
            for (int i = 128; i < 132; ++i)
                EXPECT_EQ(destination[i], 0x78787878);
            EXPECT_TRUE(task->done());
            EXPECT_EQ(done.count, 1);
        }
}
}  // namespace rtp_llm::transfer::barex_rdma

namespace rtp_llm::transfer::barex_rdma {
TEST(IndexerScaleTransferTest, DuplicateReceivesOwnCopyAndConversionTogether) {
    alignas(4) uint32_t source[132];
    alignas(4) uint32_t destination[132] = {};
    std::fill(source, source + 128, 0x5a5a5a5a);
    std::fill(source + 128, source + 132, 0x7a7a7a7a);
    auto      target = std::make_shared<KeyBlockInfo>();
    BlockInfo info;
    info.addr           = destination;
    info.size_bytes     = sizeof(destination);
    info.indexer_layout = {IndexerScaleFormat::FP32, 4, 528};
    target->blocks.push_back(info);
    auto task = std::make_shared<TransferTask>(KeyBlockInfoMap{{42, target}}, currentTimeMs() + 5000);
    rdma_transfer::RdmaLayerBlockTransferRequest request;
    request.set_unique_key("duplicate-indexer");
    request.set_timeout_ms(5000);
    auto* key = request.add_blocks();
    key->set_key(42);
    auto* block = key->add_blocks();
    block->set_len(sizeof(source));
    block->mutable_rdma_info()->set_addr(reinterpret_cast<uintptr_t>(source));
    auto* rkey = block->mutable_rdma_info()->add_nic_rkeys();
    rkey->set_nicid(0);
    rkey->set_rkey(1);
    IndexerCacheLayout{IndexerScaleFormat::UE8M0, 4, 528}.toWire(block);
    rdma_transfer::RdmaLayerBlockTransferResponse response_a, response_b;
    Done                                          done_a, done_b;
    RdmaTaskContext                               a(nullptr, &request, &response_a, &done_a, nullptr);
    RdmaTaskContext                               b(nullptr, &request, &response_b, &done_b, nullptr);
    a.setTask(task);
    b.setTask(task);
    ASSERT_TRUE(a.buildRdmaBlockPairs().success);
    ASSERT_TRUE(b.buildRdmaBlockPairs().success);
    ASSERT_TRUE(a.startTransfer());
    std::promise<void> attempting;
    auto               second = std::async(std::launch::async, [&] {
        attempting.set_value();
        return b.startTransfer();
    });
    attempting.get_future().wait();
    const bool waiting = second.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout;
    // Without copy ownership, reproduce both raw writes before either conversion.
    if (!waiting) {
        ASSERT_TRUE(second.get());
        std::memcpy(destination, source, sizeof(source));
    }
    std::memcpy(destination, source, sizeof(source));
    a.run(true);
    EXPECT_TRUE(waiting);
    if (waiting) {
        ASSERT_TRUE(second.get());
        std::memcpy(destination, source, sizeof(source));
    }
    b.run(true);
    EXPECT_TRUE(task->success());
    for (int i = 0; i < 128; ++i)
        EXPECT_EQ(destination[i], source[i]);
    for (int i = 128; i < 132; ++i)
        EXPECT_EQ(destination[i], 0x3d000000);
    EXPECT_EQ(done_a.count, 1);
    EXPECT_EQ(done_b.count, 1);
}
}  // namespace rtp_llm::transfer::barex_rdma

#ifdef TEST_RDMA_SENDER
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheSender.h"
namespace rtp_llm::transfer::barex_rdma {
namespace {
class TestRegistration: public IRdmaMemoryManager {
public:
    int      registered = 0, deregistered = 0;
    uint64_t minRegistrationSize() const override {
        return 1024;
    }
    bool regUserMr(const BlockInfo& info, uint64_t alignment) override {
        EXPECT_EQ(info.size_bytes, 1024);
        EXPECT_EQ(alignment, 1024);
        ++registered;
        return true;
    }
    bool deregUserMr(const BlockInfo&) override {
        ++deregistered;
        return true;
    }
    std::shared_ptr<RemoteBuffer> findMemoryMr(const BlockInfo& block) override {
        return std::make_shared<RemoteBuffer>(
            reinterpret_cast<int64_t>(block.addr),
            block.size_bytes,
            std::make_shared<std::map<uint32_t, uint32_t>>(std::map<uint32_t, uint32_t>{{0, 1}}));
    }
};
class TestServer: public IRdmaServer {
public:
    uint32_t getListenPort() const override {
        return 1;
    }
};
}  // namespace
TEST(IndexerScaleTransferTest, RdmaSenderOwnsRegisteredLegacyCopy) {
    alignas(4) uint32_t source[132];
    std::fill(source, source + 128, 0x5a5a5a5a);
    std::fill(source + 128, source + 132, 0x7a7a7a7a);
    auto key       = std::make_shared<KeyBlockInfo>();
    key->cache_key = 42;
    BlockInfo block;
    block.addr           = source;
    block.size_bytes     = sizeof(source);
    block.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
    key->blocks.push_back(block);
    for (bool supported : {false, true}) {
        auto              memory = std::make_shared<TestRegistration>();
        RdmaKVCacheSender sender(memory);
        sender.rdma_server_ = std::make_shared<TestServer>();
        SendRequest request;
        request.supports_indexer_layout = supported;
        request.block_info.emplace(42, key);
        std::vector<std::shared_ptr<void>> owners;
        auto wire = sender.makeTransferRequest(request, std::make_shared<TransferClientMetricsCollector>(), owners);
        ASSERT_TRUE(wire);
        const auto& offered = wire->blocks(0).blocks(0);
        const auto* copied  = reinterpret_cast<const uint32_t*>(offered.rdma_info().addr());
        EXPECT_EQ(copied[128], supported ? 0x7a7a7a7a : 0x3d000000);
        EXPECT_EQ(memory->registered, supported ? 0 : 1);
        EXPECT_EQ(memory->deregistered, 0);
        EXPECT_EQ(source[128], 0x7a7a7a7a);
        owners.clear();
        EXPECT_EQ(memory->deregistered, supported ? 0 : 1);
    }
}
}  // namespace rtp_llm::transfer::barex_rdma
#endif

namespace rtp_llm::transfer::barex_rdma {
TEST(IndexerScaleTransferTest, OrdinaryOpaqueCopyDoesNotEnterFormatConversionGuard) {
    const std::string source(17, 'x');
    char              destination[17] = {};
    auto              key             = std::make_shared<KeyBlockInfo>();
    BlockInfo         info;
    info.addr       = destination;
    info.size_bytes = sizeof(destination);
    key->blocks.push_back(info);
    auto task = std::make_shared<TransferTask>(KeyBlockInfoMap{{42, key}}, currentTimeMs() + 5000);
    rdma_transfer::RdmaLayerBlockTransferRequest request;
    request.set_unique_key("ordinary");
    request.set_timeout_ms(1000);
    auto* offered = request.add_blocks();
    offered->set_key(42);
    auto* block = offered->add_blocks();
    block->set_len(source.size());
    block->mutable_rdma_info()->set_addr(reinterpret_cast<uintptr_t>(source.data()));
    auto* rkey = block->mutable_rdma_info()->add_nic_rkeys();
    rkey->set_nicid(0);
    rkey->set_rkey(1);
    rdma_transfer::RdmaLayerBlockTransferResponse response;
    Done                                          done;
    RdmaTaskContext                               context(nullptr, &request, &response, &done, nullptr);
    context.setTask(task);
    auto built = context.buildRdmaBlockPairs();
    ASSERT_TRUE(built.success) << built.error_message;
    ASSERT_EQ(built.pairs.size(), 1);
    ASSERT_TRUE(context.startTransfer());
    const bool unguarded = task->beginCacheWrite(true, currentTimeMs() + 20);
    if (unguarded)
        task->endCacheWrite(true);
    ASSERT_TRUE(unguarded);
    std::memcpy(built.pairs[0].first.data_ptr(), source.data(), source.size());
    context.run(true);
    EXPECT_TRUE(task->success());
    EXPECT_EQ(std::memcmp(destination, source.data(), source.size()), 0);
    EXPECT_EQ(done.count, 1);
}
}  // namespace rtp_llm::transfer::barex_rdma
