#include <gtest/gtest.h>
#include <cstring>
#include <future>
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreLoadServiceClosure.h"

namespace rtp_llm {
namespace {
class Done: public google::protobuf::Closure {
public:
    int  count = 0;
    void Run() override {
        ++count;
    }
};
class Registration: public MemoryUtil {
public:
    int      registered = 0, deregistered = 0;
    uint64_t minRegistrationSize() const override {
        return 1024;
    }
    bool regUserMr(void*, uint64_t size, bool, uint64_t stride) override {
        EXPECT_EQ(size, 1024);
        EXPECT_EQ(stride, 1024);
        ++registered;
        return true;
    }
    bool deregUserMr(void*, bool) override {
        ++deregistered;
        return true;
    }
    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return false;
    }
    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return false;
    }
    bool isRdmaMode() override {
        return true;
    }
};
}  // namespace
TEST(IndexerScaleConversionTest, LegacyCopyRegistrationLivesWithTransferOwner) {
    auto bytes = std::make_shared<std::vector<uint32_t>>(132, 0x5a5a5a5a);
    std::fill(bytes->begin() + 128, bytes->end(), 0x7a7a7a7a);
    auto block        = std::make_shared<BlockBuffer>("indexer",
                                               std::shared_ptr<void>(bytes, bytes->data()),
                                               528,
                                               false,
                                               true,
                                               IndexerCacheLayout{IndexerScaleFormat::UE8M0, 4, 528});
    auto registration = std::make_shared<Registration>();
    auto wire         = legacyIndexerBlock(block, registration);
    EXPECT_EQ(registration->registered, 1);
    EXPECT_EQ(registration->deregistered, 0);
    EXPECT_EQ(wire->indexer_layout.format, IndexerScaleFormat::FP32);
    EXPECT_EQ(static_cast<uint32_t*>(wire->addr.get())[128], 0x3d000000);
    EXPECT_EQ((*bytes)[128], 0x7a7a7a7a);
    auto retained = wire->addr;
    wire.reset();
    EXPECT_EQ(registration->deregistered, 0);
    retained.reset();
    EXPECT_EQ(registration->deregistered, 1);
}
TEST(IndexerScaleConversionTest, TcpLoadNegotiatesNativeAndLegacyEncoding) {
    auto bytes = std::make_shared<std::vector<uint32_t>>(132, 0x5a5a5a5a);
    std::fill(bytes->begin() + 128, bytes->end(), 0x7a7a7a7a);
    auto block = std::make_shared<BlockBuffer>("indexer",
                                               std::shared_ptr<void>(bytes, bytes->data()),
                                               528,
                                               false,
                                               true,
                                               IndexerCacheLayout{IndexerScaleFormat::UE8M0, 4, 528});
    for (bool supported : {false, true}) {
        CacheLoadRequest request;
        request.set_requestid("mixed-version");
        auto* offered = request.add_blocks();
        offered->set_key("indexer");
        offered->set_len(528);
        if (supported)
            block->indexer_layout.toWire(offered);
        CacheLoadResponse response;
        Done              done;
        auto              context = std::make_shared<TcpCacheStoreServiceImplContext>(
            &request,
            &response,
            std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 1, 528, 0),
            &done,
            nullptr);
        context->loadBlockOnTcp(true, {block});
        ASSERT_EQ(done.count, 1);
        ASSERT_EQ(response.blocks_size(), 1);
        uint32_t scale;
        std::memcpy(&scale, response.blocks(0).content().data() + 512, 4);
        EXPECT_EQ(scale, supported ? 0x7a7a7a7a : 0x3d000000);
        EXPECT_EQ((*bytes)[128], 0x7a7a7a7a);
    }
}
TEST(IndexerScaleConversionTest, RdmaCapabilityProbePreservesUnsupportedBlockReads) {
    RdmaCacheStoreServiceImpl service(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    for (bool supported : {false, true}) {
        for (bool empty : {false, true}) {
            BlockReadRequest  request;
            BlockReadResponse response;
            Done              done;
            request.set_supports_indexer_layout(supported);
            if (!empty)
                request.add_blocks()->set_key("indexer");
            service.blockRead(nullptr, &request, &response, &done);
            EXPECT_EQ(done.count, 1);
            EXPECT_TRUE(response.supports_indexer_layout());
            EXPECT_EQ(response.error_code(), supported && empty ? EC_SUCCESS : EC_FAILED_INTERNAL);
            EXPECT_EQ(response.blocks_size(), 0);
        }
    }
}

namespace {
class ProbeMessager: public RdmaMessager {
public:
    using RdmaMessager::RdmaMessager;
    using Messager::makeLoadRequest;
    void setClient(const std::shared_ptr<TcpClient>& client) {
        tcp_client_ = client;
    }

protected:
    bool generateBlockInfo(BlockBufferInfo*                    info,
                           const std::shared_ptr<BlockBuffer>& block,
                           uint32_t                            count,
                           uint32_t                            id) override {
        info->set_key(block->key);
        info->set_len(block->len / count);
        info->set_addr(reinterpret_cast<int64_t>(block->addr.get()) + id * info->len());
        block->indexer_layout.toWire(info);
        return true;
    }
};
class ProbeService: public KvCacheStoreService {
public:
    bool supported = false;
    int  probes = 0, transfers = 0;
    void blockRead(google::protobuf::RpcController*,
                   const BlockReadRequest*    request,
                   BlockReadResponse*         response,
                   google::protobuf::Closure* done) override {
        ++probes;
        EXPECT_EQ(request->blocks_size(), 0);
        if (supported)
            response->set_supports_indexer_layout(true);
        response->set_error_code(supported ? EC_SUCCESS : EC_FAILED_INTERNAL);
        done->Run();
    }
    void transfer(google::protobuf::RpcController*,
                  const CacheTransferRequest* request,
                  CacheTransferResponse*      response,
                  google::protobuf::Closure*  done) override {
        ++transfers;
        EXPECT_EQ(request->blocks_size(), 1);
        const auto& block = request->blocks(0);
        const auto* bytes = reinterpret_cast<const uint32_t*>(block.addr());
        EXPECT_EQ(bytes[128], supported ? 0x7a7a7a7a : 0x3d000000);
        EXPECT_EQ(block.indexer_scale_format(),
                  static_cast<uint32_t>(supported ? IndexerScaleFormat::UE8M0 : IndexerScaleFormat::FP32));
        EXPECT_GT(request->timeout_ms(), 0);
        EXPECT_LT(request->timeout_ms(), 5000);
        response->set_error_code(EC_SUCCESS);
        done->Run();
    }
};
}  // namespace
TEST(IndexerScaleConversionTest, RemoteLoadNegotiatesBeforeExposingMxAddress) {
    for (bool supported : {false, true}) {
        SCOPED_TRACE(supported);
        ProbeService service;
        service.supported = supported;
        TcpServer server;
        ASSERT_TRUE(server.init(1, 1, false));
        ASSERT_TRUE(server.registerService(&service));
        const auto port = autil::NetUtil::randomPort();
        ASSERT_TRUE(server.start(port));
        auto client = std::make_shared<TcpClient>();
        ASSERT_TRUE(client->init(1));
        auto memory = std::make_shared<Registration>();
        auto store  = std::make_shared<RequestBlockBufferStore>(memory);
        auto bytes  = std::make_shared<std::vector<uint32_t>>(132, 0x5a5a5a5a);
        std::fill(bytes->begin() + 128, bytes->end(), 0x7a7a7a7a);
        auto block = std::make_shared<BlockBuffer>("indexer",
                                                   std::shared_ptr<void>(bytes, bytes->data()),
                                                   528,
                                                   false,
                                                   true,
                                                   IndexerCacheLayout{IndexerScaleFormat::UE8M0, 4, 528});
        ASSERT_TRUE(store->regUserBuffers({block}));
        auto sender = std::make_shared<ProbeMessager>(memory, store, nullptr);
        sender->setClient(client);
        auto request = std::make_shared<TransferRequest>("mixed", "127.0.0.1", port, 0, 5000);
        request->buffer_pairs.emplace("indexer", "destination");
        std::promise<bool> completion;
        request->callback = [&](bool ok, CacheStoreErrorCode, const auto&) { completion.set_value(ok); };
        auto ready        = completion.get_future();
        sender->transfer(request);
        ASSERT_EQ(ready.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        EXPECT_TRUE(ready.get());
        EXPECT_EQ(service.probes, 1);
        EXPECT_EQ(service.transfers, 1);
        EXPECT_EQ(memory->registered, supported ? 0 : 1);
        EXPECT_EQ((*bytes)[128], 0x7a7a7a7a);
    }
}
TEST(IndexerScaleConversionTest, TcpLoadRequestAdvertisesLocalEncoding) {
    auto memory  = std::make_shared<Registration>();
    auto sender  = std::make_shared<ProbeMessager>(memory, nullptr, nullptr);
    auto bytes   = std::make_shared<std::vector<uint32_t>>(132, 0);
    auto buffers = std::make_shared<RequestBlockBuffer>("native");
    buffers->addBlock(
        "indexer", std::shared_ptr<void>(bytes, bytes->data()), 528, false, true, {IndexerScaleFormat::UE8M0, 4, 528});
    auto request = std::make_shared<LoadRequest>("127.0.0.1", 1, 0, buffers, nullptr, 1000, 1, 0);
    std::unique_ptr<CacheLoadRequest> wire(sender->makeLoadRequest(request));
    ASSERT_EQ(wire->blocks_size(), 1);
    EXPECT_EQ(wire->blocks(0).indexer_scale_format(), static_cast<uint32_t>(IndexerScaleFormat::UE8M0));
    EXPECT_EQ(wire->blocks(0).indexer_page_tokens(), 4);
    EXPECT_EQ(wire->blocks(0).indexer_page_stride(), 528);
}
TEST(IndexerScaleConversionTest, ConvertsSourceScalesBeforeLoadCallback) {
    for (bool mx_source : {false, true}) {
        SCOPED_TRACE(mx_source);
        auto        bytes           = std::make_shared<std::vector<uint32_t>>(132, mx_source ? 0x5a5a5a5a : 0x7e7e7e7e);
        const float arbitrary_scale = 3.0f / 448;
        uint32_t    arbitrary_bits;
        std::memcpy(&arbitrary_bits, &arbitrary_scale, 4);
        for (int i = 128; i < 132; ++i)
            (*bytes)[i] = mx_source ? 0x7a7a7a7a : arbitrary_bits;
        const auto source  = mx_source ? IndexerScaleFormat::UE8M0 : IndexerScaleFormat::FP32;
        const auto target  = mx_source ? IndexerScaleFormat::FP32 : IndexerScaleFormat::UE8M0;
        auto       buffers = std::make_shared<RequestBlockBuffer>("indexer");
        buffers->addBlock("indexer", std::shared_ptr<void>(bytes, bytes->data()), 528, false, true, {target, 4, 528});
        auto* response = new CacheLoadResponse;
        response->set_error_code(EC_SUCCESS);
        response->set_direct_write_response(true);
        auto* info = response->add_blocks();
        info->set_key("indexer");
        info->set_len(528);
        IndexerCacheLayout{source, 4, 528}.toWire(info);
        bool ready    = false;
        auto callback = [&](bool ok, CacheStoreErrorCode error) {
            ready = true;
            EXPECT_TRUE(ok);
            EXPECT_EQ(error, CacheStoreErrorCode::None);
            for (int i = 0; i < 128; ++i)
                EXPECT_EQ((*bytes)[i], mx_source ? 0x5a5a5a5a : 0x7c7c7c7c);
            for (int i = 128; i < 132; ++i)
                EXPECT_EQ((*bytes)[i], mx_source ? 0x3d000000 : 0x78787878);
        };
        auto* closure =
            new RdmaCacheStoreLoadServiceClosure(nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 buffers,
                                                 new arpc::ANetRPCController,
                                                 new CacheLoadRequest,
                                                 response,
                                                 callback,
                                                 std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1));
        closure->Run();
        EXPECT_TRUE(ready);
    }
}
TEST(IndexerScaleConversionTest, LegacyResponseWithoutDescriptorsUsesFp32) {
    for (auto target : {IndexerScaleFormat::FP32, IndexerScaleFormat::UE8M0}) {
        auto bytes   = std::make_shared<std::vector<uint32_t>>(132, 0);
        auto buffers = std::make_shared<RequestBlockBuffer>("missing-format");
        buffers->addBlock("indexer", std::shared_ptr<void>(bytes, bytes->data()), 528, false, true, {target, 4, 528});
        for (int i = 128; i < 132; ++i)
            (*bytes)[i] = 0x3d000000;
        auto* response = new CacheLoadResponse;
        response->set_error_code(EC_SUCCESS);
        bool  called  = false;
        auto* closure = new RdmaCacheStoreLoadServiceClosure(
            nullptr,
            nullptr,
            nullptr,
            buffers,
            new arpc::ANetRPCController,
            new CacheLoadRequest,
            response,
            [&](bool ok, CacheStoreErrorCode error) {
                called = true;
                EXPECT_TRUE(ok);
                EXPECT_EQ(error, CacheStoreErrorCode::None);
            },
            std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1));
        closure->Run();
        EXPECT_TRUE(called);
        EXPECT_EQ((*bytes)[128], target == IndexerScaleFormat::FP32 ? 0x3d000000 : 0x7a7a7a7a);
    }
}
}  // namespace rtp_llm
