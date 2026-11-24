#include <gtest/gtest.h>

#include <string>

#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.h"

namespace rtp_llm {
namespace {

class CountingDone: public google::protobuf::Closure {
public:
    void Run() override {
        ++calls;
    }
    int calls = 0;
};

std::shared_ptr<BlockBuffer> hostBlock(const std::string& key, const std::shared_ptr<std::string>& owner) {
    return std::make_shared<BlockBuffer>(key, std::shared_ptr<void>(owner, owner->data()), owner->size(), false, true);
}

TEST(TcpHostPayloadTest, HostBlockReadUsesRegisteredOwner) {
    auto                       store          = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto                       owner          = std::make_shared<std::string>("ab\0cDEfg", 8);
    std::weak_ptr<std::string> borrowed_owner = owner;
    auto                       block          = hostBlock("snapshot", owner);
    auto                       address        = reinterpret_cast<int64_t>(block->addr.get());
    ASSERT_TRUE(store->regUserBuffers({block}));
    owner.reset();
    block.reset();
    ASSERT_FALSE(borrowed_owner.expired());

    TcpCacheStoreServiceImpl service(nullptr, store, nullptr, nullptr, nullptr, nullptr, -1);
    BlockReadRequest         request;
    auto                     info = request.add_blocks();
    info->set_key("snapshot");
    info->set_addr(address + 2);
    info->set_len(4);
    BlockReadResponse response;
    CountingDone      done;
    EXPECT_NO_THROW(service.blockRead(nullptr, &request, &response, &done));
    EXPECT_EQ(1, done.calls);
    ASSERT_EQ(EC_SUCCESS, response.error_code());
    ASSERT_EQ(1, response.blocks_size());
    EXPECT_EQ("snapshot", response.blocks(0).key());
    EXPECT_EQ(address + 2, response.blocks(0).addr());
    EXPECT_EQ(4, response.blocks(0).len());
    EXPECT_EQ(std::string("\0cDE", 4),
              std::string(response.blocks(0).content().data(), response.blocks(0).content().size()));
}

TEST(TcpHostPayloadTest, GpuBlockReadPreservesPayload) {
    auto owner = std::make_shared<torch::Tensor>(
        torch::tensor(std::vector<uint8_t>{0, 66, 255, 49}, torch::TensorOptions().dtype(torch::kUInt8))
            .to(torch::kCUDA));
    auto store = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto block =
        std::make_shared<BlockBuffer>("gpu-snapshot", std::shared_ptr<void>(owner, owner->data_ptr()), 4, true, true);
    ASSERT_TRUE(store->regUserBuffers({block}));
    TcpCacheStoreServiceImpl service(nullptr, store, nullptr, nullptr, nullptr, nullptr, 0);
    BlockReadRequest         request;
    auto                     info = request.add_blocks();
    info->set_key("gpu-snapshot");
    info->set_addr(reinterpret_cast<int64_t>(block->addr.get()));
    info->set_len(4);
    BlockReadResponse response;
    CountingDone      done;
    service.blockRead(nullptr, &request, &response, &done);
    EXPECT_EQ(1, done.calls);
    ASSERT_EQ(EC_SUCCESS, response.error_code());
    ASSERT_EQ(1, response.blocks_size());
    EXPECT_EQ(std::string("\0B\xff\x31", 4),
              std::string(response.blocks(0).content().data(), response.blocks(0).content().size()));
}

TEST(TcpHostPayloadTest, RejectReadPastRegisteredExtent) {
    auto store = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto owner = std::make_shared<std::string>("snapshot");
    auto block = hostBlock("snapshot", owner);
    block->len = 4;
    ASSERT_TRUE(store->regUserBuffers({block}));
    TcpCacheStoreServiceImpl service(nullptr, store, nullptr, nullptr, nullptr, nullptr, -1);
    BlockReadRequest         request;
    auto                     info = request.add_blocks();
    info->set_key("snapshot");
    info->set_addr(reinterpret_cast<int64_t>(owner->data()) + 2);
    info->set_len(4);
    BlockReadResponse response;
    CountingDone      done;
    EXPECT_NO_THROW(service.blockRead(nullptr, &request, &response, &done));
    EXPECT_EQ(1, done.calls);
    EXPECT_NE(EC_SUCCESS, response.error_code());
    EXPECT_EQ(0, response.blocks_size());
    EXPECT_EQ("snapshot", *owner);
}

TEST(TcpHostPayloadTest, HostLoadCopiesExactPayload) {
    auto owner       = std::make_shared<std::string>("????");
    auto destination = std::make_shared<RequestBlockBuffer>("load");
    destination->addBlock(hostBlock("payload", owner));
    auto response = new CacheLoadResponse;
    response->set_error_code(EC_SUCCESS);
    auto block = response->add_blocks();
    block->set_key("payload");
    block->set_len(4);
    block->set_content("A\0z!", 4);
    int  calls   = 0;
    auto closure = new TcpCacheStoreLoadServiceClosure(
        nullptr,
        destination,
        new arpc::ANetRPCController,
        new CacheLoadRequest,
        response,
        [&](bool ok, CacheStoreErrorCode ec) {
            ++calls;
            EXPECT_TRUE(ok);
            EXPECT_EQ(CacheStoreErrorCode::None, ec);
        },
        std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 4),
        -1);
    closure->Run();
    EXPECT_EQ(1, calls);
    EXPECT_EQ(std::string("A\0z!", 4), *owner);
}

// Every malformed second block must be detected before the valid first block is copied.
void checkRejectedLoad(const std::function<void(CacheLoadResponse*)>& corrupt) {
    auto first       = std::make_shared<std::string>("1111");
    auto second      = std::make_shared<std::string>("2222");
    auto destination = std::make_shared<RequestBlockBuffer>("load");
    destination->addBlock(hostBlock("first", first));
    destination->addBlock(hostBlock("second", second));
    auto response = new CacheLoadResponse;
    response->set_error_code(EC_SUCCESS);
    for (const auto& key : {"first", "second"}) {
        auto block = response->add_blocks();
        block->set_key(key);
        block->set_len(4);
        block->set_content("ABCD");
    }
    corrupt(response);
    int  calls   = 0;
    auto closure = new TcpCacheStoreLoadServiceClosure(
        nullptr,
        destination,
        new arpc::ANetRPCController,
        new CacheLoadRequest,
        response,
        [&](bool ok, CacheStoreErrorCode ec) {
            ++calls;
            EXPECT_FALSE(ok);
            EXPECT_NE(CacheStoreErrorCode::None, ec);
        },
        std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 2, 8),
        -1);
    closure->Run();
    EXPECT_EQ(1, calls);
    EXPECT_EQ("1111", *first);
    EXPECT_EQ("2222", *second);
}

TEST(TcpHostPayloadTest, RejectTruncatedLoadPayload) {
    checkRejectedLoad([](auto response) { response->mutable_blocks(1)->set_content("X"); });
}

TEST(TcpHostPayloadTest, RejectMalformedLoadPayloadBeforeAnyCopy) {
    checkRejectedLoad([](auto response) { response->mutable_blocks()->RemoveLast(); });
    checkRejectedLoad([](auto response) { response->mutable_blocks(1)->set_key("unknown"); });
    checkRejectedLoad([](auto response) { response->mutable_blocks(1)->set_key("first"); });
    checkRejectedLoad([](auto response) { response->mutable_blocks(1)->set_len(3); });
    checkRejectedLoad([](auto response) { response->mutable_blocks(1)->set_content("ABCDE"); });
}

}  // namespace
}  // namespace rtp_llm
