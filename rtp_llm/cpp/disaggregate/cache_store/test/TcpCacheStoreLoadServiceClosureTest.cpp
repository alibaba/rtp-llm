#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class TcpCacheStoreLoadServiceClosureTest: public CacheStoreTestBase {
protected:
    TcpCacheStoreLoadServiceClosure*
    makeClosure(arpc::ErrorCode arpc_ec, KvCacheStoreServiceErrorCode resp_ec, CacheStoreLoadDoneCallback callback);
};

TcpCacheStoreLoadServiceClosure* TcpCacheStoreLoadServiceClosureTest::makeClosure(arpc::ErrorCode              arpc_ec,
                                                                                  KvCacheStoreServiceErrorCode resp_ec,
                                                                                  CacheStoreLoadDoneCallback callback) {
    auto request_buffer = std::make_shared<RequestBlockBuffer>("request-id");
    auto controller     = new arpc::ANetRPCController();
    auto request        = new CacheLoadRequest;
    auto response       = new CacheLoadResponse;
    auto collector      = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);

    if (arpc_ec != arpc::ARPC_ERROR_NONE) {
        controller->SetFailed("failed");
        controller->SetErrorCode(arpc_ec);
    }

    response->set_error_code(resp_ec);

    return new TcpCacheStoreLoadServiceClosure(
        memory_util_, request_buffer, controller, request, response, callback, collector);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_Success) {
    bool called   = false;
    auto callback = [&called](bool ok, CacheStoreErrorCode ec) {
        called = true;
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);
    closure->Run();

    ASSERT_TRUE(called);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_ControllerFailed) {
    bool called   = false;
    auto callback = [&called](bool ok, CacheStoreErrorCode ec) {
        called = true;
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed, ec);
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_CONNECTION_CLOSED, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);
    closure->Run();

    ASSERT_TRUE(called);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_ResponseFailed) {
    bool called   = false;
    auto callback = [&called](bool ok, CacheStoreErrorCode ec) {
        called = true;
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER, callback);
    ASSERT_TRUE(closure != nullptr);
    closure->Run();

    ASSERT_TRUE(called);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_BlockSizeError) {
    bool called   = false;
    auto callback = [&called](bool ok, CacheStoreErrorCode ec) {
        called = true;
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);

    uint32_t block_size = 16;
    closure->request_block_buffer_->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    closure->Run();

    ASSERT_TRUE(called);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_BlockContentError) {
    bool called   = false;
    auto callback = [&called](bool ok, CacheStoreErrorCode ec) {
        called = true;
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);

    uint32_t block_size = 16;
    closure->request_block_buffer_->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    closure->response_->add_blocks()->set_len(0);
    closure->Run();

    ASSERT_TRUE(called);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_MultipleCpuBlocksReadyBeforeCallback) {
    auto first    = std::make_shared<std::vector<char>>(16, '.');
    auto second   = std::make_shared<std::vector<char>>(17, '.');
    bool called   = false;
    auto callback = [&](bool ok, CacheStoreErrorCode ec) {
        called = true;
        ASSERT_TRUE(ok);
        ASSERT_EQ(ec, CacheStoreErrorCode::None);
        ASSERT_EQ(*first, std::vector<char>(16, 'a'));
        ASSERT_EQ(*second, std::vector<char>(17, 'b'));
    };
    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    closure->request_block_buffer_->addBlock("a", std::shared_ptr<void>(first, first->data()), 16, false, false);
    closure->request_block_buffer_->addBlock("b", std::shared_ptr<void>(second, second->data()), 17, false, false);
    for (const auto& key : {"b", "a"}) {
        auto block = closure->response_->add_blocks();
        block->set_key(key);
        block->set_len(key[0] == 'a' ? 16 : 17);
        block->set_content(std::string(block->len(), key[0]));
    }
    closure->Run();
    ASSERT_TRUE(called);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_GpuBlocksReadyBeforeCallback) {
    auto gpu      = torch::zeros({257, 128}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto expected = torch::zeros({257, 128}, torch::kUInt8);
    int  calls    = 0;
    auto callback = [&](bool ok, CacheStoreErrorCode ec) {
        ++calls;
        ASSERT_TRUE(ok);
        ASSERT_EQ(ec, CacheStoreErrorCode::None);
        ASSERT_TRUE(torch::equal(gpu.cpu(), expected));
    };
    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    for (int i = 0; i < 257; ++i) {
        auto key = std::to_string(i);
        // Gaps between destination ranges must remain unchanged.
        auto addr = std::shared_ptr<void>(gpu.data_ptr<uint8_t>() + i * 128, [gpu](void*) {});
        closure->request_block_buffer_->addBlock(key, addr, 64, true, false);
        auto block = closure->response_->add_blocks();
        block->set_key(key);
        block->set_len(64);
        block->set_content(std::string(64, static_cast<char>(i % 251)));
        expected[i].slice(0, 0, 64).fill_(i % 251);
    }
    runtimeSyncAndCheck();  // Complete the poison producer before the copy-pool stream.
    closure->Run();
    ASSERT_EQ(calls, 1);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_RejectsMalformedLastBlockBeforeAnyCopy) {
    for (int malformed = 0; malformed < 4; ++malformed) {
        auto first    = std::make_shared<std::vector<char>>(16, '.');
        auto second   = std::make_shared<std::vector<char>>(16, '.');
        bool called   = false;
        auto callback = [&](bool ok, CacheStoreErrorCode ec) {
            called = true;
            ASSERT_FALSE(ok);
            ASSERT_EQ(ec, CacheStoreErrorCode::LoadBufferTimeout);
        };
        auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
        closure->request_block_buffer_->addBlock("a", std::shared_ptr<void>(first, first->data()), 16, false, false);
        closure->request_block_buffer_->addBlock("b", std::shared_ptr<void>(second, second->data()), 16, false, false);
        auto a = closure->response_->add_blocks();
        a->set_key("a");
        a->set_len(16);
        a->set_content(std::string(16, 'a'));
        auto b = closure->response_->add_blocks();
        b->set_key(malformed == 2 ? "a" : malformed == 3 ? "unknown" : "b");
        b->set_len(malformed == 0 ? 17 : 16);
        b->set_content(std::string(malformed == 1 ? 15 : b->len(), 'b'));
        closure->Run();
        ASSERT_TRUE(called);
        ASSERT_EQ(*first, std::vector<char>(16, '.'));
        ASSERT_EQ(*second, std::vector<char>(16, '.'));
    }
}

}  // namespace rtp_llm
