#include <gtest/gtest.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {
#if USING_ROCM
class TcpIsolatedCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        initRuntime(0, false, false, MlaOpsType::AUTO);
    }
    void exercise(bool malformed) {
        auto dst = torch::zeros({8192}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        auto owner = std::make_shared<torch::Tensor>(dst);
        auto buffers = std::make_shared<RequestBlockBuffer>("isolated-copy");
        auto* response = new CacheLoadResponse;
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
        for (int i = 0; i < 2; ++i) {
            const auto key = std::to_string(i);
            std::shared_ptr<void> address(owner, dst.data_ptr<uint8_t>() + i * 4096);
            buffers->addBlock(key, address, 4096, true, true);
            auto* block = response->add_blocks();
            block->set_key(key);
            block->set_len(malformed && i == 1 ? 4095 : 4096);
            block->set_content(std::string(block->len(), static_cast<char>(i + 31)));
        }
        bool called = false;
        auto callback = [&](bool ok, CacheStoreErrorCode error) {
            called = true;
            EXPECT_EQ(ok, !malformed);
            EXPECT_EQ(error, malformed ? CacheStoreErrorCode::LoadBufferTimeout : CacheStoreErrorCode::None);
            // Response memory is released immediately after this callback.
            // The data must already be complete, not merely enqueued.
            auto host = dst.cpu();
            if (malformed) {
                EXPECT_TRUE(host.eq(0).all().item<bool>());
            } else {
                EXPECT_TRUE(host.slice(0, 0, 4096).eq(31).all().item<bool>());
                EXPECT_TRUE(host.slice(0, 4096, 8192).eq(32).all().item<bool>());
            }
        };
        auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);
        auto* closure = new TcpCacheStoreLoadServiceClosure(nullptr, buffers, new arpc::ANetRPCController,
            new CacheLoadRequest, response, callback, collector, 0);
        closure->Run();
        EXPECT_TRUE(called);
    }
};
TEST_F(TcpIsolatedCopyTest, CompletesAllCopiesBeforeCallback) { exercise(false); }
TEST_F(TcpIsolatedCopyTest, RejectsMalformedBatchBeforeWriting) { exercise(true); }
#else
TEST(TcpIsolatedCopyTest, RocmOnly) { GTEST_SKIP() << "ROCm-only isolated cache copy"; }
#endif
}  // namespace rtp_llm
