#include "gtest/gtest.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"

namespace rtp_llm {

class PrefillGenerateContextTest: public CacheStoreTestBase {
protected:
    bool initCacheStore() {
        CacheStoreInitParams params;
        params.listen_port   = autil::NetUtil::randomPort();
        params.enable_metric = false;
        params.memory_util   = memory_util_;

        cache_store_ = NormalCacheStore::createNormalCacheStore(params);
        return cache_store_ != nullptr;
    }

    std::shared_ptr<NormalCacheStore> cache_store_;
};

TEST_F(PrefillGenerateContextTest, markRequestEndClearsLocalCacheStoreForTensorParallel) {
    ASSERT_TRUE(initCacheStore());

    const int64_t     request_id = 12345;
    const std::string request_key = std::to_string(request_id);
    auto              request_buffer = std::make_shared<RequestBlockBuffer>(request_key);
    request_buffer->addBlock(block_buffer_util_->makeBlockBuffer("block", 16, '0', false));
    ASSERT_TRUE(cache_store_->getRequestBlockBufferStore()->setRequestBlockBuffer(request_buffer));
    ASSERT_NE(nullptr, cache_store_->getRequestBlockBufferStore()->getBlockBuffer(request_key, "block"));

    RemoteServerResource resource;
    resource.cache_store  = cache_store_;
    resource.workers      = {"rank0", "rank1"};
    resource.grpc_workers = {};

    GenerateInputPB request;
    request.set_request_id(request_id);
    RPCContext rpc_context{&request, nullptr};

    grpc::ServerContext                  server_context;
    kmonitor::MetricsReporterPtr         metrics_reporter;
    auto                                 meta = std::make_shared<RpcServerRuntimeMeta>();
    PrefillGenerateContext prefill_context(
        &resource, rpc_context, /*timeout_ms=*/0, &server_context, metrics_reporter, meta);

    prefill_context.markRequestEnd();

    EXPECT_EQ(nullptr, cache_store_->getRequestBlockBufferStore()->getBlockBuffer(request_key, "block"));
}

}  // namespace rtp_llm
