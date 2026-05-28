#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class RecordingCacheWriter: public PdKvWritebackCacheWriter {
public:
    explicit RecordingCacheWriter(std::vector<std::string>* events): events_(events) {}

    absl::Status mallocWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                       size_t                         block_count) override {
        events_->push_back("malloc");
        batch_kv_cache_resource->resetBatchSize(1);
        batch_kv_cache_resource->initGroups(1, 1);
        batch_kv_cache_resource->mutableBlockIds(0, 0).add({7, 8});
        return malloc_status_;
    }

    void commitWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                               const CacheKeysType&           cache_keys,
                               bool                           is_resident) override {
        (void)batch_kv_cache_resource;
        (void)cache_keys;
        (void)is_resident;
        events_->push_back("commit");
    }

    void freeWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource) override {
        (void)batch_kv_cache_resource;
        events_->push_back("free");
    }

    absl::Status malloc_status_ = absl::OkStatus();

private:
    std::vector<std::string>* events_;
};

class RecordingTransferClient: public PdKvWritebackTransferClient {
public:
    explicit RecordingTransferClient(std::vector<std::string>* events): events_(events) {}

    absl::Status transfer(const PdKvWritebackTransferPlan& plan) override {
        events_->push_back("transfer");
        last_plan = plan;
        return transfer_status;
    }

    PdKvWritebackTransferPlan last_plan;
    absl::Status              transfer_status = absl::OkStatus();

private:
    std::vector<std::string>* events_;
};

PdKvWritebackLaunchRequest makeReceiveRequest() {
    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = 7001;
    request.manifest.request_key          = "writeback_request";
    request.manifest.reusable_block_count = 2;
    request.manifest.cache_keys           = {1001, 1002};
    request.manifest.group_block_ids      = {{4, 5}};
    request.source.seq_size_per_block     = 16;
    request.source.layer_count            = 1;
    request.source.group_count            = 1;
    request.source.partition_count        = 1;
    request.source.layer_to_group_id      = {0};
    request.source.group_types            = {0};
    request.destination                   = request.source;
    request.prefill_worker_addrs          = {"127.0.0.1:12345"};
    return request;
}

TEST(PdKvWritebackManagerTest, DisabledConfigSkipsLaunch) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = false;
    PdKvWritebackManager manager(pd_config, nullptr);

    PdKvWritebackLaunchRequest request;
    request.manifest.reusable_block_count = 1;

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Skipped);
    EXPECT_EQ(result.reason, "disabled");
}

TEST(PdKvWritebackManagerTest, RejectsIncompatibleSeqSize) {
    PdKvWritebackCompatibility source;
    source.seq_size_per_block = 16;
    PdKvWritebackCompatibility destination;
    destination.seq_size_per_block = 32;

    auto status = validatePdKvWritebackCompatibility(source, destination);

    EXPECT_FALSE(status.ok());
    EXPECT_NE(std::string(status.message()).find("seq_size_per_block"), std::string::npos);
}

TEST(PdKvWritebackManagerTest, EnabledCompatibleLaunchStarts) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    PdKvWritebackManager manager(pd_config, nullptr);

    PdKvWritebackLaunchRequest request;
    request.manifest.reusable_block_count = 2;
    request.source.seq_size_per_block     = 16;
    request.source.layer_count            = 4;
    request.source.group_count            = 1;
    request.source.partition_count        = 1;
    request.source.layer_to_group_id      = {0, 0, 0, 0};
    request.source.group_types            = {0};
    request.destination                   = request.source;
    request.source_prefill_grpc_addrs     = {"127.0.0.1:9000"};

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Started);
}

TEST(PdKvWritebackManagerTest, ReceiveCommitsOnlyAfterTransferSucceeds) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string> events;
    RecordingCacheWriter     cache_writer(&events);
    RecordingTransferClient  transfer_client(&events);
    PdKvWritebackManager     manager(pd_config, &cache_writer, &transfer_client);

    auto destination_resource = std::make_shared<BatchKVCacheResource>();
    auto status               = manager.receiveOnPrefill(makeReceiveRequest(), destination_resource);

    EXPECT_TRUE(status.ok()) << status;
    EXPECT_EQ(events, std::vector<std::string>({"malloc", "transfer", "commit"}));
    EXPECT_EQ(transfer_client.last_plan.decode_group_block_ids, std::vector<BlockIndicesType>({{4, 5}}));
    EXPECT_EQ(transfer_client.last_plan.prefill_group_block_ids, std::vector<BlockIndicesType>({{7, 8}}));
}

TEST(PdKvWritebackManagerTest, ReceiveFreesAllocatedBlocksWhenTransferFails) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string> events;
    RecordingCacheWriter     cache_writer(&events);
    RecordingTransferClient  transfer_client(&events);
    transfer_client.transfer_status = absl::InternalError("transfer failed");
    PdKvWritebackManager manager(pd_config, &cache_writer, &transfer_client);

    auto destination_resource = std::make_shared<BatchKVCacheResource>();
    auto status               = manager.receiveOnPrefill(makeReceiveRequest(), destination_resource);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(events, std::vector<std::string>({"malloc", "transfer", "free"}));
}

}  // namespace
}  // namespace rtp_llm
