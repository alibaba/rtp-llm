#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

class RecordingCacheWriter: public PdKvWritebackCacheWriter {
public:
    explicit RecordingCacheWriter(std::vector<std::string>* events): events_(events) {}

    absl::Status mallocWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                       size_t                         block_count,
                                       size_t                         start_block_index) override {
        events_->push_back("malloc");
        last_start_block_index = start_block_index;
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

    absl::Status malloc_status_         = absl::OkStatus();
    size_t       last_start_block_index = 0;

private:
    std::vector<std::string>* events_;
};

class RecordingTransferClient: public PdKvWritebackTransferClient {
public:
    explicit RecordingTransferClient(std::vector<std::string>* events = nullptr): events_(events) {}

    absl::Status transfer(const PdKvWritebackTransferPlan& plan) override {
        if (events_) {
            events_->push_back("transfer");
        }
        last_plan = plan;
        return transfer_status;
    }

    PdKvWritebackTransferPlan last_plan;
    absl::Status              transfer_status = absl::OkStatus();

private:
    std::vector<std::string>* events_;
};

class RecordingRpcClient: public PdKvWritebackRpcClient {
public:
    explicit RecordingRpcClient(std::vector<std::string>* events): events_(events) {}

    absl::Status requestPrefillReceive(const PdKvWritebackLaunchRequest& request,
                                       const PdKvWritebackTopologyPlan&  topology) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_->push_back("prefill_rpc");
        last_request = request;
        prefill_targets.clear();
        for (const auto& mapping : topology.mappings) {
            prefill_targets.push_back(mapping.prefill_grpc_addr);
        }
        return rpc_status;
    }

    absl::Status requestPrefillCommit(const PdKvWritebackLaunchRequest& request,
                                      const PdKvWritebackTopologyPlan&  topology) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_->push_back("prefill_commit_rpc");
        last_request = request;
        prefill_targets.clear();
        for (const auto& mapping : topology.mappings) {
            prefill_targets.push_back(mapping.prefill_grpc_addr);
        }
        return rpc_status;
    }

    absl::Status requestPrefillAbort(const PdKvWritebackLaunchRequest& request,
                                     const PdKvWritebackTopologyPlan&  topology) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_->push_back("prefill_abort_rpc");
        last_request = request;
        prefill_targets.clear();
        for (const auto& mapping : topology.mappings) {
            prefill_targets.push_back(mapping.prefill_grpc_addr);
        }
        return rpc_status;
    }

    absl::Status requestDecodeSend(const PdKvWritebackLaunchRequest& request,
                                   const PdKvWritebackTopologyPlan&  topology) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_->push_back("decode_rpc");
        last_request = request;
        decode_targets.clear();
        for (const auto& mapping : topology.mappings) {
            decode_targets.push_back(mapping.decode_grpc_addr);
        }
        return rpc_status;
    }

    PdKvWritebackLaunchRequest last_request;
    absl::Status               rpc_status = absl::OkStatus();
    std::vector<std::string>   prefill_targets;
    std::vector<std::string>   decode_targets;

private:
    std::vector<std::string>* events_;
    std::mutex                mutex_;
};

class BlockingPrefillRpcClient: public PdKvWritebackRpcClient {
public:
    absl::Status requestPrefillReceive(const PdKvWritebackLaunchRequest& request,
                                       const PdKvWritebackTopologyPlan&  topology) override {
        (void)request;
        (void)topology;
        std::unique_lock<std::mutex> lock(mutex_);
        prefill_started = true;
        cv_.notify_all();
        if (!cv_.wait_for(lock, std::chrono::seconds(1), [&]() { return local_decode_started; })) {
            prefill_timed_out = true;
            return absl::DeadlineExceededError("decode send did not start while prefill receive was waiting");
        }
        return absl::OkStatus();
    }

    absl::Status requestPrefillCommit(const PdKvWritebackLaunchRequest& request,
                                      const PdKvWritebackTopologyPlan&  topology) override {
        (void)request;
        (void)topology;
        std::lock_guard<std::mutex> lock(mutex_);
        prefill_commit_started = true;
        cv_.notify_all();
        return absl::OkStatus();
    }

    absl::Status requestPrefillAbort(const PdKvWritebackLaunchRequest& request,
                                     const PdKvWritebackTopologyPlan&  topology) override {
        (void)request;
        (void)topology;
        std::lock_guard<std::mutex> lock(mutex_);
        prefill_abort_started = true;
        cv_.notify_all();
        return absl::OkStatus();
    }

    absl::Status requestDecodeSend(const PdKvWritebackLaunchRequest& request,
                                   const PdKvWritebackTopologyPlan&  topology) override {
        (void)request;
        (void)topology;
        std::lock_guard<std::mutex> lock(mutex_);
        remote_decode_started = true;
        cv_.notify_all();
        return absl::OkStatus();
    }

    void markLocalDecodeStarted() {
        std::lock_guard<std::mutex> lock(mutex_);
        local_decode_started = true;
        cv_.notify_all();
    }

    bool prefill_started        = false;
    bool prefill_commit_started = false;
    bool prefill_abort_started  = false;
    bool local_decode_started   = false;
    bool remote_decode_started  = false;
    bool prefill_timed_out      = false;

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
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

PdKvWritebackLaunchRequest makeFourRankLaunchRequest() {
    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = 7002;
    request.manifest.request_key          = "writeback_request_tp4";
    request.manifest.final_token_count    = 128;
    request.manifest.reusable_block_count = 2;
    request.manifest.cache_keys           = {1001, 1002};
    request.manifest.group_block_ids      = {{4, 5}};
    request.source.seq_size_per_block     = 64;
    request.source.layer_count            = 1;
    request.source.group_count            = 1;
    request.source.partition_count        = 4;
    request.source.layer_to_group_id      = {0};
    request.source.group_types            = {0};
    request.destination                   = request.source;
    request.source_prefill_grpc_addrs     = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    request.prefill_worker_addrs          = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};
    return request;
}

template<typename Predicate>
bool waitUntil(Predicate predicate, int timeout_ms = 1000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
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

TEST(PdKvWritebackManagerTest, EnabledCompatibleLaunchFailsWithoutRpcClient) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    PdKvWritebackManager manager(pd_config, nullptr);

    auto request                      = makeReceiveRequest();
    request.decode_worker_addrs       = {"127.0.0.1:8000"};
    request.source_prefill_grpc_addrs = {"127.0.0.1:9000"};

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Failed);
    EXPECT_EQ(result.reason, "rpc_client_null");
}

TEST(PdKvWritebackManagerTest, LaunchTpEqualSendsLocalRankToPrefillAndUsesLocalDecodeTransfer) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string>       events;
    auto                           transfer_client          = std::make_shared<RecordingTransferClient>(&events);
    auto                           rpc_client               = std::make_shared<RecordingRpcClient>(&events);
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, rpc_client, decode_worker_grpc_addrs, nullptr);

    auto request          = makeFourRankLaunchRequest();
    request.local_tp_rank = 2;

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Started);
    manager.waitForWritebackTasksForTest();

    ASSERT_EQ(events.size(), 4);
    EXPECT_NE(std::find(events.begin(), events.end(), "prefill_rpc"), events.end());
    EXPECT_NE(std::find(events.begin(), events.end(), "transfer"), events.end());
    EXPECT_NE(std::find(events.begin(), events.end(), "decode_rpc"), events.end());
    EXPECT_NE(std::find(events.begin(), events.end(), "prefill_commit_rpc"), events.end());
    EXPECT_EQ(transfer_client->last_plan.request_key, request.manifest.request_key);
    EXPECT_EQ(transfer_client->last_plan.decode_group_block_ids, request.manifest.group_block_ids);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_servers.size(), 4);
    ASSERT_EQ(transfer_client->last_plan.prefill_transfer_targets.size(), 1);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].ip, "p2");
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].port, 2001);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].decode_rank, 2);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].prefill_rank, 2);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].local_partition_count, 1);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].local_partition_id, 0);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].remote_partition_count, 1);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].remote_partition_id, 0);
    EXPECT_EQ(rpc_client->last_request.manifest.request_id, request.manifest.request_id);
    EXPECT_EQ(rpc_client->last_request.decode_worker_addrs, decode_worker_grpc_addrs);
    EXPECT_EQ(rpc_client->prefill_targets, std::vector<std::string>({"p0:1000", "p1:1000", "p2:1000", "p3:1000"}));
    EXPECT_EQ(rpc_client->decode_targets, std::vector<std::string>({"d0:1000", "d1:1000", "d3:1000"}));
}

TEST(PdKvWritebackManagerTest, LaunchSkipsBeforePrefillReceiveWhenSourceHasMissingLayerBlocks) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string>       events;
    auto                           transfer_client          = std::make_shared<RecordingTransferClient>(&events);
    auto                           rpc_client               = std::make_shared<RecordingRpcClient>(&events);
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, rpc_client, decode_worker_grpc_addrs, nullptr);

    auto request                     = makeFourRankLaunchRequest();
    request.source.layer_count       = 2;
    request.source.group_count       = 2;
    request.source.layer_to_group_id = {0, 1};
    request.source.group_types       = {0, 0};
    request.destination              = request.source;
    request.manifest.group_block_ids = {{4, 5}, {NULL_BLOCK_IDX, NULL_BLOCK_IDX}};
    request.local_tp_rank            = 0;

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Skipped);
    EXPECT_EQ(result.reason, "source_incomplete");
    EXPECT_TRUE(events.empty());
}

TEST(PdKvWritebackManagerTest, LaunchSkipsWhenLocalTpRankHasNoMapping) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string>       events;
    auto                           rpc_client               = std::make_shared<RecordingRpcClient>(&events);
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    PdKvWritebackManager           manager(pd_config, nullptr, nullptr, rpc_client, decode_worker_grpc_addrs, nullptr);

    auto request          = makeFourRankLaunchRequest();
    request.local_tp_rank = 4;

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Skipped);
    EXPECT_NE(result.reason.find("local tp rank"), std::string::npos);
    EXPECT_TRUE(events.empty());
}

TEST(PdKvWritebackManagerTest, LaunchStartsDecodeSendWhilePrefillReceiveIsWaiting) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback                  = true;
    auto                           rpc_client               = std::make_shared<BlockingPrefillRpcClient>();
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    class NotifyingTransferClient: public PdKvWritebackTransferClient {
    public:
        explicit NotifyingTransferClient(std::shared_ptr<BlockingPrefillRpcClient> rpc_client):
            rpc_client_(std::move(rpc_client)) {}
        absl::Status transfer(const PdKvWritebackTransferPlan& plan) override {
            (void)plan;
            rpc_client_->markLocalDecodeStarted();
            return absl::OkStatus();
        }

    private:
        std::shared_ptr<BlockingPrefillRpcClient> rpc_client_;
    };
    auto                 transfer_client = std::make_shared<NotifyingTransferClient>(rpc_client);
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, rpc_client, decode_worker_grpc_addrs, nullptr);

    auto request = makeFourRankLaunchRequest();

    auto result = manager.launchFromDecode(request);
    ASSERT_EQ(result.status, PdKvWritebackLaunchStatus::Started);
    manager.waitForWritebackTasksForTest();

    EXPECT_TRUE(rpc_client->prefill_started);
    EXPECT_TRUE(rpc_client->local_decode_started);
    EXPECT_TRUE(rpc_client->remote_decode_started);
    EXPECT_TRUE(rpc_client->prefill_commit_started);
    EXPECT_FALSE(rpc_client->prefill_abort_started);
    EXPECT_FALSE(rpc_client->prefill_timed_out);
}

TEST(PdKvWritebackManagerTest, LaunchUsesWritebackTimeoutInsteadOfExpiredRequestDeadline) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    pd_config.load_cache_timeout_ms        = 12345;
    std::vector<std::string>       events;
    auto                           transfer_client          = std::make_shared<RecordingTransferClient>(&events);
    auto                           rpc_client               = std::make_shared<RecordingRpcClient>(&events);
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, rpc_client, decode_worker_grpc_addrs, nullptr);

    auto request        = makeFourRankLaunchRequest();
    request.deadline_ms = currentTimeMs() - 1000;
    const auto before   = currentTimeMs();

    auto result = manager.launchFromDecode(request);
    ASSERT_EQ(result.status, PdKvWritebackLaunchStatus::Started);
    manager.waitForWritebackTasksForTest();

    EXPECT_GE(rpc_client->last_request.deadline_ms, before + pd_config.load_cache_timeout_ms);
    EXPECT_LT(rpc_client->last_request.deadline_ms, before + pd_config.load_cache_timeout_ms + 1000);
}

TEST(PdKvWritebackManagerTest, LaunchPrunesCompletedTasksBeforeTrackingNewOne) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string>       events;
    auto                           transfer_client          = std::make_shared<RecordingTransferClient>(&events);
    auto                           rpc_client               = std::make_shared<RecordingRpcClient>(&events);
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, rpc_client, decode_worker_grpc_addrs, nullptr);

    auto request = makeFourRankLaunchRequest();

    auto first_result = manager.launchFromDecode(request);
    ASSERT_EQ(first_result.status, PdKvWritebackLaunchStatus::Started);
    ASSERT_TRUE(waitUntil([&]() { return manager.completedWritebackTaskCountForTest() == 1; }));
    EXPECT_EQ(manager.trackedWritebackTaskCountForTest(), 1);

    auto second_result = manager.launchFromDecode(request);
    ASSERT_EQ(second_result.status, PdKvWritebackLaunchStatus::Started);
    EXPECT_EQ(manager.trackedWritebackTaskCountForTest(), 1);

    manager.waitForWritebackTasksForTest();
    EXPECT_EQ(manager.trackedWritebackTaskCountForTest(), 0);
}

TEST(PdKvWritebackTransferTest, ParsesDedicatedWritebackPortFromWorkerAddr) {
    auto servers = parsePdKvWritebackTransferServers({"10.0.0.1:10102:10104", "bad", "10.0.0.2:10202:10204"});

    ASSERT_EQ(servers.size(), 2);
    EXPECT_EQ(servers[0].first, "10.0.0.1");
    EXPECT_EQ(servers[0].second, 10103);
    EXPECT_EQ(servers[1].first, "10.0.0.2");
    EXPECT_EQ(servers[1].second, 10203);

    auto explicit_servers = parsePdKvWritebackTransferServers({"127.0.0.1:12345"});
    ASSERT_EQ(explicit_servers.size(), 1);
    EXPECT_EQ(explicit_servers[0].second, 12345);
}

TEST(PdKvWritebackTransferTest, ParsesExplicitTopologyTarget) {
    auto target = parsePdKvWritebackTransferTarget("10.0.0.3:10302:10304", 1, 0, 1, 0, 3, 3);

    ASSERT_TRUE(target.ok()) << target.status();
    EXPECT_EQ(target->ip, "10.0.0.3");
    EXPECT_EQ(target->port, 10303);
    EXPECT_EQ(target->local_partition_count, 1);
    EXPECT_EQ(target->local_partition_id, 0);
    EXPECT_EQ(target->remote_partition_count, 1);
    EXPECT_EQ(target->remote_partition_id, 0);
    EXPECT_EQ(target->decode_rank, 3);
    EXPECT_EQ(target->prefill_rank, 3);
}

TEST(PdKvWritebackManagerTest, ReceiveCommitsOnlyAfterTransferSucceeds) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string> events;
    RecordingCacheWriter     cache_writer(&events);
    RecordingTransferClient  transfer_client(&events);
    PdKvWritebackManager     manager(pd_config, &cache_writer, &transfer_client);

    auto destination_resource          = std::make_shared<BatchKVCacheResource>();
    auto request                       = makeReceiveRequest();
    request.manifest.start_block_index = 3;
    auto status                        = manager.receiveOnPrefill(request, destination_resource);

    EXPECT_TRUE(status.ok()) << status;
    EXPECT_EQ(events, std::vector<std::string>({"malloc", "transfer", "commit", "free"}));
    EXPECT_EQ(cache_writer.last_start_block_index, 3);
    EXPECT_EQ(transfer_client.last_plan.decode_group_block_ids, std::vector<BlockIndicesType>({{4, 5}}));
    EXPECT_EQ(transfer_client.last_plan.prefill_group_block_ids, std::vector<BlockIndicesType>({{7, 8}}));
}

TEST(PdKvWritebackManagerTest, PrepareReceiveKeepsBlocksPendingUntilCommit) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    std::vector<std::string> events;
    RecordingCacheWriter     cache_writer(&events);
    RecordingTransferClient  transfer_client(&events);
    PdKvWritebackManager     manager(pd_config, &cache_writer, &transfer_client);

    auto destination_resource           = std::make_shared<BatchKVCacheResource>();
    auto request                        = makeReceiveRequest();
    request.destination.partition_count = 2;
    request.source.partition_count      = 2;

    auto prepare_status = manager.prepareReceiveOnPrefill(request, destination_resource);

    EXPECT_TRUE(prepare_status.ok()) << prepare_status;
    EXPECT_EQ(events, std::vector<std::string>({"malloc", "transfer"}));

    auto commit_status = manager.commitReceiveOnPrefill(request);

    EXPECT_TRUE(commit_status.ok()) << commit_status;
    EXPECT_EQ(events, std::vector<std::string>({"malloc", "transfer", "commit", "free"}));
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

TEST(PdKvWritebackManagerTest, SendOnDecodeTransfersFromHeldSourceBlocks) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    auto                 transfer_client   = std::make_shared<RecordingTransferClient>();
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, nullptr, {}, nullptr);

    auto source_resource = std::make_shared<BatchKVCacheResource>();
    source_resource->resetBatchSize(1);
    source_resource->initBatchGroups(0, 1, 1, {0});
    source_resource->setBatchBlocks(0, 0, {101, 102});
    source_resource->setBatchCacheKeys(0, {11, 12});

    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = 9;
    request.manifest.request_key          = "request_9";
    request.manifest.final_token_count    = 130;
    request.manifest.reusable_block_count = 2;
    request.manifest.cache_keys           = {11, 12};
    request.manifest.group_block_ids      = {{101, 102}};
    request.source.seq_size_per_block     = 64;
    request.source.layer_count            = 1;
    request.source.group_count            = 1;
    request.source.partition_count        = 4;
    request.source.layer_to_group_id      = {0};
    request.source.group_types            = {0};
    request.destination                   = request.source;
    request.source_prefill_grpc_addrs     = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    request.decode_worker_addrs           = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    request.prefill_worker_addrs          = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};
    request.local_tp_rank                 = 2;

    auto status = manager.sendOnDecode(request, source_resource);

    EXPECT_TRUE(status.ok()) << status;
    EXPECT_EQ(transfer_client->last_plan.request_id, 9);
    EXPECT_EQ(transfer_client->last_plan.request_key, "request_9");
    EXPECT_EQ(transfer_client->last_plan.decode_group_block_ids, std::vector<BlockIndicesType>({{101, 102}}));
    EXPECT_TRUE(transfer_client->last_plan.prefill_group_block_ids.empty());
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_servers.size(), 4);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_servers[0].first, "p0");
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_servers[0].second, 2001);
    ASSERT_EQ(transfer_client->last_plan.prefill_transfer_targets.size(), 1);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].ip, "p2");
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].port, 2001);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].decode_rank, 2);
    EXPECT_EQ(transfer_client->last_plan.prefill_transfer_targets[0].prefill_rank, 2);
}

TEST(PdKvWritebackManagerTest, SendOnDecodeRejectsIncompleteSourceBeforeTransfer) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    auto                 transfer_client   = std::make_shared<RecordingTransferClient>();
    PdKvWritebackManager manager(pd_config, nullptr, transfer_client, nullptr, {}, nullptr);

    auto source_resource = std::make_shared<BatchKVCacheResource>();
    source_resource->resetBatchSize(1);
    source_resource->initBatchGroups(0, 2, 2, {0, 1});
    source_resource->setBatchBlocks(0, 0, {101, 102});
    source_resource->setBatchBlocks(0, 1, {NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    source_resource->setBatchCacheKeys(0, {11, 12});

    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = 10;
    request.manifest.request_key          = "request_10";
    request.manifest.final_token_count    = 130;
    request.manifest.reusable_block_count = 2;
    request.manifest.cache_keys           = {11, 12};
    request.manifest.group_block_ids      = {{101, 102}, {NULL_BLOCK_IDX, NULL_BLOCK_IDX}};
    request.source.seq_size_per_block     = 64;
    request.source.layer_count            = 2;
    request.source.group_count            = 2;
    request.source.partition_count        = 4;
    request.source.layer_to_group_id      = {0, 1};
    request.source.group_types            = {0, 0};
    request.destination                   = request.source;
    request.source_prefill_grpc_addrs     = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    request.decode_worker_addrs           = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    request.prefill_worker_addrs          = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};

    auto status = manager.sendOnDecode(request, source_resource);

    EXPECT_FALSE(status.ok());
    EXPECT_NE(std::string(status.message()).find("source_incomplete"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
