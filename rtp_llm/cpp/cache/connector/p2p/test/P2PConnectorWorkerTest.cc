#include <algorithm>
#include <atomic>
#include <thread>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <map>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include <set>

#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
namespace rtp_llm {

namespace test {

namespace {

constexpr int kDsv4CpSize = 4;
constexpr int kDsv4CpRank = kDsv4CpSize - 1;

class ObservedLayerCacheBufferUtil final: public LayerCacheBufferUtil {
public:
    class Observer final: public ConversionObserver {
    public:
        void onLayerCacheBufferConstructed() override {
            ++constructed;
        }

        void onLayerCacheBufferPublished() override {
            ++published;
        }

        size_t constructed = 0;
        size_t published   = 0;
    };

    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(const CacheConfig& config,
                                                                  KVCacheResource&   resource,
                                                                  int                batch_id,
                                                                  int                start_key_ordinal,
                                                                  int                key_count,
                                                                  int                cp_rank,
                                                                  int                cp_size,
                                                                  Observer&          observer) {
        return LayerCacheBufferUtil::convert(
            config, resource, batch_id, start_key_ordinal, key_count, cp_rank, cp_size, &observer);
    }
};

CacheConfig makeRealDsv4P2PCacheConfig() {
    ModelConfig model_config;
    model_config.num_layers                                                = 43;
    model_config.hidden_size                                               = 4096;
    model_config.attn_config.head_num                                      = 64;
    model_config.attn_config.kv_head_num                                   = 1;
    model_config.attn_config.size_per_head                                 = 512;
    model_config.attn_config.rope_head_dim                                 = 64;
    model_config.attn_config.indexer_head_dim                              = 128;
    model_config.attn_config.indexer_head_num                              = 64;
    model_config.attn_config.indexer_topk                                  = 512;
    model_config.attn_config.tokens_per_block                              = 128;
    model_config.hybrid_attention_config.enable_hybrid_attention           = true;
    model_config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    std::vector<int> ratios                                                = {0, 0};
    for (int layer_id = 2; layer_id < model_config.num_layers; ++layer_id) {
        ratios.push_back(layer_id % 2 == 0 ? 4 : 128);
    }
    setDsv4KvCacheSpecs(model_config, ratios);

    ParallelismConfig parallelism_config;
    parallelism_config.role_type                          = RoleType::PREFILL;
    parallelism_config.tp_size                            = kDsv4CpSize;
    parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    return CacheConfigCreator::createBasicConfig(model_config, parallelism_config, false, 0);
}

std::pair<BlockIdxType, BlockIdxType> realDsv4BlockIds(const CacheConfig& config, std::string_view tag) {
    // Distinct block-id ranges per tag; the offset is a test fixture detail derived
    // from the tag's position among the sorted unique tags, never a routing key.
    const auto tags = groupTagSet(config);
    const auto pos  = static_cast<BlockIdxType>(std::distance(tags.begin(), tags.find(std::string(tag))));
    const auto base = static_cast<BlockIdxType>(100 + pos * 10);
    return {tag == "indexer_kv" ? NULL_BLOCK_IDX : base, static_cast<BlockIdxType>(base + 1)};
}

KVCacheResourcePtr makeRealDsv4P2PResource(const CacheConfig& config) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(config);
    for (const auto& group : config.topology().groups()) {
        const auto& tag            = group.tag;
        const auto [first, second] = realDsv4BlockIds(config, tag);
        resource->mutableBlockIds(tag).assign({first, second});
    }
    CacheKeysType keys;
    for (int key_ordinal = 0; key_ordinal < 32; ++key_ordinal) {
        keys.push_back(5000 + key_ordinal);
    }
    resource->setCacheKeys(std::move(keys));
    return resource;
}

std::vector<size_t> expectedRealDsv4KeyOrdinals(std::string_view tag) {
    if (tag == "hca_state") {
        return {31};
    }
    if (tag == "indexer_state" || tag == "csa_state" || tag == "swa_kv") {
        return {15, 31};
    }
    return {3, 7};
}

void expectRealDsv4WireMap(const CacheConfig& config, const LayerCacheBuffer& buffer) {
    const auto& tag            = buffer.cacheTag();
    const auto [first, second] = realDsv4BlockIds(config, tag);
    const auto ordinals        = expectedRealDsv4KeyOrdinals(tag);
    size_t     expected_size   = 0;
    for (size_t ordinal : ordinals) {
        const auto block_id = ordinal == ordinals.back() ? second : first;
        if (isNullBlockIdx(block_id)) {
            EXPECT_EQ(buffer.blockIdMap().count(5000 + ordinal), 0u) << tag;
            continue;
        }
        ++expected_size;
        EXPECT_EQ(buffer.blockIdMap().at(5000 + ordinal), block_id) << tag;
    }
    EXPECT_EQ(buffer.blockIdMap().size(), expected_size) << tag;
}

}  // namespace

TEST(P2PKeyUtilTest, LayerCacheBufferUsesTagIdentity) {
    LayerCacheBuffer tagged(/*layer_id=*/3, "full");
    const auto       key = P2PKeyUtil::makePartitionLayerTagKey("request", tagged.getLayerId(), tagged.cacheTag(), 1);

    EXPECT_NE(key, P2PKeyUtil::makePartitionLayerTagKey("request", 3, "linear", 1));
}

// Mock LayerBlockConverter for testing
class MockLayerBlockConverter: public LayerBlockConverter {
public:
    std::vector<BlockInfo> convertIndexToBufferByTag(int, const std::string&, int, int, int) const override {
        return {};
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

// Mock IKVCacheSender for testing (replaces old TransferClient mock)
class MockIKVCacheSender: public transfer::IKVCacheSender {
public:
    struct SendCallInfo {
        std::string ip;
        uint32_t    port;
        std::string layer_key;
        int         layer_id;  // parsed from layer_key
        std::string cache_tag;
        int64_t     deadline_ms = 0;
    };

    bool regMem(const BlockInfo& /*block_info*/, uint64_t /*aligned_size*/ = 0) override {
        return true;
    }

    void send(const transfer::SendRequest&                                         request,
              std::function<void(transfer::TransferErrorCode, const std::string&)> callback) override {
        SendCallInfo info;
        info.ip          = request.ip;
        info.port        = request.port;
        info.layer_key   = request.unique_key;
        info.layer_id    = parseLayerId(request.unique_key);
        info.cache_tag   = parseCacheTag(request.unique_key);
        info.deadline_ms = request.deadline_ms;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            send_calls_.push_back(info);
        }

        bool success = should_succeed_;
        if (layer_success_map_.count(info.layer_id)) {
            success = layer_success_map_.at(info.layer_id);
        }

        int delay_ms = callback_delay_ms_;
        if (use_staggered_callback_delay_) {
            const int n = stagger_callback_counter_.fetch_add(1);
            delay_ms    = stagger_callback_base_ms_ * (n + 1);
        }
        auto run_cb = [callback, success, delay_ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            auto code = success ? transfer::TransferErrorCode::OK : transfer::TransferErrorCode::UNKNOWN;
            callback(code, success ? "" : "mock send failed");
        };

        if (async_callback_) {
            std::thread(run_cb).detach();
        } else {
            run_cb();
        }
    }

    void setShouldSucceed(bool v) {
        should_succeed_ = v;
    }
    void setLayerSuccess(int layer_id, bool success) {
        layer_success_map_[layer_id] = success;
    }
    void setAsyncCallback(bool v) {
        async_callback_ = v;
    }
    void setCallbackDelayMs(int ms) {
        callback_delay_ms_ = ms;
    }

    /// 每次 send 的回调延迟递增（1*base, 2*base, ...），用于覆盖 wait_for + 多次 notify
    void setStaggeredCallbackDelays(bool enable, int base_ms = 15) {
        use_staggered_callback_delay_ = enable;
        stagger_callback_base_ms_     = base_ms;
        stagger_callback_counter_.store(0);
    }

    int getTransferCallCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(send_calls_.size());
    }
    std::vector<SendCallInfo> getTransferCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return send_calls_;
    }
    void clearTransferCalls() {
        std::lock_guard<std::mutex> lock(mutex_);
        send_calls_.clear();
    }

private:
    /// @brief Parse layer_id from layer_key = base_key + "_" + layer_id + "_" + partition_id
    /// or base_key + "_" + layer_id + "_tag" + cache_tag + "_" + partition_id.
    static int parseLayerId(const std::string& layer_key) {
        auto last = layer_key.rfind('_');
        if (last == std::string::npos || last == 0) {
            return -1;
        }
        auto second_last = layer_key.rfind('_', last - 1);
        if (second_last == std::string::npos) {
            return -1;
        }
        auto third_last = layer_key.rfind('_', second_last - 1);
        try {
            const bool has_identity = layer_key.compare(second_last + 1, 3, "tag") == 0;
            const auto layer_begin = has_identity && third_last != std::string::npos ? third_last + 1 : second_last + 1;
            const auto layer_end   = has_identity ? second_last : last;
            return std::stoi(layer_key.substr(layer_begin, layer_end - layer_begin));
        } catch (...) {
            return -1;
        }
    }

    static std::string parseCacheTag(const std::string& layer_key) {
        const auto last = layer_key.rfind('_');
        if (last == std::string::npos || last == 0) {
            return {};
        }
        const auto second_last = layer_key.rfind('_', last - 1);
        if (second_last == std::string::npos || layer_key.compare(second_last + 1, 3, "tag") != 0) {
            return {};
        }
        return layer_key.substr(second_last + 4, last - second_last - 4);
    }

    bool                      should_succeed_ = true;
    std::map<int, bool>       layer_success_map_;
    bool                      async_callback_    = true;
    int                       callback_delay_ms_ = 1;
    bool                      use_staggered_callback_delay_{false};
    int                       stagger_callback_base_ms_{15};
    std::atomic<int>          stagger_callback_counter_{0};
    mutable std::mutex        mutex_;
    std::vector<SendCallInfo> send_calls_;
};

// Mock IKVCacheRecvTask for testing
class MockIKVCacheRecvTask: public transfer::IKVCacheRecvTask {
public:
    bool done() const override {
        return done_.load();
    }
    bool success() const override {
        return success_.load();
    }
    void cancel() override {
        success_.store(false);
        error_code_ = transfer::TransferErrorCode::CANCELLED;
        done_.store(true);
    }
    void forceCancel() override {
        cancel();
    }
    transfer::TransferErrorCode errorCode() const override {
        return error_code_;
    }
    std::string errorMessage() const override {
        return success_.load() ? "" : "mock recv task failed";
    }

    void setDone(bool success) {
        success_.store(success);
        if (!success) {
            error_code_ = transfer::TransferErrorCode::UNKNOWN;
        }
        done_.store(true);
    }

private:
    std::atomic<bool>           done_{false};
    std::atomic<bool>           success_{true};
    transfer::TransferErrorCode error_code_{transfer::TransferErrorCode::OK};
};

// Mock IKVCacheReceiver for testing
class MockIKVCacheReceiver: public transfer::IKVCacheReceiver {
public:
    bool regMem(const BlockInfo& /*block_info*/, uint64_t /*aligned_size*/ = 0) override {
        return true;
    }

    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        auto                        task = std::make_shared<MockIKVCacheRecvTask>();
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_[request.unique_key] = task;
        return task;
    }

    /// @brief Signal a specific layer_key task as done
    void setTaskDone(const std::string& layer_key, bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = tasks_.find(layer_key); it != tasks_.end()) {
            it->second->setDone(success);
        }
    }

    void stealTask(const std::string& /*unique_key*/) override {
        steal_task_count_.fetch_add(1, std::memory_order_relaxed);
    }
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = tasks_.find(unique_key); it != tasks_.end()) {
            return it->second;
        }
        return nullptr;
    }

    int stealTaskCount() const {
        return steal_task_count_.load(std::memory_order_relaxed);
    }

    /// @brief Check if tasks for a base_key have been created (>= expected_count)
    bool hasEnoughTasks(const std::string& base_key, int expected_count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        int                         count = 0;
        for (const auto& [key, task] : tasks_) {
            if (key.size() > base_key.size() && key.substr(0, base_key.size()) == base_key
                && key[base_key.size()] == '_') {
                ++count;
            }
        }
        return count >= expected_count;
    }

private:
    mutable std::mutex                                                     mutex_;
    std::unordered_map<std::string, std::shared_ptr<MockIKVCacheRecvTask>> tasks_;
    std::atomic<int>                                                       steal_task_count_{0};
};

// Test fixture for P2PConnectorWorker (tests Prefill and Decode sub-objects directly)
class P2PConnectorWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        worker_config_.transfer_backend_config.cache_store_rdma_mode        = false;
        worker_config_.transfer_backend_config.messager_io_thread_count     = 1;
        worker_config_.transfer_backend_config.messager_worker_thread_count = 1;
        worker_config_.tp_size                                              = 2;
        worker_config_.tp_rank                                              = 0;
        worker_config_.transfer_backend_config.cache_store_listen_port      = 0;
        worker_config_.layer_all_num                                        = 2;
        cache_config_ = makeTestCacheConfig(makeTestCacheTopologyByTag(/*group_num=*/2,
                                                                       /*layer_num=*/2,
                                                                       {{"group0", "group1"}, {"group1"}}));

        mock_layer_block_converter_ = std::make_shared<MockLayerBlockConverter>();

        mock_sender_   = std::make_shared<MockIKVCacheSender>();
        mock_receiver_ = std::make_shared<MockIKVCacheReceiver>();

        prefill_ = std::make_unique<P2PConnectorWorkerPrefill>(
            worker_config_, cache_config_, mock_layer_block_converter_, nullptr, mock_sender_);
        prefill_->init(10 * 1000);

        decode_ = std::make_unique<P2PConnectorWorkerDecode>(
            worker_config_, mock_layer_block_converter_, nullptr, mock_receiver_);

        computed_buffers_ = prefill_->getComputedBuffersStore();
    }

    void TearDown() override {
        prefill_.reset();
        decode_.reset();
    }

    KVCacheResourcePtr createKVCacheResource(int layer_id, int num_blocks = 2) {
        auto                                  resource  = std::make_shared<KVCacheResource>();
        int                                   layer_num = static_cast<int>(worker_config_.layer_all_num);
        std::vector<std::vector<std::string>> layer_to_group_tags(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_to_group_tags[i] = {"group" + std::to_string(i)};
        }
        resource->initGroups(
            makeTestCacheConfig(makeTestCacheTopologyByTag(layer_num, layer_num, layer_to_group_tags)));

        for (int i = 0; i < layer_num; ++i) {
            if (i == layer_id) {
                // Each layer owns exactly one group; layer_to_group_tags records its tag.
                const auto& tag = layer_to_group_tags[i].front();
                for (int j = 0; j < num_blocks; ++j) {
                    resource->mutableBlockIds(tag).add({j});
                }
            }
        }

        for (int i = 0; i < num_blocks; ++i) {
            resource->appendCacheKey(layer_id * 1000 + i);
        }

        return resource;
    }

    // Create a c10::Event that is immediately queryable (already recorded on current stream).
    std::optional<c10::Event> createReadyEvent() {
        return std::nullopt;  // nullopt means "immediately ready" in StoreWaitContext logic
    }

    void addComputedBuffer(int64_t request_id, int layer_id, int64_t deadline_ms) {
        addComputedBuffer(request_id, layer_id, deadline_ms, "group0");
    }

    void addComputedBuffer(int64_t request_id, int layer_id, int64_t deadline_ms, const std::string& cache_tag) {
        auto computed_buffer = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
        if (!computed_buffer->expectedBufferCount().has_value()) {
            computed_buffer->setExpectedBufferCount(static_cast<size_t>(worker_config_.layer_all_num));
        }
        auto layer_cache_buffer = createTaggedLayerCacheBuffer(layer_id, cache_tag, 2);
        computed_buffers_->addBuffer(request_id, layer_cache_buffer, deadline_ms);
    }

    void setExpectedComputedBufferCount(int64_t request_id, int64_t deadline_ms, size_t expected_buffer_count) {
        auto computed_buffer = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
        computed_buffer->setExpectedBufferCount(expected_buffer_count);
    }

    std::shared_ptr<LayerCacheBuffer> createTaggedLayerCacheBuffer(int layer_id, std::string tag, int num_blocks) {
        auto buffer = std::make_shared<LayerCacheBuffer>(layer_id, std::move(tag));
        for (int i = 0; i < num_blocks; ++i) {
            buffer->addBlockId(layer_id * 1000 + i, i);
        }
        return buffer;
    }

    void simulateTaskDone(const std::string& base_key, const std::vector<int>& layer_ids, bool all_success = true) {
        for (int layer_id : layer_ids) {
            std::string layer_key =
                P2PKeyUtil::makePartitionLayerTagKey(base_key, layer_id, /*cache_tag=*/"group2", /*partition_id=*/0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            mock_receiver_->setTaskDone(layer_key, all_success);
        }
    }

    void setTransferWaitTimeout(int64_t timeout_ms) {
        worker_config_.transfer_backend_config.rdma_transfer_wait_timeout_ms = timeout_ms;
        prefill_.reset();
        decode_.reset();
        prefill_ = std::make_unique<P2PConnectorWorkerPrefill>(
            worker_config_, cache_config_, mock_layer_block_converter_, nullptr, mock_sender_);
        prefill_->init(10 * 1000);
        decode_ = std::make_unique<P2PConnectorWorkerDecode>(
            worker_config_, mock_layer_block_converter_, nullptr, mock_receiver_);
        computed_buffers_ = prefill_->getComputedBuffersStore();
    }

protected:
    P2PConnectorWorkerConfig                       worker_config_;
    CacheConfig                                    cache_config_;
    std::shared_ptr<LayerBlockConverter>           mock_layer_block_converter_;
    std::unique_ptr<P2PConnectorWorkerPrefill>     prefill_;
    std::unique_ptr<P2PConnectorWorkerDecode>      decode_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
    std::shared_ptr<MockIKVCacheSender>            mock_sender_;
    std::shared_ptr<MockIKVCacheReceiver>          mock_receiver_;
};

// ==================== writeByLayer 测试 (Prefill 端) ====================

TEST_F(P2PConnectorWorkerTest, WriteByLayer_ReturnTrue_WithReadyEvent) {
    int     layer_id   = 0;
    int64_t request_id = 1002;
    auto    resource   = std::make_shared<KVCacheResource>();
    resource->initGroups(makeTestCacheConfig(
        makeTestCacheTopologyByTag(/*group_num=*/2, /*layer_num=*/2, {{"group0", "group1"}, {"group1"}})));
    for (const auto& tag : {"group0", "group1"}) {
        resource->mutableBlockIds(tag).add({0, 1});
    }
    resource->setCacheKeys({0, 1});

    // Pass nullopt — means "immediately ready" in StoreWaitContext logic
    bool success = prefill_->writeByLayer(layer_id, resource, request_id, std::nullopt);
    EXPECT_TRUE(success);

    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_NE(computed_buffer, nullptr);
    ASSERT_TRUE(computed_buffer->expectedBufferCount().has_value());
    EXPECT_EQ(*computed_buffer->expectedBufferCount(), 3u);

    // Wait for cleanup thread to check once — event is immediately ready so buffer should appear
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    ASSERT_NE(computed_buffers_->getBuffer(request_id), nullptr);
}

TEST_F(P2PConnectorWorkerTest, WriteByLayerCountsOnlyTransferableSparseGroups) {
    constexpr int64_t request_id = 1003;
    auto              resource   = std::make_shared<KVCacheResource>();
    resource->initGroups(makeTestCacheConfig(
        makeTestCacheTopologyByTag(/*group_num=*/2, /*layer_num=*/2, {{"group0", "group1"}, {"group1"}})));
    resource->mutableBlockIds("group0").add({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    resource->mutableBlockIds("group1").add({3, 4});
    resource->setCacheKeys({10, 11});

    EXPECT_TRUE(prefill_->writeByLayer(/*layer_id=*/0, resource, request_id, std::nullopt));

    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_NE(computed_buffer, nullptr);
    ASSERT_TRUE(computed_buffer->expectedBufferCount().has_value());
    EXPECT_EQ(*computed_buffer->expectedBufferCount(), 2u);
}

TEST_F(P2PConnectorWorkerTest, WriteByLayerUsesTaggedFullAndCompactSwaPoliciesForExpectedBuffers) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 2;
    config.layer_all_num      = 2;

    GroupBase full;
    full.tag                       = "group0";
    full.spec                      = std::make_shared<MHAKVCacheSpec>();
    full.layer_ids                 = {0};
    full.seq_size_per_block        = 24;
    full.kernel_seq_size_per_block = 8;
    full.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);

    GroupBase swa;
    swa.tag                       = "group1";
    swa.spec                      = std::make_shared<MHAKVCacheSpec>();
    swa.layer_ids                 = {0, 1};
    swa.seq_size_per_block        = 32;
    swa.kernel_seq_size_per_block = 8;
    swa.policy                    = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa.policy.active_tail_blocks = 1;
    config.setTopology({std::move(full), std::move(swa)}, {{0, {"group0", "group1"}}, {1, {"group1"}}});

    auto cp_config    = worker_config_;
    cp_config.cp_rank = 3;
    cp_config.cp_size = 4;
    auto prefill      = std::make_unique<P2PConnectorWorkerPrefill>(
        cp_config, config, mock_layer_block_converter_, nullptr, mock_sender_);
    ASSERT_TRUE(prefill->init(10 * 1000));

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(config);
    resource->mutableBlockIds("group0").assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    resource->mutableBlockIds("group1").assign({70, 71});
    CacheKeysType keys;
    for (int i = 0; i < 32; ++i) {
        keys.push_back(1000 + i);
    }
    resource->setCacheKeys(std::move(keys));

    constexpr int64_t request_id = 1004;
    ASSERT_TRUE(prefill->writeByLayer(/*layer_id=*/0, resource, request_id, std::nullopt));
    auto computed = prefill->getComputedBuffersStore()->getBuffer(request_id);
    ASSERT_NE(computed, nullptr);
    ASSERT_TRUE(computed->expectedBufferCount().has_value());
    EXPECT_EQ(*computed->expectedBufferCount(), 2u);

    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    auto [count, buffers] = computed->getBuffers({0});
    EXPECT_EQ(count, 1);
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers[0]->cacheTag(), "group1");
    ASSERT_EQ(buffers[0]->blockIdMap().size(), 1u);
    EXPECT_EQ(buffers[0]->blockIdMap().at(1031), 71);
}

TEST_F(P2PConnectorWorkerTest, WriteByLayerUsesRealDsv4TopologyAndTagMappedBlocks) {
    auto config                    = makeRealDsv4P2PCacheConfig();
    auto cp_worker_config          = worker_config_;
    cp_worker_config.layer_all_num = config.layer_all_num;
    cp_worker_config.cp_rank       = kDsv4CpRank;
    cp_worker_config.cp_size       = kDsv4CpSize;
    auto prefill                   = std::make_unique<P2PConnectorWorkerPrefill>(
        cp_worker_config, config, mock_layer_block_converter_, nullptr, mock_sender_);
    ASSERT_TRUE(prefill->init(10 * 1000));

    auto              resource   = makeRealDsv4P2PResource(config);
    constexpr int64_t request_id = 1005;
    ASSERT_TRUE(prefill->writeByLayer(/*layer_id=*/2, resource, request_id, std::nullopt));
    auto computed = prefill->getComputedBuffersStore()->getBuffer(request_id);
    ASSERT_NE(computed, nullptr);
    ASSERT_TRUE(computed->expectedBufferCount().has_value());
    EXPECT_EQ(*computed->expectedBufferCount(), 167u);

    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    auto [count, buffers] = computed->getBuffers({2});
    EXPECT_EQ(count, 5);
    ASSERT_EQ(buffers.size(), 5u);
    ASSERT_EQ(config.groupsForLayer(2).size(), buffers.size());
    for (const auto& tag : config.groupsForLayer(2)) {
        const auto buffer_it = std::find_if(
            buffers.begin(), buffers.end(), [&tag](const auto& buffer) { return buffer->cacheTag() == tag; });
        ASSERT_NE(buffer_it, buffers.end()) << tag;
        expectRealDsv4WireMap(config, **buffer_it);
    }
}

// ==================== sendKVCache 测试 (Prefill 端) ====================

TEST_F(P2PConnectorWorkerTest, SendKVCache_SendRequestDeadline_AlignedWithReturnBefore) {
    int64_t     request_id  = 2000;
    std::string unique_key  = "test_send_deadline_align";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(false);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    const int64_t expected_transfer_deadline = deadline_ms - worker_config_.p2p_read_return_before_deadline_ms;
    ErrorInfo     result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
    EXPECT_TRUE(result.ok());

    for (const auto& c : mock_sender_->getTransferCalls()) {
        EXPECT_EQ(c.deadline_ms, expected_transfer_deadline)
            << "SendRequest.deadline_ms should match decode recv_task_deadline (D - return_before)";
    }
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_AllLayersTransferSuccess) {
    int64_t     request_id  = 2001;
    std::string unique_key  = "test_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_EQ("127.0.0.1", call.ip);
        EXPECT_EQ(12345u, call.port);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
    }
    EXPECT_EQ(transferred_layers.size(), 2u);
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_PartialLayersTransferFailed) {
    int64_t     request_id  = 2002;
    std::string unique_key  = "test_partial_fail";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setLayerSuccess(1, false);
    mock_sender_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_EQ(transferred_layers.size(), 2u);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
    EXPECT_TRUE(transferred_layers.find(1) != transferred_layers.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_SendsAllGroupsForSameLayer) {
    int64_t     request_id  = 2006;
    std::string unique_key  = "test_same_layer_groups";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(false);

    setExpectedComputedBufferCount(request_id, deadline_ms, 3);
    addComputedBuffer(request_id, 0, deadline_ms, "group0");
    addComputedBuffer(request_id, 0, deadline_ms, "group1");
    addComputedBuffer(request_id, 1, deadline_ms, "group0");

    ErrorInfo result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 3);

    auto                                  calls = mock_sender_->getTransferCalls();
    std::set<std::pair<int, std::string>> transferred_layer_groups;
    for (const auto& call : calls) {
        transferred_layer_groups.insert({call.layer_id, call.cache_tag});
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_TRUE(transferred_layer_groups.count({0, "group0"}));
    EXPECT_TRUE(transferred_layer_groups.count({0, "group1"}));
    EXPECT_TRUE(transferred_layer_groups.count({1, "group0"}));
}

TEST_F(P2PConnectorWorkerTest, SendKVCache_WaitsForDelayedSecondTagOnLastLayer) {
    const int64_t                                       request_id              = 2007;
    const std::string                                   unique_key              = "test_delayed_last_layer_tag";
    const int64_t                                       deadline_ms             = currentTimeMs() + 5000;
    const std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {{"127.0.0.1", 12345}};

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(false);

    setExpectedComputedBufferCount(request_id, deadline_ms, 3);
    addComputedBuffer(request_id, 0, deadline_ms, "group0");
    addComputedBuffer(request_id, 1, deadline_ms, "group0");

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       send_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done.store(true);
    });

    for (int retry = 0; retry < 100 && mock_sender_->getTransferCallCount() < 2; ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(done.load());

    addComputedBuffer(request_id, 1, deadline_ms, "group1");
    send_thread.join();

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(done.load());
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 3);
    const auto calls = mock_sender_->getTransferCalls();
    EXPECT_EQ(std::count_if(calls.begin(),
                            calls.end(),
                            [](const auto& call) { return call.layer_id == 1 && call.cache_tag == "group1"; }),
              1);
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_SomeLayersNotTransferred) {
    int64_t     request_id = 2003;
    std::string unique_key = "test_some_layers_missing";
    // D 须足够大，使 return_deadline_ms=D-100 仍晚于 now，才能先发出已有 layer 再因缺层失败
    int64_t deadline_ms = currentTimeMs() + 150;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    // 只添加 layer 0
    addComputedBuffer(request_id, 0, deadline_ms);

    // wait till deadline
    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 100);

    if (write_thread.joinable()) {
        write_thread.join();
    }
    EXPECT_TRUE(result.hasError());

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_EQ(transferred_layers.size(), 1u);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_AsymmetricTP_2P4D_Success) {
    int64_t     request_id  = 2004;
    std::string unique_key  = "test_asymmetric_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});
    decode_transfer_servers.push_back({"127.0.0.1", 12347});
    decode_transfer_servers.push_back({"127.0.0.1", 12348});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(done);
    // 2P4D: tp_size=2, tp_rank=0 负责 decode 节点 0,1（端口 12345, 12346），每层各发 1 次 → 共 4 次
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 4);

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    std::set<int> transferred_ports;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        transferred_ports.insert(call.port);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
    }
    EXPECT_EQ(transferred_layers.size(), 2u);
    EXPECT_EQ(transferred_ports.size(), 2u);
    EXPECT_TRUE(transferred_ports.find(12345) != transferred_ports.end());
    EXPECT_TRUE(transferred_ports.find(12346) != transferred_ports.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_TransferTimeout) {
    int64_t     request_id = 2005;
    std::string unique_key = "test_transfer_timeout";
    // 足够长的 D，使 return_deadline 晚于 200ms 回调延迟，仍能等到 mock 回调
    int64_t deadline_ms = currentTimeMs() + 500;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 设置 transfer 回调延迟并返回失败，模拟超时场景
    mock_sender_->setShouldSucceed(false);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setCallbackDelayMs(200);  // 延迟 200ms，超过 deadline

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    auto              start_time_ms = currentTimeMs();

    std::thread write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }
    EXPECT_TRUE(wait_count < 500);

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证返回错误（因为 transfer 失败）
    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);

    // 验证 worker 等待了 transfer 回调返回后才结束（elapsed >= 回调延迟时间）
    EXPECT_GE(elapsed_ms, 150);  // 允许一些误差

    // 验证 transfer 被调用了
    EXPECT_GT(mock_sender_->getTransferCallCount(), 1);
}

// ==================== read 测试 (Decode 端) ====================

TEST_F(P2PConnectorWorkerTest, Read_ReturnTrue_AllLayersSuccess) {
    std::string unique_key  = "test_read_success";
    int64_t     request_id  = 3001;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(0, "group2", 2));
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(1, "group2", 2));

    std::thread completion_thread([this, unique_key]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        simulateTaskDone(unique_key, {0, 1}, true);
    });

    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    EXPECT_TRUE(error_info.ok());
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_PartialLayersFailed) {
    std::string unique_key  = "test_read_partial_fail";
    int64_t     request_id  = 3002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(0, "group2", 2));
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(1, "group2", 2));

    std::thread completion_thread([this, unique_key]() {
        // Layer 0 成功，layer 1 失败
        mock_receiver_->setTaskDone(unique_key + "_0_0", true);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mock_receiver_->setTaskDone(unique_key + "_1_0", false);
    });

    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    EXPECT_TRUE(error_info.hasError());
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_Timeout) {
    std::string unique_key  = "test_read_timeout";
    int64_t     request_id  = 3003;
    int64_t     deadline_ms = currentTimeMs() + 10;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(0, "group2", 2));

    auto start_time_ms = currentTimeMs();

    // 不调用 simulateTaskDone；return_deadline = D - return_before_ms 已过，尽快以 TRANSFER_NOT_DONE 返回
    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    auto end_time_ms = currentTimeMs();
    EXPECT_LE(end_time_ms - start_time_ms, 300);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnTrue_EmptyBuffers) {
    std::string unique_key  = "test_read_empty";
    int64_t     request_id  = 3004;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    EXPECT_TRUE(error_info.ok());
}

// ==================== rdma_transfer_wait_timeout_ms 超时测试 ====================

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_RdmaTransferWaitTimeout) {
    // 设置很短的 rdma_transfer_wait_timeout_ms；return_deadline 须较近，否则 callback 等待会持续到 D-return_before
    setTransferWaitTimeout(50);  // 50ms

    int64_t     request_id = 4001;
    std::string unique_key = "test_rdma_transfer_wait_timeout_handleread";
    // return_deadline = deadline_ms - return_before_ms(100) ≈ now + 50ms，与 rdma cap 同量级
    int64_t deadline_ms = currentTimeMs() + 150;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 设置 transfer 回调延迟，超过 rdma_transfer_wait_timeout_ms
    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setCallbackDelayMs(500);  // 500ms，超过 50ms 的超时

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    auto              start_time_ms = currentTimeMs();

    std::thread write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证返回错误（因为 rdma_transfer_wait_timeout_ms 超时）
    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);

    // 验证等待时间约为 rdma_transfer_wait_timeout_ms（50ms），而不是回调延迟（500ms）
    EXPECT_GE(elapsed_ms, 40);   // 允许一些误差
    EXPECT_LE(elapsed_ms, 300);  // 应该远小于 500ms

    // 验证 transfer 被调用了
    EXPECT_GT(mock_sender_->getTransferCallCount(), 0);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_RdmaTransferWaitTimeout) {
    std::string unique_key = "test_rdma_transfer_wait_timeout_read";
    int64_t     request_id = 4002;
    // return_deadline = D - return_before_ms ≈ now + 100ms
    int64_t deadline_ms = currentTimeMs() + 200;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(0, "group2", 2));
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(1, "group2", 2));

    auto start_time_ms = currentTimeMs();

    // 不调用 simulateTaskDone：在 return 截止前退出，返回 TRANSFER_NOT_DONE，不 forceCancel
    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    // 至少等到接近 return 截止（约 100ms），允许退避误差
    EXPECT_GE(elapsed_ms, 85);
    EXPECT_LE(elapsed_ms, 400);
    EXPECT_GE(mock_receiver_->stealTaskCount(), 1);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_CancelRead) {
    std::string unique_key  = "test_read_cancel";
    int64_t     request_id  = 3005;
    int64_t     deadline_ms = currentTimeMs() + 5000;  // 足够长的 deadline

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(0, "group2", 2));
    layer_cache_buffers.push_back(createTaggedLayerCacheBuffer(1, "group2", 2));

    std::atomic<bool> done{false};
    ErrorInfo         result;

    // 启动 read 线程
    std::thread read_thread([&]() {
        result = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);
        done   = true;
    });

    // 等待 mock receiver 收到 recv() 请求（表示 read() 已在等待中）
    int wait_count = 0;
    while (!mock_receiver_->hasEnoughTasks(unique_key, static_cast<int>(layer_cache_buffers.size()))
           && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        wait_count++;
    }

    // 调用 cancelRead 取消任务（重试直到找到任务，因为 read_tasks_ 在 recv 后才插入）
    bool cancel_result = false;
    for (int i = 0; i < 50 && !cancel_result; ++i) {
        cancel_result = decode_->cancelRead(unique_key);
        if (!cancel_result) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    EXPECT_TRUE(cancel_result);

    // 等待 read 完成
    wait_count = 0;
    while (!done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (read_thread.joinable()) {
        read_thread.join();
    }

    // 验证返回错误（因为被取消）
    EXPECT_TRUE(result.hasError());
    EXPECT_EQ(result.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);
    EXPECT_TRUE(done);
}

TEST_F(P2PConnectorWorkerTest, CancelRead_ReturnFalse_TaskNotFound) {
    std::string unique_key = "test_cancel_not_found";

    // 不创建任务，直接调用 cancelRead
    bool cancel_result = decode_->cancelRead(unique_key);

    // 验证返回 false（任务不存在）
    EXPECT_FALSE(cancel_result);
}

TEST_F(P2PConnectorWorkerTest, CancelHandleRead_ReturnTrue_ContextNotFound) {
    std::string unique_key = "test_cancel_handle_read_not_found";

    // 不创建 context，直接调用 cancelSend
    // 由于 cancel 是尽力而为，即使 context 不存在也返回 true
    bool cancel_result = prefill_->cancelSend(unique_key);

    // 验证返回 true（因为 cancel 是 best-effort）
    EXPECT_TRUE(cancel_result);
}

TEST_F(P2PConnectorWorkerTest, CancelHandleRead_ReturnTrue_ContextFound) {
    int64_t     request_id  = 3006;
    std::string unique_key  = "test_cancel_handle_read_found";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    std::atomic<bool> done{false};
    ErrorInfo         result;

    // 启动 sendKVCache 线程（未添加 computed buffer，sendKVCache 会阻塞等待）
    std::thread handle_read_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    // 稍等，让 sendKVCache 线程启动并注册 cancel flag
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // 调用 cancelSend 取消 context
    bool cancel_result = prefill_->cancelSend(unique_key);
    EXPECT_TRUE(cancel_result);

    // 等待 sendKVCache 完成
    int wait_count = 0;
    while (!done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (handle_read_thread.joinable()) {
        handle_read_thread.join();
    }

    // 验证返回错误（因为被取消）
    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);
}

// 异步回调错峰完成，覆盖 waitSendCallbacksWithTimeout 中 result_cv 多次 wait_for 唤醒
TEST_F(P2PConnectorWorkerTest, SendKVCache_Succeeds_StaggeredAsyncCallbacks) {
    int64_t     request_id  = 2009;
    std::string unique_key  = "test_staggered_cv";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setStaggeredCallbackDelays(true, 25);
    mock_sender_->setCallbackDelayMs(0);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }
    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(done);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);
    mock_sender_->setStaggeredCallbackDelays(false);
}

// ==================== sendKVCache callback wait timeout 测试 ====================

// 测试：send callback 超时未回调 -> sendKVCache 应报告失败而非成功
TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_CallbackWaitTimeout) {
    int64_t     request_id  = 4001;
    std::string unique_key  = "test_callback_wait_timeout";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // callback 永不触发（大 delay），验证超时保护下不会误判成功
    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setCallbackDelayMs(10000);  // 10s delay，远超 rdma_transfer_wait_timeout_ms

    // 设置很短的 rdma_transfer_wait_timeout_ms
    setTransferWaitTimeout(50);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(done);
    // callback 未收齐，必须报告失败（fix: 修复前此处可能误判成功）
    EXPECT_TRUE(result.hasError());
    EXPECT_EQ(result.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT);
}

// ==================== LayerCacheBufferUtil 边界测试 ====================

class LayerCacheBufferUtilTest: public ::testing::Test {
protected:
    KVCacheResourcePtr createResource(int num_layers, int blocks_per_layer) {
        auto                                  resource = std::make_shared<KVCacheResource>();
        std::vector<std::vector<std::string>> layer_to_group_tags(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_to_group_tags[i] = {"group" + std::to_string(i)};
        }
        config_ = makeTestCacheConfig(makeTestCacheTopologyByTag(num_layers, num_layers, layer_to_group_tags));
        resource->initGroups(config_);
        for (int layer = 0; layer < num_layers; ++layer) {
            // Each layer owns exactly one group; layer_to_group_tags records its tag.
            const auto& tag = layer_to_group_tags[layer].front();
            for (int i = 0; i < blocks_per_layer; ++i) {
                resource->mutableBlockIds(tag).add({i});
            }
        }
        for (int i = 0; i < blocks_per_layer; ++i) {
            resource->appendCacheKey(1000 + i);
        }
        return resource;
    }

    CacheConfig config_;
};

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_StartIdxEqualActualCount) {
    auto resource = createResource(2, 3);
    // start_block_idx == actual_block_count (3) -> out of range
    auto buf = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 3, -1, 0, 1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_StartIdxGreaterThanActualCount) {
    auto resource = createResource(2, 3);
    // start_block_idx > actual_block_count
    auto buf = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 10, -1, 0, 1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_BlockCountLessThanNegativeOne) {
    auto resource = createResource(2, 3);
    // block_count < -1 is undefined/illegal
    auto buf = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 0, -2, 0, 1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_BlockCountZero) {
    auto resource = createResource(2, 3);
    auto buf      = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 0, 0, 0, 1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnPartial_BlockCountLimitsResult) {
    auto resource = createResource(2, 4);
    // start=1, count=2 -> should return 2 blocks
    auto buf = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 1, 2, 0, 1);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(static_cast<int>(buf->blockIdMap().size()), 2);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnAll_BlockCountNegativeOne) {
    auto resource = createResource(2, 3);
    // block_count=-1 means "all remaining"
    auto buf = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 0, -1, 0, 1);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(static_cast<int>(buf->blockIdMap().size()), 3);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_SkipsSparseNullBlocks) {
    auto resource = createResource(2, 3);
    resource->mutableBlockIds("group0").assign({NULL_BLOCK_IDX, 7, NULL_BLOCK_IDX});

    auto buf = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 0, -1, 0, 1);
    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(buf->blockIdMap().size(), 1u);
    EXPECT_EQ(buf->blockIdMap().at(1001), 7);

    resource->mutableBlockIds("group0").assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    EXPECT_EQ(LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", 0, -1, 0, 1), nullptr);
}

TEST_F(LayerCacheBufferUtilTest, HasTransferableBlocksHonorsSparseStartAndCountWindow) {
    auto resource = createResource(1, 3);
    resource->mutableBlockIds("group0").assign({NULL_BLOCK_IDX, 7, NULL_BLOCK_IDX});
    EXPECT_TRUE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -1, 0, 1));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, 1, 0, 1));
    EXPECT_TRUE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 1, 1, 0, 1));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 2, -1, 0, 1));

    resource->mutableBlockIds("group0").assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -1, 0, 1));
}

TEST_F(LayerCacheBufferUtilTest, HasTransferableBlocksHonorsCpKeyBoundsAndValidation) {
    auto resource = createResource(1, 3);
    resource->mutableBlockIds("group0").assign({NULL_BLOCK_IDX, 7, 8});
    resource->setCacheKeys(CacheKeysType(1));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -1, 1, 2));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 1, -1, 0, 2));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", -1, -1, 0, 2));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, 0, 0, 2));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -2, 0, 2));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -1, 0, 0));
    EXPECT_FALSE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -1, 2, 2));

    resource->mutableBlockIds("group0").setAt(0, 9);
    EXPECT_TRUE(LayerCacheBufferUtil::hasTransferableBlocks(config_, *resource, 0, "group0", 0, -1, 0, 2));
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_StartIdxNegative) {
    auto resource = createResource(2, 3);
    auto buf      = LayerCacheBufferUtil::convertLayer(config_, *resource, 0, 0, "group0", -1, -1, 0, 1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, DirectConversionSortsTagsAtWireBoundary) {
    config_.seq_size_per_block = 8;
    config_.layer_num          = 1;
    config_.layer_all_num      = 1;
    GroupBase z_group;
    z_group.tag                       = "z_group";
    z_group.spec                      = std::make_shared<MHAKVCacheSpec>();
    z_group.layer_ids                 = {0};
    z_group.seq_size_per_block        = 8;
    z_group.kernel_seq_size_per_block = 8;
    z_group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    GroupBase a_group                 = z_group;
    a_group.tag                       = "a_group";
    config_.setTopology({std::move(z_group), std::move(a_group)}, {{0, {"z_group", "a_group"}}});

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(config_);
    resource->mutableBlockIds("z_group").assign({10, 11});
    resource->mutableBlockIds("a_group").assign({20, 21});
    resource->setCacheKeys({100, 101});

    auto buffers = LayerCacheBufferUtil::convert(config_, *resource, 0);

    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0]->cacheTag(), "a_group");
    EXPECT_EQ(buffers[1]->cacheTag(), "z_group");
    EXPECT_EQ(buffers[0]->blockIdMap().at(101), 21);
    EXPECT_EQ(buffers[1]->blockIdMap().at(100), 10);
}

TEST_F(LayerCacheBufferUtilTest, FullRoundRobinUsesGlobalKeyOrdinalsWithDifferentPhysicalSpan) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 1;
    config.layer_all_num      = 1;
    GroupBase full;
    full.tag                       = "full";
    full.spec                      = std::make_shared<MHAKVCacheSpec>();
    full.layer_ids                 = {0};
    full.seq_size_per_block        = 24;
    full.kernel_seq_size_per_block = 8;
    full.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    config.setTopology({std::move(full)}, {{0, {"full"}}});

    KVCacheResource resource;
    resource.initGroups(config);
    resource.mutableBlockIds("full").assign({41, 43});
    CacheKeysType keys;
    for (int i = 0; i < 12; ++i) {
        keys.push_back(1000 + i);
    }
    resource.setCacheKeys(std::move(keys));

    auto buffer = LayerCacheBufferUtil::convertLayer(
        config, resource, 0, 0, "full", /*start_key_ordinal=*/0, /*key_count=*/-1, /*cp_rank=*/1, /*cp_size=*/2);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->blockIdMap().at(1005), 41);
    EXPECT_EQ(buffer->blockIdMap().at(1011), 43);

    resource.mutableBlockIds("full").setAt(0, NULL_BLOCK_IDX);
    buffer = LayerCacheBufferUtil::convertLayer(config, resource, 0, 0, "full", 0, -1, 1, 2);
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(buffer->blockIdMap().size(), 1u);
    EXPECT_EQ(buffer->blockIdMap().at(1011), 43);
}

TEST_F(LayerCacheBufferUtilTest, CompactSwaUsesConfigurableTailAndTransientPhysicalOrdinal) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 1;
    config.layer_all_num      = 1;
    GroupBase swa;
    swa.tag                       = "swa";
    swa.spec                      = std::make_shared<MHAKVCacheSpec>();
    swa.layer_ids                 = {0};
    swa.seq_size_per_block        = 32;
    swa.kernel_seq_size_per_block = 8;
    swa.policy                    = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa.policy.active_tail_blocks = 2;
    config.setTopology({std::move(swa)}, {{0, {"swa"}}});

    KVCacheResource resource;
    resource.initGroups(config);
    resource.mutableBlockIds("swa").assign({70, 71});
    CacheKeysType keys;
    for (int i = 0; i < 32; ++i) {
        keys.push_back(2000 + i);
    }
    resource.setCacheKeys(std::move(keys));

    auto buffer = LayerCacheBufferUtil::convertLayer(
        config, resource, 0, 0, "swa", /*start_key_ordinal=*/0, /*key_count=*/-1, /*cp_rank=*/0, /*cp_size=*/4);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->blockIdMap().at(2015), 70);
    EXPECT_EQ(buffer->blockIdMap().at(2031), 71);

    resource.mutableBlockIds("swa").setAt(0, NULL_BLOCK_IDX);
    buffer = LayerCacheBufferUtil::convertLayer(config, resource, 0, 0, "swa", 0, -1, 0, 4);
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(buffer->blockIdMap().size(), 1u);
    EXPECT_EQ(buffer->blockIdMap().at(2031), 71);
}

TEST_F(LayerCacheBufferUtilTest, RealDsv4SevenTagsIgnoreReversedRecordsAndFilterNullWireBlocks) {
    const auto                     config        = makeRealDsv4P2PCacheConfig();
    const std::vector<std::string> topology_tags = {
        "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state", "hca_kv", "hca_state"};
    EXPECT_EQ(groupTagSet(config), std::set<std::string>(topology_tags.begin(), topology_tags.end()));
    EXPECT_EQ(config.seq_size_per_block, 128u);

    const auto expect_group = [&config](std::string_view   tag,
                                        CacheGroupType     type,
                                        CpBlockMappingMode cp_mapping,
                                        uint32_t           active_tail_blocks,
                                        size_t             physical_block_size,
                                        size_t             kernel_block_size) {
        const auto& group = config.group(tag);
        EXPECT_EQ(group.policy.group_type, type) << tag;
        EXPECT_EQ(group.policy.cp_mapping, cp_mapping) << tag;
        EXPECT_EQ(group.policy.active_tail_blocks, active_tail_blocks) << tag;
        EXPECT_EQ(group.seq_size_per_block, physical_block_size) << tag;
        EXPECT_EQ(group.kernel_seq_size_per_block, kernel_block_size) << tag;
    };
    expect_group("swa_kv", CacheGroupType::SWA, CpBlockMappingMode::COMPACT_LAST_RANK, 2, 512, 512);
    expect_group("csa_kv", CacheGroupType::FULL, CpBlockMappingMode::BLOCK_ROUND_ROBIN, 0, 128, 128);
    expect_group("indexer_kv", CacheGroupType::FULL, CpBlockMappingMode::BLOCK_ROUND_ROBIN, 0, 128, 128);
    expect_group("indexer_state", CacheGroupType::SWA, CpBlockMappingMode::COMPACT_LAST_RANK, 2, 512, 512);
    expect_group("csa_state", CacheGroupType::SWA, CpBlockMappingMode::COMPACT_LAST_RANK, 2, 512, 512);
    expect_group("hca_kv", CacheGroupType::FULL, CpBlockMappingMode::BLOCK_ROUND_ROBIN, 0, 128, 128);
    expect_group("hca_state", CacheGroupType::SWA, CpBlockMappingMode::COMPACT_LAST_RANK, 1, 512, 512);

    auto resource = makeRealDsv4P2PResource(config);
    auto buffers  = LayerCacheBufferUtil::convert(
        config, *resource, 0, /*start_key_ordinal=*/0, /*key_count=*/-1, kDsv4CpRank, kDsv4CpSize);
    ASSERT_EQ(buffers.size(), 167u);
    size_t buffer_index   = 0;
    size_t expected_count = 0;
    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto sorted_tags = config.groupsForLayer(layer_id);
        std::sort(sorted_tags.begin(), sorted_tags.end());
        for (const auto& tag : sorted_tags) {
            expected_count += LayerCacheBufferUtil::hasTransferableBlocks(
                config, *resource, layer_id, tag, 0, -1, kDsv4CpRank, kDsv4CpSize);

            ASSERT_LT(buffer_index, buffers.size());
            EXPECT_EQ(buffers[buffer_index]->getLayerId(), layer_id);
            EXPECT_EQ(buffers[buffer_index]->cacheTag(), tag);
            expectRealDsv4WireMap(config, *buffers[buffer_index]);
            ++buffer_index;
        }
    }
    EXPECT_EQ(buffer_index, buffers.size());
    EXPECT_EQ(expected_count, 167u);
}

TEST_F(LayerCacheBufferUtilTest, InvalidDirectRangeProducesNoOutput) {
    auto                                   resource = createResource(1, 2);
    ObservedLayerCacheBufferUtil::Observer observer;
    const auto buffers = ObservedLayerCacheBufferUtil::convert(config_, *resource, 0, -1, -1, 0, 1, observer);
    EXPECT_TRUE(buffers.empty());
    EXPECT_EQ(observer.constructed, 0u);
    EXPECT_EQ(observer.published, 0u);
}

TEST_F(LayerCacheBufferUtilTest, ValidFirstInvalidLaterGroupPublishesNoPartialBuffers) {
    const auto make_config = [](size_t second_group_block_size) {
        CacheConfig config;
        config.seq_size_per_block = 8;
        config.layer_num          = 1;
        config.layer_all_num      = 1;
        GroupBase first;
        first.tag                       = "first";
        first.spec                      = std::make_shared<MHAKVCacheSpec>();
        first.layer_ids                 = {0};
        first.seq_size_per_block        = 8;
        first.kernel_seq_size_per_block = 8;
        first.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        GroupBase second                = first;
        second.tag                      = "second";
        second.seq_size_per_block       = second_group_block_size;
        config.setTopology({std::move(first), std::move(second)}, {{0, {"first", "second"}}});
        return config;
    };

    const auto resource_config = make_config(8);
    auto       resource        = std::make_shared<KVCacheResource>();
    resource->initGroups(resource_config);
    resource->mutableBlockIds("first").assign({10, 11});
    resource->mutableBlockIds("second").assign({20, 21});
    resource->setCacheKeys({100, 101});

    // The first group is valid. The second config expects two kernel blocks per
    // physical block while the already-created resource stores one. Validation
    // must reach that later group before constructing the first output buffer.
    const auto                             invalid_config = make_config(16);
    ObservedLayerCacheBufferUtil::Observer observer;
    EXPECT_ANY_THROW(ObservedLayerCacheBufferUtil::convert(invalid_config, *resource, 0, 0, -1, 0, 1, observer));
    EXPECT_EQ(observer.constructed, 0u);
    EXPECT_EQ(observer.published, 0u);
}

}  // namespace test
}  // namespace rtp_llm
