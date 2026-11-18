#pragma once

#include <memory>
#include <chrono>
#include <cstdint>
#include <optional>
#include <functional>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
class DeviceBlockPool;
// One actual complete endpoint. No schema, per-token history or Graph inputs.
struct DecodeEndpoint {
    size_t end{0};
    std::unordered_map<std::string, BlockIndicesType> blocks;
    // The original per-page pin, shared only for matching FULL slots. Holding
    // a pool/block prevents recycling; it is not a writer-completion proof.
    struct PageHold {
        std::shared_ptr<DeviceBlockPool> pool;
        BlockIdxType block{NULL_BLOCK_IDX};
        RemotePublicationPin pin;
        PageHold() = default;
        PageHold(const PageHold&) = delete;
        PageHold& operator=(const PageHold&) = delete;
        PageHold(PageHold&&) = delete;
        PageHold& operator=(PageHold&&) = delete;
        ~PageHold() = default;
    };
    using PageHolds = std::vector<std::shared_ptr<const PageHold>>;
    // FULL holds are slot-indexed. STATE/SWA hold only this endpoint's required
    // tail pages and are never inherited from another endpoint.
    std::unordered_map<std::string, PageHolds> pages;
};
class BatchKVCacheResource {
public:
    using DecodePublicationClock = std::chrono::steady_clock;
    BatchKVCacheResource() {}

    int batchSize() const {
        return static_cast<int>(batch_resource.size());
    }

    void resetBatchSize(size_t batch_size) {
        if (batch_size != batch_resource.size()) {
            resetDecodePublication();
            decode_endpoint.reset(); decode_round_endpoints.clear();
        }
        batch_resource.resize(batch_size);
        cache_keys_initialized_ = false;
    }

    void initGroups(std::shared_ptr<const CacheTopology> topology) {
        RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "BatchKVCacheResource::initGroups requires a topology");
        for (auto& batch : batch_resource) {
            batch.initGroups(topology);
        }
    }

    int groupNums() const {
        RTP_LLM_CHECK(!batch_resource.empty());
        return batch_resource[0].groupNums();
    }

    void resizeBlocks(int reserver_blocks, int value = 0) {
        for (auto& resource : batch_resource) {
            resource.resizeBlocks(reserver_blocks, value);
        }
    }

    int blocksNum(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).blocksNum(tag);
    }

    int curBlocksNum() const {
        if (batch_resource.empty()) {
            return 0;
        }

        return batch_resource[0].maxBlocksNum();
    }

    const BlockIndicesType& blocks(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).blocks(tag);
    }

    const BlockIndicesType& blocksForLayer(int batch_id, int layer_id, std::string_view tag) const {
        return cacheResource(batch_id).blocksForLayer(layer_id, tag);
    }

    const BlockIndicesType& kernelBlocks(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).kernelBlocks(tag);
    }

    const BlockIndicesType& kernelBlocksForLayer(int batch_id, int layer_id, std::string_view tag) const {
        return cacheResource(batch_id).kernelBlocksForLayer(layer_id, tag);
    }

    BlockIds& mutableBlockIds(int batch_id, std::string_view tag) {
        return cacheResource(batch_id).mutableBlockIds(tag);
    }

    BlockIds& mutableBlockIdsForLayer(int batch_id, int layer_id, std::string_view tag) {
        return cacheResource(batch_id).mutableBlockIdsForLayer(layer_id, tag);
    }

    std::shared_ptr<BlockIds> groupBlockIds(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).groupBlockIds(tag);
    }

    const KVCacheResource& cacheResource(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id];
    }

    KVCacheResource& cacheResource(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id];
    }

    void clearBlocks() {
        resetDecodePublication();
        decode_endpoint.reset(); decode_round_endpoints.clear();
        resizeBlocks(0, 0);
    }

    const CacheKeysType& cacheKeys(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].cacheKeys();
    }

    void popBackCacheKey(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        auto& keys = batch_resource[batch_id].cacheKeys();
        if (!keys.empty()) {
            keys.pop_back();
        }
    }

    void popBackAllBatchCacheKeys() {
        for (auto& resource : batch_resource) {
            auto& keys = resource.cacheKeys();
            if (!keys.empty()) {
                keys.pop_back();
            }
        }
    }

    void clearCacheKeys(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].cacheKeys().clear();
        cache_keys_initialized_ = false;
    }

    void pushBackCacheKey(int batch_id, CacheKeyType key) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        auto& resource = batch_resource[batch_id];
        auto& keys     = resource.cacheKeys();
        keys.push_back(key);
    }

    void setBatchBlocks(int batch_id, std::string_view tag, const BlockIndicesType& blocks) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].mutableBlockIds(tag).assign(blocks);
    }

    void setBatchCacheKeys(int batch_id, const CacheKeysType& keys) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].cacheKeys() = keys;
    }

    void check() const {
        RTP_LLM_CHECK(!batch_resource.empty());
        const auto& first = batch_resource[0];
        const auto  tags  = first.groupTags();
        for (const auto& resource : batch_resource) {
            RTP_LLM_CHECK(resource.groupNums() == first.groupNums());
            for (const auto& tag : tags) {
                RTP_LLM_CHECK(resource.blocksNum(tag) == first.blocksNum(tag));
            }
        }
    }

    std::string debugString() const {
        std::stringstream debug_string, batch_resource_string;
        for (size_t i = 0; i < batch_resource.size(); i++) {
            batch_resource_string << "batch:[" << i << "], detail info: ";
            batch_resource_string << batch_resource[i].debugString();
        }

        debug_string << "BatchKVCacheResource {" << batch_resource_string.str() << "}";
        return debug_string.str();
    }

    void resetAndReturnOldResources(int new_batch_size, std::vector<KVCacheResource>& old_resources) {
        old_resources = std::move(batch_resource);
        batch_resource.clear();
        batch_resource.resize(new_batch_size);
        cache_keys_initialized_ = false;
    }

    void moveBatchResource(int batch_idx, KVCacheResource&& resource) {
        RTP_LLM_CHECK(batch_idx >= 0 && static_cast<size_t>(batch_idx) < batch_resource.size());
        batch_resource[batch_idx] = std::move(resource);
    }

    void swap(BatchKVCacheResource& other) noexcept {
        decode_endpoint.swap(other.decode_endpoint);
        decode_round_endpoints.swap(other.decode_round_endpoints);
        std::swap(decode_completion_submitted_, other.decode_completion_submitted_);
        std::swap(decode_preparing_, other.decode_preparing_);
        std::swap(decode_attempts_, other.decode_attempts_);
        std::swap(decode_eos_at_, other.decode_eos_at_);
        std::swap(decode_deadline_, other.decode_deadline_);
        std::swap(prefill_computed_end_, other.prefill_computed_end_);
        swapBlockStorage(other);
    }

    // An atomic block-table remap stages new storage for the same request. It
    // must not exchange the logical EOS/once-only state with an empty staging
    // object, or release-time fallback could publish a second time.
    void swapBlockStorage(BatchKVCacheResource& other) noexcept {
        batch_resource.swap(other.batch_resource);
        std::swap(cache_keys_initialized_, other.cache_keys_initialized_);
    }

    std::vector<BlockIndicesType> getAllBatchBlocks(std::string_view tag) const {
        std::vector<BlockIndicesType> all_blocks;
        all_blocks.reserve(batch_resource.size());
        for (const auto& resource : batch_resource) {
            all_blocks.push_back(resource.blocks(tag));
        }
        return all_blocks;
    }

    bool hasCacheKeys() const {
        if (batch_resource.empty()) {
            return false;
        }
        for (const auto& resource : batch_resource) {
            if (!resource.cacheKeys().empty()) {
                return true;
            }
        }
        return false;
    }

    // Set by initCacheKeys()/updateCacheKeys() once the rolling cache keys have been computed for
    // this resource. A first malloc that fails with MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED
    // re-enters KVCacheManager::malloc() having allocated nothing, so curBlocksNum() is still zero
    // and the attempt still looks like a first malloc. This flag tells the retry apart from a
    // genuine first attempt, so the keys are not recomputed and the prefill-cache-hit metric is not
    // double-counted for the same request.
    bool cacheKeysInitialized() const {
        return cache_keys_initialized_;
    }

    void markCacheKeysInitialized() {
        cache_keys_initialized_ = true;
    }

    bool lastBlockAligned() const {
        for (const auto& resource : batch_resource) {
            if (!resource.lastBlockAligned()) {
                return false;
            }
        }
        return true;
    }

    void setLastBlockAligned(bool last_block_aligned) {
        for (auto& resource : batch_resource) {
            resource.setLastBlockAligned(last_block_aligned);
        }
    }

    void swapBlocks(int32_t batch_id, std::string_view group_tag, size_t rhs, size_t lhs) {
        cacheResource(batch_id).swapBlocks(group_tag, rhs, lhs);
    }

    std::shared_ptr<const DecodeEndpoint> decode_endpoint;
    // Native endpoints retained through EOS, plus not-yet-validated candidates
    // from the current verify packet. They carry pages, never token snapshots.
    std::vector<std::shared_ptr<const DecodeEndpoint>> decode_round_endpoints;

    // Scheduler-owned idempotency only. Payloads stay in their native GPU pools;
    // StorageBackend owns the retained references after EOS admission.
    bool& decodeCompletionSubmitted() const {
        return decode_completion_submitted_;
    }
    bool mayAttemptDecodePublication(DecodePublicationClock::time_point now = DecodePublicationClock::now()) const {
        return !decode_completion_submitted_ && !decode_preparing_ && decode_attempts_ < 2
               && (!decode_deadline_ || now < *decode_deadline_);
    }
    bool beginDecodePublication(DecodePublicationClock::time_point now = DecodePublicationClock::now()) const {
        if (!mayAttemptDecodePublication(now)) return false;
        ++decode_attempts_;
        decode_preparing_ = true;
        return true;
    }
    void endDecodePublication() const noexcept { decode_preparing_ = false; }
    size_t decodePublicationAttempts() const { return decode_attempts_; }
    DecodePublicationClock::time_point decodePublicationDeadline(std::chrono::milliseconds timeout,
        DecodePublicationClock::time_point now = DecodePublicationClock::now()) const {
        if (!decode_deadline_) decode_deadline_ = decode_eos_at_.value_or(now) + timeout;
        return *decode_deadline_;
    }
    void recordDecodePublicationEos(DecodePublicationClock::time_point at = DecodePublicationClock::now()) const {
        if (!decode_eos_at_) decode_eos_at_ = at;
    }
    const auto& decodePublicationEos() const { return decode_eos_at_; }
    bool recordPrefillComputedEnd(int64_t end) const {
        if (end < 0 || (prefill_computed_end_ && *prefill_computed_end_ != end)) return false;
        prefill_computed_end_ = end;
        return true;
    }
    const auto& prefillComputedEnd() const { return prefill_computed_end_; }

private:
    mutable bool                 decode_completion_submitted_{false};
    mutable bool                 decode_preparing_{false};
    mutable size_t               decode_attempts_{0};
    mutable std::optional<DecodePublicationClock::time_point> decode_eos_at_, decode_deadline_;
    mutable std::optional<int64_t> prefill_computed_end_;
    void resetDecodePublication() {
        decode_completion_submitted_ = decode_preparing_ = false;
        decode_attempts_ = 0;
        decode_eos_at_.reset(); decode_deadline_.reset(); prefill_computed_end_.reset();
    }
    std::vector<KVCacheResource> batch_resource;  // [batch_size]
    bool                         cache_keys_initialized_{false};
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
