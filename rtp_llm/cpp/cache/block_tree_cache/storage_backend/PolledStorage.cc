#include "PolledStorage.h"

#include "../BlockTreeCacheCInternal.h"
#include "StorageBackend.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <stdexcept>

namespace rtp_llm::c_api {
namespace {

// Polled operations enqueue value snapshots and do not do I/O on this thread.
// The common backend accounts for admitted work until its last safety receipt.
class PollExecutor final: public StorageBackendExecutor {
public:
    bool start() override {
        return true;
    }
    bool submit(Task task, BlockTreeTaskClass) override {
        if (!accepting_.load()) {
            return false;
        }
        task();
        return true;
    }
    bool submitCompletion(Task task) override {
        task();
        return true;
    }
    void shutdown() noexcept override {
        accepting_.store(false);
    }

private:
    std::atomic<bool> accepting_{true};
};

using Position = std::pair<uint32_t, uint64_t>;

struct PolledMatchMeta final: StorageBackendMatchMeta {
    using Positions = std::vector<std::unordered_map<std::string, Position>>;
    PolledMatchMeta(Positions positions, PolledOutcome::Objects descriptors):
        positions(std::move(positions)), descriptors(std::move(descriptors)) {}

    const std::string* descriptor(Position position) const {
        auto found = descriptors.find(position);
        return found == descriptors.end() ? nullptr : &found->second;
    }

    std::string replicaDescriptor(size_t row, const std::vector<std::string>& tags) const override {
        if (row >= positions.size() || tags.empty()) {
            return {};
        }
        const auto& row_positions = positions[row];
        auto        first         = row_positions.find(tags.front());
        if (first == row_positions.end()) {
            return {};
        }
        for (const auto& tag : tags) {
            auto found = row_positions.find(tag);
            if (found == row_positions.end() || found->second != first->second) {
                return {};
            }
        }
        const auto* proof = descriptor(first->second);
        return proof ? *proof : std::string{};
    }

    const Positions              positions;
    const PolledOutcome::Objects descriptors;
};

StorageBackend::PageAccess pageAccess(std::shared_ptr<IBlockPool> pool) {
    if (!pool) {
        return {};
    }
    return {pool, [pool](BlockIdxType block) {
                std::function<void()> release = [pool, block] { pool->decTreeRef(block, BlockTreeRefType::LOAD); };
                pool->incTreeRef(block, BlockTreeRefType::LOAD);
                return release;
            }};
}


void checkPayload(const PolledOutcome& outcome, bool require_commit) {
    if (outcome.result == BTC_TIMEOUT) {
        throw StorageDeadlineExceeded("remote payload timed out before confirmed completion");
    }
    if (outcome.result != BTC_SUCCESS || (require_commit && !outcome.committed)) {
        throw StorageOperationBufferSafeFailure("remote payload was rejected or failed");
    }
}

class PolledStorage final: public StorageBackend {
public:
    explicit PolledStorage(btc_cache& cache):
        // The external executor supplies both deadline and buffer-safety receipts.
        StorageBackend(std::make_shared<PollExecutor>(), nullptr, -1, std::nullopt), cache_(cache) {
        Groups groups;
        for (size_t index = 0; index < cache_.groups.size(); ++index) {
            const auto& group  = cache_.groups[index];
            const auto& tags   = group.set->groupTags();
            const auto& pools  = group.set->pools();
            const auto  stride = group.page_tokens / cache_.key_tokens;
            for (size_t member = 0; member < tags.size(); ++member) {
                indices_.emplace(tags[member], index);
                groups.emplace(tags[member],
                               GroupAccess{
                                   pageAccess(pools.device.at(member)),
                                   pageAccess(pools.host),
                                   [set = group.set, stride](size_t row, size_t matched_rows) {
                                       const auto pages     = matched_rows / stride;
                                       const auto this_page = (row + 1) / stride;
                                       return this_page <= pages
                                              && pages - this_page < set->computeReuseBlockCount(pages);
                                   },
                               });
            }
        }
        if (!StorageBackend::init(std::move(groups))) {
            throw std::runtime_error("could not initialize polled remote storage");
        }
    }

    bool supportsPositionalReplicas() const noexcept override {
        return true;
    }
    bool supportsWriteTier(Tier tier) const noexcept override {
        if (tier != Tier::DEVICE && tier != Tier::HOST) {
            return false;
        }
        return cache_.transfers->supports(tier == Tier::HOST ? BTC_WORK_HOST_TO_REMOTE : BTC_WORK_DEVICE_TO_REMOTE);
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        return {request.local_matched_blocks_num, nullptr};
    }

    bool usesAsyncMatch(const StorageRequest&) const noexcept override {
        return cache_.transfers->supports(BTC_WORK_REMOTE_QUERY);
    }

    bool matchAsyncImpl(const StorageRequest& request, MatchCompletion done) override {
        if (!cache_.transfers->supports(BTC_WORK_REMOTE_QUERY)) {
            return false;
        }
        auto pages = snapshot(request, BTC_WORK_REMOTE_QUERY);
        if (pages.empty()) {
            return false;
        }
        const auto request_id = request.request_id.value_or(0);
        // MatchCompletion retains the original StorageRequest through settlement.
        cache_.transfers->submit(
            BTC_WORK_REMOTE_QUERY,
            request_id,
            std::move(pages),
            [this, &request, request_id, done = std::move(done), reported = std::make_shared<std::atomic<bool>>(false)](
                PolledOutcome outcome) {
                if (reported->exchange(true)) {
                    return;
                }
                if (request.prefetch_notice) {
                    request.prefetch_notice(false);
                } else {
                    cache_.event(request_id, BTC_EVENT_PROGRESS);
                }
                done([this, &request, outcome = std::move(outcome)]() mutable {
                    checkPayload(outcome, false);
                    PolledMatchMeta::Positions positions(request.handles.size());
                    for (size_t row = request.local_matched_blocks_num; row < request.handles.size(); ++row) {
                        for (const auto& handle : request.handles[row]) {
                            positions[row].emplace(handle.tag, position(row, handle));
                        }
                    }
                    auto match   = std::make_shared<PolledMatchMeta>(std::move(positions), std::move(outcome.objects));
                    auto matched = coveredPrefix(request, [&](size_t row, const StorageBlockHandle& handle) {
                        return match->descriptor(position(row, handle)) != nullptr;
                    });
                    return StorageMatchResult{matched, std::move(match)};
                });
            });
        return true;
    }

    bool usesAsyncPayload(const StorageRequest&, bool) const noexcept override {
        return true;
    }
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        throw std::logic_error("polled reads require asynchronous execution");
    }
    bool readAsyncImpl(const StorageRequest&                           request,
                       const std::shared_ptr<StorageBackendMatchMeta>& meta,
                       ReadCompletion                                  done) override {
        const auto kind  = payloadTier(request) == Tier::HOST ? BTC_WORK_REMOTE_TO_HOST : BTC_WORK_REMOTE_TO_DEVICE;
        auto       pages = snapshot(request, kind, dynamic_cast<const PolledMatchMeta*>(meta.get()));
        const auto request_id = request.request_id.value_or(0);
        cache_.transfers->submit(kind,
                                 request_id,
                                 std::move(pages),
                                 [this, kind, request_id, notice = request.prefetch_notice,
                                  done = std::move(done)](PolledOutcome outcome) {
                                     if (!outcome.safe) {
                                         if (kind == BTC_WORK_REMOTE_TO_HOST && notice) {
                                             notice(true);
                                         } else {
                                             cache_.event(request_id, BTC_EVENT_RECOMPUTE);
                                         }
                                         return;
                                     }
                                     done([outcome = std::move(outcome)] { checkPayload(outcome, false); });
                                 });
        return true;
    }

    void writeImpl(const StorageRequest&) override {
        throw std::logic_error("polled writes require asynchronous execution");
    }
    bool startWriteImpl(const StorageRequest& request, WriteCompletion done, WritePinFilter) override {
        const auto kind  = payloadTier(request) == Tier::HOST ? BTC_WORK_HOST_TO_REMOTE : BTC_WORK_DEVICE_TO_REMOTE;
        auto       pages = snapshot(request, kind);
        // Host writeback batches address one physical pool. An unsafe failure
        // wakes cold reads waiting on it, while storage retains the DMA pins.
        const auto pool = kind == BTC_WORK_HOST_TO_REMOTE
                              ? cache_.groups.at(cache_.groupIndex(pages.front().position.group)).set->hostPool()
                              : nullptr;
        cache_.transfers->submit(kind, request.request_id.value_or(0), std::move(pages),
                                 [this, pool, done = std::move(done)](PolledOutcome outcome) {
                                     if (outcome.safe) {
                                         done([outcome = std::move(outcome)](bool) { checkPayload(outcome, true); });
                                     } else if (pool && cache_.native) {
                                         cache_.native->notifyRemoteHostPressureFailure(*pool);
                                     }
                                 });
        return true;
    }

private:
    Position position(size_t row, const StorageBlockHandle& handle) const {
        const size_t index = indices_.at(handle.tag);
        const auto&  group = cache_.groups[index];
        return {group.id, handle.end_token ? handle.end_token : (row + 1) * cache_.key_tokens};
    }

    static Tier payloadTier(const StorageRequest& request) {
        for (const auto& row : request.handles) {
            if (!row.empty()) {
                return row.front().tier;
            }
        }
        throw std::invalid_argument("polled remote payload has no bound pages");
    }

    std::vector<PolledPage>
    snapshot(const StorageRequest& request, btc_work_kind kind, const PolledMatchMeta* meta = nullptr) const {
        const bool query = kind == BTC_WORK_REMOTE_QUERY;
        const bool read  = kind == BTC_WORK_REMOTE_TO_HOST || kind == BTC_WORK_REMOTE_TO_DEVICE;
        if (!request.opaque_keys || (read && !meta)) {
            throw std::invalid_argument("polled remote work requires opaque keys and matched read descriptors");
        }
        const auto                     first = query ? request.local_matched_blocks_num : 0;
        const auto                     tier  = query ? Tier::NONE : payloadTier(request);
        std::vector<PolledPage> pages;
        for (size_t row = first; row < request.handles.size(); ++row) {
            // Distinct eviction victims can share a group and token depth.
            // Coalesce member buffers within this logical key only.
            std::map<Position, PolledPage> work;
            for (const auto& handle : request.handles[row]) {
                const auto matched_position = position(row, handle);
                auto [it, inserted]         = work.try_emplace(matched_position);
                auto& item                  = it->second;
                if (inserted) {
                    item.position = {matched_position.first, matched_position.second};
                    item.key      = (*request.opaque_keys)[row].bytes();
                    if (read) {
                        const auto* proof = meta->descriptor(matched_position);
                        if (!proof) {
                            throw std::invalid_argument("polled remote read lost its matched position");
                        }
                        item.remote = *proof;
                    }
                }
                if (!query) {
                    if (handle.tier != tier) {
                        throw std::invalid_argument("polled remote payload cannot mix buffer tiers");
                    }
                    const auto page    = cache_.describePage(indices_.at(handle.tag), handle.tier, handle.block);
                    auto&      buffers = read ? item.target : item.source;
                    if (std::none_of(
                            buffers.begin(), buffers.end(), [&](const auto& bound) { return samePage(bound, page); })) {
                        buffers.push_back(page);
                    }
                }
            }
            for (auto& [_, page] : work)
                pages.push_back(std::move(page));
        }
        return pages;
    }

    btc_cache&                              cache_;
    std::unordered_map<std::string, size_t> indices_;
};

}  // namespace

std::shared_ptr<StorageBackend> makeHostStorage(btc_cache& cache, const btc_config&) {
    const auto& routes = *cache.transfers;
    if (routes.supports(BTC_WORK_REMOTE_QUERY) || routes.supports(BTC_WORK_HOST_TO_REMOTE)
        || routes.supports(BTC_WORK_DEVICE_TO_REMOTE)) {
        return std::make_shared<PolledStorage>(cache);
    }
    return nullptr;
}

}  // namespace rtp_llm::c_api
