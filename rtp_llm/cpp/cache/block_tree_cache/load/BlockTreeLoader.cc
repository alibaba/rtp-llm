#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CompletionStage.h"

namespace rtp_llm {
namespace {

bool participates(const GroupSet& group, bool skip_linear, const std::vector<bool>& selected_groups) {
    return (selected_groups.empty() || selected_groups[group.groupSetId()])
           && (!skip_linear || group.groupType() != CacheGroupType::LINEAR);
}

size_t firstPageRow(size_t first, size_t stride) {
    return first + (stride - 1 - first % stride);
}

size_t reuseBegin(const GroupSet& group, size_t end, size_t stride) {
    const size_t pages = end / stride;
    return (pages - std::min(group.computeReuseBlockCount(pages), pages)) * stride;
}

// A staged lower-tier match must fit the same Host pools that will supply
// its eventual Device load. Each group's native rule chooses its suffix.
class StagedHostBudget {
public:
    explicit StagedHostBudget(const std::vector<GroupSetPtr>& groups): groups_(groups.size()) {
        std::unordered_map<const IBlockPool*, size_t> pool_indices;
        for (size_t group = 0; group < groups.size(); ++group) {
            const auto* host = groups[group]->pools().host.get();
            const auto [index, inserted] = pool_indices.emplace(host, pools_.size());
            if (inserted) {
                pools_.push_back({host ? host->totalBlocksNum() : 0, 0});
            }
            groups_[group].pool = index->second;
        }
    }

    void beginCandidate() {
        for (auto& pool : pools_) {
            pool.used = 0;
        }
    }

    void append(size_t group, bool needs_host, size_t required) {
        auto& current = groups_[group];
        auto& prefix = current.prefix;
        prefix.push_back(prefix.back() + needs_host);
        pools_[current.pool].used += prefix.back() - prefix[prefix.size() - 1 - required];
    }

    bool fits() const {
        return std::all_of(pools_.begin(), pools_.end(), [](const auto& pool) {
            return pool.used <= pool.capacity;
        });
    }

private:
    struct Group {
        size_t pool = 0;
        std::vector<size_t> prefix{0};
    };
    struct Pool {
        size_t capacity;
        size_t used;
    };
    std::vector<Group> groups_;
    std::vector<Pool> pools_;
};

struct PrefetchPins {
    ~PrefetchPins() { clear(); }
    void add(const std::shared_ptr<IBlockPool>& pool, BlockIdxType block) {
        slots.emplace_back(pool, block);
        block_tree_cache_detail::ScopeRollback rollback([&] { slots.pop_back(); });
        pool->incTreeRef(block, BlockTreeRefType::LOAD);
        rollback.dismiss();
    }
    void addDevice(const std::shared_ptr<IBlockPool>& pool, BlockIdxType block) {
        requests.emplace_back(pool, BlockIdList{block});
        block_tree_cache_detail::ScopeRollback rollback([&] { requests.pop_back(); });
        pool->referenceRequestBlocks(requests.back().second);
        rollback.dismiss();
    }
    void swap(PrefetchPins& other) {
        slots.swap(other.slots);
        requests.swap(other.requests);
    }
    void clear() {
        for (const auto& [pool, block] : slots) {
            pool->decTreeRef(block, BlockTreeRefType::LOAD);
        }
        slots.clear();
        for (const auto& [pool, blocks] : requests) {
            pool->unreferenceRequestBlocks(blocks);
        }
        requests.clear();
    }
    std::vector<std::pair<std::shared_ptr<IBlockPool>, BlockIdxType>> slots;
    std::vector<std::pair<std::shared_ptr<IBlockPool>, BlockIdList>> requests;
};

struct AdmissionPage {
    uint64_t node_identity;
    uint64_t generation;
    Tier tier;
    BlockIdList blocks;
    BlockIdList joined_targets;
};
using AdmissionPages = std::vector<std::unordered_map<size_t, AdmissionPage>>;

bool pinAdmissionPage(TreeNode* node, size_t row, const GroupSet& group,
                      const LoadJoinRegistry& joins, AdmissionPages& pages, PrefetchPins& pins) {
    const auto id = group.groupSetId();
    auto& group_pages = pages[id];
    if (group_pages.count(row)) {
        return false;
    }
    const auto& resource = node->group_set_resources[id];
    const auto tier = resource.getTopTier();
    BTC_CHECK(resource.isMatchUsable() && (tier != Tier::DEVICE || resource.hasCompleteDeviceValue()));
    auto it = group_pages.emplace(
        row, AdmissionPage{node->identity, resource.generation, tier, resource.getBlocks(tier), {}}).first;
    auto& page = it->second;
    const auto& pools = group.pools();
    if (tier == Tier::DEVICE) {
        for (size_t member = 0; member < pools.device.size(); ++member) {
            pins.addDevice(pools.device[member], page.blocks[member]);
        }
    } else {
        pins.add(tier == Tier::HOST ? pools.host : pools.disk, page.blocks.front());
        if (resource.transfer_state == GroupSetTransferState::LOADING) {
            if (const auto* targets = joins.targetBlocks(node, id, resource.generation)) {
                page.joined_targets = *targets;
                for (size_t member = 0; member < pools.device.size(); ++member) {
                    pins.addDevice(pools.device[member], (*targets)[member]);
                }
            }
        }
    }
    return true;
}

struct HostPrefetch {
    std::shared_ptr<const BlockTreeKeys> keys;
    std::vector<GroupSetPtr> groups;
    std::vector<std::vector<GroupSetResource>> resources;
    std::vector<std::pair<size_t, size_t>> local_positions;
    std::vector<std::pair<size_t, size_t>> destinations;

    HostPrefetch(std::shared_ptr<const BlockTreeKeys> keys, const std::vector<GroupSetPtr>& groups):
        keys(std::move(keys)), groups(groups), resources(this->keys->size(), std::vector<GroupSetResource>(groups.size())) {}

    ~HostPrefetch() {
        for (const auto& row : resources) {
            for (size_t group = 0; group < row.size(); ++group) {
                if (!isNullBlockIdx(row[group].host_block)) {
                    groups[group]->releaseSingleBlock(Tier::HOST, row[group].host_block, BlockTreeRefType::LOAD);
                }
            }
        }
    }
};

}  // namespace

struct LocalMatchAdmission::State {
    std::weak_ptr<LoadContextCoordinator> owner;
    BlockTreeKeys keys;
    std::vector<bool> selected_groups;
    std::function<bool(size_t)> valid_prefix;
    std::optional<int64_t> request_id;
    BlockTreeMatchResult local;
    AdmissionPages pages;
    PrefetchPins pins;
    bool consumed{false};
};

LocalMatchAdmission::LocalMatchAdmission(): state_(std::make_unique<State>()) {}
LocalMatchAdmission::~LocalMatchAdmission() = default;
size_t LocalMatchAdmission::readyRows() const { return state_->local.matched_device_blocks; }
size_t LocalMatchAdmission::matchedRows() const { return state_->local.matched_recoverable_blocks; }

class HostPrefetchMatch {
public:
    HostPrefetchMatch(const std::shared_ptr<LoadContextCoordinator>& owner, size_t end, size_t groups):
        owner(owner), end(end), pages(groups) {}
    const std::weak_ptr<LoadContextCoordinator> owner;
    const size_t end;
    AdmissionPages pages;
    PrefetchPins pins;
};

namespace {

std::shared_ptr<HostPrefetchMatch> captureWarmRows(
    const std::shared_ptr<LoadContextCoordinator>& owner, size_t end,
    const std::vector<GroupSetPtr>& groups, const LoadJoinRegistry& joins,
    const std::vector<TreeNode*>& path, const std::vector<std::pair<size_t, size_t>>& local,
    const std::vector<std::pair<size_t, size_t>>& warmed) {
    auto result = std::make_shared<HostPrefetchMatch>(owner, end, groups.size());
    const auto capture = [&](const auto& positions) {
        for (const auto& [row, group] : positions) {
            if (row >= path.size() || !path[row]->group_set_resources[group].isMatchUsable()) {
                return false;
            }
            pinAdmissionPage(path[row], row, *groups[group], joins, result->pages, result->pins);
        }
        return true;
    };
    return capture(local) && capture(warmed) ? std::move(result) : nullptr;
}

}  // namespace

struct BlockTreeLoader::RemotePrefetch {
    enum class Phase { QUERY, CAPACITY, READING, DONE };

    bool live() const {
        return TransferTask::Clock::now() < deadline && still_needed();
    }

    void finish(bool warmed, std::shared_ptr<HostPrefetchMatch> residency = {}) {
        if (phase == Phase::DONE) {
            return;
        }
        phase = Phase::DONE;
        pins.clear();
        on_ready(warmed, std::move(residency));
    }

    Phase phase{Phase::QUERY};
    StorageRequest request;
    std::shared_ptr<StorageBackendMatchMeta> meta;
    size_t matched{0};
    size_t max_load_pages{0};
    TransferTask::Clock::time_point deadline;
    std::vector<bool> selected;
    std::function<bool(size_t)> valid;
    PublishHost publish;
    std::function<bool()> still_needed;
    RemotePrefetchDone on_ready;
    PrefetchPins pins;
    // The last free count for each deficient pool. New physical capacity can
    // expose another FULL leaf without issuing the same failed write again.
    struct CapacityWait {
        size_t free;
        bool retried_after_contention{false};
    };
    std::map<const HostBlockPool*, CapacityWait> waiting;
};

BlockTreeLoader::BlockTreeLoader(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 BlockTransferDispatcher*        transfer_dispatcher,
                                 BlockTreeTaskPool*              task_pool,
                                 CacheObserver&  metrics_reporter,
                                 std::mutex&                     mutex,
                                 int                             disk_timeout_ms,
                                 int                             host_timeout_ms,
                                 bool                            enable_device_cache,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled):
    tree_(tree),
    evictor_(evictor),
    transfer_dispatcher_(transfer_dispatcher),
    task_pool_(task_pool),
    metrics_reporter_(metrics_reporter),
    mutex_(mutex),
    enable_device_cache_(enable_device_cache),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)),
    load_task_runner_(tree_->groupSets(), host_timeout_ms, disk_timeout_ms),
    load_join_registry_(tree),
    load_context_coordinator_(std::make_shared<LoadContextCoordinator>(
        [this](const std::shared_ptr<LoadAsyncContext>& context) { return commitLoad(context); },
        [this](LoadAsyncContext& context) {
            CacheLockGuard lock(mutex_, metrics_reporter_, "load_abort");
            abortLoadLocked(
                context.loadDescs(), context.joinedLoads(), 0, context.contextId(), /*release_transferred_refs=*/false);
        },
        [task_pool](std::function<void()> cleanup) { return task_pool->submitCompletion(std::move(cleanup)); })) {}

BlockTreeMatchResult BlockTreeLoader::matchLocked(const CacheKeysType&               cache_keys,
                                                  const std::function<bool(size_t)>& valid_prefix,
                                                  bool                               enable_storage_backend,
                                                  std::optional<int64_t>             request_id,
                                                  bool                               skip_linear) {
    return matchLockedImpl(cache_keys, valid_prefix, enable_storage_backend, request_id, skip_linear, {},
                           std::numeric_limits<size_t>::max(), false);
}

BlockTreeMatchResult BlockTreeLoader::matchLocked(const BlockTreeKeys& cache_keys,
                                                  const std::function<bool(size_t)>& valid_prefix,
                                                  bool enable_storage_backend,
                                                  std::optional<int64_t> request_id,
                                                  bool skip_linear,
                                                  const std::vector<bool>& selected_groups,
                                                  size_t max_load_pages,
                                                  bool stage_disk_through_host) {
    return matchLockedImpl(cache_keys, valid_prefix, enable_storage_backend, request_id,
                           skip_linear, selected_groups, max_load_pages, stage_disk_through_host);
}

std::shared_ptr<LocalMatchAdmission> BlockTreeLoader::prepareHostAdmissionLocked(
    const BlockTreeKeys& keys, const std::function<bool(size_t)>& valid_prefix,
    std::optional<int64_t> request_id, const std::vector<bool>& selected_groups,
    size_t max_load_pages, bool stage_disk_through_host, size_t max_load_rows) {
    auto admission = std::shared_ptr<LocalMatchAdmission>(new LocalMatchAdmission);
    auto& state = *admission->state_;
    state.owner = load_context_coordinator_;
    state.keys = keys;
    state.selected_groups = selected_groups;
    state.valid_prefix = valid_prefix;
    state.request_id = request_id;
    state.pages.resize(tree_->groupSets().size());
    state.local = matchLockedImpl(keys, valid_prefix, false, request_id, false, selected_groups,
                                  max_load_pages, stage_disk_through_host, admission.get(), max_load_rows);
    return admission;
}

std::optional<BlockTreeMatchResult> BlockTreeLoader::completeHostAdmissionLocked(
    LocalMatchAdmission& admission, const std::shared_ptr<HostPrefetchMatch>& warmed) {
    auto& captured = *admission.state_;
    if (captured.owner.lock() != load_context_coordinator_
        || (warmed && warmed->owner.lock() != load_context_coordinator_)) {
        throw std::invalid_argument("Cache admission belongs to a different BlockTreeCache");
    }
    if (captured.consumed) {
        return std::nullopt;
    }
    captured.consumed = true;
    block_tree_cache_detail::ScopeRollback release([&] { captured.pins.clear(); });
    auto result = std::move(captured.local);
    const auto& groups = tree_->groupSets();
    const size_t ready = result.matched_device_blocks;
    const size_t local_end = result.matched_recoverable_blocks;
    const size_t warm_end = warmed ? warmed->end : 0;
    // This lookup only resolves the captured physical sources. Endpoints and
    // required group rows were selected by the initial match or remote lookup.
    const auto path = tree_->findNode(captured.keys);
    const auto source_matches = [&](const GroupSet& group, size_t row, const AdmissionPage& page) {
        const auto group_id = group.groupSetId();
        if (row >= path.size() || path[row]->identity != page.node_identity) {
            return false;
        }
        const auto& resource = path[row]->group_set_resources[group_id];
        if (!resource.isMatchUsable() || resource.generation != page.generation) {
            return false;
        }
        if (resource.hasCompleteDeviceValue()) {
            const auto& device = resource.device_blocks;
            // A concurrent successful load can settle before a waiter claims
            // its plan. Only that exact JOIN proves the Device value.
            const auto& expected = page.tier == Tier::DEVICE ? page.blocks : page.joined_targets;
            return device == expected && group.canPublishDevice(device);
        }
        if (resource.getTopTier() != page.tier || resource.getBlocks(page.tier) != page.blocks) {
            return false;
        }
        if (resource.transfer_state == GroupSetTransferState::LOADING) {
            const auto* target = load_join_registry_.targetBlocks(path[row], group_id, resource.generation);
            return target && (page.joined_targets.empty() || *target == page.joined_targets)
                   && group.canPublishDevice(*target);
        }
        return true;
    };

    struct SelectedRow {
        size_t group;
        size_t row;
        bool added;
    };
    std::vector<SelectedRow> selected;
    const auto select = [&](size_t end, bool include_warm) {
        selected.clear();
        for (const auto& group : groups) {
            if (!participates(*group, false, captured.selected_groups)) {
                continue;
            }
            const auto id = group->groupSetId();
            const auto stride = group->pageTokens() / tree_->keyTokens();
            const auto& old_pages = captured.pages[id];
            const auto& new_pages = include_warm ? warmed->pages[id] : old_pages;
            const auto check_range = [&](size_t first, size_t last, bool ready_only) {
                for (size_t row = first; row < last; row += stride) {
                    auto old = old_pages.find(row);
                    const bool added = old == old_pages.end();
                    auto page = added ? new_pages.find(row) : old;
                    if (page == (added ? new_pages.end() : old_pages.end())
                        || !source_matches(*group, row, page->second)) {
                        return false;
                    }
                    if (!ready_only && row >= ready) {
                        selected.push_back({id, row, added});
                    }
                }
                return true;
            };
            if (!check_range(firstPageRow(reuseBegin(*group, ready, stride), stride), ready, true)
                || !check_range(firstPageRow(reuseBegin(*group, end, stride), stride), end, false)) {
                return false;
            }
        }
        return true;
    };

    const bool use_warm = warm_end > local_end && warm_end <= captured.keys.size()
                         && (!captured.valid_prefix || captured.valid_prefix(warm_end)) && select(warm_end, true);
    if (!use_warm && !select(local_end, false)) {
        return std::nullopt;
    }
    const auto end = use_warm ? warm_end : local_end;
    result.matched_recoverable_blocks = end;
    if (std::none_of(selected.begin(), selected.end(), [&](const auto& entry) {
            return path[entry.row]->group_set_resources[entry.group].getTopTier() == Tier::DISK;
        })) {
        result.matched_device_host_blocks = end;
    }
    result.matched_device_host_blocks = std::min(result.matched_device_host_blocks, end);

    // Acquire the allocator and transfer references before releasing any
    // captured pins, under the same cache lock.
    block_tree_cache_detail::ScopeRollback release_device([&] {
        for (const auto& resource : result.matched_device_resources) {
            groups[resource.group_set_id]->unreferenceBlocks(resource);
        }
    });
    result.matched_device_resources.reserve(groups.size());
    for (const auto& group : groups) {
        if (!participates(*group, false, captured.selected_groups)) {
            continue;
        }
        const auto id = group->groupSetId();
        const auto stride = group->pageTokens() / tree_->keyTokens();
        MultiNodeResource device{id, Tier::DEVICE};
        for (size_t row = firstPageRow(reuseBegin(*group, ready, stride), stride); row < ready; row += stride) {
            device.node_blocks.emplace_back(nullptr, path[row]->group_set_resources[id].device_blocks);
        }
        if (!device.node_blocks.empty()) {
            group->referenceBlocks(device);
            result.matched_device_resources.push_back(std::move(device));
        }
    }
    if (!selected.empty()) {
        std::vector<TransferDescriptor> descriptors;
        std::vector<bool> joins;
        descriptors.reserve(selected.size());
        joins.reserve(selected.size());
        block_tree_cache_detail::ScopeRollback release_descriptors([&] {
            abortLoadLocked(descriptors, joins, 0, 0, /*release_transferred_refs=*/true);
        });
        const int64_t access_time = cacheNowUs();
        const uint64_t access_seq = evictor_.nextAccessSeq();
        for (const auto& entry : selected) {
            appendLoadDescriptor(path[entry.row], entry.group, entry.row, descriptors, joins);
            if (entry.added) {
                collectReuseTimeMetrics(path[entry.row], entry.group, access_seq, access_time,
                                        result.reuse_time_metrics);
            }
        }
        StorageRequest request;
        request.request_id = captured.request_id;
        release_descriptors.dismiss();
        result.async_context = createLoadContextLocked(descriptors, joins, end, std::move(request), false);
        if (!result.async_context) {
            return std::nullopt;
        }
        result.async_context->setLocalMatchLengths(ready, result.matched_device_host_blocks);
    }
    release_device.dismiss();
    return result;
}

template<class Keys>
BlockTreeMatchResult BlockTreeLoader::matchLockedImpl(const Keys& cache_keys,
                                                      const std::function<bool(size_t)>& valid_prefix,
                                                      bool enable_storage_backend,
                                                      std::optional<int64_t> request_id,
                                                      bool skip_linear,
                                                      const std::vector<bool>& selected_groups,
                                                      size_t max_load_pages,
                                                      bool stage_disk_through_host,
                                                      LocalMatchAdmission* admission,
                                                      size_t max_load_rows) {
    if (cache_keys.empty()) {
        BTC_LOG_DEBUG("empty cache_keys, returning empty result");
        return {};
    }

    std::vector<TreeNode*> path = tree_->findNode(cache_keys);
    BlockTreeMatchResult result =
        createMatchResult(path, cache_keys, valid_prefix, enable_storage_backend, request_id,
                          skip_linear, selected_groups, max_load_pages, stage_disk_through_host, admission,
                          max_load_rows);
    BTC_LOG_DEBUG("matched %zu device blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_device_blocks,
                      cache_keys.size(),
                      tree_->size());
    return result;
}

bool BlockTreeLoader::validMatch(std::vector<TreeNode*>&            path,
                                 std::vector<bool>&                 candidate_valid,
                                 const std::function<bool(size_t)>& valid_prefix,
                                 bool                               skip_linear,
                                 const std::vector<bool>&           selected_groups) const {
    size_t valid_block_count = 0;
    candidate_valid.reserve(path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(tree_->groupSets().size());
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    bool has_considered_group_set = false;
    for (size_t i = 0; i < path.size(); ++i) {
        TreeNode* node             = path[i];
        bool      all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
            const auto& group = *tree_->groupSets()[group_set_id];
            if (!participates(group, skip_linear, selected_groups)) {
                continue;
            }
            has_considered_group_set = true;
            if ((i + 1) % (group.pageTokens() / tree_->keyTokens()) != 0
                || !match_validators[group_set_id]->validate(node->group_set_resources[group_set_id])) {
                all_groups_valid = false;
            }
        }
        all_groups_valid = all_groups_valid && (!valid_prefix || valid_prefix(i + 1));
        if (all_groups_valid) {
            valid_block_count = i + 1;
        }
        candidate_valid.push_back(all_groups_valid);
    }

    if (!has_considered_group_set) {
        path.clear();
        candidate_valid.clear();
        return false;
    }
    path.resize(valid_block_count);
    candidate_valid.resize(valid_block_count);
    return valid_block_count > 0;
}

BlockIdList BlockTreeLoader::matchedBlocksForGroup(std::string_view                      group_tag,
                                                        const std::vector<MultiNodeResource>& matched_resources) const {
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
    if (location == nullptr) {
        return {};
    }
    for (const MultiNodeResource& resource : matched_resources) {
        if (resource.group_set_id != location->group_set_id) {
            continue;
        }
        BlockIdList blocks;
        blocks.reserve(resource.node_blocks.size());
        for (const auto& [_, node_blocks] : resource.node_blocks) {
            blocks.push_back(node_blocks[location->member_group_id]);
        }
        return blocks;
    }
    return {};
}

void BlockTreeLoader::collectReuseTimeMetrics(TreeNode*                                    node,
                                              size_t                                       group_set_id,
                                              uint64_t                                     access_seq,
                                              int64_t                                      access_time_us,
                                              std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list) {
    GroupSetResource& resource = node->group_set_resources[group_set_id];
    const Tier        tier     = resource.getTopTier();
    if (tier == Tier::NONE) {
        return;
    }
    CandidateMeta&                 meta             = resource.candidate_meta;
    const bool                     first_hit        = meta.tier_last_hit_time_us == 0;
    const int64_t                  previous_time_us = first_hit ? meta.tier_enter_time_us : meta.tier_last_hit_time_us;
    BlockTreeCacheReuseTimeMetrics metrics_data;
    metrics_data.tier                        = tier;
    metrics_data.group_type                  = tree_->groupSets()[group_set_id]->groupType();
    metrics_data.reuse_block_hit_idletime_ms = std::max<int64_t>(0, access_time_us - previous_time_us) / 1000;
    if (first_hit) {
        metrics_data.reuse_block_first_hit_idletime_ms = metrics_data.reuse_block_hit_idletime_ms;
    }
    metrics_list.push_back(metrics_data);

    meta.tier_last_hit_time_us = access_time_us;
    meta.last_access_time_us   = access_time_us;
    meta.last_access_seq       = access_seq;
    ++meta.hit_count;
    evictor_.onMatched(node, group_set_id);
}

template<class Keys>
BlockTreeMatchResult BlockTreeLoader::createMatchResult(std::vector<TreeNode*>&            path,
                                                        const Keys&                       cache_keys,
                                                        const std::function<bool(size_t)>& valid_prefix,
                                                        bool                               enable_storage_backend,
                                                        std::optional<int64_t>             request_id,
                                                        bool                               skip_linear,
                                                        const std::vector<bool>&           selected_groups,
                                                        size_t                             max_load_pages,
                                                        bool                               stage_disk_through_host,
                                                        LocalMatchAdmission*               admission,
                                                        size_t                             max_load_rows) {
    BlockTreeMatchResult result;
    const bool           has_considered_group_set =
        std::any_of(tree_->groupSets().begin(), tree_->groupSets().end(), [&](const GroupSetPtr& group_set) {
            return participates(*group_set, skip_linear, selected_groups);
        });
    // A tail-only request on an all-LINEAR topology has no reusable prefix.
    // Return an empty match instead of treating every key as FULL-reusable.
    if (!has_considered_group_set) {
        path.clear();
        return result;
    }
    std::vector<bool>    candidate_valid;
    if (!path.empty() && !validMatch(path, candidate_valid, valid_prefix, skip_linear, selected_groups)
        && !storage_backend_) {
        return result;
    }
    const int64_t  access_time_us = cacheNowUs();
    const uint64_t access_seq     = evictor_.nextAccessSeq();
    result.reuse_time_metrics.reserve(path.size() * tree_->groupSets().size());

    std::vector<size_t> device_runs(tree_->groupSets().size(), 0);
    std::vector<size_t> device_host_runs(tree_->groupSets().size(), 0);
    const bool bounded = max_load_pages != std::numeric_limits<size_t>::max();
    const bool bounded_rows = max_load_rows != std::numeric_limits<size_t>::max();
    std::vector<size_t> host_ends;
    std::vector<size_t> recoverable_ends;
    std::vector<std::vector<size_t>> load_counts;
    if (bounded) {
        load_counts.resize(tree_->groupSets().size(), {0});
    }
    std::optional<StagedHostBudget> host_budget;
    if (stage_disk_through_host) {
        host_budget.emplace(tree_->groupSets());
    }
    std::vector<size_t> load_begins;
    std::vector<size_t> strides;
    load_begins.reserve(tree_->groupSets().size());
    strides.reserve(tree_->groupSets().size());
    for (const auto& group_set : tree_->groupSets()) {
        const auto stride = group_set->pageTokens() / tree_->keyTokens();
        strides.push_back(stride);
        load_begins.push_back(reuseBegin(*group_set, path.size(), stride));
    }
    for (size_t i = 0; i < path.size(); ++i) {
        const size_t candidate_count        = i + 1;
        bool         all_groups_device      = true;
        bool         all_groups_device_host = true;
        size_t       loads                  = 0;
        if (host_budget) {
            host_budget->beginCandidate();
        }
        for (size_t group_id = 0; group_id < tree_->groupSets().size(); ++group_id) {
            // PD decode handoff: LINEAR groups never gate prefix matching when
            // the caller supplies the linear state out of band.
            const auto& group = *tree_->groupSets()[group_id];
            if (!participates(group, skip_linear, selected_groups) || candidate_count % strides[group_id] != 0) {
                continue;
            }
            const auto& resource       = path[i]->group_set_resources[group_id];
            const bool  device_ready   = resource.hasCompleteDeviceValue();
            device_runs[group_id]      = device_ready ? device_runs[group_id] + 1 : 0;
            const bool host_will_load  = i >= load_begins[group_id] && resource.hasTier(Tier::HOST);
            device_host_runs[group_id] = device_ready || host_will_load ? device_host_runs[group_id] + 1 : 0;
            const size_t pages = candidate_count / strides[group_id];
            const size_t required = std::min(group.computeReuseBlockCount(pages), pages);
            if (bounded) {
                auto& counts = load_counts[group_id];
                counts.push_back(counts.back() + !device_ready);
                loads += counts.back() - counts[pages - required];
            }
            if (host_budget) {
                const bool needs_host = !device_ready && (resource.hasTier(Tier::HOST) || resource.hasTier(Tier::DISK));
                host_budget->append(group_id, needs_host, required);
            }
            all_groups_device      = all_groups_device && device_runs[group_id] >= required;
            all_groups_device_host = all_groups_device_host && device_host_runs[group_id] >= required;
        }
        if (candidate_valid[i]) {
            if (all_groups_device) {
                result.matched_device_blocks = candidate_count;
            }
            if (all_groups_device_host) {
                result.matched_device_host_blocks = candidate_count;
                if (bounded_rows) {
                    host_ends.push_back(candidate_count);
                }
            }
            if (loads <= max_load_pages && (!host_budget || host_budget->fits())) {
                result.matched_recoverable_blocks = candidate_count;
                if (bounded_rows) {
                    recoverable_ends.push_back(candidate_count);
                }
            }
        }
    }
    if (bounded_rows) {
        const size_t ready = result.matched_device_blocks;
        const size_t limit = ready + std::min(max_load_rows, path.size() - ready);
        const auto allowed = [limit](const auto& endpoints) {
            const auto upper = std::upper_bound(endpoints.begin(), endpoints.end(), limit);
            return upper == endpoints.begin() ? size_t{0} : *std::prev(upper);
        };
        result.matched_device_host_blocks = allowed(host_ends);
        result.matched_recoverable_blocks = allowed(recoverable_ends);
    }
    if (path.size() > result.matched_recoverable_blocks) {
        path.resize(result.matched_recoverable_blocks);
        for (size_t group = 0; group < strides.size(); ++group) {
            load_begins[group] = reuseBegin(*tree_->groupSets()[group], path.size(), strides[group]);
        }
    }

    if (admission) {
        auto& captured = *admission->state_;
        for (const auto& group : tree_->groupSets()) {
            const auto id = group->groupSetId();
            if (!participates(*group, skip_linear, selected_groups)) {
                continue;
            }
            const auto stride = strides[id];
            const auto pin_range = [&](size_t begin, size_t end) {
                for (size_t row = firstPageRow(begin, stride); row < end; row += stride) {
                    if (pinAdmissionPage(path[row], row, *group, load_join_registry_, captured.pages, captured.pins)) {
                        collectReuseTimeMetrics(path[row], id, access_seq, access_time_us, result.reuse_time_metrics);
                    }
                }
            };
            const auto ready = result.matched_device_blocks;
            pin_range(reuseBegin(*group, ready, stride), ready);
            pin_range(load_begins[id], path.size());
        }
        return result;
    }

    std::vector<TransferDescriptor> pending_load_descs;
    std::vector<bool>               joined_loads;
    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
        if (!participates(*group_set, skip_linear, selected_groups)) {
            continue;
        }
        const auto stride = strides[group_set_id];
        const auto reuse_rows = [&](size_t rows) {
            const auto pages = rows / stride;
            return std::min(group_set->computeReuseBlockCount(pages), pages) * stride;
        };
        const auto ready_reuse_rows = reuse_rows(result.matched_device_blocks);
        MultiNodeResource matched_device_resource{group_set_id, Tier::DEVICE};
        for (size_t i = firstPageRow(result.matched_device_blocks - ready_reuse_rows, stride);
             i < result.matched_device_blocks; i += stride) {
            collectReuseTimeMetrics(path[i], group_set_id, access_seq, access_time_us, result.reuse_time_metrics);
            const GroupSetResource& resource = path[i]->group_set_resources[group_set_id];
            matched_device_resource.node_blocks.emplace_back(nullptr, resource.getBlocks(Tier::DEVICE));
        }
        if (!matched_device_resource.node_blocks.empty()) {
            group_set->referenceBlocks(matched_device_resource);
            result.matched_device_resources.push_back(std::move(matched_device_resource));
        }

        const size_t host_prefix      = std::min(result.matched_device_host_blocks, path.size());
        const auto   append_range     = [&](size_t begin, size_t end, bool device_only = false) {
            for (size_t i = firstPageRow(std::max(begin, result.matched_device_blocks), stride); i < end; i += stride) {
                if (device_only && !path[i]->group_set_resources[group_set_id].hasCompleteDeviceValue()) {
                    continue;
                }
                appendLoadDescriptor(path[i], group_set_id, i, pending_load_descs, joined_loads);
                collectReuseTimeMetrics(path[i], group_set_id, access_seq, access_time_us, result.reuse_time_metrics);
            }
        };
        // Outside the normal load window, fallback only pins existing device values.
        append_range(host_prefix - reuse_rows(host_prefix), std::min(host_prefix, load_begins[group_set_id]),
                     std::is_same_v<Keys, BlockTreeKeys>);
        append_range(load_begins[group_set_id], path.size());
    }
    StorageRequest storage_request;
    if (max_load_pages != 0 && enable_storage_backend && storage_backend_ && path.size() < cache_keys.size()) {
        storage_request = makeStorageRequest(cache_keys, path.size(), skip_linear, selected_groups);
    }
    const bool use_storage     = !storage_request.empty();
    storage_request.request_id = request_id;
    if (!pending_load_descs.empty() || use_storage) {
        result.async_context = createLoadContextLocked(
            pending_load_descs, joined_loads, path.size(), std::move(storage_request), use_storage);
    }
    if (result.async_context) {
        result.async_context->setLocalMatchLengths(result.matched_device_blocks,
                                                   std::min(result.matched_device_host_blocks, path.size()));
        result.async_context->setValidPrefix(valid_prefix);
    }
    return result;
}

void BlockTreeLoader::appendLoadDescriptor(TreeNode* node, size_t group, size_t row,
                                           std::vector<TransferDescriptor>& descriptors, std::vector<bool>& joins) {
    const auto& group_set = tree_->groupSets()[group];
    auto& resource = node->group_set_resources[group];
    const auto tier = resource.getTopTier();
    auto* source = tier == Tier::DEVICE ? nullptr : node;
    const bool owner = source && resource.transfer_state == GroupSetTransferState::IDLE;
    TransferDescriptor desc{source, group, row, tier, Tier::DEVICE, resource.getBlocks(tier), owner};
    const bool joined = source && resource.transfer_state == GroupSetTransferState::LOADING;
    if (!joined) {
        const MultiNodeResource resource_ref{group, tier, {{source, desc.source_blocks}}};
        if (tier == Tier::DEVICE) {
            group_set->referenceBlocks(resource_ref);
        } else {
            group_set->referenceBlocks(resource_ref, BlockTreeRefType::LOAD);
            if (owner) {
                resource.transfer_state = GroupSetTransferState::LOAD_PENDING;
                evictor_.suspendCandidate(node, group, tier);
            } else {
                tree_->referenceNodeForNonOwnerTransfer(source);
            }
        }
    }
    descriptors.emplace_back(std::move(desc));
    joins.push_back(joined);
}

std::shared_ptr<LoadAsyncContext> BlockTreeLoader::createLoadContextLocked(
    const std::vector<TransferDescriptor>& descriptors, const std::vector<bool>& joins,
    size_t matched_rows, StorageRequest request, bool use_storage) {
    std::shared_ptr<LoadAsyncContext> context;
    block_tree_cache_detail::ScopeRollback rollback([&] {
        if (context) {
            abortLoadLocked(context->loadDescs(), context->joinedLoads(), 0, context->contextId(),
                            /*release_transferred_refs=*/true);
        } else {
            abortLoadLocked(descriptors, joins, 0, 0, /*release_transferred_refs=*/true);
        }
    });
    context = load_context_coordinator_->create(
        descriptors, joins, matched_rows, use_storage ? storage_backend_ : nullptr, std::move(request));
    if (!context || !load_join_registry_.join(context)) {
        return {};
    }
    const auto count = static_cast<size_t>(std::count(joins.begin(), joins.end(), true));
    if (count != 0) {
        context->startJoinWait(cacheNowUs());
        metrics_reporter_.reportLoadJoin(count);
    }
    if (!load_context_coordinator_->registerContext(context)) {
        return {};
    }
    rollback.dismiss();
    return context;
}

template<class Keys>
StorageRequest BlockTreeLoader::makeStorageRequest(const Keys& cache_keys,
                                                   size_t               local_matched_blocks_num,
                                                   bool                 skip_linear,
                                                   const std::vector<bool>& selected_groups) const {
    StorageRequest request;
    if constexpr (std::is_same_v<Keys, CacheKeysType>) {
        request.keys = std::make_shared<Keys>(cache_keys);
    } else {
        request.opaque_keys = std::make_shared<Keys>(cache_keys);
        request.positional_replicas = true;
    }
    request.local_matched_blocks_num = local_matched_blocks_num;
    request.handles.resize(cache_keys.size());
    for (size_t row = 0; row < request.handles.size(); ++row) {
        auto& key_handles = request.handles[row];
        for (const auto& group_set : tree_->groupSets()) {
            if (!participates(*group_set, skip_linear, selected_groups)
                || (row + 1) % (group_set->pageTokens() / tree_->keyTokens()) != 0) {
                continue;
            }
            for (const auto& tag : group_set->groupTags()) {
                key_handles.push_back({tag, NULL_BLOCK_IDX});
            }
        }
    }
    return request;
}

bool BlockTreeLoader::prefetchRemote(const BlockTreeKeys& cache_keys,
                                     size_t locally_matched_rows,
                                     std::function<bool(size_t)> valid_prefix,
                                     std::vector<bool> selected_groups,
                                     size_t max_load_pages,
                                     std::optional<int64_t> request_id,
                                     std::chrono::milliseconds timeout,
                                     PublishHost publish,
                                     std::function<bool()> still_needed,
                                     std::function<void(bool)> notice,
                                     RemotePrefetchDone on_ready) {
    if (!storage_backend_ || !storage_backend_->remoteCacheEnabled() || locally_matched_rows >= cache_keys.size()) {
        return false;
    }
    for (const auto& group : tree_->groupSets()) {
        if (participates(*group, false, selected_groups) && !group->pools().host) {
            return false;
        }
    }
    auto attempt = std::make_shared<RemotePrefetch>();
    attempt->request = makeStorageRequest(cache_keys, locally_matched_rows, false, selected_groups);
    if (attempt->request.empty()) {
        return false;
    }
    attempt->request.request_id = request_id;
    attempt->request.prefetch_notice = std::move(notice);
    attempt->max_load_pages = max_load_pages;
    attempt->deadline = TransferTask::Clock::now() + timeout;
    attempt->selected = std::move(selected_groups);
    attempt->valid = std::move(valid_prefix);
    attempt->publish = std::move(publish);
    attempt->still_needed = std::move(still_needed);
    attempt->on_ready = std::move(on_ready);
    {
        std::lock_guard lock(mutex_);
        remote_prefetches_.push_back(attempt);
    }
    storage_backend_->match(attempt->request,
        [this, attempt](size_t matched, std::shared_ptr<StorageBackendMatchMeta> meta, bool success) {
            std::lock_guard lock(mutex_);
            if (attempt->phase != RemotePrefetch::Phase::QUERY) {
                return;
            }
            if (!success || !meta || !attempt->live()) {
                attempt->finish(false);
                return;
            }
            attempt->meta = std::move(meta);
            attempt->matched = std::min(matched, attempt->request.cacheKeyCount());
            attempt->phase = RemotePrefetch::Phase::CAPACITY;
            advanceRemotePrefetchLocked(attempt);
        });
    return true;
}

void BlockTreeLoader::resumeRemotePrefetchesLocked(bool capacity_changed) {
    for (const auto& attempt : remote_prefetches_) {
        if (attempt->phase == RemotePrefetch::Phase::DONE) {
            continue;
        }
        if (!attempt->live()) {
            attempt->finish(false);
        } else if (capacity_changed && attempt->phase == RemotePrefetch::Phase::CAPACITY) {
            const bool can_progress = std::any_of(attempt->waiting.begin(), attempt->waiting.end(),
                [&](const auto& entry) {
                    return entry.first->freeBlocksNum() != entry.second.free || !evictor_.pendingHostReleases(*entry.first);
                });
            if (can_progress) {
                advanceRemotePrefetchLocked(attempt);
            }
        }
    }
    remote_prefetches_.erase(
        std::remove_if(remote_prefetches_.begin(), remote_prefetches_.end(), [](const auto& attempt) {
            return attempt->phase == RemotePrefetch::Phase::DONE;
        }),
        remote_prefetches_.end());
}

void BlockTreeLoader::failRemotePrefetchesLocked(const HostBlockPool& pool) {
    for (const auto& attempt : remote_prefetches_) {
        if (attempt->phase == RemotePrefetch::Phase::CAPACITY && attempt->waiting.count(&pool)) {
            attempt->finish(false);
        }
    }
    resumeRemotePrefetchesLocked(false);
}

void BlockTreeLoader::advanceRemotePrefetchLocked(const std::shared_ptr<RemotePrefetch>& attempt) {
    // No destination slots are reserved until every physical pool can satisfy
    // the chosen native prefix at once.
    block_tree_cache_detail::ScopeRollback fail([&] { attempt->finish(false); });
    const auto& query = attempt->request;
    if (attempt->matched <= query.local_matched_blocks_num) {
        return;
    }
    const auto& groups = tree_->groupSets();
    const auto path = tree_->findNode(*query.opaque_keys);
    struct Counts {
        size_t host{0};
        size_t missing{0};
        size_t load{0};
    };
    struct Page {
        Tier tier;
        std::string replica;
    };
    struct GroupPlan {
        GroupSetPtr group;
        size_t stride;
        std::vector<Page> pages;
        std::vector<Counts> prefix{{}};
    };
    std::vector<GroupPlan> plans;
    std::map<const HostBlockPool*, size_t> demands;
    for (const auto& group : groups) {
        if (!participates(*group, false, attempt->selected)) {
            continue;
        }
        const size_t stride = group->pageTokens() / tree_->keyTokens();
        auto& plan = plans.emplace_back(GroupPlan{group, stride, {}, {{}}});
        demands.try_emplace(group->pools().host.get(), 0);
        for (size_t row = stride - 1; row < attempt->matched; row += stride) {
            const auto id = group->groupSetId();
            const auto* resource = row < path.size() ? &path[row]->group_set_resources[id] : nullptr;
            Tier tier = Tier::NONE;
            std::string replica;
            if (resource && resource->isMatchUsable()) {
                tier = resource->hasCompleteDeviceValue() ? Tier::DEVICE
                    : resource->hasTier(Tier::HOST) ? Tier::HOST : Tier::DISK;
            } else {
                replica = attempt->meta->replicaDescriptor(row, group->groupTags());
                if (!replica.empty()) {
                    tier = Tier::REMOTE;
                }
            }
            auto count = plan.prefix.back();
            count.host += tier == Tier::HOST || tier == Tier::REMOTE;
            count.missing += tier == Tier::NONE;
            count.load += tier != Tier::DEVICE;
            plan.pages.push_back({tier, std::move(replica)});
            plan.prefix.push_back(count);
        }
    }
    const auto required = [](const GroupPlan& plan, size_t n) {
        return std::min(n, plan.group->computeReuseBlockCount(n));
    };
    size_t end = attempt->matched;
    for (; end > query.local_matched_blocks_num; --end) {
        if ((attempt->valid && !attempt->valid(end))
            || std::any_of(plans.begin(), plans.end(), [&](const auto& plan) { return end % plan.stride; })) {
            continue;
        }
        for (auto& [pool, count] : demands) {
            count = 0;
        }
        size_t loads = 0;
        bool complete = true;
        for (const auto& plan : plans) {
            const auto n = end / plan.stride;
            const auto& last = plan.prefix[n];
            const auto& first = plan.prefix[n - required(plan, n)];
            complete &= last.missing == first.missing;
            loads += last.load - first.load;
            demands.at(plan.group->pools().host.get()) += last.host - first.host;
        }
        if (complete && loads <= attempt->max_load_pages
            && std::all_of(demands.begin(), demands.end(), [](const auto& entry) {
                return entry.second <= entry.first->totalBlocksNum();
            })) {
            break;
        }
    }
    if (end <= query.local_matched_blocks_num) {
        return;
    }
    struct Destination {
        GroupSetPtr group;
        size_t row;
        std::string replica;
    };
    std::vector<Destination> destinations;
    std::vector<std::pair<size_t, size_t>> local;
    std::set<std::pair<const IBlockPool*, BlockIdxType>> seen;
    std::map<const HostBlockPool*, size_t> protected_evictions;
    PrefetchPins pins;
    const auto pin = [&](const std::shared_ptr<IBlockPool>& pool, BlockIdxType block, bool device = false) {
        if (seen.emplace(pool.get(), block).second) {
            if (device) {
                pins.addDevice(pool, block);
            } else {
                pins.add(pool, block);
            }
        }
    };
    for (auto& [pool, count] : demands) {
        count = 0;
    }
    for (auto& plan : plans) {
        const auto& group = plan.group;
        const auto id = group->groupSetId();
        const auto& pools = group->pools();
        const auto n = end / plan.stride;
        for (auto page = n - required(plan, n); page < n; ++page) {
            const auto row = (page + 1) * plan.stride - 1;
            auto& source = plan.pages[page];
            if (source.tier == Tier::REMOTE) {
                destinations.push_back({group, row, std::move(source.replica)});
                ++demands.at(pools.host.get());
                continue;
            }
            auto& resource = path[row]->group_set_resources[id];
            local.emplace_back(row, id);
            if (source.tier == Tier::HOST) {
                pin(pools.host, resource.host_block);
                if (resource.transfer_state == GroupSetTransferState::EVICTING) {
                    ++protected_evictions[pools.host.get()];
                }
            } else if (source.tier == Tier::DISK) {
                pin(pools.disk, resource.disk_block);
            } else {
                for (size_t member = 0; member < pools.device.size(); ++member) {
                    pin(pools.device[member], resource.device_blocks[member], true);
                }
            }
            if (resource.transfer_state == GroupSetTransferState::EVICTING) {
                resource.candidate_meta.last_access_seq = evictor_.nextAccessSeq();
            }
        }
    }
    attempt->pins.swap(pins);
    pins.clear();
    if (destinations.empty()) {
        auto captured = captureWarmRows(load_context_coordinator_, end, groups, load_join_registry_, path, local, {});
        const bool warmed = static_cast<bool>(captured);
        attempt->finish(warmed, std::move(captured));
        fail.dismiss();
        return;
    }
    std::map<const HostBlockPool*, RemotePrefetch::CapacityWait> waiting;
    for (const auto& [pool, needed] : demands) {
        const auto free = pool->freeBlocksNum();
        if (free >= needed) {
            continue;
        }
        const auto old = attempt->waiting.find(pool);
        const auto protected_count = protected_evictions[pool];
        size_t scheduled = evictor_.pendingHostReleases(*pool);
        bool retried = old != attempt->waiting.end() && old->second.retried_after_contention;
        if (old == attempt->waiting.end() || free > old->second.free) {
            scheduled = evictor_.scheduleHostReleaseLocked(*pool, needed - free + protected_count);
            retried = false;
        } else if (scheduled <= protected_count && !retried) {
            // Another admission may have taken the newly freed slots. One
            // fresh pass can use other victims without looping on contention.
            scheduled = evictor_.scheduleHostReleaseLocked(*pool, needed - free + protected_count);
            retried = true;
        }
        if (scheduled <= protected_count) {
            return;
        }
        waiting.emplace(pool, RemotePrefetch::CapacityWait{free, retried});
    }
    attempt->waiting.swap(waiting);
    if (!attempt->waiting.empty()) {
        fail.dismiss();
        return;
    }
    StorageRequest payload;
    payload.opaque_keys = std::make_shared<BlockTreeKeys>(query.opaque_keys->begin(), query.opaque_keys->begin() + end);
    payload.request_id = query.request_id;
    payload.prefetch_notice = query.prefetch_notice;
    payload.positional_replicas = true;
    payload.handles.resize(end);
    payload.replica_descriptors.resize(end);
    auto staged = std::make_shared<HostPrefetch>(payload.opaque_keys, groups);
    staged->local_positions = std::move(local);
    for (const auto& dest : destinations) {
        const auto slot = dest.group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::LOAD);
        if (isNullBlockIdx(slot)) {
            return;
        }
        staged->resources[dest.row][dest.group->groupSetId()].host_block = slot;
        staged->destinations.emplace_back(dest.row, dest.group->groupSetId());
        payload.handles[dest.row].push_back({dest.group->groupTags().front(), slot, Tier::HOST});
        payload.replica_descriptors[dest.row].push_back(dest.replica);
    }
    attempt->phase = RemotePrefetch::Phase::READING;
    if (!task_pool_->submitCompletion([this, attempt, staged, payload = std::move(payload)]() mutable {
            {
                std::lock_guard lock(mutex_);
                if (attempt->phase == RemotePrefetch::Phase::DONE || !attempt->live()) {
                    attempt->finish(false);
                    return;
                }
            }
            storage_backend_->readOwned(std::move(payload), attempt->meta, [this, attempt, staged](StorageTransferOutcome result) {
                std::lock_guard lock(mutex_);
                if (attempt->phase == RemotePrefetch::Phase::DONE) {
                    return;
                }
                bool warmed = result.safe && result.error.ok() && attempt->live();
                if (warmed) {
                    const auto path = tree_->findNode(*staged->keys);
                    warmed = std::all_of(staged->local_positions.begin(), staged->local_positions.end(),
                                         [&](const auto& key) {
                        const auto [row, group] = key;
                        return row < path.size() && path[row]->group_set_resources[group].isMatchUsable();
                    });
                }
                std::shared_ptr<HostPrefetchMatch> captured;
                if (warmed) {
                    try {
                        attempt->publish(*staged->keys, staged->resources);
                        captured = captureWarmRows(load_context_coordinator_, staged->keys->size(), tree_->groupSets(),
                                                   load_join_registry_,
                                                   tree_->findNode(*staged->keys), staged->local_positions,
                                                   staged->destinations);
                    } catch (const std::exception& error) {
                        BTC_LOG_ERROR("failed to publish completed remote Host prefetch: %s", error.what());
                    }
                }
                const bool ready = static_cast<bool>(captured);
                attempt->finish(ready, std::move(captured));
            });
        })) {
        return;
    }
    fail.dismiss();
}

bool BlockTreeLoader::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(context);
    if (load_context == nullptr) {
        BTC_LOG_WARNING("context is not owned by BlockTreeCache");
        return false;
    }
    return load_context->abortPending();
}

void BlockTreeLoader::shutdown() {
    load_context_coordinator_->shutdown();
}

bool BlockTreeLoader::commitLoad(const std::shared_ptr<LoadAsyncContext>& context) {
    CacheLockGuard                         lock(mutex_, metrics_reporter_, "load_prepare");
    const std::vector<TransferDescriptor>& load_descs          = context->loadDescs();
    const std::vector<bool>&               joined_loads        = context->joinedLoads();
    const uint64_t                         context_id          = context->contextId();
    size_t                                 prepared_desc_count = 0;
    block_tree_cache_detail::ScopeRollback rollback_guard(
        [this, &load_descs, &joined_loads, &prepared_desc_count, context_id]() {
            abortLoadLocked(load_descs,
                            joined_loads,
                            prepared_desc_count,
                            context_id,
                            /*release_transferred_refs=*/false);
        });

    for (const auto& desc : load_descs) {
        if (desc.isCurrentOwner() && desc.node->group_set_resources[desc.group_set_id].transfer_detached) {
            abortLoadLocked(load_descs, joined_loads, 0, context_id, false);
            rollback_guard.dismiss();
            context->setSettlementReadyCallback(
                [this](const auto& ready) { scheduleContextSettlement(nullptr, ready); });
            context->cancelUnsubmittedTransfers();
            return true;
        }
    }

    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc = load_descs[desc_index];
        if (desc.source_tier == Tier::DEVICE || joined_loads[desc_index]) {
            ++prepared_desc_count;
            continue;
        }
        if (desc.isCurrentOwner()
            && !changeTransferState(
                desc.node, desc.group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING)) {
            BTC_LOG_ERROR("committed load source is not LOAD_PENDING, group_set_id=%zu", desc.group_set_id);
            return false;
        }
        if (desc.initial_owner) {
            BTC_CHECK(
                load_join_registry_.start(desc.node, desc.group_set_id, desc.generation, desc.target_blocks, context));
        }
        tree_->groupSets()[desc.group_set_id]->referenceBlocks(
            MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}},
            BlockTreeRefType::LOAD);
        ++prepared_desc_count;
    }

    LoadTaskRunner::TaskPtr task = load_task_runner_.createTask(context);
    context->setSettlementReadyCallback([this, task](const std::shared_ptr<LoadAsyncContext>& ready_context) {
        scheduleContextSettlement(task, ready_context);
    });
    if (task) {
        auto queue_wait = TransferStageTimer::create(
            CacheStageContext::forWorkflow(
                metrics_reporter_, task->load_descs, "load", context->requestId().value_or(-1), context->workflowId()),
            "business_queue_wait",
            "business_task",
            AsyncProfileKind::Wait);
        block_tree_cache_detail::ScopeRollback discard_wait([&] { queue_wait.discard(); });
        auto                                   on_timeout = [this, task, queue_wait]() {
            queue_wait.finish(false);
            const size_t descriptor_count = task->load_descs.size();
            BTC_LOG_WARNING("load expired in business queue, descriptor_count=%zu", descriptor_count);
            if (!task->context->completeTransfers(descriptor_count, false)) {
                BTC_LOG_WARNING("failed to record expired load, descriptor_count=%zu", descriptor_count);
            }
        };
        const TransferTask& queue_task =
            task->disk_to_device_task.descriptors().empty() ? task->host_to_device_task : task->disk_to_device_task;
        if (queue_task.expired()) {
            on_timeout();
            rollback_guard.dismiss();
            return true;
        }
        const auto deadline = queue_task.deadline();
        if (!task_pool_->submit(
                BlockTreeTaskClass::LOAD,
                [this, task, queue_wait]() {
                    queue_wait.finish(true);
                    runLoadTask(task);
                },
                deadline,
                std::move(on_timeout))) {
            return false;
        }
        discard_wait.dismiss();
    }

    rollback_guard.dismiss();
    return true;
}

void BlockTreeLoader::abortLoadLocked(const std::vector<TransferDescriptor>& load_descs,
                                      const std::vector<bool>&               joined_loads,
                                      size_t                                 prepared_desc_count,
                                      uint64_t                               context_id,
                                      bool                                   release_transferred_refs) {
    bool                                  device_refs_released = false;
    std::unordered_map<TreeNode*, size_t> pending_nodes;
    std::vector<TransferDescriptor>       detached_descs;
    for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
        const TransferDescriptor& desc           = load_descs[desc_index];
        const bool                joined_load    = joined_loads[desc_index];
        const bool                fully_prepared = desc_index < prepared_desc_count;
        if (joined_load || (desc.source_tier != Tier::DEVICE && fully_prepared)) {
            if ((joined_load || desc.initial_owner)
                && !load_join_registry_.eraseForContext(desc.node, desc.group_set_id, desc.generation, context_id)) {
                if (joined_load) {
                    BTC_LOG_DEBUG("joined load context is no longer registered, group_set=%zu", desc.group_set_id);
                } else {
                    BTC_LOG_WARNING("failed to erase aborted load context, group_set=%zu", desc.group_set_id);
                }
            }
            if (!joined_load || release_transferred_refs) {
                const MultiNodeResource target_resource{
                    desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}};
                if (joined_load) {
                    tree_->groupSets()[desc.group_set_id]->unreferenceBlocks(target_resource);
                } else {
                    tree_->groupSets()[desc.group_set_id]->unreferenceBlocks(target_resource, BlockTreeRefType::LOAD);
                }
            }
        }
        if (joined_load) {
            device_refs_released = device_refs_released || release_transferred_refs;
            continue;
        }

        MultiNodeResource resource{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
        if (desc.source_tier == Tier::DEVICE) {
            if (release_transferred_refs) {
                tree_->groupSets()[desc.group_set_id]->unreferenceBlocks(resource);
                device_refs_released = true;
            }
            continue;
        }
        tree_->groupSets()[desc.group_set_id]->unreferenceBlocks(resource, BlockTreeRefType::LOAD);
        if (!desc.isCurrentOwner()) {
            ++pending_nodes[desc.node];
            continue;
        }
        if (desc.node->group_set_resources[desc.group_set_id].transfer_detached) {
            detached_descs.push_back(desc);
            pending_nodes.try_emplace(desc.node, 0);
            continue;
        }
        const GroupSetTransferState expected_state =
            fully_prepared ? GroupSetTransferState::LOADING : GroupSetTransferState::LOAD_PENDING;
        if (!changeTransferState(desc.node, desc.group_set_id, expected_state, GroupSetTransferState::IDLE)) {
            BTC_LOG_WARNING(
                "load rollback state mismatch, group_set=%zu source=%s", desc.group_set_id, tierName(desc.source_tier));
        } else {
            evictor_.admitCandidate(
                desc.node, desc.group_set_id, desc.node->group_set_resources[desc.group_set_id].getTopTier());
        }
    }
    evictor_.discardDetachedTransfer(detached_descs);
    tree_->removePendingNodes(pending_nodes);
    if (device_refs_released) {
        settled_(true);
    }
}

void BlockTreeLoader::runLoadTask(const LoadTaskRunner::TaskPtr& task) {
    const auto complete = [task](ErrorInfo) noexcept {
        task->phase = LoadTaskRunner::Task::Phase::FINISHED;
        for (const Tier tier : {Tier::HOST, Tier::DISK}) {
            const auto&  transfer = tier == Tier::HOST ? task->host_to_device_task : task->disk_to_device_task;
            const auto&  result   = tier == Tier::HOST ? task->host_result : task->disk_result;
            const size_t count    = transfer.descriptors().size();
            const bool   success  = result.confirmedCopy(count);
            if (count != 0 && !task->context->completeTransfers(count, success, tier)) {
                BTC_LOG_WARNING("failed to record load copy completion, descriptor_count=%zu", count);
            }
        }
    };
    try {
        load_task_runner_.runTransfer(task, *transfer_dispatcher_, metrics_reporter_, complete);
    } catch (const std::exception& error) {
        BTC_LOG_ERROR("load task runner failed with exception: %s", error.what());
        complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        BTC_LOG_ERROR("load task runner failed with unknown exception");
        complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown load task runner exception"));
    }
}

void BlockTreeLoader::scheduleContextSettlement(const LoadTaskRunner::TaskPtr&           task,
                                                const std::shared_ptr<LoadAsyncContext>& context) {
    const auto& descriptors = task ? task->load_descs : context->loadDescs();
    uint64_t    transfer_id = context->workflowId();
    if (task) {
        transfer_id = task->completionTransfer().traceId();
    }
    auto stage_context = CacheStageContext::forWorkflow(
        metrics_reporter_, descriptors, "load", context->requestId().value_or(-1), transfer_id);
    const bool transfer_success = context->localTransfersSucceeded();
    auto       timers           = CompletionStageTimers::create(stage_context);
    auto       settle           = [this, task, context, stage_context, timers, transfer_success]() {
        timers.finishQueue(transfer_success);
        bool                                   success = false;
        block_tree_cache_detail::ScopeRollback finish([&] { timers.finish(success); });
        auto                                   execute =
            TransferStageTimer::scope(stage_context, "completion_execute", "business_completion", transfer_success);
        bool                              settlement_success = context->localTransfersSucceeded();
        bool                              host_success       = context->hostTransfersSucceeded();
        std::vector<JoinedLoadCompletion> joined_contexts;
        if (task) {
            {
                CacheLockGuard lock(mutex_, "load_settle", stage_context);
                settlement_success =
                    settleLoadLocked(*task, settlement_success, host_success, joined_contexts, stage_context);
            }
            auto notify = TransferStageTimer::scope(
                stage_context, "completion_join_notify", "business_completion", transfer_success);
            for (const auto& joined_context : joined_contexts) {
                bool    join_completed       = false;
                int64_t join_wait_latency_us = 0;
                if (!joined_context.context->completeJoinedOne(
                        joined_context.success, join_completed, join_wait_latency_us, joined_context.source_tier)) {
                    BTC_LOG_WARNING("failed to complete joined load context");
                } else if (join_completed) {
                    metrics_reporter_.reportLoadJoinWait(join_wait_latency_us);
                }
            }
        }
        {
            auto notify =
                TransferStageTimer::scope(stage_context, "completion_notify", "business_completion", transfer_success);
            if (!context->settle(settlement_success, host_success)) {
                BTC_LOG_WARNING("failed to publish load context settlement, context_id=%lu", context->contextId());
            }
            success = context->success();
            notify.setSuccess(success);
        }
        execute.setSuccess(success);
    };
    if (!task_pool_->submitCompletion(settle)) {
        BTC_LOG_WARNING("load completion queue is closed; settling context inline");
        settle();
    }
}

bool BlockTreeLoader::validateLoadTaskLocked(const LoadTaskRunner::Task& task) const {
    bool valid = true;
    for (const TransferDescriptor& desc : task.load_descs) {
        if (!tree_->groupSets()[desc.group_set_id]->canPublishDevice(desc.target_blocks)) {
            valid = false;
        }
        if (!desc.isCurrentOwner()) {
            continue;
        }
        const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        if (resource.transfer_detached) {
            BTC_LOG_WARNING("load transfer detached before completion, group_set=%zu", desc.group_set_id);
            valid = false;
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::LOADING) {
            BTC_LOG_ERROR("load state mismatch during settlement, group_set_id=%zu", desc.group_set_id);
            valid = false;
        }
    }
    return valid;
}

bool BlockTreeLoader::settleLoadLocked(LoadTaskRunner::Task&              task,
                                       bool                               aggregate_success,
                                       bool&                              host_success,
                                       std::vector<JoinedLoadCompletion>& joined_contexts,
                                       const CacheStageContext&           stage_context) {
    std::vector<BlockTreeEvictor::InvalidationSource> corrupted;
    const auto collect_corrupted = [this, &corrupted](const TransferTask& transfer, const TransferResult& result) {
        if (result.block_errors.size() != transfer.descriptors().size()) {
            return;
        }
        for (size_t index = 0; index < result.block_errors.size(); ++index) {
            const auto& desc = transfer.descriptors()[index];
            if (result.block_errors[index] != CacheCopyError::CRC_MISMATCH || !desc.node->parent) {
                continue;
            }
            auto& resource = desc.node->group_set_resources[desc.group_set_id];
            if (desc.generation != resource.generation || resource.transfer_detached
                || resource.getBlocks(desc.source_tier) != desc.source_blocks) {
                continue;
            }
            BlockTreeKeys cache_keys;
            for (TreeNode* node = desc.node; node != tree_->root(); node = node->parent) {
                cache_keys.emplace_back(node->key());
            }
            std::reverse(cache_keys.begin(), cache_keys.end());
            corrupted.push_back({std::move(cache_keys), desc.group_set_id, desc.source_tier});
        }
    };
    // Capture both tiers before resource settlement or a cascading drop can destroy a source node.
    collect_corrupted(task.host_to_device_task, task.host_result);
    collect_corrupted(task.disk_to_device_task, task.disk_result);

    const bool valid                            = validateLoadTaskLocked(task);
    host_success                                = host_success && valid;
    const bool                            settlement_success = aggregate_success && valid;
    bool                                  state_settled = false;
    std::unordered_map<TreeNode*, size_t> pending_nodes;

    const auto& descriptors = task.load_descs;
    {
        auto update = TransferStageTimer::scope(
            stage_context, "completion_tree_update", "business_completion", settlement_success);
        std::vector<TransferDescriptor> detached_descs;
        for (size_t desc_index = 0; desc_index < descriptors.size(); ++desc_index) {
            const TransferDescriptor& desc      = descriptors[desc_index];
            const GroupSetPtr&        group_set = tree_->groupSets()[desc.group_set_id];
            MultiNodeResource source_protection{desc.group_set_id, desc.source_tier, {{desc.node, desc.source_blocks}}};
            group_set->unreferenceBlocks(source_protection, BlockTreeRefType::LOAD);

            if (!desc.isCurrentOwner()) {
                ++pending_nodes[desc.node];
                state_settled = true;
                continue;
            }
            GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
            if (resource.transfer_detached) {
                detached_descs.push_back(desc);
                pending_nodes.try_emplace(desc.node, 0);
                state_settled = true;
                continue;
            }
            if (desc.source_tier == Tier::HOST ? host_success : settlement_success) {
                const auto* previous = tree_->previousGroupPage(desc.node, desc.group_set_id);
                const bool install_target =
                    enable_device_cache_
                    && (group_set->groupType() != CacheGroupType::FULL || previous == tree_->root()
                        || (previous && previous->group_set_resources[desc.group_set_id].hasCompleteDeviceValue()));
                if (install_target) {
                    MultiNodeResource target_holder{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}};
                    resource.setBlocks(Tier::DEVICE, desc.target_blocks);
                    group_set->referenceBlocks(target_holder, BlockTreeRefType::CACHE);
                    group_set->unreferenceBlocks(target_holder, BlockTreeRefType::LOAD);
                    resource.evictFromTier(desc.source_tier);
                    group_set->unreferenceBlocks(source_protection, BlockTreeRefType::CACHE);
                    task.target_installed[desc_index] = true;
                }
                const bool state_changed = changeTransferState(
                    desc.node, desc.group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE);
                assert(state_changed);
                (void)state_changed;
                state_settled = true;
                if (install_target) {
                    evictor_.onLoaded(desc.node, desc.group_set_id);
                } else {
                    evictor_.admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
                }
                continue;
            }

            // Ordinary copy failures still retain a reusable source.
            if (!changeTransferState(
                    desc.node, desc.group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE)) {
                BTC_LOG_WARNING(
                    "loading state mismatch, group_set=%zu source=%s", desc.group_set_id, tierName(desc.source_tier));
            } else {
                evictor_.admitCandidate(desc.node, desc.group_set_id, desc.source_tier);
                state_settled = true;
            }
        }
        evictor_.discardDetachedTransfer(detached_descs);
    }
    {
        auto refs = TransferStageTimer::scope(
            stage_context, "completion_release_refs", "business_completion", settlement_success);
        load_task_runner_.releaseTaskResources(task);
    }
    {
        auto joins = TransferStageTimer::scope(
            stage_context, "completion_join_registry", "business_completion", settlement_success);
        for (const TransferDescriptor& desc : task.load_descs) {
            if (!desc.initial_owner) {
                continue;
            }
            std::vector<std::shared_ptr<LoadAsyncContext>> contexts;
            if (!load_join_registry_.finish(desc.node, desc.group_set_id, desc.generation, contexts)) {
                BTC_LOG_WARNING("failed to finish loading record, group_set=%zu", desc.group_set_id);
            }
            const bool success = desc.source_tier == Tier::HOST ? host_success : settlement_success;
            for (auto& context : contexts) {
                joined_contexts.push_back({std::move(context), desc.source_tier, success});
            }
        }
    }
    {
        auto cleanup =
            TransferStageTimer::scope(stage_context, "completion_cleanup", "business_completion", settlement_success);
        tree_->removePendingNodes(pending_nodes);
        evictor_.invalidateSources(corrupted);
    }
    {
        auto watermark =
            TransferStageTimer::scope(stage_context, "completion_watermark", "business_completion", settlement_success);
        settled_(state_settled);
    }
    return settlement_success;
}

bool BlockTreeLoader::changeTransferState(TreeNode*             node,
                                          size_t                group_set_id,
                                          GroupSetTransferState expected_state,
                                          GroupSetTransferState target_state) {
    GroupSetResource& resource = node->group_set_resources[group_set_id];
    if (resource.transfer_state != expected_state) {
        return false;
    }
    resource.transfer_state = target_state;
    tree_->refreshPublishedState(node);
    return true;
}

}  // namespace rtp_llm
