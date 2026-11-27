#include "BlockTreeCacheCInternal.h"

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

namespace rtp_llm::c_api {
namespace {

struct MatchLimits {
    std::vector<bool> selected;
    std::vector<size_t> groups;
    std::function<bool(size_t)> boundary;
    std::vector<uint64_t> boundaries;
    bool constrained = false;
    size_t max_rows;
};

MatchLimits readLimits(const btc_cache& cache, const btc_limits& limits, size_t keys) {
    if ((!limits.groups && limits.group_count) || (!limits.boundaries && limits.boundary_count)) {
        throw std::invalid_argument("Missing BlockTreeCache match selection");
    }
    MatchLimits result;
    result.selected.resize(cache.groups.size(), limits.groups == nullptr);
    if (limits.groups) {
        for (size_t i = 0; i < limits.group_count; ++i) {
            const size_t group = cache.groupIndex(limits.groups[i]);
            if (result.selected[group]) {
                throw std::invalid_argument("Duplicate selected BlockTreeCache group");
            }
            result.selected[group] = true;
            result.groups.push_back(group);
        }
        if (result.groups.empty()) {
            throw std::invalid_argument("BlockTreeCache match requires at least one group");
        }
    } else {
        for (size_t group = 0; group < cache.groups.size(); ++group) {
            result.groups.push_back(group);
        }
    }
    if (limits.boundaries) {
        result.constrained = true;
        for (size_t i = 0; i < limits.boundary_count; ++i) {
            const auto boundary = limits.boundaries[i];
            if (!boundary || boundary % cache.key_tokens || (i && boundary <= result.boundaries.back())) {
                throw std::invalid_argument("Invalid BlockTreeCache match boundaries");
            }
            result.boundaries.push_back(boundary);
        }
        auto allowed_rows = std::unordered_set<size_t>();
        allowed_rows.reserve(result.boundaries.size());
        for (auto boundary : result.boundaries) {
            if (boundary / cache.key_tokens <= keys) {
                allowed_rows.insert(static_cast<size_t>(boundary / cache.key_tokens));
            }
        }
        result.boundary = [allowed_rows = std::move(allowed_rows)](size_t rows) {
            return allowed_rows.count(rows) != 0;
        };
    }
    result.max_rows = static_cast<size_t>(std::min<uint64_t>(keys, limits.max_tokens / cache.key_tokens));
    return result;
}

btc_work_kind loadRoute(Tier source) {
    return source == Tier::HOST ? BTC_WORK_HOST_TO_DEVICE : BTC_WORK_DISK_TO_DEVICE;
}

struct Row {
    btc_plan_page view{};
    std::vector<btc_page> members;
    std::vector<size_t> remote_handles;
    size_t descriptor = std::numeric_limits<size_t>::max();
};

std::vector<PolledLoadLookup> joinLookups(const std::vector<Row>& rows) {
    std::vector<PolledLoadLookup> result;
    for (const auto& row : rows) {
        if (row.view.state == BTC_PAGE_JOIN) {
            result.push_back({row.view.route, row.view.position, &row.members});
        }
    }
    return result;
}

// Hold direct read targets through native publication, even when the
// originating scheduler request cancels after handing off its allocation.
class DirectPublication {
public:
    DirectPublication(const BlockTreeKeys& keys, size_t rows, size_t group_count, size_t remote_begin):
        keys_(keys.begin(), keys.begin() + rows),
        remote_begin_(remote_begin),
        resources_(rows, std::vector<GroupSetResource>(group_count)) {}

    ~DirectPublication() {
        for (const auto& [group, resource] : targets_) {
            group->unreferenceBlocks(resource);
        }
    }

    void add(const GroupSetPtr& group, size_t row, const BlockIdList& blocks) {
        const size_t index = group->groupSetId();
        resources_[row][index].setBlocks(Tier::DEVICE, blocks);
        targets_.emplace_back(group, MultiNodeResource{index, Tier::DEVICE, {{nullptr, blocks}}});
        block_tree_cache_detail::ScopeRollback rollback([&] { targets_.pop_back(); });
        group->referenceBlocks(targets_.back().second);
        rollback.dismiss();
    }

    bool publish(BlockTreeCache& native, uint64_t request) {
        for (const auto& [group, resource] : targets_) {
            if (!group->canPublishDevice(resource.node_blocks[0].second)) {
                return false;
            }
        }
        native.insert(keys_, resources_, Tier::DEVICE, false, static_cast<int64_t>(request),
                      remote_begin_, keys_.size());
        return true;
    }

private:
    BlockTreeKeys keys_;
    size_t remote_begin_;
    std::vector<std::vector<GroupSetResource>> resources_;
    std::vector<std::pair<GroupSetPtr, MultiNodeResource>> targets_;
};

// Native matching gives the allocator request refs for READY and JOIN
// pages. These leases bridge the two C calls while the host allocates targets.
struct RequestState {
    explicit RequestState(const btc_cache& cache): groups(cache.native->groupSets()) {}
    ~RequestState() { abandon(); }

    void retain(BlockTreeMatchResult result) {
        context = std::move(result.async_context);
        device_refs = std::move(result.matched_device_resources);
    }

    bool abandon() {
        std::shared_ptr<LoadAsyncContext> pending;
        bool active;
        {
            std::lock_guard lock(mutex);
            active = !cancelled && !completed;
            cancelled = true;
            pending = context;
        }
        if (pending && !submitted) {
            pending->abortPending();
        }
        std::lock_guard lock(mutex);
        releaseRefs();
        return active;
    }

    void complete(btc_cache& cache, uint64_t request, uint64_t ready, uint64_t host, uint64_t recoverable,
                  const std::shared_ptr<DirectPublication>& remote) {
        bool success = context->success();
        if (success && remote) {
            try {
                success = remote->publish(*cache.native, request);
            } catch (const std::exception& error) {
                BTC_LOG_ERROR("failed to publish completed remote BlockTreeCache load: %s", error.what());
                success = false;
            }
        }
        std::lock_guard lock(mutex);
        releaseRefs();
        if (!cancelled) {
            cache.event(request, success ? BTC_EVENT_READY : BTC_EVENT_RECOMPUTE,
                        success ? recoverable : (context->hostTransfersSucceeded() ? host : ready));
        }
        completed = true;
    }

    void releaseRefs() {
        for (const auto& resource : device_refs) {
            groups[resource.group_set_id]->unreferenceBlocks(resource);
        }
        device_refs.clear();
        if (!context || released_context_refs) {
            return;
        }
        released_context_refs = true;
        const auto& descriptors = context->loadDescs();
        const auto& joins = context->joinedLoads();
        for (size_t i = 0; i < descriptors.size(); ++i) {
            const auto& desc = descriptors[i];
            if (joins[i] || desc.source_tier == Tier::DEVICE) {
                groups[desc.group_set_id]->unreferenceBlocks(
                    {desc.group_set_id, Tier::DEVICE, {{nullptr, joins[i] ? desc.target_blocks : desc.source_blocks}}});
            }
        }
    }

    const std::vector<GroupSetPtr> groups;
    std::shared_ptr<LoadAsyncContext> context;
    std::vector<MultiNodeResource> device_refs;
    std::mutex mutex;
    bool submitted = false;  // written by the serial ABI thread before installing completion
    bool cancelled = false;
    bool released_context_refs = false;
    std::atomic<bool> completed{false};
    std::atomic<size_t> remote_end{std::numeric_limits<size_t>::max()};
};

// Identical pending requests subscribe to one native prefetch. Membership
// only controls scheduler notifications; the storage workflow owns its I/O.
struct RemoteAttempt {
    bool attach(uint64_t request) {
        std::lock_guard lock(mutex);
        if (finished) {
            return false;
        }
        waiters.insert(request);
        return true;
    }

    bool detach(uint64_t request) {
        std::lock_guard lock(mutex);
        const bool removed = waiters.erase(request) != 0;
        if (waiters.empty()) {
            residency.reset();
        }
        return removed;
    }

    bool pending() {
        std::lock_guard lock(mutex);
        return !finished;
    }

    bool needed() {
        std::lock_guard lock(mutex);
        return !finished && !waiters.empty();
    }

    std::shared_ptr<HostPrefetchMatch> takeResidency(uint64_t request) {
        std::lock_guard lock(mutex);
        if (!finished || !waiters.erase(request)) {
            return {};
        }
        return waiters.empty() ? std::move(residency) : residency;
    }

    void progress(btc_cache& cache, bool terminal, std::shared_ptr<HostPrefetchMatch> hold = {}) {
        std::lock_guard lock(mutex);
        if (!finished) {
            for (auto request : waiters) {
                cache.event(request, BTC_EVENT_PROGRESS);
            }
            if (terminal) {
                residency = std::move(hold);
                if (!residency) {
                    waiters.clear();
                }
                finished = true;
            }
        }
    }

    std::mutex mutex;
    bool finished = false;
    std::unordered_set<uint64_t> waiters;
    std::shared_ptr<HostPrefetchMatch> residency;
};

void cancelRequest(btc_cache& cache, uint64_t request);

}  // namespace
}  // namespace rtp_llm::c_api

struct btc_plan {
    uint64_t instance = 0;
    uint64_t epoch = 0;
    uint64_t request = 0;
    uint64_t ready = 0;
    uint64_t recoverable = 0;
    uint64_t host = 0;
    rtp_llm::BlockTreeKeys keys;
    std::vector<rtp_llm::c_api::Row> rows, ready_rows;
    std::vector<btc_plan_page> views, ready_views;
    std::shared_ptr<rtp_llm::c_api::RequestState> native;
};

namespace rtp_llm::c_api {

// Owned by the request table until the caller collects the plan or cancels.
struct MatchAdmission {
    std::unique_ptr<btc_plan> plan;
    MatchLimits limits;
    size_t max_load_pages;
    size_t max_load_rows;
    bool allow_remote;
    std::shared_ptr<LocalMatchAdmission> local;
    std::shared_ptr<RemoteAttempt> remote;
    std::shared_ptr<RemoteAttempt> disk;
    bool direct_query = false;
};

struct RequestTable {
    struct Request {
        uint64_t epoch = 0;
        std::weak_ptr<RequestState> proposed;
        std::shared_ptr<RequestState> running;
        std::unique_ptr<MatchAdmission> pending;
    };
    std::unordered_map<uint64_t, Request> requests;
    uint64_t epoch = 0;
    std::mutex warm_mutex;
    std::unordered_map<std::string, std::weak_ptr<RemoteAttempt>> warming;
    std::unordered_set<uint64_t> awaiting_local_slot;
};

std::shared_ptr<RequestTable> makeRequests() {
    return std::make_shared<RequestTable>();
}

void cancelRequests(btc_cache& cache) {
    while (!cache.requests->requests.empty()) {
        cancelRequest(cache, cache.requests->requests.begin()->first);
    }
}

void notifyLocalCapacity(btc_cache& cache) {
    if (cache.native) {
        cache.native->advanceRemotePrefetch();
    }
    std::unordered_set<uint64_t> retry;
    {
        std::lock_guard lock(cache.requests->warm_mutex);
        retry.swap(cache.requests->awaiting_local_slot);
    }
    for (const auto request : retry) {
        cache.event(request, BTC_EVENT_PROGRESS);
    }
}

namespace {

void cancelRequest(btc_cache& cache, uint64_t request) {
    auto& requests = cache.requests->requests;
    auto found = requests.find(request);
    if (found == requests.end()) {
        return;
    }
    auto previous = found->second.proposed.lock();
    const bool proposed_cancelled = previous && previous->abandon();
    const auto& running = found->second.running;
    const auto& pending = found->second.pending;
    if (pending && pending->remote) {
        pending->remote->detach(request);
    }
    if (pending && pending->disk) {
        pending->disk->detach(request);
    }
    bool retry_cancelled;
    {
        std::lock_guard lock(cache.requests->warm_mutex);
        retry_cancelled = cache.requests->awaiting_local_slot.erase(request) != 0;
    }
    const bool running_cancelled = running && running->abandon();
    if (proposed_cancelled || running_cancelled || retry_cancelled || pending) {
        cache.event(request, BTC_EVENT_CANCELLED);
    }
    requests.erase(found);
}

std::string prefetchIdentity(const BlockTreeKeys& keys, uint64_t local,
                             const MatchLimits& limits, size_t max_load_pages) {
    std::string value;
    const auto append = [&](uint64_t n) {
        value.append(reinterpret_cast<const char*>(&n), sizeof(n));
    };
    append(max_load_pages);
    append(local);
    append(limits.selected.size());
    for (bool selected : limits.selected) {
        value.push_back(selected);
    }
    append(limits.constrained);
    append(limits.boundaries.size());
    for (auto boundary : limits.boundaries) {
        append(boundary);
    }
    for (const auto& key : keys) {
        append(key.bytes().size());
        value.append(key.bytes());
    }
    return value;
}

bool startPrefetch(btc_cache& cache, MatchAdmission& request, const BlockTreeKeys& path, size_t local_rows) {
    const auto& table = cache.requests;
    const auto& plan = *request.plan;
    const auto& limits = request.limits;
    const auto identity = prefetchIdentity(path, local_rows, limits, request.max_load_pages);
    auto attempt = std::make_shared<RemoteAttempt>();
    {
        std::lock_guard lock(table->warm_mutex);
        auto& entry = table->warming[identity];
        if (auto current = entry.lock(); current && current->attach(plan.request)) {
            request.remote = std::move(current);
            return true;
        }
        attempt->attach(plan.request);
        entry = attempt;
    }

    std::weak_ptr<RequestTable> table_weak = table;
    auto retire = [attempt, table_weak, identity] {
        if (auto table = table_weak.lock()) {
            std::lock_guard lock(table->warm_mutex);
            const auto it = table->warming.find(identity);
            if (it != table->warming.end() && it->second.lock() == attempt) {
                table->warming.erase(it);
            }
        }
    };
    block_tree_cache_detail::ScopeRollback rollback(retire);
    const bool started = cache.native->prefetchRemote(
        path, local_rows, limits.boundary, limits.selected, request.max_load_pages,
        static_cast<int64_t>(plan.request),
        [attempt] { return attempt->needed(); },
        [attempt, &cache](bool terminal) { attempt->progress(cache, terminal); },
        [attempt, &cache, retire](bool, std::shared_ptr<HostPrefetchMatch> residency) {
            attempt->progress(cache, true, std::move(residency));
            retire();
        });
    if (started) {
        request.remote = std::move(attempt);
        rollback.dismiss();
    }
    return started;
}

std::vector<btc_plan_page> pageViews(const std::vector<Row>& rows) {
    std::vector<btc_plan_page> result;
    result.reserve(rows.size());
    for (const auto& row : rows) {
        result.push_back(row.view);
        result.back().pages = row.members.data();
        result.back().page_count = row.members.size();
    }
    return result;
}

std::vector<btc_page> describe(const btc_cache& cache, size_t group, Tier tier, const BlockIdList& blocks) {
    std::vector<btc_page> result;
    result.reserve(blocks.size());
    for (auto block : blocks) {
        result.push_back(cache.describePage(group, tier, block));
    }
    return result;
}

bool serialize(const btc_cache& cache, const MatchLimits& limits, btc_plan& output) {
    const auto& state = *output.native;
    const auto& context = state.context;
    const size_t no_desc = std::numeric_limits<size_t>::max();
    std::vector<const MultiNodeResource*> independent(cache.groups.size(), nullptr);
    for (const auto& resource : state.device_refs) {
        independent[resource.group_set_id] = &resource;
    }
    std::vector<std::vector<size_t>> descriptors(cache.groups.size());
    if (context) {
        const auto& loads = context->loadDescs();
        for (size_t i = 0; i < loads.size(); ++i) {
            const auto& load = loads[i];
            const auto stride = cache.groups.at(load.group_set_id).page_tokens / cache.key_tokens;
            auto& group = descriptors.at(load.group_set_id);
            const size_t row = (load.path_index + 1) / stride - 1;
            if (row >= group.size()) {
                group.resize(row + 1, no_desc);
            }
            group[row] = i;
        }
    }
    const auto fill = [&](uint64_t tokens, std::vector<Row>& destination) {
        for (auto group : limits.groups) {
            const auto& spec = cache.groups[group];
            const size_t end = tokens / spec.page_tokens;
            const size_t required = spec.set->computeReuseBlockCount(end);
            const auto* resource = independent[group];
            const size_t ready_end = output.ready / spec.page_tokens;
            const size_t ready_begin = ready_end - (resource ? resource->node_blocks.size() : 0);
            for (size_t index = end - required; index < end; ++index) {
                Row row;
                auto& view = row.view;
                view.position = {spec.id, (index + 1) * spec.page_tokens};
                view.state = BTC_PAGE_READY;
                auto tier = Tier::DEVICE;
                const BlockIdList* blocks = nullptr;
                if (resource && index >= ready_begin && index < ready_end) {
                    blocks = &resource->node_blocks[index - ready_begin].second;
                } else {
                    const size_t key_row = view.position.end_token / cache.key_tokens - 1;
                    if (context && key_row >= context->localMatchedBlocks()
                        && key_row < context->backendHandles().size()
                        && output.recoverable > context->localMatchedBlocks() * cache.key_tokens) {
                        const auto& handles = context->backendHandles()[key_row];
                        for (const auto& tag : spec.set->groupTags()) {
                            const auto handle = std::find_if(handles.begin(), handles.end(),
                                                             [&](const auto& member) { return member.tag == tag; });
                            if (handle == handles.end()) {
                                return false;
                            }
                            row.remote_handles.push_back(static_cast<size_t>(handle - handles.begin()));
                        }
                        view.route = BTC_WORK_REMOTE_TO_DEVICE;
                        view.state = BTC_PAGE_LOAD;
                        destination.push_back(std::move(row));
                        continue;
                    }
                    if (index >= descriptors[group].size() || descriptors[group][index] == no_desc) {
                        return false;
                    }
                    const size_t i = descriptors[group][index];
                    const auto& desc = context->loadDescs()[i];
                    row.descriptor = i;
                    tier = desc.source_tier;
                    blocks = &desc.source_blocks;
                    if (tier != Tier::DEVICE) {
                        view.route = loadRoute(tier);
                        view.state = BTC_PAGE_LOAD;
                    }
                    if (context->joinedLoads()[i]) {
                        view.state = BTC_PAGE_JOIN;
                        tier = Tier::DEVICE;
                        blocks = &desc.target_blocks;
                    }
                }
                row.members = describe(cache, group, tier, *blocks);
                destination.push_back(std::move(row));
            }
        }
        return true;
    };
    if ((output.ready != output.recoverable && !fill(output.ready, output.ready_rows))
        || !fill(output.recoverable, output.rows)) {
        return false;
    }
    const auto operations = cache.transfers->findLoads(joinLookups(output.rows));
    size_t join = 0;
    for (auto& row : output.rows) {
        if (row.view.state == BTC_PAGE_JOIN) {
            row.view.operation = operations[join++];
            if (!row.view.operation) {
                return false;
            }
        }
    }
    // Native loading may protect additional READY rows for tier fallback.
    // Every actual copy requires an allocator target represented in the ABI plan.
    if (context) {
        std::vector<bool> bound(context->loadDescs().size());
        for (const auto& row : output.rows) {
            if (row.view.state == BTC_PAGE_LOAD && row.remote_handles.empty()) {
                bound[row.descriptor] = true;
            }
        }
        for (size_t i = 0; i < bound.size(); ++i) {
            if (!bound[i] && !context->joinedLoads()[i] && context->loadDescs()[i].source_tier != Tier::DEVICE) {
                return false;
            }
        }
    }
    output.views = pageViews(output.rows);
    output.ready_views = pageViews(output.ready_rows);
    return true;
}

bool snapshotMatch(btc_cache& cache, const MatchLimits& limits, btc_plan& output, BlockTreeMatchResult matched) {
    output.ready = matched.matched_device_blocks * cache.key_tokens;
    output.recoverable = matched.matched_recoverable_blocks * cache.key_tokens;
    output.host = matched.matched_device_host_blocks * cache.key_tokens;
    output.native->retain(std::move(matched));
    output.rows.clear();
    output.ready_rows.clear();
    return serialize(cache, limits, output);
}

bool projectNativeMatch(btc_cache& cache, const MatchLimits& limits, btc_plan& output,
                        size_t load_limit, bool direct, bool stage_disk_through_host) {
    for (;;) {
        output.native = std::make_shared<RequestState>(cache);
        auto matched = cache.native->match(output.keys, limits.boundary, direct && load_limit,
                                           static_cast<int64_t>(output.request), false, limits.selected, load_limit,
                                           stage_disk_through_host);
        if (snapshotMatch(cache, limits, output, std::move(matched))) {
            return true;
        }
        // A concurrent unsafe or completed load can leave the executor
        // ledger while the plan is taking its snapshot.
        output.native.reset();
        if (!load_limit) {
            return false;
        }
        load_limit = 0;
    }
}

DiskStage startDiskRestore(btc_cache& cache, MatchAdmission& request, const btc_plan& plan,
                           const std::vector<std::pair<size_t, size_t>>& positions) {
    auto attempt = std::make_shared<RemoteAttempt>();
    attempt->attach(plan.request);
    const auto stage = cache.native->prefetchDisk(
        plan.keys, positions, static_cast<int64_t>(plan.request),
        [attempt, &cache](bool) {
            attempt->progress(cache, true);
        });
    if (stage == DiskStage::Pending) {
        request.disk = std::move(attempt);
    } else if (stage == DiskStage::Retry) {
        std::lock_guard lock(cache.requests->warm_mutex);
        cache.requests->awaiting_local_slot.insert(plan.request);
    }
    return stage;
}

bool currentRows(btc_cache& cache, const btc_plan& plan) {
    const auto operations = cache.transfers->findLoads(joinLookups(plan.rows));
    size_t join = 0;
    bool valid = true;
    cache.native->visitTreeLocked([&](const BlockTree& tree) {
        const auto path = tree.findNode(plan.keys);
        const auto check = [&](const std::vector<Row>& rows) {
            return std::all_of(rows.begin(), rows.end(), [&](const Row& row) {
                const auto position = row.view.position;
                const size_t index = position.end_token / cache.key_tokens - 1;
                const size_t group = cache.groupIndex(position.group);
                if (!row.remote_handles.empty()) {
                    return index >= path.size() || !path[index]->group_set_resources[group].isMatchUsable();
                }
                if (index >= path.size()) {
                    return false;
                }
                const auto& resource = path[index]->group_set_resources[group];
                if (resource.transfer_detached) {
                    return false;
                }
                if (row.view.state == BTC_PAGE_JOIN) {
                    return resource.transfer_state == GroupSetTransferState::LOADING
                           && operations[join++] == row.view.operation;
                }
                const Tier tier = row.view.state == BTC_PAGE_READY ? Tier::DEVICE : resource.getTopTier();
                if (!resource.hasTier(tier)
                    || (tier == Tier::DEVICE && !resource.hasCompleteDeviceValue())
                    || (row.view.state == BTC_PAGE_LOAD
                        && (tier == Tier::DEVICE || loadRoute(tier) != row.view.route
                            || resource.generation != plan.native->context->loadDescs()[row.descriptor].generation))) {
                    return false;
                }
                if (tier == Tier::DEVICE) {
                    const auto matches = [&](BlockIdxType block, const btc_page& expected) {
                        return samePage(cache.describePage(group, tier, block), expected);
                    };
                    return std::equal(resource.device_blocks.begin(), resource.device_blocks.end(),
                                      row.members.begin(), row.members.end(), matches);
                }
                const auto block = tier == Tier::HOST ? resource.host_block : resource.disk_block;
                return row.members.size() == 1 && samePage(cache.describePage(group, tier, block), row.members.front());
            });
        };
        valid = check(plan.rows) && check(plan.ready_rows);
    });
    return valid;
}

// Select Device/Host once. The native admission owns the exact chosen pages
// even while remote I/O or the engine allocator is busy.
btc_status advanceHostAdmission(btc_cache& cache, MatchAdmission& admission) {
    auto& plan = *admission.plan;
    if (!admission.local) {
        admission.local = cache.native->prepareHostAdmission(
            plan.keys, admission.limits.boundary, static_cast<int64_t>(plan.request), admission.limits.selected,
            admission.max_load_pages, false, admission.max_load_rows);
        const auto ready = admission.local->readyRows();
        const auto local = admission.local->matchedRows();
        const auto end = ready + std::min(plan.keys.size() - ready, admission.max_load_rows);
        if (admission.allow_remote && admission.max_load_pages && local < end
            && cache.transfers->supports(BTC_WORK_REMOTE_TO_HOST)) {
            const auto& path = plan.keys;
            if (end == path.size() ? startPrefetch(cache, admission, path, local)
                                   : startPrefetch(cache, admission, BlockTreeKeys(path.begin(), path.begin() + end), local)) {
                return BTC_PENDING;
            }
        }
    }
    const auto warm = admission.remote ? admission.remote->takeResidency(plan.request) : nullptr;
    // Once the loader hands off physical references, adopting them must not allocate.
    plan.native = std::make_shared<RequestState>(cache);
    auto matched = cache.native->completeHostAdmission(admission.local, warm);
    return matched && snapshotMatch(cache, admission.limits, plan, std::move(*matched)) ? BTC_OK : BTC_RETRY;
}

// Direct remote reads bind Device targets to their paused backend context.
// Disk-to-Host staging has a separate scheduler and can retry on local capacity.
btc_status advanceStagedAdmission(btc_cache& cache, MatchAdmission& admission, bool stage_disk) {
    auto& plan = *admission.plan;
    const auto& limits = admission.limits;
    if (admission.direct_query) {
        const auto end = plan.native->remote_end.load();
        if (end == std::numeric_limits<size_t>::max() && !plan.native->context->done()) {
            return BTC_PENDING;
        }
        if (end != std::numeric_limits<size_t>::max() && end > plan.native->context->localMatchedBlocks()) {
            plan.recoverable = end * cache.key_tokens;
            plan.rows.clear();
            plan.ready_rows.clear();
            if (serialize(cache, limits, plan)
                && static_cast<size_t>(std::count_if(plan.rows.begin(), plan.rows.end(),
                       [](const Row& row) { return row.view.state != BTC_PAGE_READY; })) <= admission.max_load_pages) {
                return BTC_OK;
            }
        }
        plan.native.reset();
    }
    const bool direct = admission.allow_remote && !admission.direct_query
        && cache.transfers->supports(BTC_WORK_REMOTE_TO_DEVICE) && !cache.transfers->supports(BTC_WORK_REMOTE_TO_HOST);
    const auto warm = admission.remote ? admission.remote->takeResidency(plan.request) : nullptr;
    if (!projectNativeMatch(cache, limits, plan, admission.max_load_pages, direct, stage_disk)) {
        return BTC_RETRY;
    }
    if (stage_disk) {
        std::vector<std::pair<size_t, size_t>> positions;
        bool restore = false;
        for (const auto& row : plan.rows) {
            if ((row.view.state == BTC_PAGE_LOAD || row.view.state == BTC_PAGE_JOIN)
                && (row.view.route == BTC_WORK_DISK_TO_DEVICE || row.view.route == BTC_WORK_HOST_TO_DEVICE)) {
                positions.emplace_back(cache.groupIndex(row.view.position.group), row.view.position.end_token / cache.key_tokens - 1);
                restore |= row.view.state == BTC_PAGE_LOAD && row.view.route == BTC_WORK_DISK_TO_DEVICE;
            }
        }
        if (restore) {
            plan.native.reset();
            if (!admission.disk && startDiskRestore(cache, admission, plan, positions) != DiskStage::None) {
                return BTC_PENDING;
            }
            if (!projectNativeMatch(cache, limits, plan, 0, false, false)) {
                return BTC_RETRY;
            }
        }
    }
    if (const auto& ctx = plan.native->context; ctx && ctx->needBackendMatch()) {
        admission.direct_query = true;
        ctx->startBackendMatchPaused([weak = std::weak_ptr(plan.native)](LoadAsyncContext&, size_t end) {
            if (auto state = weak.lock()) {
                state->remote_end = end;
            }
            return true;
        });
        cache.flush();
        return BTC_PENDING;
    }
    const auto local = plan.recoverable / cache.key_tokens;
    if (admission.allow_remote && admission.max_load_pages && !admission.remote && local < plan.keys.size()
        && cache.transfers->supports(BTC_WORK_REMOTE_TO_HOST) && startPrefetch(cache, admission, plan.keys, local)) {
        plan.native.reset();
        return BTC_PENDING;
    }
    return BTC_OK;
}

btc_status advanceAdmission(btc_cache& cache, uint64_t request, btc_plan** result) {
    const auto found = cache.requests->requests.find(request);
    if (found == cache.requests->requests.end() || !found->second.pending) {
        return BTC_RETRY;
    }
    auto& record = found->second;
    auto& pending = *record.pending;
    if ((pending.remote && pending.remote->pending()) || (pending.disk && pending.disk->pending())) {
        return BTC_PENDING;
    }
    const bool stage_disk = cache.transfers->supports(BTC_WORK_DISK_TO_HOST)
        && !cache.transfers->supports(BTC_WORK_DISK_TO_DEVICE);
    const bool direct = pending.allow_remote && cache.transfers->supports(BTC_WORK_REMOTE_TO_DEVICE)
        && !cache.transfers->supports(BTC_WORK_REMOTE_TO_HOST);
    if (stage_disk) {
        std::lock_guard lock(cache.requests->warm_mutex);
        if (cache.requests->awaiting_local_slot.count(request)) {
            return BTC_PENDING;
        }
    }
    block_tree_cache_detail::ScopeRollback discard([&] { cancelRequest(cache, request); });
    const auto status = stage_disk || direct ? advanceStagedAdmission(cache, pending, stage_disk)
                                             : advanceHostAdmission(cache, pending);
    if (status == BTC_RETRY) {
        return status;
    }
    if (status == BTC_OK) {
        pending.plan->epoch = record.epoch = ++cache.requests->epoch;
        record.proposed = pending.plan->native;
        *result = pending.plan.release();
        record.pending.reset();
    }
    discard.dismiss();
    return status;
}

bool sameInput(const MatchAdmission& current, const BlockTreeKeys& path, const MatchLimits& limits,
               size_t max_load_pages, size_t max_load_rows, bool allow_remote) {
    return std::equal(current.plan->keys.begin(), current.plan->keys.end(), path.begin(), path.end(),
                      [](const BlockTreeKey& a, const BlockTreeKey& b) { return a.bytes() == b.bytes(); })
        && current.limits.groups == limits.groups && current.limits.constrained == limits.constrained
        && current.limits.boundaries == limits.boundaries && current.max_load_pages == max_load_pages
        && current.max_load_rows == max_load_rows && current.allow_remote == allow_remote;
}

btc_bytes bytes(std::string_view key) {
    return {key.data(), key.size()};
}

}  // namespace
}  // namespace rtp_llm::c_api

using namespace rtp_llm;
using namespace rtp_llm::c_api;

namespace {

btc_status beginMatch(btc_cache* cache, uint64_t request, btc_keys keys, const btc_limits* input,
                      uint64_t max_load_tokens, btc_plan** result) {
    return locked(cache, [&](btc_cache& handle) -> btc_status {
        require(result) = nullptr;
        if (handle.draining) {
            return BTC_CLOSED;
        }
        handle.flush();
        auto path = copyKeys(keys);
        const auto& config = require(input);
        auto limits = readLimits(handle, config, path.size());
        path.erase(path.begin() + limits.max_rows, path.end());
        const size_t max_rows = max_load_tokens == UINT64_MAX ? SIZE_MAX :
            static_cast<size_t>(std::min<uint64_t>(SIZE_MAX, max_load_tokens / handle.key_tokens));
        const bool allow_remote = config.allow_remote && max_rows;
        if (max_load_tokens != UINT64_MAX && (handle.transfers->supports(BTC_WORK_DISK_TO_HOST)
            || (allow_remote && handle.transfers->supports(BTC_WORK_REMOTE_TO_DEVICE)
                && !handle.transfers->supports(BTC_WORK_REMOTE_TO_HOST)))) {
            throw std::invalid_argument("A relative token budget requires Device/Host admission");
        }
        const auto found = handle.requests->requests.find(request);
        if (found != handle.requests->requests.end()) {
            const auto& record = found->second;
            if (record.running && !record.running->completed) {
                return BTC_WAITING;
            }
            if (record.pending && sameInput(*record.pending, path, limits, config.max_load_pages, max_rows, allow_remote)) {
                return advanceAdmission(handle, request, result);
            }
            if (record.pending) {
                cancelRequest(handle, request);
            } else if (const auto previous = record.proposed.lock()) {
                previous->abandon();
            }
        }
        auto plan = std::make_unique<btc_plan>();
        plan->instance = handle.instance;
        plan->request = request;
        plan->keys = std::move(path);
        auto& record = handle.requests->requests[request];
        record.running.reset();
        record.proposed.reset();
        record.pending = std::make_unique<MatchAdmission>(MatchAdmission{
            std::move(plan), std::move(limits), config.max_load_pages, max_rows, allow_remote, {}, {}, {}, false});
        return advanceAdmission(handle, request, result);
    });
}

}  // namespace

btc_status btc_match_capped(btc_cache* cache, uint64_t request, btc_keys keys, const btc_limits* input,
                            uint64_t max_load_tokens, btc_plan** result) {
    return beginMatch(cache, request, keys, input, max_load_tokens, result);
}

btc_status btc_match(btc_cache* cache, uint64_t request, btc_keys keys, const btc_limits* input, btc_plan** result) {
    return beginMatch(cache, request, keys, input, UINT64_MAX, result);
}

btc_status btc_poll_match(btc_cache* cache, uint64_t request, btc_plan** result) {
    return locked(cache, [&](btc_cache& handle) -> btc_status {
        require(result) = nullptr;
        if (handle.draining) {
            return BTC_CLOSED;
        }
        handle.flush();
        return advanceAdmission(handle, request, result);
    });
}

btc_plan_view btc_plan_get(const btc_plan* plan) {
    if (!plan) {
        return {};
    }
    const auto& ready = plan->ready == plan->recoverable ? plan->views : plan->ready_views;
    return {plan->request, plan->ready, plan->recoverable, plan->views.data(), plan->views.size(),
            ready.data(), ready.size(), plan->host};
}

void btc_plan_free(btc_plan* plan) {
    delete plan;
}

btc_status btc_bind(btc_cache* cache, btc_plan* input, const btc_binding* bindings, size_t count) {
    return locked(cache, [&](btc_cache& handle) -> btc_status {
        auto& plan = require(input);
        if (handle.draining) {
            return BTC_CLOSED;
        }
        auto& requests = handle.requests->requests;
        auto found = requests.find(plan.request);
        if (plan.instance != handle.instance || found == requests.end()
            || found->second.epoch != plan.epoch || found->second.pending || found->second.running) {
            return BTC_RETRY;
        }
        handle.flush();
        const auto refuse = [&]() -> btc_status {
            plan.native->abandon();
            requests.erase(found);
            return BTC_RETRY;
        };
        if (!currentRows(handle, plan)) {
            return refuse();
        }
        PreparedPages prepared(handle);
        if (!prepared.prepare(bindings, count, true)) {
            return BTC_RETRY;
        }
        size_t required = 0;
        const auto& state = plan.native;
        std::shared_ptr<DirectPublication> remote;
        for (const auto& row : plan.rows) {
            if (row.view.state != BTC_PAGE_LOAD) {
                continue;
            }
            const auto pos = row.view.position;
            auto target = prepared.rows.find({pos.group, pos.end_token});
            if (target == prepared.rows.end()) {
                throw std::invalid_argument("Missing BlockTreeCache device target");
            }
            ++required;
            if (row.remote_handles.empty()) {
                state->context->setTargetBlocks(row.descriptor, target->second);
                continue;
            }
            const size_t index = pos.end_token / handle.key_tokens - 1;
            if (!remote) {
                remote = std::make_shared<DirectPublication>(plan.keys, plan.recoverable / handle.key_tokens,
                                                             handle.groups.size(), state->context->localMatchedBlocks());
            }
            remote->add(handle.groups[handle.groupIndex(pos.group)].set, index, target->second);
            for (size_t i = 0; i < row.remote_handles.size(); ++i) {
                state->context->setBackendTargetBlock(index, row.remote_handles[i], target->second[i]);
            }
        }
        if (prepared.rows.size() != required) {
            throw std::invalid_argument("Unexpected BlockTreeCache device target");
        }
        if (!state->context) {
            state->abandon();
            requests.erase(found);
            return BTC_OK;
        }
        if (!state->context->commit()) {
            return refuse();
        }
        state->submitted = true;
        prepared.adopt();
        // The caller has installed READY and JOIN pages in its request table.
        // The native transfer context independently owns all in-flight buffers.
        state->releaseRefs();
        found->second.proposed.reset();
        found->second.running = state;
        state->context->onDone([weak = std::weak_ptr(state), keep = remote ? state : nullptr,
                                remote, &handle, request = plan.request, ready = plan.ready,
                                host = plan.host, recoverable = plan.recoverable](ErrorInfo) {
            if (auto locked = keep ? keep : weak.lock()) {
                locked->complete(handle, request, ready, host, recoverable, remote);
            }
        });
        if (remote && !state->context->resumeBackendRead()) {
            state->context->onTaskFail();
        }
        handle.flush();
        return BTC_WAITING;
    });
}

btc_status btc_cancel(btc_cache* cache, uint64_t request) {
    return locked(cache, [&](btc_cache& handle) {
        cancelRequest(handle, request);
        return BTC_OK;
    });
}

btc_status btc_read_device_catalog(btc_cache* cache, btc_catalog_reader reader, void* opaque) {
    return status([&] {
        if (!reader) {
            throw std::invalid_argument("Missing BlockTreeCache catalog reader");
        }
        std::vector<std::pair<std::string, uint32_t>> rows;
        uint64_t version = 0;
        auto& handle = require(cache);
        {
            std::lock_guard lock(handle.api_mutex);
            if (!handle.native) {
                return BTC_CLOSED;
            }
            handle.flush();
            handle.native->visitTreeLocked([&](const BlockTree& tree) {
                version = tree.opaqueCatalogVersion();
                tree.visitNodeRangeLocked(0, tree.size(), tree.size(), [&](const TreeNode& node) {
                    if (!node.opaque) {
                        return;
                    }
                    bool empty = true;
                    for (size_t group = 0; group < handle.groups.size(); ++group) {
                        if (node.opaque->device_groups[group]) {
                            rows.emplace_back(node.opaque->key, handle.groups[group].id);
                            empty = false;
                        }
                    }
                    if (empty) {
                        rows.emplace_back(node.opaque->key, BTC_NO_DEVICE_GROUP);
                    }
                });
            });
        }
        std::vector<btc_catalog_item> output;
        output.reserve(rows.size());
        for (const auto& [key, group] : rows) {
            output.push_back({bytes(key), group});
        }
        reader(opaque, version, output.data(), output.size());
        return BTC_OK;
    });
}

btc_status btc_track_device_changes(btc_cache* cache, int enabled) {
    return locked(cache, [&](btc_cache& handle) {
        if (!handle.native) {
            return BTC_CLOSED;
        }
        handle.flush();
        handle.native->visitTreeLocked([&](const BlockTree&) {
            handle.native->tree()->trackOpaqueDeviceChanges(enabled != 0);
        });
        return BTC_OK;
    });
}

btc_status btc_read_device_changes(btc_cache* cache, btc_device_change_reader reader, void* opaque) {
    return status([&] {
        if (!reader) {
            throw std::invalid_argument("Missing BlockTreeCache change reader");
        }
        auto& handle = require(cache);
        std::vector<OpaqueDeviceChange> changes;
        std::vector<btc_device_change> output;
        {
            std::lock_guard lock(handle.api_mutex);
            if (!handle.native) {
                return BTC_CLOSED;
            }
            handle.flush();
            handle.native->visitTreeLocked([&](const BlockTree&) {
                changes = handle.native->tree()->takeOpaqueDeviceChanges();
            });
            output.reserve(changes.size());
            for (const auto& change : changes) {
                output.push_back({bytes(change.key), handle.groups[change.group].id, change.added});
            }
        }
        reader(opaque, output.data(), output.size());
        return BTC_OK;
    });
}

btc_status btc_device_stats(btc_cache* cache, size_t* keys, size_t* groups) {
    return locked(cache, [&](btc_cache& handle) {
        auto& key_count = require(keys);
        auto& group_count = require(groups);
        if (!handle.native) {
            return BTC_CLOSED;
        }
        handle.flush();
        handle.native->visitTreeLocked([&](const BlockTree& tree) {
            key_count = tree.size();
            group_count = tree.opaqueDeviceGroups();
        });
        return BTC_OK;
    });
}

btc_status btc_enable_prefix_scan(btc_cache* cache) {
    // The native tree's ordinary node pool already supplies the scan index.
    return locked(cache, [](btc_cache& handle) { return handle.native ? BTC_OK : BTC_CLOSED; });
}

namespace {
struct ScanNode {
    std::string key;
    uint32_t tiers;
    uint32_t activity;
};

ScanNode scanNode(const TreeNode& node, size_t group) {
    const auto& resource = node.group_set_resources[group];
    const auto tiers = (resource.hasTier(Tier::DEVICE) ? BTC_ON_DEVICE : 0)
                     | (resource.hasTier(Tier::HOST) ? BTC_ON_HOST : 0)
                     | (resource.hasTier(Tier::DISK) ? BTC_ON_DISK : 0);
    const auto state = resource.transfer_state;
    const auto activity = state == GroupSetTransferState::IDLE ? BTC_IDLE
                        : state == GroupSetTransferState::EVICTING ? BTC_EVICTING : BTC_LOADING;
    return {std::string(node.key().bytes()), static_cast<uint32_t>(tiers), static_cast<uint32_t>(activity)};
}
}  // namespace

btc_status btc_scan_prefix(btc_cache* cache, size_t cursor, size_t end, size_t budget,
                           size_t* tree_size, size_t* visited, btc_prefix_issue_fn report, void* opaque) {
    return status([&] {
        if (budget && !report) {
            throw std::invalid_argument("Missing BlockTreeCache prefix scan visitor");
        }
        auto& size = require(tree_size);
        auto& count = require(visited);
        auto& handle = require(cache);
        struct Issue {
            uint32_t group;
            ScanNode parent;
            ScanNode child;
        };
        std::vector<Issue> issues;
        std::vector<FullViolationDetail> violations;
        {
            std::lock_guard lock(handle.api_mutex);
            if (!handle.native) {
                return BTC_CLOSED;
            }
            handle.native->visitTreeLocked([&](const BlockTree& tree) {
                const auto range = tree.visitNodeRangeLocked(cursor, end, budget, [&](const TreeNode& node) {
                    violations.clear();
                    detectNodeViolations(tree, node, violations);
                    for (const auto& issue : violations) {
                        if (issue.type != FullViolationType::INVALID_RESOURCE) {
                            const auto group = issue.group_set_id;
                            const auto* previous = tree.previousGroupPage(&node, group);
                            issues.push_back({handle.groups[group].id, scanNode(*previous, group),
                                              scanNode(node, group)});
                        }
                    }
                });
                size = range.tree_size;
                count = range.visited;
            });
        }
        for (const auto& issue : issues) {
            const auto& parent = issue.parent;
            const auto& child = issue.child;
            report(opaque, issue.group, {bytes(parent.key), parent.tiers, parent.activity},
                   {bytes(child.key), child.tiers, child.activity});
        }
        return BTC_OK;
    });
}

