#include <algorithm>
#include <array>
#include <cstdio>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {
namespace {

struct Key {
    const char* data;
    size_t      size;

    std::string_view view() const {
        return {data, size};
    }
};

struct Binding {
    size_t  group;
    size_t  row;
    int     tier;
    int32_t slot;
};

struct DevicePublication {
    size_t   group;
    size_t   row;
    int32_t  slot;
    uint64_t generation;
};

struct DeviceLoadTarget {
    size_t   group;
    size_t   row;
    int      source_tier;
    int32_t  source_slot;
    int32_t  target_slot;
    uint64_t generation;
};

struct JoinBinding {
    size_t   group;
    size_t   row;
    int      kind;
    int      tier;
    int32_t  slot;
    uint64_t generation;
    uint64_t operation;
};

struct DeviceResidency {
    size_t   group;
    int32_t  slot;
    uint64_t generation;

    bool operator<(const DeviceResidency& other) const {
        return std::tie(group, slot, generation) < std::tie(other.group, other.slot, other.generation);
    }
};

struct ReclaimView {
    uint64_t               transaction;
    const DeviceResidency* selected;
    size_t                 count;
};

struct ReclaimPlan {
    uint64_t                     transaction{0};
    std::vector<DeviceResidency> selected;

    ReclaimView view() const {
        return {transaction, selected.data(), selected.size()};
    }
};

struct DiskResidency {
    size_t  group;
    int32_t slot;

    bool operator<(const DiskResidency& other) const {
        return std::tie(group, slot) < std::tie(other.group, other.slot);
    }
};

struct DiskReclaimView {
    bool                 has_victim;
    DiskResidency        victim;
    const DiskResidency* order;
    size_t               count;
};

struct DiskReclaimPlan {
    std::optional<DiskResidency> victim;
    std::vector<DiskResidency>   order;

    DiskReclaimView view() const {
        return {victim.has_value(), victim.value_or(DiskResidency{}), order.data(), order.size()};
    }
};

#include "rtp_llm/cpp/cache/block_tree_cache/external/ExternalDiskProtocol.h"

struct Endpoint {
    size_t                       tokens;
    const ExternalMatchPosition* needs;
    size_t                       count;
};

struct MatchView {
    size_t         device_tokens;
    size_t         total_tokens;
    const Binding* device_bindings;
    size_t         num_device_bindings;
    const Binding* bindings;
    size_t         num_bindings;
};

struct JoinMatchView {
    size_t             device_tokens;
    size_t             total_tokens;
    const JoinBinding* device_bindings;
    size_t             num_device_bindings;
    const JoinBinding* bindings;
    size_t             num_bindings;
};

int externalTier(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            return 0;
        case Tier::HOST:
            return 1;
        case Tier::DISK:
            return 2;
        default:
            throw std::logic_error("unexpected native tier in external match");
    }
}

struct MatchPlan {
    size_t               device_tokens{0};
    size_t               total_tokens{0};
    std::vector<Binding> device_bindings;
    std::vector<Binding> bindings;

    explicit MatchPlan(const ExternalBlockTreeMatchResult& result):
        device_tokens(result.device_tokens), total_tokens(result.total_tokens) {
        auto convert = [](const ExternalMatchBinding& binding) {
            return Binding{binding.group, binding.row, externalTier(binding.tier), binding.slot};
        };
        device_bindings.reserve(result.device_bindings.size());
        bindings.reserve(result.bindings.size());
        std::transform(result.device_bindings.begin(),
                       result.device_bindings.end(),
                       std::back_inserter(device_bindings),
                       convert);
        std::transform(result.bindings.begin(), result.bindings.end(), std::back_inserter(bindings), convert);
    }

    MatchView view() const {
        return {device_tokens,
                total_tokens,
                device_bindings.data(),
                device_bindings.size(),
                bindings.data(),
                bindings.size()};
    }
};

struct JoinMatchPlan {
    size_t                   device_tokens;
    size_t                   total_tokens;
    std::vector<JoinBinding> device_bindings;
    std::vector<JoinBinding> bindings;

    explicit JoinMatchPlan(const ExternalBlockTreeMatchResult& result):
        device_tokens(result.device_tokens),
        total_tokens(result.total_tokens),
        device_bindings(convert(result.device_bindings)),
        bindings(convert(result.bindings)) {}

    JoinMatchView view() const {
        return {device_tokens,
                total_tokens,
                device_bindings.data(),
                device_bindings.size(),
                bindings.data(),
                bindings.size()};
    }

private:
    static std::vector<JoinBinding> convert(const std::vector<ExternalMatchBinding>& bindings) {
        std::vector<JoinBinding> result;
        result.reserve(bindings.size());
        for (const auto& binding : bindings) {
            result.push_back({binding.group,
                              binding.row,
                              static_cast<int>(binding.kind),
                              externalTier(binding.tier),
                              binding.slot,
                              binding.generation,
                              binding.operation});
        }
        return result;
    }
};

// vLLM owns the worker buffers and protects their IDs. This is the tree-side
// catalog for acknowledged pages, including lower-tier copies shadowed by GPU.
#include "rtp_llm/cpp/cache/block_tree_cache/external/ExternalHostTypes.inc"

class ExternalBlockTreeDriver {
public:
    explicit ExternalBlockTreeDriver(size_t groups): groups_(groups), cache_(createCache(groups)) {
        device_heaps_.reserve(groups);
        disk_heaps_.reserve(groups);
        for (size_t group = 0; group < groups; ++group) {
            device_heaps_.emplace_back(EvictionPolicy::LRU);
            disk_heaps_.emplace_back(EvictionPolicy::LRU);
        }
    }

#include "rtp_llm/cpp/cache/block_tree_cache/external/ExternalHostOps.inc"

    void publish(const Key* keys, size_t count, const Binding* bindings, size_t num_bindings) {
        const std::lock_guard lock(mutex_);
        for (size_t i = 0; i < num_bindings; ++i) {
            const auto& binding = bindings[i];
            checkPublication(binding.group, binding.row, binding.tier, binding.slot, count);
            if (binding.tier == kDiskTier && disk_capacity_ != 0) {
                throw std::logic_error("managed BTC Disk pages require transaction publication");
            }
            if (binding.tier == kHostTier && host_pool_) {
                throw std::logic_error("managed BTC Host rows require a store ticket");
            }
            if (binding.tier == kDeviceTier
                && (device_slots_.count(binding.slot) != 0
                    || hasGuardedProjection(binding.group, keys[binding.row].view())
                    || isLoadingProjection(binding.group, keys[binding.row].view()))) {
                throw std::logic_error("guarded BTC device publication requires a generation");
            }
        }
        publishLocked(keys, bindings, num_bindings);
    }

    void publishDevice(const Key* keys, size_t count, const DevicePublication* bindings, size_t num_bindings) {
        const std::lock_guard                          lock(mutex_);
        std::map<BlockIdxType, DevicePublication>      pending;
        std::set<std::pair<size_t, std::string_view>> projections;
        std::vector<Binding>                           publications;
        publications.reserve(num_bindings);
        for (size_t i = 0; i < num_bindings; ++i) {
            const auto& binding = bindings[i];
            checkPublication(binding.group, binding.row, kDeviceTier, binding.slot, count);
            if (binding.generation == 0 || binding.slot == 0) {
                throw std::invalid_argument("guarded BTC device publications need a nonzero slot and generation");
            }
            const auto name    = keys[binding.row].view();
            const auto current = device_slots_.find(binding.slot);
            if (current != device_slots_.end()) {
                const auto& slot = current->second;
                if (binding.generation < slot.generation
                    || (binding.generation == slot.generation && slot.node == nullptr)) {
                    continue;  // A delayed publication cannot resurrect a retired slot.
                }
                if (slot.node != nullptr) {
                    const auto key = key_ids_.find(name);
                    if (slot.load_operation == 0 && binding.generation == slot.generation && binding.group == slot.group
                        && key != key_ids_.end() && key->second == slot.node->cache_key) {
                        continue;  // Retry of the current publication.
                    }
                    throw std::logic_error("BTC device slot still has a live owner");
                }
            }
            const auto [position, inserted] = pending.emplace(binding.slot, binding);
            if (!inserted) {
                const auto& previous = position->second;
                if (binding.group == previous.group && binding.row == previous.row
                    && binding.generation == previous.generation) {
                    continue;
                }
                throw std::invalid_argument("BTC device publication gives one slot multiple owners");
            }
            if (hasGuardedProjection(binding.group, name) || isLoadingProjection(binding.group, name)
                || !projections.emplace(binding.group, name).second) {
                throw std::logic_error("BTC device projection already has a guarded copy");
            }
            publications.push_back({binding.group, binding.row, kDeviceTier, binding.slot});
        }
        publishLocked(keys, publications.data(), publications.size());
        for (const auto& publication : publications) {
            auto* node = slots_.at({publication.group, kDeviceTier, publication.slot});
            promoteDevice(node, publication.group, publication.slot, pending.at(publication.slot).generation);
        }
    }

    void unpublish(size_t group, int tier, BlockIdxType slot) {
        const std::lock_guard lock(mutex_);
        validate(group, tier, slot);
        if (tier == kHostTier && host_pool_) {
            throw std::logic_error("managed BTC Host rows are reclaimed by the tree");
        }
        const auto current = slots_.find({group, tier, slot});
        if (current == slots_.end()) {
            return;
        }
        if (tier == kDiskTier && disk_capacity_ != 0) {
            throw std::logic_error("managed BTC Disk pages require transaction retirement");
        }
        if (tier == kDeviceTier && device_slots_.count(slot) != 0) {
            throw std::logic_error("guarded BTC device unpublish requires a generation");
        }
        eraseSlot(current);
    }

    void unpublishDevice(DeviceResidency residency) {
        const std::lock_guard lock(mutex_);
        checkResidency(residency);
        if (currentDevice(residency) != nullptr) {
            eraseSlot(slots_.find({residency.group, kDeviceTier, residency.slot}));
        }
    }

    void beginDeviceLoad(const Key*              keys,
                         size_t                  count,
                         const DeviceLoadTarget* targets,
                         size_t                  target_count,
                         uint64_t                operation) {
        const std::lock_guard lock(mutex_);
        if (operation == 0 || target_count == 0) {
            throw std::invalid_argument("BTC load requires a nonzero operation and at least one target");
        }
        const auto existing = loads_.find(operation);
        const auto path     = cache_->tree()->findNode(knownKeys(keys, count));
        if (existing != loads_.end()) {
            if (existing->second.size() != target_count) {
                throw std::logic_error("BTC load operation already has different targets");
            }
            for (size_t i = 0; i < target_count; ++i) {
                const auto& target = targets[i];
                if (target.row >= path.size() || !existing->second[i].sameTarget(path[target.row], target)) {
                    throw std::logic_error("BTC load operation already has different targets");
                }
            }
            return;
        }
        if (operation <= last_load_operation_) {
            throw std::logic_error("BTC load operation has already retired");
        }

        std::vector<PendingLoad> pending;
        std::set<Projection>    projections;
        std::set<BlockIdxType>  target_slots;
        pending.reserve(target_count);
        for (size_t i = 0; i < target_count; ++i) {
            const auto& target = targets[i];
            checkResidency({target.group, target.target_slot, target.generation});
            validate(target.group, target.source_tier, target.source_slot);
            if ((target.source_tier != 1 && target.source_tier != 2) || target.row >= path.size()) {
                throw std::invalid_argument("BTC load target needs a valid Host or Disk source row");
            }
            auto*            node = path[target.row];
            const Projection projection{node->cache_key, target.group};
            const auto       copy = copies_.find(projection);
            if (copy == copies_.end() || copy->second.slots[target.source_tier] != target.source_slot
                || copy->second.slots[kDeviceTier] != NULL_BLOCK_IDX) {
                throw std::logic_error("BTC load source changed or GPU became ready");
            }
            const auto slot = device_slots_.find(target.target_slot);
            if ((slot != device_slots_.end()
                 && (slot->second.node != nullptr || target.generation <= slot->second.generation))
                || slots_.count({target.group, kDeviceTier, target.target_slot}) != 0) {
                throw std::logic_error("BTC load target GPU slot already has an owner or a newer generation");
            }
            if (loading_.count(projection) != 0 || !projections.insert(projection).second
                || !target_slots.insert(target.target_slot).second) {
                throw std::logic_error("BTC load already has a producer for this projection or physical target");
            }
            pending.push_back({node, target});
        }

        auto& registered = loads_.emplace(operation, std::move(pending)).first->second;
        for (const auto& record : registered) {
            const auto& target = record.target;
            device_slots_[target.target_slot] =
                DeviceSlot{record.node, target.group, target.generation, 0, operation};
            loading_.emplace(Projection{record.node->cache_key, target.group},
                             ExternalJoinTarget{target.target_slot, target.generation, operation});
        }
        last_load_operation_ = operation;
    }

    bool settleDeviceLoad(uint64_t operation, bool success) {
        const std::lock_guard lock(mutex_);
        const auto            load = loads_.find(operation);
        if (load == loads_.end()) {
            return false;
        }
        for (const auto& record : load->second) {
            const auto& target = record.target;
            const auto& slot   = device_slots_.at(target.target_slot);
            if (slot.node != record.node || slot.group != target.group || slot.generation != target.generation
                || slot.load_operation != operation) {
                throw std::logic_error("BTC load operation lost its registered target generation");
            }
        }
        for (const auto& record : load->second) {
            const auto&      target = record.target;
            const Projection projection{record.node->cache_key, target.group};
            loading_.erase(projection);
            if (success) {
                auto& copy = copies_[projection];
                copy.slots[kDeviceTier] = target.target_slot;
                slots_.emplace(SlotKey{target.group, kDeviceTier, target.target_slot}, record.node);
                promoteDevice(record.node, target.group, target.target_slot, target.generation);
                refresh(record.node, target.group, copy);
                continue;
            }
            auto& slot = device_slots_.at(target.target_slot);
            slot.node           = nullptr;
            slot.load_operation = 0;
            const auto copy = copies_.find(projection);
            if (copy == copies_.end()) {
                refresh(record.node, target.group, Copies{});
                pruneEmptyNode(record.node);
            } else {
                refresh(record.node, target.group, copy->second);
            }
        }
        loads_.erase(load);
        return true;
    }

    std::vector<DeviceResidency> selectDeviceWritebacks(const DeviceResidency* eligible, size_t count, size_t limit) {
        const std::lock_guard        lock(mutex_);
        std::vector<DeviceResidency> selected;
        std::set<DeviceResidency>    seen;
        selected.reserve(std::min(count, limit));
        for (size_t i = 0; i < count; ++i) {
            const auto& residency = eligible[i];
            validate(residency.group, kDeviceTier, residency.slot);
            if (residency.slot == 0) {
                throw std::invalid_argument("BTC device writeback needs a nonzero slot");
            }
            if (selected.size() == limit) {
                continue;
            }

            TreeNode* node;
            if (residency.generation == 0) {
                // Ordinary publishes have no generation. Never let an unguarded
                // offer alias a slot tracked by the guarded publication path.
                const auto published = slots_.find({residency.group, kDeviceTier, residency.slot});
                if (published == slots_.end() || device_slots_.count(residency.slot) != 0) {
                    continue;
                }
                node = published->second;
            } else {
                const auto* slot = currentDevice(residency);
                if (slot == nullptr || slot->transaction != 0) {
                    continue;
                }
                node = slot->node;
            }
            const auto& copies = copies_.at({node->cache_key, residency.group});
            if (copies.slots[kHostTier] == NULL_BLOCK_IDX && copies.slots[kDiskTier] == NULL_BLOCK_IDX
                && seen.insert(residency).second) {
                selected.push_back(residency);
            }
        }
        return selected;
    }

    ReclaimPlan planDeviceReclaim(const DeviceResidency* eligible, size_t count, size_t limit) {
        const std::lock_guard     lock(mutex_);
        std::set<DeviceResidency> allowed;
        for (size_t i = 0; i < count; ++i) {
            checkResidency(eligible[i]);
            const auto* slot = currentDevice(eligible[i]);
            if (slot != nullptr && slot->transaction == 0) {
                allowed.insert(eligible[i]);
            }
        }
        ReclaimPlan plan;
        plan.selected.reserve(std::min(limit, allowed.size()));
        while (plan.selected.size() < limit && !allowed.empty()) {
            std::optional<EvictionEntry> best;
            size_t                       best_group = 0;
            for (size_t group = 0; group < groups_; ++group) {
                auto candidate = device_heaps_[group].best([&](TreeNode* node) {
                    const auto& resource = node->group_set_resources[group];
                    return resource.hasCompleteDeviceValue()
                           && allowed.count(deviceIdentity(group, resource.device_blocks.front())) != 0;
                });
                if (candidate
                    && (!best
                        || std::tie(candidate->primary_key, candidate->secondary_key, candidate->cache_key, group)
                            < std::tie(best->primary_key, best->secondary_key, best->cache_key, best_group))) {
                    best = candidate;
                    best_group = group;
                }
            }
            if (!best) {
                break;
            }
            const auto& resource = best->node->group_set_resources[best_group];
            const auto  residency = deviceIdentity(best_group, resource.device_blocks.front());
            plan.selected.push_back(residency);
            allowed.erase(residency);
        }
        if (plan.selected.empty()) {
            return plan;
        }
        if (next_transaction_ == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("BTC device reclaim transaction IDs exhausted");
        }
        plan.transaction = next_transaction_++;
        reclaims_.emplace(plan.transaction, plan.selected);
        for (const auto& residency : plan.selected) {
            auto& slot = device_slots_.at(residency.slot);
            slot.transaction = plan.transaction;
            refresh(slot.node, slot.group, copies_.at({slot.node->cache_key, slot.group}));
        }
        return plan;
    }

    void settleDeviceReclaim(uint64_t transaction, const DeviceResidency* released, size_t count) {
        const std::lock_guard lock(mutex_);
        const auto            plan = reclaims_.find(transaction);
        if (plan == reclaims_.end()) {
            return;  // Includes empty transaction and a retried acknowledgement.
        }
        std::set<DeviceResidency> confirmed;
        for (size_t i = 0; i < count; ++i) {
            checkResidency(released[i]);
            confirmed.insert(released[i]);
        }
        for (const auto& residency : plan->second) {
            auto* slot = currentDevice(residency);
            if (slot == nullptr || slot->transaction != transaction) {
                continue;
            }
            if (confirmed.count(residency) != 0) {
                eraseSlot(slots_.find({residency.group, kDeviceTier, residency.slot}));
                continue;
            }
            slot->transaction = 0;
            refresh(slot->node, slot->group, copies_.at({slot->node->cache_key, slot->group}));
        }
        reclaims_.erase(plan);
    }

    void touchDisk(const DiskResidency* pages, size_t count) {
        const std::lock_guard lock(mutex_);
        std::vector<TreeNode*> nodes;
        nodes.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& page = pages[i];
            validate(page.group, kDiskTier, page.slot);
            const auto slot = slots_.find({page.group, kDiskTier, page.slot});
            if (slot == slots_.end()) {
                throw std::logic_error("cannot touch an unpublished BTC Disk page");
            }
            nodes.push_back(slot->second);
        }
        for (size_t i = 0; i < count; ++i) {
            recordDiskAccess(nodes[i], pages[i].group);
        }
    }

    DiskReclaimPlan reclaimDisk(const DiskResidency* protected_pages, size_t count, bool snapshot) {
        const std::lock_guard   lock(mutex_);
        if (disk_capacity_ != 0) {
            throw std::logic_error("managed BTC Disk pages require transaction reclaim");
        }
        std::set<DiskResidency> protected_slots;
        for (size_t i = 0; i < count; ++i) {
            const auto& page = protected_pages[i];
            validate(page.group, kDiskTier, page.slot);
            protected_slots.insert(page);
        }
        auto plan = selectDiskVictimLocked(protected_slots, snapshot);
        if (plan.victim) {
            // Withdraw the page before the owner returns the rank-local slots.
            eraseSlot(slots_.find({plan.victim->group, kDiskTier, plan.victim->slot}));
        }
        return plan;
    }

private:
    DiskReclaimPlan selectDiskVictimLocked(const std::set<DiskResidency>& protected_slots, bool snapshot) const {
        using Candidate = std::pair<EvictionEntry, size_t>;
        auto older = [](const Candidate& left, const Candidate& right) {
            const auto& [a, a_group] = left;
            const auto& [b, b_group] = right;
            return std::tie(a.primary_key, a.secondary_key, a.cache_key, a_group)
                 < std::tie(b.primary_key, b.secondary_key, b.cache_key, b_group);
        };
        std::optional<Candidate> best;
        std::vector<Candidate>   ordered;
        for (size_t group = 0; group < groups_; ++group) {
            const auto& heap = disk_heaps_[group];
            const auto candidate = heap.best([&](TreeNode* node) {
                const auto page = diskResidency(node, group);
                return protected_slots.count(page) == 0 && disk_load_refs_.count(page) == 0;
            });
            if (candidate && (!best || older({*candidate, group}, *best))) {
                best.emplace(*candidate, group);
            }
            if (snapshot) {
                for (const auto& entry : heap.orderedEntries()) {
                    ordered.emplace_back(entry, group);
                }
            }
        }
        DiskReclaimPlan plan;
        if (snapshot) {
            std::sort(ordered.begin(), ordered.end(), older);
            plan.order.reserve(ordered.size());
            for (const auto& [entry, group] : ordered) {
                plan.order.push_back(diskResidency(entry.node, group));
            }
        }
        if (best) {
            plan.victim = diskResidency(best->first.node, best->second);
        }
        return plan;
    }

    void publishLocked(const Key* keys, const Binding* bindings, size_t num_bindings) {
        size_t                last_row = 0;
        for (size_t i = 0; i < num_bindings; ++i) {
            last_row = std::max(last_row, bindings[i].row + 1);
        }
        for (size_t i = 0; i < num_bindings; ++i) {
            const auto& binding = bindings[i];
            const auto  current = slots_.find({binding.group, binding.tier, binding.slot});
            if (current != slots_.end()) {
                const auto key = key_ids_.find(keys[binding.row].view());
                if (key == key_ids_.end() || key->second != current->second->cache_key) {
                    eraseSlot(current);
                }
            }
        }
        if (last_row == 0) {
            return;
        }

        CacheKeysType key_ids;
        key_ids.reserve(last_row);
        for (size_t i = 0; i < last_row; ++i) {
            key_ids.push_back(intern(keys[i].view()));
        }
        const auto path = cache_->tree()->ensureExternalPath(key_ids);
        for (size_t i = 0; i < num_bindings; ++i) {
            const auto& binding = bindings[i];
            auto*       node    = path[binding.row];
            auto&       copies  = copies_[{node->cache_key, binding.group}];
            auto&       previous = copies.slots[binding.tier];
            const bool  new_disk = binding.tier == kDiskTier && previous != binding.slot;
            if (previous != NULL_BLOCK_IDX && previous != binding.slot) {
                slots_.erase({binding.group, binding.tier, previous});
            }
            previous = binding.slot;
            slots_.insert_or_assign({binding.group, binding.tier, binding.slot}, node);
            refresh(node, binding.group, copies);
            if (new_disk) {
                recordDiskAccess(node, binding.group);
            }
        }
    }

public:
    ExternalBlockTreeMatchResult match(
        const Key* keys, size_t count, const Endpoint* endpoints, size_t endpoint_count, bool include_join) {
        const std::lock_guard lock(mutex_);
        std::vector<ExternalMatchEndpoint> requirements;
        requirements.reserve(endpoint_count);
        size_t previous = 0;
        for (size_t i = 0; i < endpoint_count; ++i) {
            const auto& endpoint = endpoints[i];
            if (endpoint.tokens <= previous) {
                throw std::invalid_argument("BTC match endpoints must be in increasing token order");
            }
            previous = endpoint.tokens;
            auto& requirement = requirements.emplace_back(ExternalMatchEndpoint{endpoint.tokens, {}});
            requirement.needs.reserve(endpoint.count);
            for (size_t need = 0; need < endpoint.count; ++need) {
                requirement.needs.push_back(endpoint.needs[need]);
            }
        }
        ExternalJoinLookup lookup;
        if (include_join) {
            lookup = [this](const TreeNode& node, size_t group) -> std::optional<ExternalJoinTarget> {
                const auto match = loading_.find({node.cache_key, group});
                return match == loading_.end() ? std::nullopt : std::make_optional(match->second);
            };
        }
        auto result = cache_->match(knownKeys(keys, count), requirements, ExternalMatchMode::BORROWED_SLOTS, lookup);
        touchMatchedDevice(result.bindings);
        if (include_join) {
            fillDeviceGenerations(result.device_bindings);
            fillDeviceGenerations(result.bindings);
        }
        return result;
    }

    void reset() {
        const std::lock_guard lock(mutex_);
        if (!host_stores_.empty() || !host_loads_.empty() || !disk_jobs_.empty() || !disk_reads_.empty()) {
            throw std::logic_error("BTC tree reset requires every transfer to settle");
        }
        auto cache = createCache(groups_);
        resetDiskLocked();
        resetHostLocked();
        for (auto& heap : device_heaps_) {
            heap = EvictionHeap(EvictionPolicy::LRU);
        }
        for (auto& heap : disk_heaps_) {
            heap = EvictionHeap(EvictionPolicy::LRU);
        }
        for (auto& [id, slot] : device_slots_) {
            slot.node        = nullptr;
            slot.transaction = 0;
            slot.load_operation = 0;
        }
        reclaims_.clear();
        loading_.clear();
        loads_.clear();
        slots_.clear();
        copies_.clear();
        key_names_.clear();
        key_ids_.clear();
        next_key_ = 1;
        cache_ = std::move(cache);
    }

private:
    static constexpr int kDeviceTier = 0;
    static constexpr int kHostTier   = 1;
    static constexpr int kDiskTier   = 2;
    using KeyIds  = std::map<std::string, CacheKeyType, std::less<>>;
    using SlotKey = std::tuple<size_t, int, BlockIdxType>;
    using Slots   = std::map<SlotKey, TreeNode*>;
    using Projection = std::pair<CacheKeyType, size_t>;

    struct Copies {
        std::array<BlockIdxType, 3> slots{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX};

        bool empty() const {
            return std::all_of(slots.begin(), slots.end(), isNullBlockIdx);
        }
    };

    struct DeviceSlot {
        TreeNode* node{nullptr};
        size_t    group{0};
        uint64_t  generation{0};
        uint64_t  transaction{0};
        uint64_t  load_operation{0};
    };

    struct PendingLoad {
        TreeNode*        node;
        DeviceLoadTarget target;

        bool sameTarget(const TreeNode* other_node, const DeviceLoadTarget& other) const {
            return node == other_node && target.group == other.group && target.source_tier == other.source_tier
                   && target.source_slot == other.source_slot && target.target_slot == other.target_slot
                   && target.generation == other.generation;
        }
    };

#include "rtp_llm/cpp/cache/block_tree_cache/external/ExternalDiskTransactions.inc"

    static std::unique_ptr<BlockTreeCache> createCache(size_t groups) {
        BlockTreeCacheConfig config;
        config.full_prefix_scan_interval_ms = 0;
        return std::make_unique<BlockTreeCache>(
            std::make_unique<BlockTree>(BlockTree::ExternalGroups{groups}),
            config,
            nullptr,
            nullptr,
            std::make_unique<BlockTreeTaskPool>(1, 1, "external_btc"),
            std::make_shared<BlockTreeCacheMetricsReporter>(nullptr, /*enable_stage_metrics=*/false));
    }

    void validate(size_t group, int tier, BlockIdxType slot) const {
        if (group >= groups_ || tier < 0 || tier > 2 || slot < 0) {
            throw std::invalid_argument("BTC publication has an invalid group, tier or slot");
        }
    }

    void checkPublication(size_t group, size_t row, int tier, BlockIdxType slot, size_t key_count) const {
        validate(group, tier, slot);
        if (row >= key_count) {
            throw std::invalid_argument("BTC published row is outside the hash path");
        }
    }

    void checkResidency(DeviceResidency residency) const {
        validate(residency.group, kDeviceTier, residency.slot);
        if (residency.slot == 0 || residency.generation == 0) {
            throw std::invalid_argument("BTC device residency requires a nonzero slot and generation");
        }
    }

    DeviceSlot* currentDevice(DeviceResidency residency) {
        auto entry = device_slots_.find(residency.slot);
        if (entry == device_slots_.end()) {
            return nullptr;
        }
        auto& slot = entry->second;
        return slot.node != nullptr && slot.load_operation == 0 && slot.group == residency.group
                   && slot.generation == residency.generation
                   ? &slot : nullptr;
    }

    void promoteDevice(TreeNode* node, size_t group, BlockIdxType id, uint64_t generation) {
        device_slots_[id] = DeviceSlot{node, group, generation, 0, 0};
        auto& meta = node->group_set_resources[group].candidate_meta;
        meta = {};
        meta.admission_seq   = ++device_clock_;
        meta.last_access_seq = device_clock_;
        device_heaps_[group].upsert(node, meta);
    }

    DeviceResidency deviceIdentity(size_t group, BlockIdxType physical_id) const {
        return {group, physical_id, device_slots_.at(physical_id).generation};
    }

    DiskResidency diskResidency(TreeNode* node, size_t group) const {
        return {group, copies_.at({node->cache_key, group}).slots[kDiskTier]};
    }

    void recordDiskAccess(TreeNode* node, size_t group) {
        CandidateMeta meta{};
        meta.admission_seq = meta.last_access_seq = ++disk_clock_;
        disk_heaps_[group].upsert(node, meta);
    }

    bool isLoadingProjection(size_t group, std::string_view name) const {
        const auto key = key_ids_.find(name);
        return key != key_ids_.end() && loading_.count({key->second, group}) != 0;
    }

    bool hasGuardedProjection(size_t group, std::string_view name) const {
        const auto key = key_ids_.find(name);
        if (key == key_ids_.end()) {
            return false;
        }
        const auto copy = copies_.find({key->second, group});
        if (copy == copies_.end()) {
            return false;
        }
        const auto slot = device_slots_.find(copy->second.slots[kDeviceTier]);
        return slot != device_slots_.end() && slot->second.node != nullptr && slot->second.group == group
               && slot->second.node->cache_key == key->second;
    }

    void touchMatchedDevice(const std::vector<ExternalMatchBinding>& bindings) {
        const auto access = ++device_clock_;
        for (const auto& binding : bindings) {
            if (binding.kind != ExternalBindingKind::READY) {
                continue;
            }
            const auto slot = device_slots_.find(binding.slot);
            if (slot == device_slots_.end() || slot->second.node == nullptr || slot->second.group != binding.group) {
                continue;
            }
            auto* node = slot->second.node;
            auto& meta = node->group_set_resources[binding.group].candidate_meta;
            meta.last_access_seq = access;
            ++meta.hit_count;
            device_heaps_[binding.group].updateIfPresent(node, meta);
        }
    }

    void fillDeviceGenerations(std::vector<ExternalMatchBinding>& bindings) const {
        for (auto& binding : bindings) {
            if (binding.kind != ExternalBindingKind::READY) {
                continue;
            }
            const auto slot = device_slots_.find(binding.slot);
            if (slot != device_slots_.end() && slot->second.node != nullptr && slot->second.load_operation == 0
                && slot->second.group == binding.group) {
                binding.generation = slot->second.generation;
            }
        }
    }

    CacheKeysType knownKeys(const Key* keys, size_t count) const {
        CacheKeysType ids;
        ids.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto key = key_ids_.find(keys[i].view());
            if (key == key_ids_.end()) {
                break;
            }
            ids.push_back(key->second);
        }
        return ids;
    }

    CacheKeyType intern(std::string_view name) {
        auto entry = key_ids_.lower_bound(name);
        if (entry != key_ids_.end() && entry->first == name) {
            return entry->second;
        }
        if (next_key_ == std::numeric_limits<CacheKeyType>::max()) {
            throw std::overflow_error("BTC external key IDs exhausted");
        }
        const CacheKeyType id = next_key_++;
        entry = key_ids_.emplace_hint(entry, name, id);
        key_names_.emplace(id, entry);
        return id;
    }

    void refresh(TreeNode* node, size_t group, const Copies& copies) {
        auto&      resource = node->group_set_resources[group];
        const auto device_id = copies.slots[kDeviceTier];
        const auto device = device_slots_.find(device_id);
        const bool pending = device != device_slots_.end() && device->second.node == node
                             && device->second.group == group && device->second.transaction != 0;
        resource.device_blocks.clear();
        resource.host_block = NULL_BLOCK_IDX;
        resource.disk_block = NULL_BLOCK_IDX;
        resource.transfer_state = GroupSetTransferState::IDLE;
        if (device_id != NULL_BLOCK_IDX && !pending) {
            resource.device_blocks.push_back(device_id);
        } else if (copies.slots[1] != NULL_BLOCK_IDX) {
            resource.host_block = copies.slots[1];
        } else if (copies.slots[2] != NULL_BLOCK_IDX) {
            resource.disk_block = copies.slots[2];
        } else if (pending) {
            // Keep the node in the tree through a pending eviction with no lower copy.
            resource.transfer_state = GroupSetTransferState::EVICTING;
        } else if (loading_.count({node->cache_key, group}) != 0) {
            // A submitted job retains its physical source independently of the catalog.
            resource.transfer_state = GroupSetTransferState::LOADING;
        }
    }

    void eraseSlot(Slots::iterator slot) {
        const auto group = std::get<0>(slot->first);
        const auto tier  = std::get<1>(slot->first);
        const auto block = std::get<2>(slot->first);
        auto* node = slot->second;
        const bool managed_host = tier == kHostTier && host_pool_;
        if (managed_host) {
            beforeHostUnpublishDiskLocked(node, group, block);
            host_heaps_[group].erase(node);
        }
        if (tier == kDiskTier) {
            disk_heaps_[group].erase(node);
        }
        if (tier == kDeviceTier) {
            auto device = device_slots_.find(std::get<2>(slot->first));
            if (device != device_slots_.end() && device->second.node == node && device->second.group == group) {
                device_heaps_[group].erase(node);
                device->second.node        = nullptr;
                device->second.transaction = 0;
            }
        }
        auto copies = copies_.find({node->cache_key, group});
        copies->second.slots[tier] = NULL_BLOCK_IDX;
        refresh(node, group, copies->second);
        slots_.erase(slot);
        if (managed_host) {
            host_pool_->decTreeRef(block + 1, BlockTreeRefType::CACHE);
        }
        if (copies->second.empty()) {
            copies_.erase(copies);
            pruneEmptyNode(node);
        }
    }

    void pruneEmptyNode(TreeNode* node) {
        CacheKeysType removed;
        cache_->tree()->removeNodeAndEmptyAncestors(node, &removed);
        for (const auto key : removed) {
            const auto name = key_names_.find(key);
            key_ids_.erase(name->second);
            key_names_.erase(name);
        }
    }

    const size_t                                                groups_;
    std::mutex                                                  mutex_;
    std::unique_ptr<BlockTreeCache>                             cache_;
    Slots                                                       slots_;
    std::map<std::pair<CacheKeyType, size_t>, Copies>            copies_;
    KeyIds                                                      key_ids_;
    std::map<CacheKeyType, KeyIds::iterator>                     key_names_;
    CacheKeyType                                                next_key_{1};
    std::vector<EvictionHeap>                                   device_heaps_;
    std::vector<EvictionHeap>                                   disk_heaps_;
    // Retired IDs retain their last generation across resets; their count is bounded by the external pool.
    std::map<BlockIdxType, DeviceSlot>                          device_slots_;
    std::map<uint64_t, std::vector<DeviceResidency>>            reclaims_;
    // External loader follows the stock LoadJoinRegistry split between one producer
    // and independent request waiters. The vLLM scheduler owns the physical refs.
    std::map<Projection, ExternalJoinTarget>                    loading_;
    std::map<uint64_t, std::vector<PendingLoad>>                 loads_;
    uint64_t                                                   device_clock_{0};
    uint64_t                                                   disk_clock_{0};
    uint64_t                                                   next_transaction_{1};
    uint64_t                                                   last_load_operation_{0};
};

thread_local char last_error[512];

template<typename Fn, typename Result>
Result guard(Fn&& fn, Result failure) noexcept {
    try {
        return fn();
    } catch (const std::exception& error) {
        std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "unknown external BTC error");
    }
    return failure;
}

}  // namespace

extern "C" {

const char* btc_tree_last_error() {
    return last_error;
}

void* btc_tree_create(size_t groups) {
    return guard([&]() -> void* { return new ExternalBlockTreeDriver(groups); }, static_cast<void*>(nullptr));
}

void btc_tree_destroy(void* tree) {
    delete static_cast<ExternalBlockTreeDriver*>(tree);
}

int btc_tree_publish(void* tree, const Key* keys, size_t count, const Binding* bindings, size_t num_bindings) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->publish(keys, count, bindings, num_bindings);
            return 0;
        },
        -1);
}

int btc_tree_unpublish(void* tree, size_t group, int tier, int32_t slot) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->unpublish(group, tier, slot);
            return 0;
        },
        -1);
}

int btc_tree_configure_host(void* tree, size_t capacity) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->configureHost(capacity);
            return 0;
        },
        -1);
}

int btc_tree_prepare_host_store(void*                     tree,
                                const Key*                hashes,
                                size_t                    hash_count,
                                const HostStoreCandidate* candidates,
                                size_t                    candidate_count,
                                Key                       request_id,
                                HostStoreView*            view,
                                void**                    result) {
    return guard(
        [&] {
            auto plan = static_cast<ExternalBlockTreeDriver*>(tree)->prepareHostStore(
                hashes, hash_count, candidates, candidate_count, request_id);
            if (!plan) {
                return 0;
            }
            *view   = plan->view();
            *result = plan.release();
            return 1;
        },
        -1);
}

void btc_tree_destroy_host_store_plan(void* plan) {
    delete static_cast<HostStorePlan*>(plan);
}

int btc_tree_settle_host_store(void* tree, uint64_t ticket, bool success) {
    return guard(
        [&] { return static_cast<ExternalBlockTreeDriver*>(tree)->settleHostStore(ticket, success) ? 1 : 0; },
        -1);
}

int btc_tree_pin_host_load(void* tree, const HostResidency* pages, size_t count, uint64_t* ticket) {
    return guard(
        [&] {
            *ticket = static_cast<ExternalBlockTreeDriver*>(tree)->pinHostLoad(pages, count);
            return 0;
        },
        -1);
}

int btc_tree_release_host_load(void* tree, uint64_t ticket) {
    return guard(
        [&] { return static_cast<ExternalBlockTreeDriver*>(tree)->releaseHostLoad(ticket) ? 1 : 0; },
        -1);
}

int btc_tree_host_slot(void* tree, size_t group, Key hash, int32_t* out) {
    return guard(
        [&] {
            const auto slot = static_cast<ExternalBlockTreeDriver*>(tree)->hostSlot(group, hash);
            if (slot == NULL_BLOCK_IDX) {
                return 0;
            }
            *out = slot;
            return 1;
        },
        -1);
}

int btc_tree_reset_host(void* tree) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->resetHost();
            return 0;
        },
        -1);
}

int btc_tree_publish_device(void*                    tree,
                            const Key*               keys,
                            size_t                   count,
                            const DevicePublication* bindings,
                            size_t                   num_bindings) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->publishDevice(keys, count, bindings, num_bindings);
            return 0;
        },
        -1);
}

int btc_tree_unpublish_device(void* tree, size_t group, int32_t slot, uint64_t generation) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->unpublishDevice({group, slot, generation});
            return 0;
        },
        -1);
}

int btc_tree_begin_device_load(void*                   tree,
                               const Key*              keys,
                               size_t                  count,
                               const DeviceLoadTarget* targets,
                               size_t                  target_count,
                               uint64_t                operation) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->beginDeviceLoad(keys, count, targets, target_count, operation);
            return 0;
        },
        -1);
}

int btc_tree_settle_device_load(void* tree, uint64_t operation, bool success) {
    return guard(
        [&] { return static_cast<ExternalBlockTreeDriver*>(tree)->settleDeviceLoad(operation, success) ? 1 : 0; },
        -1);
}

int btc_tree_select_device_writebacks(void*                  tree,
                                      const DeviceResidency* eligible,
                                      size_t                 count,
                                      size_t                 limit,
                                      DeviceResidency*       output,
                                      size_t*                output_count) {
    return guard(
        [&] {
            if (output_count == nullptr || (count != 0 && eligible == nullptr)
                || (std::min(count, limit) != 0 && output == nullptr)) {
                throw std::invalid_argument("invalid BTC writeback selection buffer");
            }
            const auto selected =
                static_cast<ExternalBlockTreeDriver*>(tree)->selectDeviceWritebacks(eligible, count, limit);
            if (!selected.empty()) {
                std::copy(selected.begin(), selected.end(), output);
            }
            *output_count = selected.size();
            return 0;
        },
        -1);
}

int btc_tree_plan_device_reclaim(void*                  tree,
                                 const DeviceResidency* eligible,
                                 size_t                 count,
                                 size_t                 limit,
                                 ReclaimView*           view,
                                 void**                 result) {
    return guard(
        [&] {
            auto plan = std::make_unique<ReclaimPlan>();
            *plan = static_cast<ExternalBlockTreeDriver*>(tree)->planDeviceReclaim(eligible, count, limit);
            *view = plan->view();
            *result = plan.release();
            return 0;
        },
        -1);
}

void btc_tree_destroy_reclaim_plan(void* result) {
    delete static_cast<ReclaimPlan*>(result);
}

int btc_tree_settle_device_reclaim(void* tree, uint64_t transaction, const DeviceResidency* released, size_t count) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->settleDeviceReclaim(transaction, released, count);
            return 0;
        },
        -1);
}

#include "rtp_llm/cpp/cache/block_tree_cache/external/ExternalDiskBindings.inc"

int btc_tree_touch_disk(void* tree, const DiskResidency* pages, size_t count) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->touchDisk(pages, count);
            return 0;
        },
        -1);
}

int btc_tree_reclaim_disk(void*                tree,
                          const DiskResidency* protected_pages,
                          size_t               count,
                          bool                 snapshot,
                          DiskReclaimView*     view,
                          void**               result) {
    return guard(
        [&] {
            auto plan = std::make_unique<DiskReclaimPlan>();
            *plan = static_cast<ExternalBlockTreeDriver*>(tree)->reclaimDisk(protected_pages, count, snapshot);
            *view = plan->view();
            *result = plan.release();
            return 0;
        },
        -1);
}

void btc_tree_destroy_disk_reclaim(void* result) {
    delete static_cast<DiskReclaimPlan*>(result);
}

int btc_tree_match(void*            tree,
                   const Key*       keys,
                   size_t           count,
                   const Endpoint*  endpoints,
                   size_t           endpoint_count,
                   MatchView*       view,
                   void**           result) {
    return guard(
        [&] {
            auto plan = std::make_unique<MatchPlan>(static_cast<ExternalBlockTreeDriver*>(tree)->match(
                keys, count, endpoints, endpoint_count, /*include_join=*/false));
            *view = plan->view();
            *result = plan.release();
            return 0;
        },
        -1);
}

void btc_tree_destroy_match(void* result) {
    delete static_cast<MatchPlan*>(result);
}

int btc_tree_match_with_join(void*           tree,
                             const Key*      keys,
                             size_t          count,
                             const Endpoint* endpoints,
                             size_t          endpoint_count,
                             JoinMatchView*  view,
                             void**          result) {
    return guard(
        [&] {
            auto plan = std::make_unique<JoinMatchPlan>(static_cast<ExternalBlockTreeDriver*>(tree)->match(
                keys, count, endpoints, endpoint_count, /*include_join=*/true));
            *view = plan->view();
            *result = plan.release();
            return 0;
        },
        -1);
}

void btc_tree_destroy_join_match(void* result) {
    delete static_cast<JoinMatchPlan*>(result);
}

int btc_tree_reset(void* tree) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->reset();
            return 0;
        },
        -1);
}

}  // extern "C"
}  // namespace rtp_llm
