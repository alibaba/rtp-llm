#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

int64_t steadyTimeUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

using Validators = std::vector<std::unique_ptr<MatchValidator>>;

Validators makeValidators(const std::vector<GroupSetPtr>& group_sets) {
    Validators validators;
    validators.reserve(group_sets.size());
    for (const GroupSetPtr& group_set : group_sets) {
        validators.push_back(group_set->createMatchValidator());
    }
    return validators;
}

Validators cloneValidators(const Validators& source) {
    Validators validators;
    validators.reserve(source.size());
    for (const auto& validator : source) {
        validators.push_back(validator->clone());
    }
    return validators;
}

bool validateObservedResource(MatchValidator& validator, const GroupSetResource& resource, bool ignore_transfer_state) {
    if (!ignore_transfer_state && !resource.transfer_detached) {
        return validator.validate(resource);
    }
    GroupSetResource observed = resource;
    if (observed.transfer_detached) {
        // Detached resources are being pruned; pinned bytes cannot become a new cache hit.
        observed.device_blocks.clear();
        observed.host_block = NULL_BLOCK_IDX;
        observed.disk_block = NULL_BLOCK_IDX;
    }
    if (ignore_transfer_state) {
        // A temporary transfer reservation does not end retained cache lifetime.
        observed.transfer_state = GroupSetTransferState::IDLE;
        if (observed.hasTier(Tier::DEVICE) && !observed.hasCompleteDeviceValue()) {
            observed.device_blocks.clear();
        }
    }
    return validator.validate(observed);
}

}  // namespace

BlockTree::BlockTree(std::vector<GroupSetPtr> group_sets, size_t reuse_unit_tokens):
    group_sets_(std::move(group_sets)), reuse_unit_tokens_(reuse_unit_tokens) {
    RTP_LLM_CHECK_WITH_INFO(reuse_unit_tokens_ > 0, "BlockTree reuse unit must be positive");
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const auto& group_tags = group_sets_[group_set_id]->groupTags();
        for (size_t member_group_id = 0; member_group_id < group_tags.size(); ++member_group_id) {
            const auto& tag = group_tags[member_group_id];
            const bool  inserted =
                reusable_group_locations_.emplace(tag, ReusableGroupLocation{group_set_id, member_group_id}).second;
            RTP_LLM_CHECK_WITH_INFO(inserted, "duplicate BlockTree reusable tag=%s", tag.c_str());
        }
    }

    root_            = std::make_unique<TreeNode>();
    root_->cache_key = 0;
    root_->parent    = nullptr;
    root_->group_set_resources.resize(group_sets_.size());
}

const ReusableGroupLocation* BlockTree::reusableGroupLocation(std::string_view group_tag) const {
    const auto location_it = reusable_group_locations_.find(std::string(group_tag));
    return location_it == reusable_group_locations_.end() ? nullptr : &location_it->second;
}

BlockTreeNodeRangeResult BlockTree::visitNodeRangeLocked(size_t                                      cursor,
                                                         size_t                                      cycle_end_index,
                                                         size_t                                      max_nodes,
                                                         const std::function<void(const TreeNode&)>& visitor) const {
    BlockTreeNodeRangeResult result;
    result.tree_size = node_pool_.size();

    const size_t end   = std::min(cycle_end_index, node_pool_.size());
    size_t       index = cursor;
    while (index < end && result.visited < max_nodes) {
        visitor(*node_pool_[index]);
        ++index;
        ++result.visited;
    }
    result.next_cursor    = index;
    result.cycle_complete = index >= end;
    return result;
}

BlockTree::~BlockTree() {
    releaseNode(root_.get());
    for (const std::unique_ptr<TreeNode>& node : node_pool_) {
        releaseNode(node.get());
    }
    for (const auto& [_, node] : pending_node_pool_) {
        releaseNode(node.get());
    }
    for (auto& node : node_pool_) {
        node->children.clear();
        node->parent = nullptr;
    }
    root_->children.clear();
}

void BlockTree::releaseNode(TreeNode* node) {
    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        GroupSetResource&  resource  = node->group_set_resources[group_set_id];
        if (resource.hasTier(Tier::DEVICE)) {
            const std::vector<BlockIdxType> device_blocks = resource.device_blocks;
            const MultiNodeResource         device_resource{group_set_id, Tier::DEVICE, {{node, device_blocks}}};
            resource.evictFromTier(Tier::DEVICE);
            group_set->unreferenceBlocks(device_resource, BlockTreeRefType::CACHE);
        }
        if (resource.hasTier(Tier::HOST)) {
            group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::HOST, {{node, {resource.host_block}}}},
                                         BlockTreeRefType::CACHE);
            resource.host_block = NULL_BLOCK_IDX;
        }
        if (resource.hasTier(Tier::DISK)) {
            group_set->unreferenceBlocks(MultiNodeResource{group_set_id, Tier::DISK, {{node, {resource.disk_block}}}},
                                         BlockTreeRefType::CACHE);
            resource.disk_block = NULL_BLOCK_IDX;
        }
        resource.transfer_state = GroupSetTransferState::IDLE;
    }
}

TreeNode* BlockTree::createNode(CacheKeyType key, TreeNode* parent) {
    auto node       = std::make_unique<TreeNode>();
    node->cache_key = key;
    node->parent    = parent;
    node->index     = node_pool_.size();
    node->group_set_resources.resize(group_sets_.size());
    node->attached_at_us = steadyTimeUs();
    auto* raw = node.get();
    node_pool_.push_back(std::move(node));
    cache_keys_.push_back(key);
    ++key_version_;
    return raw;
}

std::vector<TreeNode*> BlockTree::findNode(const CacheKeysType& cache_keys) const {
    std::vector<TreeNode*> path;
    TreeNode*              current = root_.get();

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        auto it = current->children.find(cache_keys[i]);
        if (it == current->children.end()) {
            break;
        }
        TreeNode* child = it->second;
        for (const GroupSetResource& resource : child->group_set_resources) {
            if (resource.transfer_state == GroupSetTransferState::IDLE) {
                RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState(),
                                        "BlockTree encountered invalid IDLE multi-tier resource, node_key=%ld",
                                        child->cache_key);
            }
        }
        current = child;
        path.push_back(current);
    }

    return path;
}

CacheKeysType BlockTree::cacheKeys() const {
    return cache_keys_;
}

bool BlockTree::isLeafAtTier(const TreeNode* node, size_t group_set_id, Tier tier) const {
    for (const auto& [_, child] : node->children) {
        if (child->group_set_resources[group_set_id].hasTier(tier)) {
            return false;
        }
    }
    return true;
}

BlockTreeInsertResult BlockTree::insertNode(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            bool                                              collect_path,
                                            bool                                              is_resident) {
    BlockTreeInsertResult result;
    if (resources.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("key/resource size mismatch, keys=%zu resources=%zu", cache_keys.size(), resources.size());
        return result;
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        if (resources[i].size() != group_sets_.size()) {
            RTP_LLM_LOG_WARNING("GroupSetResource mismatch, index=%zu expected=%zu actual=%zu",
                                i,
                                group_sets_.size(),
                                resources[i].size());
            return result;
        }
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& resource = resources[i][group_set_id];
            RTP_LLM_CHECK_WITH_INFO(resource.isValidSteadyState()
                                        && (!resource.hasTier(Tier::DEVICE) || resource.hasCompleteDeviceValue()),
                                    "BlockTree insert requires an IDLE steady resource and complete DEVICE value: "
                                    "key=%ld group_set_id=%zu state=%d tiers=%zu",
                                    cache_keys[i],
                                    group_set_id,
                                    static_cast<int>(resource.transfer_state),
                                    resource.servingTierCount());
        }
    }

    return insertNodeImpl(cache_keys, resources, collect_path, is_resident);
}

BlockTreeInsertResult BlockTree::insertNodeImpl(const CacheKeysType&                              cache_keys,
                                                const std::vector<std::vector<GroupSetResource>>& resources,
                                                bool                                              collect_path,
                                                bool                                              is_resident) {
    BlockTreeInsertResult result;
    if (collect_path) {
        result.path.reserve(cache_keys.size());
    }

    TreeNode* current                = root_.get();
    TreeNode* first_changed          = nullptr;
    size_t    inserted_prefix_length = 0;

    auto publishTier = [this](TreeNode* node, size_t group_set_id, const GroupSetResource& resource, Tier tier) {
        const MultiNodeResource published{group_set_id, tier, {{node, resource.getBlocks(tier)}}};
        group_sets_[group_set_id]->referenceBlocks(published, BlockTreeRefType::CACHE);
    };

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        CacheKeyType key = cache_keys[i];
        if (pending_node_pool_.count(key)) {
            break;
        }
        auto         it  = current->children.find(key);
        if (it != current->children.end()) {
            TreeNode* child = it->second;
            if (is_resident
                && std::any_of(child->group_set_resources.begin(),
                               child->group_set_resources.end(),
                               [](const GroupSetResource& resource) {
                                   return resource.transfer_state != GroupSetTransferState::IDLE
                                          || resource.transfer_detached;
                               })) {
                RTP_LLM_LOG_WARNING("resident insert stopped at busy prefix: key_index=%zu key=%ld", i, key);
                break;
            }
            bool full_path_ready = true;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                if (group_sets_[group_set_id]->groupType() != CacheGroupType::FULL) {
                    continue;
                }
                const GroupSetResource& incoming = resources[i][group_set_id];
                if (!incoming.hasTier(Tier::DEVICE)) {
                    continue;
                }
                const GroupSetResource& existing  = child->group_set_resources[group_set_id];
                const bool              can_reuse = existing.isValidSteadyState() && existing.hasCompleteDeviceValue();
                const bool              can_adopt = existing.is_removable();
                if (!can_reuse && !can_adopt) {
                    RTP_LLM_LOG_WARNING("event=block_tree_insert_hard_stop key_index=%zu key=%ld "
                                        "group_set_id=%zu existing_tier=%s transfer_state=%d serving_tiers=%zu",
                                        i,
                                        key,
                                        group_set_id,
                                        tierName(existing.getTopTier()),
                                        static_cast<int>(existing.transfer_state),
                                        existing.servingTierCount());
                    full_path_ready = false;
                    break;
                }
            }
            if (!full_path_ready) {
                if (collect_path) {
                    current = child;
                    result.path.push_back(current);
                    for (size_t path_index = i + 1; path_index < cache_keys.size(); ++path_index) {
                        auto path_it = current->children.find(cache_keys[path_index]);
                        if (path_it == current->children.end()) {
                            break;
                        }
                        current = path_it->second;
                        result.path.push_back(current);
                    }
                }
                break;
            }

            current                                = child;
            const auto&         incoming_resources = resources[i];
            std::vector<size_t> adopted_group_set_ids;
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                GroupSetResource&       existing      = current->group_set_resources[group_set_id];
                const GroupSetResource& incoming      = incoming_resources[group_set_id];
                const Tier              incoming_tier = incoming.getTopTier();
                if (!existing.is_empty() || existing.transfer_state != GroupSetTransferState::IDLE
                    || incoming_tier == Tier::NONE) {
                    continue;
                }
                existing.setBlocks(incoming_tier, incoming.getBlocks(incoming_tier));
                existing.transfer_state = GroupSetTransferState::IDLE;
                existing.candidate_meta = {};
                publishTier(current, group_set_id, existing, incoming_tier);
                ++result.accepted_resource_count;
                adopted_group_set_ids.push_back(group_set_id);
            }
            if (!adopted_group_set_ids.empty()) {
                if (first_changed == nullptr) {
                    first_changed = current;
                }
                result.adopted_nodes.emplace_back(current, std::move(adopted_group_set_ids));
            }
        } else {
            TreeNode* child              = createNode(key, current);
            current->children[key]       = child;
            current                      = child;
            current->group_set_resources = resources[i];
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                const GroupSetResource& resource = current->group_set_resources[group_set_id];
                const Tier              tier     = resource.getTopTier();
                if (tier == Tier::NONE) {
                    continue;
                }
                publishTier(current, group_set_id, resource, tier);
                ++result.accepted_resource_count;
            }
            result.inserted_nodes.push_back(current);
            if (first_changed == nullptr) {
                first_changed = current;
            }
        }
        if (is_resident && !current->is_resident) {
            assert(current->parent == root_.get() || current->parent->is_resident);
            current->is_resident = true;
            result.newly_resident_nodes.push_back(current);
        }
        if (is_resident) {
            result.resident_prefix_length = i + 1;
        }
        if (collect_path) {
            result.path.push_back(current);
        }
        refreshPublishedState(current, false);
        inserted_prefix_length = i + 1;
    }

    if (first_changed != nullptr) {
        observeCacheState(first_changed);
    }

    RTP_LLM_LOG_DEBUG("keys=%zu created=%zu adopted=%zu tree_nodes=%zu",
                      inserted_prefix_length,
                      result.inserted_nodes.size(),
                      result.adopted_nodes.size(),
                      node_pool_.size());
    return result;
}

bool BlockTree::isRemovable(TreeNode* node) const {
    return node != nullptr && node != root_.get() && !node->is_resident && node->children.empty()
           && std::all_of(node->group_set_resources.begin(),
                          node->group_set_resources.end(),
                          [](const GroupSetResource& resource) { return resource.is_removable(); });
}

void BlockTree::removeNode(TreeNode* node) {
    RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

    if (node->cache_live_since_us != 0) {
        const int64_t lifetime_us = std::max<int64_t>(steadyTimeUs() - node->cache_live_since_us, 0);
        ++completed_cache_lifetime_count_;
        completed_cache_lifetime_sum_us_ += lifetime_us;
        completed_cache_lifetime_max_us_ = std::max(completed_cache_lifetime_max_us_, lifetime_us);
        node->cache_live_since_us = 0;
    }

    if (node->parent == nullptr) {
        pending_node_pool_.erase(node->cache_key);
        return;
    }
    removePublishedKey(node->cache_key);
    node->parent->children.erase(node->cache_key);
    node->parent            = nullptr;
    const size_t index      = node->index;
    const size_t last_index = node_pool_.size() - 1;
    if (index != last_index) {
        std::swap(node_pool_[index], node_pool_[last_index]);
        node_pool_[index]->index = index;
        cache_keys_[index]       = cache_keys_.back();
    }
    node->index = std::numeric_limits<size_t>::max();
    if (!isRemovable(node)) {
        pending_node_pool_.emplace(node->cache_key, std::move(node_pool_.back()));
    }
    node_pool_.pop_back();
    cache_keys_.pop_back();
    ++key_version_;
}

void BlockTree::removePendingNodes(const std::unordered_set<TreeNode*>& nodes) {
    for (TreeNode* node : nodes) {
        if (node->parent == nullptr && isRemovable(node)) {
            removeNode(node);
        }
    }
}

TreeNode* BlockTree::removeNodeAndEmptyAncestors(TreeNode* node) {
    TreeNode* current       = node;
    size_t    removed_count = 0;
    while (isRemovable(current)) {
        TreeNode* parent = current->parent;
        removeNode(current);
        current = parent;
        ++removed_count;
    }
    if (removed_count != 0) {
        RTP_LLM_LOG_DEBUG("removed %zu nodes", removed_count);
    }
    return current;
}

void BlockTree::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                  const std::vector<std::string>& required_group_tags) {
    std::vector<ReusableGroupLocation> locations;
    if (publisher) {
        if (required_group_tags.empty()) {
            throw std::invalid_argument("cache event publication requires reusable groups");
        }
        for (const auto& tag : required_group_tags) {
            const auto* location = reusableGroupLocation(tag);
            if (!location) {
                throw std::invalid_argument("cache event publication group is absent from BlockTree");
            }
            locations.push_back(*location);
        }
    }
    event_publisher_    = std::move(publisher);
    publication_groups_ = std::move(locations);
    published_keys_.clear();
    // Initialization only. Normal mutations update a single affected key.
    // The publisher starts after attachment and reconciles this initial snapshot.
    for (const auto& node : node_pool_) {
        refreshPublishedState(node.get(), false);
    }
    for (const auto& child : root_->children) {
        observeCacheState(child.second);
    }
}

KVCacheSnapshot BlockTree::logicalCacheSnapshot() const {
    KVCacheSnapshot snapshot;
    snapshot.version = publication_version_;
    snapshot.block_keys.assign(published_keys_.begin(), published_keys_.end());
    std::sort(snapshot.block_keys.begin(), snapshot.block_keys.end());
    return snapshot;
}

void BlockTree::removePublishedKey(CacheKeyType key) {
    if (event_publisher_ && published_keys_.erase(key)) {
        ++publication_version_;
        (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_DELETE, static_cast<int64_t>(key), 0});
    }
}

void BlockTree::refreshPublishedState(const TreeNode* node, bool observe_cache_state) {
    if (node->parent == nullptr) {
        return;
    }
    if (observe_cache_state) {
        observeCacheState(node);
    }
    if (!event_publisher_) {
        return;
    }
    for (const auto& location : publication_groups_) {
        const auto& resource = node->group_set_resources.at(location.group_set_id);
        if (resource.transfer_detached || resource.transfer_state != GroupSetTransferState::IDLE
            || resource.device_blocks.size() != group_sets_[location.group_set_id]->groupTags().size()
            || !resource.hasCompleteDeviceValue()) {
            removePublishedKey(node->cache_key);
            return;
        }
    }
    if (published_keys_.insert(node->cache_key).second) {
        ++publication_version_;
        (void)event_publisher_->tryPublish({KVCacheEventType::BLOCK_ADD, static_cast<int64_t>(node->cache_key), 0});
    }
}

void BlockTree::observeCacheState(const TreeNode* node) {
    if (node == root_.get()) {
        return;
    }
    const int64_t now_us = steadyTimeUs();
    Validators initial_validators = makeValidators(group_sets_);
    std::vector<const TreeNode*> ancestors;
    for (const TreeNode* current = node->parent; current != root_.get(); current = current->parent) {
        ancestors.push_back(current);
    }
    std::reverse(ancestors.begin(), ancestors.end());
    for (const TreeNode* ancestor : ancestors) {
        for (size_t group_set_id = 0; group_set_id < initial_validators.size(); ++group_set_id) {
            validateObservedResource(
                *initial_validators[group_set_id], ancestor->group_set_resources[group_set_id], true);
        }
    }
    struct Frame {
        TreeNode*  node;
        Validators validators;
    };
    std::vector<Frame> pending;
    pending.push_back({const_cast<TreeNode*>(node), std::move(initial_validators)});
    while (!pending.empty()) {
        Frame frame = std::move(pending.back());
        pending.pop_back();
        TreeNode* current = frame.node;
        bool      valid   = !frame.validators.empty();
        for (size_t group_set_id = 0; group_set_id < frame.validators.size(); ++group_set_id) {
            valid = validateObservedResource(*frame.validators[group_set_id],
                                             current->group_set_resources[group_set_id],
                                             true)
                    && valid;
        }
        if (valid && current->cache_live_since_us == 0) {
            current->cache_live_since_us = now_us;
        } else if (!valid && current->cache_live_since_us != 0) {
            const int64_t lifetime_us = std::max<int64_t>(now_us - current->cache_live_since_us, 0);
            ++completed_cache_lifetime_count_;
            completed_cache_lifetime_sum_us_ += lifetime_us;
            completed_cache_lifetime_max_us_ = std::max(completed_cache_lifetime_max_us_, lifetime_us);
            current->cache_live_since_us = 0;
        }
        size_t remaining_children = current->children.size();
        for (const auto& child : current->children) {
            --remaining_children;
            Validators child_validators =
                remaining_children == 0 ? std::move(frame.validators) : cloneValidators(frame.validators);
            pending.push_back({child.second, std::move(child_validators)});
        }
    }
    const int64_t cpu_us = std::max<int64_t>(steadyTimeUs() - now_us, 0);
    ++observation_count_;
    observation_cpu_sum_us_ += cpu_us;
    observation_cpu_max_us_ = std::max(observation_cpu_max_us_, cpu_us);
}

BlockTreeObservabilitySnapshot BlockTree::observabilitySnapshot(bool include_detail) const {
    BlockTreeObservabilitySnapshot snapshot;
    snapshot.node_count                          = node_pool_.size();
    snapshot.root_child_count                    = root_->children.size();
    snapshot.completed_cache_lifetime_count      = completed_cache_lifetime_count_;
    snapshot.completed_cache_lifetime_sum_us     = completed_cache_lifetime_sum_us_;
    snapshot.completed_cache_lifetime_max_us     = completed_cache_lifetime_max_us_;
    snapshot.observation_count                    = observation_count_;
    snapshot.observation_cpu_sum_us               = observation_cpu_sum_us_;
    snapshot.observation_cpu_max_us               = observation_cpu_max_us_;
    const int64_t now_us = steadyTimeUs();
    for (const GroupSetPtr& group_set : group_sets_) {
        for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            snapshot.resource_leaves.push_back({group_set->groupType(), tier, 0});
        }
    }
    struct Frame {
        const TreeNode*       node;
        size_t                depth;
        size_t                segment_index;
        size_t                segment_start_tokens;
        std::optional<size_t> ancestor_point_depth;
        Validators            validators;
    };
    std::vector<Frame> pending;
    for (const auto& child : root_->children) {
        size_t segment_index = 0;
        if (include_detail) {
            segment_index = snapshot.chain_segments.size();
            snapshot.chain_segments.push_back({0, 0, false, {}});
        }
        pending.push_back({child.second, 1, segment_index, 0, std::nullopt, makeValidators(group_sets_)});
    }
    while (!pending.empty()) {
        Frame frame = std::move(pending.back());
        pending.pop_back();
        const TreeNode* node                 = frame.node;
        const size_t    depth                = frame.depth;
        const size_t    segment_index        = frame.segment_index;
        const size_t    segment_start_tokens = frame.segment_start_tokens;
        auto            ancestor_point_depth = frame.ancestor_point_depth;
        Validators&     validators           = frame.validators;
        const size_t child_count = node->children.size();
        const size_t depth_tokens = depth * reuse_unit_tokens_;
        snapshot.max_chain_tokens = std::max(snapshot.max_chain_tokens, depth_tokens - segment_start_tokens);
        snapshot.max_child_count = std::max(snapshot.max_child_count, child_count);
        if (include_detail) {
            BlockTreeChainSegmentSnapshot& segment = snapshot.chain_segments[segment_index];
            segment.end_tokens = depth_tokens;
            segment.end_is_leaf = child_count == 0;
            if (snapshot.child_count_histogram.size() <= child_count) {
                snapshot.child_count_histogram.resize(child_count + 1, 0);
            }
            ++snapshot.child_count_histogram[child_count];
        }
        snapshot.max_depth_tokens = std::max(snapshot.max_depth_tokens, depth_tokens);
        const int64_t node_age_us = std::max<int64_t>(now_us - node->attached_at_us, 0);
        snapshot.node_age_sum_us += node_age_us;
        snapshot.node_age_max_us = std::max(snapshot.node_age_max_us, node_age_us);
        if (node->cache_live_since_us != 0) {
            ++snapshot.stored_cache_count;
            const int64_t age_us = std::max<int64_t>(now_us - node->cache_live_since_us, 0);
            snapshot.live_cache_age_sum_us += age_us;
            snapshot.live_cache_age_max_us = std::max(snapshot.live_cache_age_max_us, age_us);
        }
        bool recoverable = !validators.empty();
        for (size_t group_set_id = 0; group_set_id < validators.size(); ++group_set_id) {
            recoverable = validateObservedResource(*validators[group_set_id],
                                                   node->group_set_resources[group_set_id],
                                                   false)
                          && recoverable;
        }
        if (recoverable) {
            ++snapshot.recoverable_point_count;
            snapshot.max_recoverable_depth_tokens = std::max(snapshot.max_recoverable_depth_tokens, depth_tokens);
            snapshot.recoverable_depth_sum_tokens += depth_tokens;
            if (include_detail) {
                snapshot.chain_segments[segment_index].recoverable_points_tokens.push_back(depth_tokens);
            }
            if (ancestor_point_depth.has_value()) {
                const size_t gap = depth_tokens - *ancestor_point_depth;
                ++snapshot.nearest_ancestor_gap_count;
                snapshot.nearest_ancestor_gap_sum_tokens += gap;
                snapshot.nearest_ancestor_gap_max_tokens =
                    std::max(snapshot.nearest_ancestor_gap_max_tokens, gap);
            } else {
                ++snapshot.first_recoverable_point_count;
            }
            ancestor_point_depth = depth_tokens;
        } else if (node->cache_live_since_us != 0) {
            ++snapshot.temporarily_busy_count;
        } else if (std::all_of(node->group_set_resources.begin(),
                               node->group_set_resources.end(),
                               [](const GroupSetResource& resource) { return resource.is_empty(); })) {
            ++snapshot.pure_content_count;
        } else {
            ++snapshot.incomplete_state_count;
        }
        if (child_count == 0) {
            ++snapshot.leaf_count;
            snapshot.leaf_depth_sum_tokens += depth_tokens;
            snapshot.max_leaf_depth_tokens = std::max(snapshot.max_leaf_depth_tokens, depth_tokens);
            if (include_detail) {
                snapshot.leaf_depths_tokens.push_back(depth_tokens);
            }
        } else if (child_count == 1) {
            ++snapshot.linear_count;
        } else {
            ++snapshot.branch_count;
            snapshot.branch_depth_sum_tokens += depth_tokens;
            snapshot.max_branch_depth_tokens = std::max(snapshot.max_branch_depth_tokens, depth_tokens);
            if (include_detail) {
                snapshot.branch_depths_tokens.push_back(depth_tokens);
            }
        }
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            for (size_t tier_index = 0; tier_index < 3; ++tier_index) {
                const Tier tier = snapshot.resource_leaves[group_set_id * 3 + tier_index].tier;
                if (node->group_set_resources[group_set_id].hasTier(tier)
                    && isLeafAtTier(node, group_set_id, tier)) {
                    ++snapshot.resource_leaves[group_set_id * 3 + tier_index].leaf_count;
                }
            }
        }
        if (child_count != 1) {
            ++snapshot.chain_segment_count;
        }
        size_t remaining_children = child_count;
        for (const auto& child : node->children) {
            size_t child_segment_index = segment_index;
            const size_t child_segment_start_tokens = child_count == 1 ? segment_start_tokens : depth_tokens;
            if (include_detail && child_count != 1) {
                child_segment_index = snapshot.chain_segments.size();
                snapshot.chain_segments.push_back({depth_tokens, depth_tokens, false, {}});
            }
            --remaining_children;
            Validators child_validators = remaining_children == 0 ? std::move(validators) : cloneValidators(validators);
            pending.push_back({child.second,
                               depth + 1,
                               child_segment_index,
                               child_segment_start_tokens,
                               ancestor_point_depth,
                               std::move(child_validators)});
        }
    }
    if (include_detail) {
        std::sort(snapshot.leaf_depths_tokens.begin(), snapshot.leaf_depths_tokens.end());
        std::sort(snapshot.branch_depths_tokens.begin(), snapshot.branch_depths_tokens.end());
        std::sort(snapshot.chain_segments.begin(), snapshot.chain_segments.end(), [](const auto& a, const auto& b) {
            return std::tie(a.start_tokens, a.end_tokens, a.end_is_leaf, a.recoverable_points_tokens)
                   < std::tie(b.start_tokens, b.end_tokens, b.end_is_leaf, b.recoverable_points_tokens);
        });
    }
    snapshot.snapshot_cpu_us = std::max<int64_t>(steadyTimeUs() - now_us, 0);
    return snapshot;
}

}  // namespace rtp_llm
