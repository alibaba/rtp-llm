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
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

size_t matchingTokenPrefix(const int* source, const int* request, size_t count) {
    size_t matched = 0;
    // Most unrelated children differ immediately. Keep that path cheap before
    // comparing large CP/block segments in batches.
    for (; matched < count && matched < 4; ++matched) {
        if (source[matched] != request[matched]) {
            return matched;
        }
    }
    if (count - matched >= 32) {
        if (!std::equal(source + matched, source + matched + 32, request + matched)) {
            while (matched < count && source[matched] == request[matched]) {
                ++matched;
            }
            return matched;
        }
        matched += 32;
    }
    while (count - matched >= 64 && std::equal(source + matched, source + matched + 64, request + matched)) {
        matched += 64;
    }
    // A failed batch comparison only narrows the search. Locate the first
    // unequal token exactly; callers still authenticate the source identity.
    while (matched < count && source[matched] == request[matched]) {
        ++matched;
    }
    return matched;
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

// Observation uses the same endpoint geometry as precise matching. Point-state
// groups need only their endpoint; SWA short leaves contribute their actual span.
bool observedEndpointReady(const std::vector<GroupSetPtr>& groups,
                           const std::vector<TreeNode*>&   path,
                           size_t                          endpoint) {
    if (groups.empty() || path.empty() || path.back()->token_end != endpoint) {
        return false;
    }
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const auto& group = groups[group_id];
        for (const size_t index : group->requiredPathIndices(path, endpoint)) {
            const auto& resource = path[index]->group_set_resources[group_id];
            if (!resource.isMatchUsable() || resource.is_empty()
                || (resource.hasTier(Tier::DEVICE)
                    && (!resource.hasCompleteDeviceValue()
                        || resource.device_blocks.size() != group->groupTags().size()))) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

BlockTree::BlockTree(std::vector<GroupSetPtr> group_sets,
                     size_t                   reuse_unit_tokens,
                     bool                     enable_observability,
                     std::string              observability_instance):
    group_sets_(std::move(group_sets)),
    reuse_unit_tokens_(reuse_unit_tokens),
    observability_enabled_(enable_observability),
    observability_instance_(std::move(observability_instance)) {
    RTP_LLM_CHECK_WITH_INFO(reuse_unit_tokens_ > 0, "BlockTree reuse unit must be positive");
    if (observability_enabled_) {
        tree_epoch_us_ = steadyTimeUs();
    }
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

BlockTree::ObservationBatch::ObservationBatch(BlockTree& tree): tree_(tree) {
    ++tree_.observation_batch_depth_;
}

BlockTree::ObservationBatch::~ObservationBatch() {
    tree_.endObservationBatch();
}

void BlockTree::deferObservationRoot(const TreeNode* node, const char* trigger) {
    if (!observability_enabled_ || node == nullptr || node == root_.get()) {
        return;
    }
    for (DeferredObservationRoot& root : deferred_observation_roots_) {
        if (root.node == node) {
            root.trigger = trigger;
            return;
        }
    }
    deferred_observation_roots_.push_back({node, trigger});
}

void BlockTree::endObservationBatch() {
    if (observation_batch_depth_ == 0) {
        return;
    }
    --observation_batch_depth_;
    if (observation_batch_depth_ == 0) {
        commitDeferredObservations();
    }
}

void BlockTree::commitDeferredObservations() {
    std::vector<DeferredObservationRoot> roots = std::move(deferred_observation_roots_);
    deferred_observation_roots_.clear();
    for (size_t candidate_index = 0; candidate_index < roots.size(); ++candidate_index) {
        bool covered = false;
        for (size_t other_index = 0; other_index < roots.size(); ++other_index) {
            if (candidate_index != other_index
                && isObservationAncestor(roots[other_index].node, roots[candidate_index].node)) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            observeCacheState(roots[candidate_index].node,
                              roots[candidate_index].trigger != nullptr ? roots[candidate_index].trigger : "mutation");
        }
    }
}

void BlockTree::eraseDeferredObservationRoots(const TreeNode* node) {
    if (node == nullptr || deferred_observation_roots_.empty()) {
        return;
    }
    std::vector<const TreeNode*> pending{node};
    while (!pending.empty()) {
        const TreeNode* current = pending.back();
        pending.pop_back();
        deferred_observation_roots_.erase(
            std::remove_if(deferred_observation_roots_.begin(),
                           deferred_observation_roots_.end(),
                           [current](const DeferredObservationRoot& root) { return root.node == current; }),
            deferred_observation_roots_.end());
        for (const auto& child : current->children) {
            pending.push_back(child.second);
        }
    }
}

bool BlockTree::isObservationAncestor(const TreeNode* ancestor, const TreeNode* candidate) {
    if (ancestor == nullptr || candidate == nullptr) {
        return false;
    }
    for (const TreeNode* current = candidate->parent; current != nullptr; current = current->parent) {
        if (current == ancestor) {
            return true;
        }
    }
    return false;
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

size_t BlockTree::tierIndex(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            return 0;
        case Tier::HOST:
            return 1;
        case Tier::DISK:
            return 2;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "prefix observability does not support tier=%d", static_cast<int>(tier));
            return 0;
    }
}

void BlockTree::startInterval(IntervalSet& intervals, int64_t& since_us) {
    if (!observability_enabled_ || since_us >= 0) {
        return;
    }
    since_us = steadyTimeUs() - tree_epoch_us_;
    intervals.starts.insert(since_us);
    intervals.start_sum += static_cast<long double>(since_us);
}

void BlockTree::endInterval(IntervalSet&    intervals,
                            int64_t&        since_us,
                            const TreeNode& node,
                            const char*     kind,
                            const char*     medium,
                            const char*     trigger,
                            const char*     effect) {
    if (!observability_enabled_ || since_us < 0) {
        return;
    }
    const int64_t duration_us = std::max<int64_t>(steadyTimeUs() - tree_epoch_us_ - since_us, 0);
    const auto    it          = intervals.starts.find(since_us);
    if (it != intervals.starts.end()) {
        intervals.start_sum -= static_cast<long double>(*it);
        intervals.starts.erase(it);
    }
    RTP_LLM_LOG_INFO("event=prefix_cache_observation payload={\"kind\":\"interval_end\","
                     "\"schema_version\":1,\"instance\":\"%s\",\"key\":\"%ld\","
                     "\"interval\":\"%s\",\"medium\":\"%s\",\"since_us\":%ld,"
                     "\"duration_us\":%ld,\"trigger\":\"%s\",\"effect\":\"%s\"}",
                     observability_instance_.c_str(),
                     node.cache_key,
                     kind,
                     medium ? medium : "none",
                     since_us,
                     duration_us,
                     trigger,
                     effect);
    since_us = -1;
}

void BlockTree::updateContentShape(const TreeNode& node, size_t old_child_count, size_t new_child_count) {
    if (!observability_enabled_ || &node == root_.get()) {
        return;
    }
    if (old_child_count == 0 && new_child_count != 0) {
        --leaf_count_;
    } else if (old_child_count != 0 && new_child_count == 0) {
        ++leaf_count_;
    }
    if (old_child_count <= 1 && new_child_count > 1) {
        ++branch_count_;
    } else if (old_child_count > 1 && new_child_count <= 1) {
        --branch_count_;
    }
}

void BlockTree::updateTierIntervals(TreeNode& node, const char* trigger) {
    if (!observability_enabled_ || &node == root_.get()) {
        return;
    }
    for (size_t tier_idx = 0; tier_idx < node.observation.tier_since_us.size(); ++tier_idx) {
        const Tier tier     = static_cast<Tier>(tier_idx);
        bool       has_tier = false;
        for (const GroupSetResource& resource : node.group_set_resources) {
            has_tier = resource.hasTier(tier) || has_tier;
        }
        if (has_tier) {
            startInterval(tier_intervals_[tier_idx], node.observation.tier_since_us[tier_idx]);
        } else {
            const char* medium = tier_idx == 0 ? "DEVICE" : (tier_idx == 1 ? "HOST" : "DISK");
            endInterval(tier_intervals_[tier_idx],
                        node.observation.tier_since_us[tier_idx],
                        node,
                        "tier",
                        medium,
                        trigger,
                        "tier_release");
        }
    }
}

BlockTreeAgeSummary BlockTree::intervalSummary(const IntervalSet& intervals) const {
    BlockTreeAgeSummary summary;
    summary.count = intervals.starts.size();
    if (summary.count == 0) {
        return summary;
    }
    const int64_t     now_us = steadyTimeUs() - tree_epoch_us_;
    const long double mean_age =
        static_cast<long double>(now_us) - intervals.start_sum / static_cast<long double>(summary.count);
    summary.mean_age_us = std::max<int64_t>(static_cast<int64_t>(mean_age), 0);
    summary.max_age_us  = std::max<int64_t>(now_us - *intervals.starts.begin(), 0);
    return summary;
}

TreeNode* BlockTree::createNode(CacheKeyType key, TreeNode* parent, size_t token_end) {
    auto node       = std::make_unique<TreeNode>();
    node->cache_key = key;
    node->parent    = parent;
    node->token_end = token_end;
    node->index     = node_pool_.size();
    node->group_set_resources.resize(group_sets_.size());
    if (observability_enabled_) {
        ++leaf_count_;
    }
    startInterval(content_intervals_, node->observation.content_since_us);
    auto* raw = node.get();
    node_pool_.push_back(std::move(node));
    cache_keys_.push_back(key);
    ++key_version_;
    return raw;
}

std::optional<BlockTreeContentMatch> BlockTree::contentMatchInfo(const std::vector<int>& request_tokens) const {
    if (request_tokens.empty()) {
        return std::nullopt;
    }
    struct Frame {
        const TreeNode* node;
        size_t          pos;
    };
    std::vector<Frame>    pending{{root_.get(), 0}};
    BlockTreeContentMatch best;
    while (!pending.empty()) {
        const Frame frame = pending.back();
        pending.pop_back();
        for (const auto& [key, child] : frame.node->children) {
            if (child == nullptr || child->token_ids.empty()) {
                continue;
            }
            const size_t remaining = request_tokens.size() - frame.pos;
            const size_t limit     = std::min(child->token_ids.size(), remaining);
            const size_t matched =
                matchingTokenPrefix(child->token_ids.data(), request_tokens.data() + frame.pos, limit);
            const size_t end = frame.pos + matched;
            if (end > best.pos || (end == best.pos && matched > best.node_offset)) {
                best = {end, key, matched, matched != child->token_ids.size()};
            }
            if (matched == child->token_ids.size() && !isShortLeaf(child) && end < request_tokens.size()) {
                pending.push_back({child, end});
            }
        }
    }
    if (best.pos == 0) {
        return std::nullopt;
    }
    return best;
}

std::vector<TreeNode*>
BlockTree::contentMatchCandidates(const std::vector<int>&                    request_tokens,
                                  const std::function<CacheKeyType(size_t)>& endpoint_key) const {
    std::vector<TreeNode*> candidates;
    if (request_tokens.empty() || !endpoint_key) {
        return candidates;
    }
    std::vector<const TreeNode*> pending{root_.get()};
    while (!pending.empty()) {
        const TreeNode* parent = pending.back();
        pending.pop_back();
        for (const auto& [_, child] : parent->children) {
            if (child->token_ids.empty() || child->token_end > request_tokens.size()
                || child->token_ids.size() != child->token_end - parent->token_end
                || !std::equal(
                    child->token_ids.begin(), child->token_ids.end(), request_tokens.begin() + parent->token_end)
                || child->cache_key != endpoint_key(child->token_end)) {
                continue;
            }
            candidates.push_back(child);
            if (!isShortLeaf(child)) {
                pending.push_back(child);
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const TreeNode* lhs, const TreeNode* rhs) {
        return lhs->token_end < rhs->token_end;
    });
    return candidates;
}

std::vector<BlockTreeContentRange>
BlockTree::contentMatchRanges(const std::vector<int>&                                                   request_tokens,
                              const std::function<CacheKeyType(size_t)>&                                endpoint_key,
                              const std::function<bool(size_t, const std::vector<int>&, CacheKeyType)>& source_identity,
                              std::optional<PrefixCachePlanPoint>* farthest_fork) const {
    std::vector<BlockTreeContentRange> ranges;
    if (farthest_fork != nullptr) {
        farthest_fork->reset();
    }
    if (request_tokens.empty() || !endpoint_key) {
        return ranges;
    }
    std::vector<const TreeNode*> pending{root_.get()};
    while (!pending.empty()) {
        const TreeNode* parent = pending.back();
        pending.pop_back();
        for (const auto& [_, child] : parent->children) {
            if (child->token_ids.empty() || child->token_end <= parent->token_end
                || child->token_ids.size() != child->token_end - parent->token_end
                || parent->token_end >= request_tokens.size()) {
                continue;
            }
            const size_t limit = std::min(child->token_ids.size(), request_tokens.size() - parent->token_end);
            const size_t matched =
                matchingTokenPrefix(child->token_ids.data(), request_tokens.data() + parent->token_end, limit);
            if (matched == 0 && (farthest_fork == nullptr || parent->token_end == 0)) {
                continue;
            }
            const bool whole = matched == child->token_ids.size();
            // Token equality alone is insufficient: the key also carries the
            // original hash seed and image identity. Never accept an interior
            // range unless the caller can authenticate its complete source.
            if (whole ? child->cache_key != endpoint_key(child->token_end) :
                        (!source_identity || !source_identity(parent->token_end, child->token_ids, child->cache_key))) {
                continue;
            }
            // A strict fork has a next token on both sides. The source segment
            // is authenticated above, including a mismatch at child offset 0
            // after an authenticated canonical parent.
            if (farthest_fork != nullptr && matched < limit) {
                const size_t fork_pos = parent->token_end + matched;
                if (fork_pos > 0 && (!*farthest_fork || fork_pos > (*farthest_fork)->pos)) {
                    *farthest_fork =
                        PrefixCachePlanPoint{(fork_pos - 1) / reuse_unit_tokens_, endpoint_key(fork_pos), fork_pos};
                }
            }
            if (matched == 0) {
                continue;
            }
            ranges.push_back({child, parent->token_end + matched});
            if (whole && !isShortLeaf(child)) {
                pending.push_back(child);
            }
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.matched_end > rhs.matched_end;
    });
    return ranges;
}

std::optional<std::vector<TreeNode*>> BlockTree::coveredDeviceGroupPath(
    const std::vector<BlockTreeContentRange>& ranges, size_t endpoint, size_t group_id) const {
    if (group_id >= group_sets_.size() || endpoint == 0) {
        return std::nullopt;
    }
    const auto& group = group_sets_[group_id];
    if (group->hasZeroStateDemand(endpoint)) {
        return std::any_of(ranges.begin(),
                           ranges.end(),
                           [endpoint](const auto& range) { return endpoint <= range.matched_end; }) ?
                   std::optional<std::vector<TreeNode*>>(std::vector<TreeNode*>{}) :
                   std::nullopt;
    }
    for (const auto& range : ranges) {
        if (endpoint > range.matched_end || endpoint <= range.node->parent->token_end
            || (group->requiresExactStateEndpoint() && endpoint != range.node->token_end)) {
            continue;
        }
        std::vector<TreeNode*> path;
        for (auto* cursor = range.node; cursor->parent != nullptr; cursor = cursor->parent) {
            path.push_back(cursor);
        }
        std::reverse(path.begin(), path.end());
        if (!group->hasDeviceCoverage(path, endpoint)) {
            continue;
        }
        std::vector<TreeNode*> required_nodes;
        for (const size_t index : group->requiredPathIndices(path, endpoint)) {
            required_nodes.push_back(path[index]);
        }
        return required_nodes;
    }
    return std::nullopt;
}

size_t BlockTree::contentMatchPos(const std::vector<int>& request_tokens) const {
    const auto match = contentMatchInfo(request_tokens);
    return match.has_value() ? match->pos : 0;
}

std::vector<PrefixCachePlanPoint>
BlockTree::planRecoveryPoints(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              const std::vector<std::vector<int>>* /*token_segments*/,
                              Tier                       target_tier,
                              const std::vector<size_t>* token_ends) const {
    std::vector<PrefixCachePlanPoint> points;
    if (!observability_enabled_ || cache_keys.empty() || resources.size() != cache_keys.size()
        || (token_ends != nullptr && token_ends->size() != cache_keys.size())) {
        return points;
    }

    std::vector<TreeNode>  current_nodes(cache_keys.size());
    std::vector<TreeNode>  projected_nodes(cache_keys.size());
    std::vector<TreeNode*> current_path;
    std::vector<TreeNode*> projected_path;
    const TreeNode*        current = root_.get();
    size_t                 pos     = 0;
    for (size_t i = 0; i < cache_keys.size(); ++i) {
        const TreeNode* child = current == nullptr ? nullptr : [&] {
            const auto it = current->children.find(cache_keys[i]);
            return it == current->children.end() ? nullptr : it->second;
        }();

        std::vector<GroupSetResource> current_resources(group_sets_.size());
        if (child != nullptr) {
            current_resources = child->group_set_resources;
        }
        std::vector<GroupSetResource> projected_resources = current_resources;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            if (!current_resources[group_set_id].is_empty()) {
                continue;
            }
            const GroupSetResource& incoming = resources[i][group_set_id];
            if (target_tier == Tier::DEVICE && incoming.hasTier(Tier::DEVICE)) {
                projected_resources[group_set_id].setBlocks(Tier::DEVICE, incoming.getBlocks(Tier::DEVICE));
            } else if (target_tier != Tier::DEVICE && incoming.hasTier(Tier::DEVICE)) {
                projected_resources[group_set_id].setBlocks(target_tier, incoming.getBlocks(Tier::DEVICE));
            }
        }

        // Mirror insertNode geometry. Legacy content segments may include the
        // live request tail beyond the last complete cache key.
        pos                    = token_ends != nullptr ? (*token_ends)[i] : (i + 1) * reuse_unit_tokens_;
        auto& current_node     = current_nodes[i];
        auto& projected_node   = projected_nodes[i];
        current_node.parent    = current_path.empty() ? nullptr : current_path.back();
        projected_node.parent  = projected_path.empty() ? nullptr : projected_path.back();
        current_node.token_end = projected_node.token_end = pos;
        current_node.group_set_resources                  = std::move(current_resources);
        projected_node.group_set_resources                = std::move(projected_resources);
        current_path.push_back(&current_node);
        projected_path.push_back(&projected_node);
        const bool current_valid   = observedEndpointReady(group_sets_, current_path, pos);
        const bool projected_valid = observedEndpointReady(group_sets_, projected_path, pos);
        if (projected_valid && !current_valid && pos > 0) {
            points.push_back({i, cache_keys[i], pos});
        }
        current = child;
    }
    return points;
}

std::vector<PrefixCachePlanPoint> BlockTree::unfulfilledPlanPoints(const std::vector<PrefixCachePlanPoint>& plan_points,
                                                                   const CacheKeysType& cache_keys) const {
    std::vector<PrefixCachePlanPoint> unfulfilled;
    if (!observability_enabled_) {
        return unfulfilled;
    }
    for (const PrefixCachePlanPoint& point : plan_points) {
        const TreeNode*        current = root_.get();
        std::vector<TreeNode*> path;
        bool                   valid = point.index < cache_keys.size();
        for (size_t i = 0; valid && i <= point.index; ++i) {
            const auto it = current->children.find(cache_keys[i]);
            if (it == current->children.end()) {
                valid = false;
                break;
            }
            current = it->second;
            path.push_back(it->second);
        }
        if (!valid || !observedEndpointReady(group_sets_, path, point.pos)) {
            unfulfilled.push_back(point);
        }
    }
    return unfulfilled;
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
    CacheKeysType full_keys;
    full_keys.reserve(cache_keys_.size());
    for (size_t index = 0; index < node_pool_.size(); ++index) {
        if (!isShortLeaf(node_pool_[index].get())) {
            full_keys.push_back(cache_keys_[index]);
        }
    }
    return full_keys;
}

bool BlockTree::isShortLeaf(const TreeNode* node) const {
    return node != nullptr && node->parent != nullptr && node->token_end - node->parent->token_end < reuse_unit_tokens_;
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
                                            bool                                              is_resident,
                                            const std::vector<std::vector<int>>*              token_segments,
                                            const std::vector<size_t>*                        token_ends) {
    BlockTreeInsertResult result;
    if (resources.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("key/resource size mismatch, keys=%zu resources=%zu", cache_keys.size(), resources.size());
        return result;
    }
    if (token_ends != nullptr
        && (token_ends->size() != cache_keys.size() || token_segments == nullptr
            || token_segments->size() != cache_keys.size())) {
        RTP_LLM_LOG_WARNING("exact endpoint insert requires one token segment and endpoint per key");
        return result;
    }
    if (token_ends != nullptr) {
        size_t previous = 0;
        for (size_t i = 0; i < token_ends->size(); ++i) {
            const size_t end = (*token_ends)[i];
            if (end <= previous || end - previous > reuse_unit_tokens_ || (*token_segments)[i].size() != end - previous
                || (end - previous < reuse_unit_tokens_ && i + 1 != token_ends->size())) {
                RTP_LLM_LOG_WARNING("invalid exact endpoint insert at key_index=%zu endpoint=%zu", i, end);
                return result;
            }
            previous = end;
        }
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        if (resources[i].size() != group_sets_.size()) {
            RTP_LLM_LOG_WARNING("GroupSetResource mismatch, index=%zu expected=%zu actual=%zu",
                                i,
                                group_sets_.size(),
                                resources[i].size());
            return result;
        }
        const bool short_endpoint =
            token_ends != nullptr && ((*token_ends)[i] - (i == 0 ? 0 : (*token_ends)[i - 1])) < reuse_unit_tokens_;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& resource = resources[i][group_set_id];
            if (short_endpoint && (resource.hasTier(Tier::HOST) || resource.hasTier(Tier::DISK))) {
                RTP_LLM_LOG_WARNING(
                    "short leaf requires local DEVICE resource: key_index=%zu group_set_id=%zu", i, group_set_id);
                return result;
            }
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

    return insertNodeImpl(cache_keys, resources, collect_path, is_resident, token_segments, token_ends);
}

BlockTreeInsertResult BlockTree::insertNodeImpl(const CacheKeysType&                              cache_keys,
                                                const std::vector<std::vector<GroupSetResource>>& resources,
                                                bool                                              collect_path,
                                                bool                                              is_resident,
                                                const std::vector<std::vector<int>>*              token_segments,
                                                const std::vector<size_t>*                        token_ends) {
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
        if (current != root_.get() && isShortLeaf(current)) {
            RTP_LLM_LOG_WARNING("cannot insert a child below short leaf: parent_key=%ld", current->cache_key);
            break;
        }
        const size_t token_end = token_ends != nullptr ? (*token_ends)[i] : current->token_end + reuse_unit_tokens_;
        auto         it        = current->children.find(key);
        if (it != current->children.end()) {
            TreeNode* child = it->second;
            if (child->token_end != token_end
                || (token_ends != nullptr && !child->token_ids.empty() && child->token_ids != (*token_segments)[i])) {
                RTP_LLM_LOG_WARNING(
                    "cache key identity conflict: key_index=%zu key=%ld endpoint=%zu", i, key, token_end);
                break;
            }
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

            current = child;
            if (token_segments != nullptr && i < token_segments->size() && current->token_ids.empty()) {
                current->token_ids = (*token_segments)[i];
            }
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
                updateTierIntervals(*current, "store");
                if (first_changed == nullptr) {
                    first_changed = current;
                }
                result.adopted_nodes.emplace_back(current, std::move(adopted_group_set_ids));
            }
        } else {
            const size_t old_child_count = current->children.size();
            TreeNode*    child           = createNode(key, current, token_end);
            current->children[key]       = child;
            updateContentShape(*current, old_child_count, current->children.size());
            current                      = child;
            current->group_set_resources = resources[i];
            if (token_segments != nullptr && i < token_segments->size()) {
                current->token_ids = (*token_segments)[i];
            }
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                const GroupSetResource& resource = current->group_set_resources[group_set_id];
                const Tier              tier     = resource.getTopTier();
                if (tier == Tier::NONE) {
                    continue;
                }
                publishTier(current, group_set_id, resource, tier);
                ++result.accepted_resource_count;
            }
            updateTierIntervals(*current, "store");
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
        observeCacheState(first_changed, "store");
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

void BlockTree::removeNode(TreeNode* node, const char* trigger) {
    RTP_LLM_LOG_DEBUG("removing node key=%ld, pool_size=%zu", node->cache_key, node_pool_.size());

    eraseDeferredObservationRoots(node);
    if (observability_enabled_) {
        for (size_t tier_idx = 0; tier_idx < node->observation.tier_since_us.size(); ++tier_idx) {
            const char* medium = tier_idx == 0 ? "DEVICE" : (tier_idx == 1 ? "HOST" : "DISK");
            endInterval(tier_intervals_[tier_idx],
                        node->observation.tier_since_us[tier_idx],
                        *node,
                        "tier",
                        medium,
                        trigger,
                        "tier_release");
        }
        endInterval(recovery_intervals_,
                    node->observation.recovery_since_us,
                    *node,
                    "recovery",
                    nullptr,
                    trigger,
                    "dependency_loss");
        endInterval(content_intervals_,
                    node->observation.content_since_us,
                    *node,
                    "content",
                    nullptr,
                    trigger,
                    "content_remove");
    }
    if (node->parent == nullptr) {
        pending_node_pool_.erase(node->cache_key);
        return;
    }
    removePublishedKey(node->cache_key);
    TreeNode*    parent          = node->parent;
    const size_t old_child_count = parent->children.size();
    parent->children.erase(node->cache_key);
    node->parent = nullptr;
    if (observability_enabled_) {
        if (node->children.empty()) {
            --leaf_count_;
        } else if (node->children.size() > 1) {
            --branch_count_;
        }
    }
    updateContentShape(*parent, old_child_count, parent->children.size());
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

TreeNode* BlockTree::removeNodeAndEmptyAncestors(TreeNode* node, const char* trigger) {
    TreeNode* current       = node;
    size_t    removed_count = 0;
    while (isRemovable(current)) {
        TreeNode* parent = current->parent;
        removeNode(current, trigger);
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
        observeCacheState(child.second, "publisher_attachment");
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

void BlockTree::refreshPublishedState(const TreeNode* node, bool observe_cache_state, const char* trigger) {
    if (node->parent == nullptr) {
        return;
    }
    if (observe_cache_state) {
        observeCacheState(node, trigger);
    }
    if (!event_publisher_) {
        return;
    }
    if (isShortLeaf(node)) {
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

void BlockTree::observeCacheState(const TreeNode* node, const char* trigger) {
    if (!observability_enabled_ || node == root_.get()) {
        return;
    }
    if (observation_batch_depth_ > 0) {
        deferObservationRoot(node, trigger);
        return;
    }
    Validators                   initial_validators = makeValidators(group_sets_);
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
            valid = validateObservedResource(
                        *frame.validators[group_set_id], current->group_set_resources[group_set_id], true)
                    && valid;
        }
        int64_t& recovery_since_us = current->observation.recovery_since_us;
        if (valid) {
            startInterval(recovery_intervals_, recovery_since_us);
        } else if (recovery_since_us >= 0) {
            endInterval(
                recovery_intervals_, recovery_since_us, *current, "recovery", nullptr, trigger, "dependency_loss");
        }
        updateTierIntervals(*current, trigger);
        size_t remaining_children = current->children.size();
        for (const auto& child : current->children) {
            --remaining_children;
            Validators child_validators =
                remaining_children == 0 ? std::move(frame.validators) : cloneValidators(frame.validators);
            pending.push_back({child.second, std::move(child_validators)});
        }
    }
}

BlockTreeObservabilitySnapshot BlockTree::observabilitySnapshot() const {
    BlockTreeObservabilitySnapshot snapshot;
    snapshot.node_count   = node_pool_.size();
    snapshot.leaf_count   = leaf_count_;
    snapshot.branch_count = branch_count_;
    snapshot.content      = intervalSummary(content_intervals_);
    snapshot.recovery     = intervalSummary(recovery_intervals_);
    for (size_t tier_idx = 0; tier_idx < tier_intervals_.size(); ++tier_idx) {
        snapshot.tier[tier_idx] = intervalSummary(tier_intervals_[tier_idx]);
    }
    return snapshot;
}

}  // namespace rtp_llm
