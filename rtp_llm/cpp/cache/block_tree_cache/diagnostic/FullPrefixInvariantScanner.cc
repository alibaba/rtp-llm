#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <utility>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

const char* transferStateName(GroupSetTransferState state) {
    switch (state) {
        case GroupSetTransferState::IDLE:
            return "IDLE";
        case GroupSetTransferState::DEMOTING:
            return "DEMOTING";
        case GroupSetTransferState::LOAD_PENDING:
            return "LOAD_PENDING";
        case GroupSetTransferState::LOADING:
            return "LOADING";
    }
    return "UNKNOWN";
}

bool isIdleAttached(const GroupSetResource& resource) {
    return resource.transfer_state == GroupSetTransferState::IDLE && !resource.transfer_detached;
}

std::string formatNodeBrief(const NodeBrief& brief) {
    std::ostringstream oss;
    oss << (brief.tier == Tier::NONE ? "EMPTY" : tierName(brief.tier)) << "(key=" << brief.cache_key;
    if (brief.transfer_state != GroupSetTransferState::IDLE) {
        oss << ",state=" << transferStateName(brief.transfer_state);
    }
    oss << ")";
    return oss.str();
}

std::string formatTierMask(uint8_t tier_mask) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if ((tier_mask & (1U << static_cast<uint8_t>(tier))) == 0) {
            continue;
        }
        if (!first) {
            oss << ",";
        }
        oss << tierName(tier);
        first = false;
    }
    oss << "]";
    return oss.str();
}

}  // namespace

const char* fullViolationTypeName(FullViolationType type) {
    switch (type) {
        case FullViolationType::LOWER_TO_DEVICE:
            return "lower_to_device";
        case FullViolationType::GAP_TO_DATA:
            return "gap_to_data";
        case FullViolationType::INVALID_RESOURCE:
            return "invalid_resource";
    }
    return "unknown";
}

const char* invalidResourceReasonName(InvalidResourceReason reason) {
    switch (reason) {
        case InvalidResourceReason::NONE:
            return "none";
        case InvalidResourceReason::MULTI_TIER:
            return "multi_tier";
        case InvalidResourceReason::PARTIAL_DEVICE:
            return "partial_device";
        case InvalidResourceReason::IDLE_DETACHED:
            return "idle_detached";
        case InvalidResourceReason::BUSY_EMPTY:
            return "busy_empty";
    }
    return "unknown";
}

InvalidResourceReason invalidResourceReason(const GroupSetResource& resource) {
    if (resource.servingTierCount() > 1) {
        return InvalidResourceReason::MULTI_TIER;
    }
    if (resource.hasTier(Tier::DEVICE) && !resource.hasCompleteDeviceValue()) {
        return InvalidResourceReason::PARTIAL_DEVICE;
    }
    if (resource.transfer_detached && resource.transfer_state == GroupSetTransferState::IDLE) {
        return InvalidResourceReason::IDLE_DETACHED;
    }
    if (resource.transfer_state != GroupSetTransferState::IDLE && resource.is_empty()) {
        return InvalidResourceReason::BUSY_EMPTY;
    }
    return InvalidResourceReason::NONE;
}

NodeBrief makeNodeBrief(const TreeNode& node, const GroupSetResource& resource) {
    NodeBrief brief;
    brief.cache_key      = node.cache_key;
    brief.tier           = resource.getTopTier();
    brief.transfer_state = resource.transfer_state;
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if (resource.hasTier(tier)) {
            brief.tier_mask |= static_cast<uint8_t>(1U << static_cast<uint8_t>(tier));
        }
    }
    return brief;
}

void detectNodeViolations(const BlockTree& tree, const TreeNode& node, std::vector<FullViolationDetail>& details) {
    const std::vector<GroupSetPtr>& group_sets      = tree.groupSets();
    const size_t                    group_set_count = std::min(group_sets.size(), node.group_set_resources.size());

    for (size_t group_set_id = 0; group_set_id < group_set_count; ++group_set_id) {
        const GroupSetResource& current = node.group_set_resources[group_set_id];

        const InvalidResourceReason reason = invalidResourceReason(current);
        if (reason != InvalidResourceReason::NONE) {
            FullViolationDetail detail;
            detail.group_set_id = group_set_id;
            detail.type         = FullViolationType::INVALID_RESOURCE;
            detail.reason       = reason;
            detail.stable       = current.transfer_state == GroupSetTransferState::IDLE;
            detail.current      = makeNodeBrief(node, current);
            details.push_back(detail);
            continue;
        }

        if (group_sets[group_set_id]->groupType() != CacheGroupType::FULL) {
            continue;
        }
        // The root is a sentinel: root(EMPTY) -> first data node is not a gap.
        const TreeNode* parent = node.parent;
        if (parent == nullptr || parent == tree.root() || group_set_id >= parent->group_set_resources.size()) {
            continue;
        }
        const GroupSetResource& parent_resource = parent->group_set_resources[group_set_id];
        if (invalidResourceReason(parent_resource) != InvalidResourceReason::NONE) {
            continue;
        }

        const Tier        parent_tier = parent_resource.getTopTier();
        FullViolationType type;
        if (current.hasCompleteDeviceValue() && (parent_tier == Tier::HOST || parent_tier == Tier::DISK)) {
            type = FullViolationType::LOWER_TO_DEVICE;
        } else if (!current.is_empty() && parent_resource.is_empty()
                   && parent_resource.transfer_state == GroupSetTransferState::IDLE) {
            type = FullViolationType::GAP_TO_DATA;
        } else {
            continue;
        }

        FullViolationDetail detail;
        detail.group_set_id = group_set_id;
        detail.type         = type;
        detail.stable       = isIdleAttached(parent_resource) && isIdleAttached(current);
        detail.parent       = makeNodeBrief(*parent, parent_resource);
        detail.current      = makeNodeBrief(node, current);
        details.push_back(detail);
    }
}

std::string formatViolationDetail(const FullViolationDetail& detail, int world_rank, int local_rank) {
    std::ostringstream oss;
    if (detail.type == FullViolationType::INVALID_RESOURCE) {
        oss << "event=block_tree_resource_violation"
            << " group_set_id=" << detail.group_set_id << " type=" << fullViolationTypeName(detail.type)
            << " reason=" << invalidResourceReasonName(detail.reason)
            << " status=" << (detail.stable ? "stable" : "transient") << " key=" << detail.current.cache_key
            << " tiers=" << formatTierMask(detail.current.tier_mask)
            << " transfer_state=" << transferStateName(detail.current.transfer_state);
    } else {
        oss << "event=block_tree_full_violation"
            << " group_set_id=" << detail.group_set_id << " type=" << fullViolationTypeName(detail.type)
            << " status=" << (detail.stable ? "stable" : "transient") << " edge=[" << formatNodeBrief(detail.parent)
            << " -> " << formatNodeBrief(detail.current) << "]";
    }
    oss << " world_rank=" << world_rank << " local_rank=" << local_rank;
    return oss.str();
}

FullPrefixInvariantScanner::FullPrefixInvariantScanner(const BlockTree&      tree,
                                                       std::mutex&           cache_mutex,
                                                       FullPrefixScanOptions options):
    tree_(tree), cache_mutex_(cache_mutex), options_([](FullPrefixScanOptions o) {
        o.nodes_per_round       = std::clamp(o.nodes_per_round, size_t{1}, kFullPrefixScanNodesPerRoundLimit);
        o.max_details_per_cycle = std::min(o.max_details_per_cycle, kFullPrefixScanMaxDetailsPerCycle);
        return o;
    }(std::move(options))) {}

FullPrefixInvariantScanner::~FullPrefixInvariantScanner() {
    stop();
}

bool FullPrefixInvariantScanner::start() {
    if (options_.interval_ms <= 0) {
        return false;
    }
    loop_thread_ = autil::LoopThread::createLoopThread(
        [this] { runBatchGuarded(); }, options_.interval_ms * 1000, "BlockTreeScan", /*strictMode=*/true);
    if (!loop_thread_) {
        RTP_LLM_LOG_ERROR("failed to create FULL prefix invariant scanner loop thread");
        return false;
    }
    return true;
}

void FullPrefixInvariantScanner::stop() {
    stopping_.store(true, std::memory_order_release);
    if (loop_thread_) {
        loop_thread_->stop();
        loop_thread_.reset();
    }
}

FullPrefixScanStats FullPrefixInvariantScanner::stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FullPrefixInvariantScanner::runBatchGuarded() {
    try {
        runOneBatch();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("FULL prefix invariant scan round failed, dropping details: %s", e.what());
    }
}

void FullPrefixInvariantScanner::runOneBatch() {
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.batches_started;
    }

    std::vector<FullViolationDetail> details;
    size_t                           nodes_scanned  = 0;
    size_t                           tree_size      = 0;
    bool                             cycle_complete = false;

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (stopping_.load(std::memory_order_acquire)) {
            return;
        }
        if (!cycle_active_) {
            cycle_active_    = true;
            cursor_          = 0;
            cycle_end_index_ = tree_.size();
        }
        const BlockTreeNodeRangeResult range =
            tree_.visitNodeRangeLocked(cursor_, cycle_end_index_, options_.nodes_per_round, [&](const TreeNode& node) {
                detectNodeViolations(tree_, node, details);
            });
        cursor_        = range.next_cursor;
        nodes_scanned  = range.visited;
        tree_size      = range.tree_size;
        cycle_complete = range.cycle_complete;
    }

    publishBatch(details, nodes_scanned, tree_size, cycle_complete);
}

void FullPrefixInvariantScanner::publishBatch(const std::vector<FullViolationDetail>& details,
                                              size_t                                  nodes_scanned,
                                              size_t                                  tree_size,
                                              bool                                    cycle_complete) {
    cycle_tree_size_ = tree_size;
    cycle_nodes_scanned_ += nodes_scanned;

    for (const FullViolationDetail& detail : details) {
        if (detail.stable) {
            ++cycle_stable_;
        } else {
            ++cycle_transient_;
        }
        // Past the cap the violation is dropped rather than remembered, which is what keeps
        // scanner state constant when a large part of the tree is anomalous. The summary
        // still reports the full count.
        if (cycle_details_logged_ >= options_.max_details_per_cycle) {
            ++cycle_details_suppressed_;
            continue;
        }
        ++cycle_details_logged_;
        const std::string line = formatViolationDetail(detail, options_.world_rank, options_.local_rank);
        if (detail.stable) {
            RTP_LLM_LOG_WARNING("%s", line.c_str());
        } else {
            RTP_LLM_LOG_INFO("%s", line.c_str());
        }
    }

    uint64_t cycle_index = 0;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.cycle_active         = cycle_active_ && !cycle_complete;
        stats_.nodes_scanned        = cycle_complete ? 0 : cycle_nodes_scanned_;
        stats_.stable_violations    = cycle_complete ? 0 : cycle_stable_;
        stats_.transient_violations = cycle_complete ? 0 : cycle_transient_;
        stats_.details_logged       = cycle_complete ? 0 : cycle_details_logged_;
        stats_.details_suppressed   = cycle_complete ? 0 : cycle_details_suppressed_;
        if (cycle_complete) {
            cycle_index = stats_.cycles_completed++;
        }
    }
    if (!cycle_complete) {
        return;
    }

    const std::string summary =
        "event=block_tree_full_scan cycle=" + std::to_string(cycle_index)
        + " nodes_scanned=" + std::to_string(cycle_nodes_scanned_) + " tree_nodes=" + std::to_string(cycle_tree_size_)
        + " violations=" + std::to_string(cycle_stable_ + cycle_transient_) + " stable=" + std::to_string(cycle_stable_)
        + " transient=" + std::to_string(cycle_transient_) + " details_logged=" + std::to_string(cycle_details_logged_)
        + " details_suppressed=" + std::to_string(cycle_details_suppressed_)
        + " world_rank=" + std::to_string(options_.world_rank) + " local_rank=" + std::to_string(options_.local_rank);
    if (cycle_stable_ > 0) {
        RTP_LLM_LOG_WARNING("%s", summary.c_str());
    } else {
        RTP_LLM_LOG_INFO("%s", summary.c_str());
    }

    cycle_active_             = false;
    cursor_                   = 0;
    cycle_end_index_          = 0;
    cycle_tree_size_          = 0;
    cycle_nodes_scanned_      = 0;
    cycle_stable_             = 0;
    cycle_transient_          = 0;
    cycle_details_logged_     = 0;
    cycle_details_suppressed_ = 0;
}

}  // namespace rtp_llm
