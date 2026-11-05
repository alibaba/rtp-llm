# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: python3 /tmp/strip_cache_metrics.py from the assigned public worktree.
from pathlib import Path
import re

def span(text: str, start: str, end: str, replacement: str = '') -> str:
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + replacement + text[b:]

def edit(name: str, pairs: list[tuple[str, str]]) -> None:
    path = Path(name)
    text = path.read_text()
    for old, new in pairs:
        if old not in text:
            raise RuntimeError(f'Missing edit anchor in {name}: {old}')
        text = text.replace(old, new)
    path.write_text(text)

p = Path('rtp_llm/cpp/cache/block_tree_cache/BlockTree.cc')
s = p.read_text()
s = span(s, 'namespace {\n', 'BlockTree::BlockTree')
s = span(s, 'size_t BlockTree::tierIndex', 'TreeNode* BlockTree::createNode')
s = span(s, 'std::vector<PrefixCachePlanPoint> BlockTree::planRecoveryPoints', 'std::vector<TreeNode*> BlockTree::findNode')
s = s.replace('size_t                    reuse_unit_tokens,\n                     bool                      enable_observability,\n                     std::string               observability_instance)', 'size_t                    reuse_unit_tokens)')
s = span(s, '    reuse_unit_tokens_(reuse_unit_tokens),', '    for (size_t group_set_id', '    reuse_unit_tokens_(reuse_unit_tokens) {\n    RTP_LLM_CHECK_WITH_INFO(reuse_unit_tokens_ > 0, "BlockTree reuse unit must be positive");\n')
s = s.replace('    if (observability_enabled_) {\n        ++leaf_count_;\n    }\n    startInterval(content_intervals_, node->observation.content_since_us);\n', '')
s = s.replace('(observability_enabled_ || token_ends != nullptr)', '(token_ends != nullptr)')
s = s.replace('                updateTierIntervals(*current, "store");\n', '')
s = span(s, 'void BlockTree::refreshPublishedState(', '    for (const auto& location : publication_groups_)', 'void BlockTree::refreshPublishedState(const TreeNode* node) {\n    if (!event_publisher_ || node->parent == nullptr) {\n        return;\n    }\n')
p.write_text(s)
p = Path('rtp_llm/cpp/cache/block_tree_cache/BlockTree.h')
s = p.read_text()
s = span(s, 'struct BlockTreeAgeSummary', 'class BlockTree {')
s = span(s, '    std::vector<PrefixCachePlanPoint> planRecoveryPoints', '    bool isLeafAtTier')
s = span(s, '    struct IntervalSet', '    std::vector<GroupSetPtr>')
s = s.replace('size_t                    reuse_unit_tokens = 1,\n                       bool                      enable_observability = false,\n                       std::string               observability_instance = {}', 'size_t                    reuse_unit_tokens = 1')
s = s.replace('    CacheKeysType cache_keys_;', '    size_t reuse_unit_tokens_{1};\n    CacheKeysType cache_keys_;')
p.write_text(s)
p = Path('rtp_llm/cpp/cache/Types.h')
s = p.read_text()
s = span(s, 'enum class PrefixCacheMediumStatus', 'struct PrefixCachePlanPoint')
s = span(s, 'enum class PrefixCachePlanState', 'struct MallocInfo')
s = span(s, '    // Optional request-owned plan-state sink.', '    const PrefixCacheRegistration*')
s = re.sub(r'^.*observation_request_id.*\n', '', s, flags=re.M)
s = s.replace('        // Owned full-input token snapshot, populated only when complete prefix\n        // observability is enabled. Empty means content matching is unformed.', '        // Owned token snapshot for functional local short-leaf lookup.')
p.write_text(s)
p = Path('rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc')
s = p.read_text()
s = span(s, 'std::string jsonEscape', 'std::shared_ptr<const MallocInfo::RequestLayout>')
s = span(s, '    if (prefixObservationEnabled()) {', '    if (total_blocks > 0) {')
s = span(s, 'std::string StreamCacheResource::prefixObservationSummaryPayload', 'bool StreamCacheResource::', '')
s = re.sub(r'^.*insert_info\.(prefix_plan_state|observation_request_id).*\n', '', s, flags=re.M)
s = s.replace('prefixObservationEnabled() || (is_first_malloc && partialTailReuseEnabled())', '(is_first_malloc && partialTailReuseEnabled())')
s = s.replace('prefixObservationEnabled()', 'false')
p.write_text(s)
p = Path('rtp_llm/cpp/cache/KVCacheManager.cc')
s = p.read_text()
s = span(s, '    tree_kv_cache_config.enable_prefix_cache_observability =', '    block_tree_cache_ =')
p.write_text(s)
