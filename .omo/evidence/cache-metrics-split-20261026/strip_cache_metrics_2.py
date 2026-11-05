# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: python3 /tmp/strip_cache_metrics_2.py from the assigned public worktree.
from pathlib import Path
import re
import subprocess

def span(text: str, start: str, end: str, replacement: str = '') -> str:
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + replacement + text[b:]

p = Path('rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc')
s = p.read_text()
old = subprocess.check_output(['git', 'show', '61acca1d:'+str(p)], text=True)
restored = old[old.index('std::optional<absl::Status> StreamCacheResource::pollAllocatorLoad'):old.index('bool StreamCacheResource::asyncLoadCache')].replace('prefixObservationEnabled()', 'false')
s = s.replace('bool StreamCacheResource::asyncLoadCache', restored+'bool StreamCacheResource::asyncLoadCache')
p.write_text(s)
p = Path('rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.cc')
s = p.read_text()
s = span(s, 'namespace {', 'BlockTreeStorer::BlockTreeStorer')
s = span(s, 'void BlockTreeStorer::observePublicationSkipped', 'void BlockTreeStorer::stopAdmissionLocked')
s = span(s, '    const bool plan_reached', '    const BlockTreeInsertResult insert_result')
s = span(s, '    const auto unfulfilled', '    const bool tree_data_mutated')
s = span(s, '    const bool plan_reached', '    StoreTaskPtr task', '    if (stopping_.load()) {\n        return;\n    }\n\n')
s = s.replace('token_segments ? *token_segments : std::vector<std::vector<int>>{},\n        plan_points', 'token_segments ? *token_segments : std::vector<std::vector<int>>{}')
s = re.sub(r'^.*task->observation_request_id.*\n', '', s, flags=re.M)
s = s.replace('    const char* failure_reason = "prepare_failed";\n', '')
s = s.replace('[this, &task, &failure_reason]() { settleLocked(*task, /*publish=*/false, failure_reason); }', '[this, &task]() { settleLocked(*task, /*publish=*/false); }')
s = re.sub(r'^.*task->error = ErrorInfo.*\n', '', s, flags=re.M)
s = span(s, '        unfulfilled   =', '    store_task_runner_.releaseTaskResources', '    }\n\n')
p.write_text(s)
# Remove observation-only parameters consistently from declarations and calls.
for name in ['rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.cc', 'rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h', 'rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc', 'rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h']:
    p = Path(name)
    s = p.read_text()
    s = re.sub(r',\s*PrefixCachePlanState\*\s+prefix_plan_state(?: = nullptr)?', '', s)
    s = re.sub(r',\s*std::optional<int64_t> observation_request_id(?: = std::nullopt)?', '', s)
    s = re.sub(r',\s*prefix_plan_state', '', s)
    s = re.sub(r',\s*observation_request_id', '', s)
    s = re.sub(r'^.*void observePublicationSkipped.*\n', '', s, flags=re.M)
    s = re.sub(r'^.*BlockTreeObservabilitySnapshot.*\n', '', s, flags=re.M)
    if name.endswith('BlockTreeCache.cc'):
        s = span(s, 'void BlockTreeCache::observePublicationSkipped', 'int BlockTreeCache::evictForGroup')
    if name.endswith('BlockTreeStorer.h'):
        s = re.sub(r',\s*const char\*\s+failure_reason', '', s)
        s = s.replace('settleTask(StoreTask& task', 'settleTask(const StoreTask& task').replace('settleLocked(StoreTask& task', 'settleLocked(const StoreTask& task')
    p.write_text(s)
p = Path('rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h')
s = subprocess.check_output(['git', 'show', 'ec247b9d61^:'+str(p)], text=True)
s = s.replace('Task(Tier target, CacheKeysType keys, std::chrono::milliseconds timeout):\n            target_tier(target), cache_keys(std::move(keys)), transfer_task({}, timeout) {}', 'Task(Tier target, CacheKeysType keys, std::chrono::milliseconds timeout,\n             std::vector<std::vector<int>> tokens = {}):\n            target_tier(target), cache_keys(std::move(keys)), token_segments(std::move(tokens)), transfer_task({}, timeout) {}')
s = s.replace('        CacheKeysType cache_keys;', '        CacheKeysType cache_keys;\n        std::vector<std::vector<int>> token_segments;')
p.write_text(s)
p = Path('rtp_llm/cpp/cache/CoordinatorCacheManager.cc')
s = p.read_text()
s = re.sub(r'        if \(prepared.match_observation.has_value\(\)\) \{\n.*?\n        }\n', '', s, flags=re.S)
s = re.sub(r'^.*(?:prepared.match_observation =|result.match_observation =).*\n', '', s, flags=re.M)
s = re.sub(r'            block_tree_cache_->observePublicationSkipped\([^;]+;\n', '', s)
s = re.sub(r',\s*insert_info.prefix_plan_state', '', s)
s = re.sub(r',\s*insert_info.observation_request_id', '', s)
s = span(s, '        std::vector<std::vector<int>> token_segments;\n        if (block_tree_cache_->prefixObservabilityEnabled()', '        if (insert_info.is_resident)')
s = s.replace('                                          &token_segments,\n                                          nullptr', '                                          nullptr,\n                                          nullptr')
s = s.replace('        } else {\n        }\n        return;', '        }\n        return;')
p.write_text(s)
for name in ['rtp_llm/cpp/cache/CoordinatorCacheManager.h', 'rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h', 'rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h', 'rtp_llm/cpp/engine_base/stream/StreamCacheResource.h']:
    p = Path(name)
    s = p.read_text()
    s = re.sub(r'^.*(?:PrefixCacheMatchObservation|PrefixCacheRemoteObservation|PrefixCachePlanState|prefix_summary_emitted_|prefix_plan_event_emitted_).*\n', '', s, flags=re.M)
    p.write_text(s)
for name in ['rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h', 'rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.cc']:
    p = Path(name)
    s = p.read_text()
    s = re.sub(r'^\s*std::vector<std::vector<bool>>& group_valid,\n', '', s, flags=re.M)
    s = re.sub(r'^.*group_valid(?:\.assign|\[i\]).*\n', '', s, flags=re.M)
    s = s.replace('createMatchResult(path, cache_keys, valid_prefix, request_tokens)', 'createMatchResult(path, cache_keys, valid_prefix)')
    s = re.sub(r',\n\s*const std::vector<int>\*\s+request_tokens\);', ');', s)
    p.write_text(s)
