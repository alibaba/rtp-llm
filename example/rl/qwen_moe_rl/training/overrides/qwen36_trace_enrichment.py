"""Associate gateway traces with responses actually received by the agent.

The upstream positional merger is retained after strict, session-local alignment.
This module uses only the standard library and never initializes CUDA.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time


class TraceLinkageError(RuntimeError):
    pass


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _mapping(value):
    if hasattr(value, 'model_dump'):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, dict):
        raise TraceLinkageError('Assistant/history entries must be dictionaries')
    return value


def _arguments(value):
    if isinstance(value, str):
        def reject(constant):
            raise TraceLinkageError(f'Nonfinite tool argument: {constant}')
        value = json.loads(value, parse_constant=reject)
    if not isinstance(value, dict):
        raise TraceLinkageError('Tool arguments must be a JSON object')
    # The canonical JSON serializer also rejects nonfinite values in dict input.
    _json(value)
    return value


def canonical_message(message, *, assistant=False):
    """Compare all prompt-bearing text and tool identity fields exactly.

    Reasoning/usage/transport fields are not part of this deployment's agent
    history. OpenAI JSON-string tool arguments and their dict representation are
    equivalent, but tool IDs, names, code text and content whitespace are exact.
    """
    message = _mapping(message)
    role = message.get('role', 'assistant' if assistant else None)
    if role not in ('system', 'user', 'assistant', 'tool'):
        raise TraceLinkageError(f'Unsupported history role: {role!r}')
    content = message.get('content')
    content = '' if content is None else content
    if not isinstance(content, str):
        raise TraceLinkageError('This text-only deployment requires string message content')
    result = {'role': role, 'content': content}
    if message.get('name') is not None:
        result['name'] = message['name']
    calls = message.get('tool_calls') or []
    if calls:
        if role != 'assistant' or not isinstance(calls, list):
            raise TraceLinkageError('Only assistant messages may contain a tool-call list')
        normalized = []
        for raw in calls:
            call = _mapping(raw)
            function = _mapping(call.get('function'))
            if not isinstance(call.get('id'), str) or not call['id']:
                raise TraceLinkageError('Tool call ID is missing')
            if call.get('type', 'function') != 'function' or not isinstance(function.get('name'), str) or not function['name']:
                raise TraceLinkageError('Tool function type/name is invalid')
            normalized.append({'id': call['id'], 'type': 'function', 'function': {
                'name': function['name'], 'arguments': _arguments(function.get('arguments', {})),
            }})
        result['tool_calls'] = normalized
    if role == 'tool':
        call_id = message.get('tool_call_id')
        if not isinstance(call_id, str) or not call_id:
            raise TraceLinkageError('Tool-return message is missing its call ID')
        result['tool_call_id'] = call_id
    return result


def _history(messages):
    if not isinstance(messages, list):
        raise TraceLinkageError('Complete agent/gateway history must be a list')
    return [canonical_message(message) for message in messages]


def _trace_summary(trace):
    messages = _history(_get(trace, 'messages', []))
    response = canonical_message(_get(trace, 'response_message', {}), assistant=True)
    prompt = _get(trace, 'prompt_token_ids', [])
    completion = _get(trace, 'completion_token_ids', [])
    values = _get(trace, 'logprobs') or []
    raw = struct.pack(f'<{len(values)}f', *values)
    return {
        'trace_id': _get(trace, 'trace_id'), 'session_id': _get(trace, 'session_id'),
        'response_id': (_get(trace, 'raw_response') or {}).get('id'),
        'request_history_sha256': _sha(_json(messages).encode()),
        'delivered_history_sha256': _sha(_json(messages + [response]).encode()),
        'assistant_sha256': _sha(_json(response).encode()),
        # Match the existing native generation receipt's JSON spacing.
        'prompt_sha256': _sha(json.dumps(prompt).encode()),
        'response_sha256': _sha(json.dumps(completion).encode()),
        'raw_logprobs_fp32_sha256': _sha(raw),
        'prompt_tokens': len(prompt), 'response_tokens': len(completion), 'logprobs': len(values),
        'weight_version': _get(trace, 'weight_version'), 'timestamp': _get(trace, 'timestamp'),
        'tool_call_ids': [call['id'] for call in response.get('tool_calls', [])],
    }


def align_traces(episode, traces, uid, audit):
    """Return the original trace objects in the exact delivered agent order."""
    traces = list(traces)
    summaries = [_trace_summary(trace) for trace in traces]
    audit['observed_traces'] = summaries
    ids = [item['trace_id'] for item in summaries]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise TraceLinkageError('Gateway trace IDs are missing or duplicated')
    if any(item['session_id'] != uid for item in summaries):
        raise TraceLinkageError('Gateway trace belongs to a different agent session')
    agent_steps = [step for trajectory in _get(episode, 'trajectories', [])
                   for step in _get(trajectory, 'steps', [])]
    if not agent_steps:
        raise TraceLinkageError('Delivered agent steps are required for strict trace linkage')
    audit['agent_steps'] = len(agent_steps)
    selected = []
    used = set()
    audit['selected'] = []
    for number, step in enumerate(agent_steps):
        history = _history(_get(step, 'chat_completions', []))
        if not history or history[-1]['role'] != 'assistant':
            raise TraceLinkageError(f'Agent step {number} lacks its delivered assistant/history')
        if _get(step, 'model_response', '') != history[-1]['content']:
            raise TraceLinkageError(f'Agent step {number} model_response differs from its delivered history')
        expected = _sha(_json(history).encode())
        candidates = [index for index, summary in enumerate(summaries)
                      if index not in used and summary['delivered_history_sha256'] == expected]
        if len(candidates) != 1:
            raise TraceLinkageError(f'Agent step {number}: expected one exact delivered trace, got {len(candidates)}')
        index = candidates[0]
        trace = traces[index]
        values = _get(trace, 'logprobs') or []
        if (not _get(trace, 'prompt_token_ids') or not _get(trace, 'completion_token_ids')
                or len(values) != len(_get(trace, 'completion_token_ids'))
                or any(not math.isfinite(value) for value in values)):
            raise TraceLinkageError(f'Agent step {number}: selected trace has incomplete token/logprob payload')
        used.add(index)
        selected.append(trace)
        audit['selected'].append(dict(summaries[index], agent_step=number, observed_trace_index=index))
    selected_requests = {summaries[index]['request_history_sha256'] for index in used}
    orphans = [dict(summary, observed_trace_index=index,
                    reason='same_request_retry_response_absent_from_delivered_agent_histories')
               for index, summary in enumerate(summaries) if index not in used]
    audit['orphaned'] = orphans
    if any(item['request_history_sha256'] not in selected_requests for item in orphans):
        raise TraceLinkageError('Unmatched trace is not a retry of an exactly selected request history')
    return selected


def _write_audit(audit):
    configured = os.environ.get('RLLM_QWEN36_TRACE_AUDIT_DIR') or os.environ.get('RLLM_RTP_TRACE_LINKAGE_DIR')
    if configured:
        root = Path(configured)
    else:
        parent = os.environ.get('RLLM_RTP_ACCEPTANCE_DIR')
        if not parent:
            raise TraceLinkageError('A persistent trace-linkage audit directory is required')
        root = Path(parent)/'trace-linkage'
    root.mkdir(parents=True, exist_ok=True)
    filename = f"trace-linkage-{_sha(audit['uid'].encode())[:24]}-{time.time_ns()}-{os.getpid()}.json"
    path = root/filename
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return path


def install(module):
    """Idempotent hook for the naturally imported agentflow_engine module."""
    original = module.enrich_episode_with_traces
    if getattr(original, '_qwen36_trace_linkage', False):
        return

    @functools.wraps(original)
    def enrich(episode, traces, uid, task, *, strict=True):
        audit = {'schema': 'qwen36-delivered-trace-linkage-v1', 'uid': uid,
                 'strict': strict, 'started_unix': time.time(), 'passed': False,
                 'selection': 'exact_full_delivered_history_and_assistant_tool_identity',
                 'tokens_logprobs_and_agent_artifacts_modified': False}
        try:
            selected = align_traces(episode, traces, uid, audit)
            result = original(episode, selected, uid, task, strict=strict)
            audit.update(passed=True, selected_count=len(selected), orphaned_count=len(audit['orphaned']))
        except Exception as error:
            audit.update(error_type=type(error).__name__, error=str(error), finished_unix=time.time())
            _write_audit(audit)
            raise module.EnrichMismatchError(f'[{uid}] strict delivered trace linkage failed: {error}') from error
        audit['finished_unix'] = time.time()
        _write_audit(audit)
        return result

    enrich._qwen36_trace_linkage = True
    module.enrich_episode_with_traces = enrich
