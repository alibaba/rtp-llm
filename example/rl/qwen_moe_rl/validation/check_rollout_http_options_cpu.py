#!/usr/bin/env python3
"""Check only the Python Tool HTTP option patch, with a fake CPU-only client."""
from __future__ import annotations
import argparse
import ast
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ORIGINAL_SHA = '8caa1c78082d8e0f26427f10c0cf8c36ed1568d6da3562c446b4a8e8301f135e'
SANDBOX_SHA = 'b53b81e16baaa4401c17ef0ad470a1207fe3cafda0f0a3af2ae66573990b142b'
ENV_NAMES = ('RLLM_PYTHON_TOOL_HTTP_TIMEOUT', 'RLLM_PYTHON_TOOL_HTTP_MAX_RETRIES')


def normalized_patch(tree):
    tree = copy.deepcopy(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'python_math_agent')
    for statement, name in zip(fn.body[:2], ('request_timeout', 'http_max_retries')):
        assert isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == name
    assert isinstance(fn.body[2], ast.If) and isinstance(fn.body[2].body[0], ast.Raise)
    fn.body = fn.body[3:]
    counts = {'client': 0, 'request': 0}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call): continue
        if isinstance(node.func, ast.Name) and node.func.id == 'AsyncOpenAI':
            keyword = next(k for k in node.keywords if k.arg == 'max_retries')
            assert isinstance(keyword.value, ast.Name) and keyword.value.id == 'http_max_retries'
            keyword.value = ast.Constant(1)
            counts['client'] += 1
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'create':
            keyword = next(k for k in node.keywords if k.arg == 'timeout')
            assert isinstance(keyword.value, ast.Name) and keyword.value.id == 'request_timeout'
            keyword.value = ast.Constant(180)
            counts['request'] += 1
    assert counts == {'client': 1, 'request': 1}
    return tree


def mocked_call(source, environment):
    constructors, requests = [], []
    class Client:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def create(self, **kwargs):
            requests.append(kwargs)
            message = SimpleNamespace(content=r'\boxed{2}', tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(completion_tokens=4))
    parsed = ast.parse(source)
    body = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('TOOLS', 'SYSTEM_PROMPT') for t in node.targets):
            body.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'extract_boxed': body.append(node)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'python_math_agent':
            node.decorator_list = []
            body.append(node)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *body], type_ignores=[])
    namespace = dict(os=os, AsyncOpenAI=Client, Episode=lambda **k: SimpleNamespace(**k),
                     Trajectory=lambda **k: SimpleNamespace(**k), Step=lambda **k: SimpleNamespace(**k))
    exec(compile(ast.fix_missing_locations(tree), 'actual_python_math_agent_cpu_mock', 'exec'), namespace)
    isolated = dict(os.environ)
    for name in ENV_NAMES: isolated.pop(name, None)
    isolated.update(environment)
    with patch.dict(os.environ, isolated, clear=True):
        episode = asyncio.run(namespace['python_math_agent'](
            SimpleNamespace(instruction='What is 1 + 1?'),
            SimpleNamespace(base_url='http://cpu-only.invalid/v1', model='unchanged-model')))
    assert episode.artifacts['answer'] == '2'
    assert len(constructors) == len(requests) == 1
    return constructors[0], requests[0]


def check(agent, baseline, sandbox):
    original = baseline.read_bytes()
    current = agent.read_bytes()
    assert hashlib.sha256(original).hexdigest() == ORIGINAL_SHA
    assert hashlib.sha256(sandbox.read_bytes()).hexdigest() == SANDBOX_SHA
    assert ast.dump(normalized_patch(ast.parse(current)), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    old_client, old_request = mocked_call(original, {})
    default_client, default_request = mocked_call(current, {})
    assert (old_client, old_request) == (default_client, default_request)
    client, request = mocked_call(current, dict(zip(ENV_NAMES, ('1800', '0'))))
    assert client == dict(old_client, max_retries=0)
    assert request == dict(old_request, timeout=1800.0)
    for invalid in ({ENV_NAMES[0]: 'nan'}, {ENV_NAMES[1]: '-1'}):
        try: mocked_call(current, invalid)
        except ValueError: pass
        else: raise AssertionError('Invalid HTTP options accepted')
    execute = next(n for n in ast.parse(current).body if isinstance(n, ast.FunctionDef) and n.name == 'execute_python')
    worker = next(n for n in execute.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'worker')
    selected = eval(compile(ast.Expression(worker.value), 'actual_mirror_worker_path', 'eval'), {'Path': Path, '__file__': str(agent)})
    assert selected == sandbox.resolve() and selected.parent == agent.resolve().parent
    return dict(schema='q397-rollout-http-options-cpu-v1', passed=True,
                checks=['original source SHA', 'unchanged sandbox SHA and colocated path',
                        'whole AST unchanged except two HTTP options and their validation',
                        'legacy defaults 180 seconds/one SDK retry preserved',
                        '397 client max_retries=0', '397 create timeout=1800 seconds',
                        'request payload otherwise byte-equivalent Python values', 'invalid options rejected'],
                original_agent_sha256=ORIGINAL_SHA, patched_agent_sha256=hashlib.sha256(current).hexdigest(),
                sandbox_sha256=SANDBOX_SHA, agent_source=str(agent.resolve()),
                mock_client_kwargs=client, mock_create_timeout=request['timeout'],
                network_performed=False, gpu_work_performed=False,
                scope='Mock only: actual agent body executes with a fake one-response client; no reward or native generation claimed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--agent', type=Path, default=root/'references/agent/python_math_agent.py')
    parser.add_argument('--baseline', type=Path, default=root/'provenance/original-agent/python_math_agent.py')
    parser.add_argument('--sandbox', type=Path, default=root/'references/agent/sandbox_worker.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.agent, args.baseline, args.sandbox)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Q397_ROLLOUT_HTTP_CPU_PASS', json.dumps(result), flush=True)


if __name__ == '__main__': main()
