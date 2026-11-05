"""Smoke-only KV cap for actual SFA multi-chunk coverage.

Installed only by the instrumented K3.1 Prefill smoke launcher. Keep the
normal workspace budget, Q rectangle and request inventory unchanged.
"""
import contextvars
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path

MODULE = 'rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution'
FP8_PAGE_CAPACITY = 320
BF16_PAGE_CAPACITY = 512


def install(module):
    execution = module.DsaKVExecution
    if getattr(execution, '_k31_smoke_chunk_fixture', False):
        return
    page_capacity = contextvars.ContextVar('k31_sfa_chunk_fixture', default=None)
    original_plan = module.plan_dsa_kv_chunks
    original_init = execution.__init__

    @functools.wraps(original_plan)
    def plan(*args, **kwargs):
        limit = page_capacity.get()
        if limit is not None:
            current = kwargs.get('max_page_capacity')
            kwargs['max_page_capacity'] = limit if current is None else min(current, limit)
        return original_plan(*args, **kwargs)

    @functools.wraps(original_init)
    def prepare(self, params, name):
        limit = None
        if (name == 'sfa' and params.is_prefill and not params.is_cuda_graph
                and max(params.kv_lens_host, default=0) >= 650000):
            if not params.workspace_owner.sfa_use_fp8:
                limit = BF16_PAGE_CAPACITY
            elif 1024 <= params.query_positions.numel() <= 1088:
                limit = FP8_PAGE_CAPACITY
        token = page_capacity.set(limit)
        try:
            original_init(self, params, name)
            if limit is not None:
                print('[K31_SMOKE_CHUNK_FIXTURE] ' + json.dumps({
                    'rank': os.environ.get('WORLD_RANK'), 'phase': name,
                    'local_query_rows': params.query_positions.numel(),
                    'kv_tokens': sum(params.kv_lens_host),
                    'max_page_capacity': limit,
                    'budget_bytes': params.workspace_owner.budgets[name],
                    'actual_chunks': len(self.plan.blocks),
                }), flush=True)
        finally:
            page_capacity.reset(token)

    module.plan_dsa_kv_chunks = plan
    execution.__init__ = prepare
    execution._k31_smoke_chunk_fixture = True


def register():
    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            install(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                raise ImportError(MODULE)
            spec.loader = Loader(spec.loader)
            return spec

    import sys
    sys.meta_path.insert(0, Finder())


def fixture_environment(run_dir, environment):
    """Bootstrap in every Python worker, including spawned rank processes."""
    directory = Path(run_dir) / 'chunk-fixture'
    directory.mkdir()
    source = str(Path(__file__).resolve())
    hook = directory / 'sitecustomize.py'
    # Load the stdlib-only helper before the server's imports initialize CUDA.
    hook.write_text(
        'import importlib.util\n'
        f'_s=importlib.util.spec_from_file_location("k31_chunk_fixture", {source!r})\n'
        '_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m);_m.register()\n'
    )
    previous = environment.get('PYTHONPATH', '')
    return {'PYTHONPATH': str(directory) + (os.pathsep + previous if previous else '')}
