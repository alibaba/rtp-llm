import argparse
import ast
import json
import types
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--worker", type=Path, required=True)
args = parser.parse_args()
source = args.worker.read_text()
tree = ast.parse(source)
cls = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "NixlBaseConnectorWorker"
)
method = next(
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "_compute_desc_ids"
)
method.decorator_list = []


class AttentionSpec:
    pass


class CircularBufferSpec(AttentionSpec):
    pass


class MambaSpec:
    pass


namespace = dict(
    np=np,
    CircularBufferSpec=CircularBufferSpec,
    _SHARED_REGION_GROUP_ID=-1,
    _is_attention_spec=lambda t: issubclass(t, AttentionSpec),
    _is_ssm_spec=lambda t: issubclass(t, MambaSpec),
    BlockIds=list,
)
unit = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        method,
    ],
    type_ignores=[],
)
ast.fix_missing_locations(unit)
exec(compile(unit, "actual_nixl_descriptor_method", "exec"), namespace)
x = types.SimpleNamespace(
    _has_mamba=True,
    _conv_decomp=types.SimpleNamespace(
        local_conv_offsets=[(0, 16), (16, 16), (32, 32)]
    ),
    _ssm_region_indices=[3, 4],
    block_len_per_layer=[100] * 7,
    _transfer_layer_group_ids=[],
    num_regions=7,
    region_group_ids=[0, 1, 2, 3, 3, 4, 4],
    _group_spec_types=[
        AttentionSpec,
        CircularBufferSpec,
        AttentionSpec,
        MambaSpec,
        MambaSpec,
    ],
    _scratch_region_indices=[1],
    _ple_region_indices=[5, 6],
)
fn = namespace["_compute_desc_ids"]
expected = [2, 16, 29] + [70 + i * 10 + 3 for i in range(8)] + [155, 165]
actual = fn(x, [[2], [6], [9], [3], [5]], 10, None, 1).tolist()
assert actual == expected, (actual, expected)
empty = fn(x, [[], [], [], [], []], 10, None, 1)
assert empty.size == 0
partial = fn(x, [[], [], [], [1], []], 10, None, 1).tolist()
assert partial == [70 + i * 10 + 1 for i in range(8)], partial
replicas = fn(x, [[], [], [], [], [2, 7]], 10, None, 1).tolist()
assert replicas == [152, 157, 162, 167], replicas
uneven = fn(
    x,
    [[2], [6], [9], [], []],
    10,
    None,
    1,
    region_num_blocks=[7, 10, 10, 10, 10, 10, 10],
).tolist()
assert uneven == [2, 13, 26], uneven
out = dict(
    passed=True,
    tests=5,
    scope="actual patched descriptor method AST;mainKV/compressed/ring/GDN pieces/two replicated PLE owners isolated;CPU only,no NIXL transport or model admission",
)
print(json.dumps(out, indent=2))
