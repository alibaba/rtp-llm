"""Narrow stdlib checks of actual policy/transaction code, without device imports.

The list-backed storage below tests ownership and publication only. It is not
a simulated model/provider or device correctness evidence. Native smoke is L0's.
"""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

PPU = Path(__file__).resolve().parents[1]

# Load the import-light contract directly; do not import models_py's device
# bootstrap in this stdlib-only source check.
_spec = importlib.util.spec_from_file_location(
    "rtp_llm.models_py.modules.dsv41.ppu.shared_dspark_modes", PPU / "shared_dspark_modes.py"
)
_modes = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _modes
_spec.loader.exec_module(_modes)


class Vector:
    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values), len(self.values[0])) if self.values and isinstance(self.values[0], list) else (len(self.values),)

    def cpu(self):
        return self

    def reshape(self, *shape):
        return self

    def tolist(self):
        return self.values

    def clone(self):
        return Vector(self.values)

    def data_ptr(self):
        return 123

    def any(self):
        return any(self.values)

    def numel(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, Vector):
            return Vector([v for v, keep in zip(self.values, index.values) if keep])
        return self.values[index]

    def __ge__(self, other):
        return Vector([v >= other for v in self.values])

    def __iand__(self, other):
        self.values = [a and b for a, b in zip(self.values, other.values)]
        return self


class NativeStorage:
    @staticmethod
    def _copy_rows(cache, key, request, positions, rows=None):
        addresses = cache._coordinates(key, request, positions)
        if rows is None:
            return Vector([cache.storage[a] for a in addresses])
        if cache.fail_after is not None:
            cache.fail_after -= 1
            if cache.fail_after == 0:
                cache.fail_after = None
                raise RuntimeError("fixture publication failure")
        for address, value in zip(addresses, rows.values):
            cache.storage[address] = value


def load_transaction():
    tree = ast.parse((PPU / "shared_verify.py").read_text())
    names = {"accepted_position", "SharedCacheVersion", "SharedVerifyTransaction"}
    selected = ast.Module(body=[x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names], type_ignores=[])
    torch = SimpleNamespace(tensor=lambda values, **kwargs: Vector(values), bool=bool,
                            equal=lambda a, b: a.values == b.values)
    namespace = {"dataclass": dataclass, "torch": torch, "SharedNativeCache": NativeStorage}
    exec(compile(selected, str(PPU / "shared_verify.py"), "exec"), namespace)
    return namespace


CODE = load_transaction()


def fixture(*, duplicate=False):
    keys = [(0, "swa_kv"), (1, "global_kv_2"), (1, "indexer_kv"), (1, "csa_state")]
    geometry = SimpleNamespace(ratio=lambda layer: 2 if layer else 0, engram_enabled=lambda: True, engram_max_ngram_size=4)
    pools = {key: SimpleNamespace(tensor=Vector([0]), table=Vector([1]), entries=2 if key[1] == "csa_state" else 128) for key in keys}
    cache = SimpleNamespace(pools=pools, geometry=geometry, device="cpu", journal=[], storage={}, fail_after=None)

    def coordinates(key, request, positions):
        return [(key, request, p // 2 if key[1] in ("global_kv_2", "indexer_kv") else p % pools[key].entries) for p in positions.values]

    cache._coordinates = coordinates
    for key in keys:
        positions = Vector([256, 258] if key[1] in ("global_kv_2", "indexer_kv") else [257, 258, 259, 260])
        for address in coordinates(key, 0, positions):
            cache.storage[address] = -100
        cache.journal.append(CODE["SharedCacheVersion"](key, 0, positions, Vector(list(range(1, positions.numel() + 1)))))
    layout = SimpleNamespace(prefixes=(257,), lengths=(4,), starts=(0,), rows=4)
    windows = Vector([[100, 11, 10, 9], [101, 100, 11, 10], [102, 101, 100, 11], [103, 102, 101, 100]])
    tx = CODE["SharedVerifyTransaction"](cache, Vector([7, 7] if duplicate else [7]), layout,
        windows, Vector([100, 101, 102, 103]), call_id=1, max_backtracking=4)
    return tx, cache


class SharedDSparkSourceChecks(unittest.TestCase):
    def test_accepted_scope_zero_partial_full_and_odd_pair(self):
        for count in (0, 1, 2, 4):
            with self.subTest(count=count):
                tx, cache = fixture()
                before = dict(cache.storage)
                report = tx.finish(Vector([7]), Vector([257]), Vector([count]))
                self.assertEqual(report["accepted_ends"], [257 + count])
                self.assertEqual(report["accepted_input_token_ids"], [list(range(100, 100 + count))])
                self.assertEqual(report["engram_committed_windows"], [None if count == 0 else [[100,11,10,9],[101,100,11,10],[102,101,100,11],[103,102,101,100]][count - 1]])
                if count == 0:
                    self.assertEqual(cache.storage, before)
                if count:
                    self.assertEqual(cache.storage[((1, "global_kv_2"), 0, 128)], 1)
                if count < 3:
                    self.assertEqual(cache.storage[((1, "global_kv_2"), 0, 129)], -100)
                self.assertTrue(tx.closed)
                self.assertFalse(cache.journal)

    def test_repeated_acceptance_rejected(self):
        tx, _ = fixture()
        tx.finish(Vector([7]), Vector([257]), Vector([1]))
        with self.assertRaises(RuntimeError):
            tx.finish(Vector([7]), Vector([257]), Vector([1]))

    def test_request_prefix_count_and_page_ownership(self):
        for ids, bases, counts in (([8], [257], [1]), ([7], [256], [1]), ([7], [257], [-1]), ([7], [257], [5])):
            tx, cache = fixture()
            before = dict(cache.storage)
            with self.assertRaises(ValueError):
                tx.finish(Vector(ids), Vector(bases), Vector(counts))
            self.assertEqual(before, cache.storage)
        tx, cache = fixture()
        cache.pools[(0, "swa_kv")].table.values[0] = 2
        with self.assertRaises(RuntimeError):
            tx.finish(Vector([7]), Vector([257]), Vector([1]))
        with self.assertRaises(ValueError):
            fixture(duplicate=True)

    def test_publication_failure_restores_physical_bytes(self):
        tx, cache = fixture()
        before = dict(cache.storage)
        cache.fail_after = 3
        with self.assertRaisesRegex(RuntimeError, "publication failure"):
            tx.finish(Vector([7]), Vector([257]), Vector([4]))
        self.assertEqual(cache.storage, before)
        self.assertTrue(tx.closed)

    def test_frozen_profile_actual_capacity(self):
        spec = importlib.util.spec_from_file_location("n3_replay_policy", PPU / "replay_policy.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        policy = module.SharedPrefillPolicy
        env = {"DSV41_CED": "1", "DSV41_SWA_BOUNDED_REPLAY": "1", "SP_TYPE": "dspark",
               "GEN_NUM_PER_CIRCLE": "3", "SP_DSPARK_CONFIDENCE_MODE": "observe",
               "SP_DSPARK_VERIFY_MODE": "dense", "SP_DSPARK_VERIFY_BUDGET_MODE": "fixed",
               "SP_DSPARK_VERIFY_BUDGET": "-1"}
        actual = policy.from_config(env=env)
        self.assertEqual((actual.keep_tokens, actual.window, actual.max_rollback_tokens), (140, 128, 4))
        self.assertEqual(policy.from_config(env={}).max_rollback_tokens, 0)
        self.assertEqual(policy.from_config(env={**env, "GEN_NUM_PER_CIRCLE": "5"}).max_rollback_tokens, 6)
        self.assertEqual(policy.from_config(env={**env, "SP_DSPARK_CONFIDENCE_MODE": "bounded",
                         "SP_DSPARK_VERIFY_MODE": "packed", "SP_DSPARK_VERIFY_BUDGET": "1"}).max_rollback_tokens, 4)
        for delta in ({"RTP_V41_PPU_MAX_ROLLBACK_TOKENS": "3"}, {"GEN_NUM_PER_CIRCLE": "2"},
                      {"SP_DSPARK_VERIFY_MODE": "packed"}, {"SP_DSPARK_VERIFY_BUDGET": "1"}):
            with self.assertRaises(ValueError):
                policy.from_config(env={**env, **delta})
        with self.assertRaises(ValueError):
            policy(max_rollback_tokens=13).validate()

    def test_public_loops_and_native_helpers_remain_distinct(self):
        tree = ast.parse((PPU / "shared_dspark_model.py").read_text())
        model = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        self.assertEqual(ast.unparse(model.bases[0]), "DeepSeekV41DSparkModel")
        declared = {x.name for x in model.body if isinstance(x, ast.FunctionDef)}
        self.assertFalse(declared & {"_forward_layers", "run_commit_step", "run_propose_step",
                                     "compute_draft_outputs", "combine_hidden_states", "commit_feature_rows", "map_commit_rows"})
        source = (PPU / "shared_dspark_model.py").read_text()
        self.assertNotIn("NativeModelRunner", source)
        self.assertNotIn("load_draft_weights(", source)
        idle = next(x for x in model.body if isinstance(x, ast.FunctionDef) and x.name == "forward_query_block")
        self.assertIn("self._forward_layers(", ast.unparse(idle))

    def test_idle_metadata_before_host_reads(self):
        tree = ast.parse((PPU / "shared_dspark_model.py").read_text())
        model = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        method = next(x for x in model.body if isinstance(x, ast.FunctionDef) and x.name == "_call")
        source = ast.unparse(method)
        self.assertLess(source.index("if idle:"), source.index("attn.input_lengths.tolist()"))
        self.assertIn("input_hiddens=None", source)
        self.assertIn("torch.zeros(width", source)

    def test_no_business_verify_finalization_lifecycle(self):
        # Execute the actual runtime early-return and callback methods. The
        # public fake DSpark round has4 input rows/prefix1, not zero rows;
        # its gamma+1 accept packet may be clipped by the stream preview.
        tree = ast.parse((PPU / "shared_runtime.py").read_text())
        selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)
                                   and node.name in {"_NoBusinessVerifyToken", "SharedNativeRuntime"}], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "dataclass": dataclass, "replace": replace, "Path": Path, "json": json,
                "os": SimpleNamespace(environ={"RTP_V41_SHARED_EVIDENCE_DIR": directory}),
                "dist": SimpleNamespace(is_initialized=lambda: False),
                "torch": SimpleNamespace(zeros=lambda shape, **kwargs: SimpleNamespace(shape=shape), bfloat16="bf16"),
                "PyModelOutputs": lambda hidden: SimpleNamespace(hidden_states=hidden),
                "as_tagged_attention_inputs": lambda value: value,
            }
            exec(compile(selected, str(PPU / "shared_runtime.py"), "exec"), namespace)
            runtime_class = namespace["SharedNativeRuntime"]
            runtime = runtime_class.__new__(runtime_class)
            runtime.current = runtime.pending_verify = runtime.pending_idle_verify = None
            runtime.calls = runtime._idle_verify_calls = 0
            runtime.geometry = SimpleNamespace(hidden_size=5120)
            runtime.policy = SimpleNamespace(max_rollback_tokens=4)
            runtime.device = "cpu"

            def inputs(*, fake=True, verify=True):
                return SimpleNamespace(
                    input_ids=Vector([0, 0, 0, 0]), request_id=Vector([0]), is_fake_stream=fake,
                    attention_inputs=SimpleNamespace(is_target_verify=verify, is_packed_target_verify=False,
                        input_lengths=Vector([4]), prefix_lengths=Vector([1])),
                )

            for reason, voted, cache in (("all_empty_vote", False, object()), ("no_kv_cache", True, None)):
                runtime.vote = lambda count, voted=voted: voted
                runtime._model = lambda cache=cache: SimpleNamespace(kv_cache=cache)
                for accepted in (0, 2, 4):
                    with self.subTest(reason=reason, accepted=accepted):
                        current = inputs()
                        output = runtime.dispatch(current)
                        self.assertEqual(output.hidden_states.shape, (4, 5120))
                        token = runtime.pending_idle_verify
                        self.assertEqual((token.request_ids, token.prefixes, token.lengths), ((0,), (1,), (4,)))
                        self.assertEqual(token.reason, reason)
                        with self.assertRaisesRegex(RuntimeError, "previous shared verify"):
                            runtime.dispatch(current)
                        self.assertIs(runtime.pending_idle_verify, token)
                        runtime.finish_speculative_forward(Vector([0]), Vector([1]), Vector([accepted]))
                        self.assertIsNone(runtime.pending_idle_verify)
                        with self.assertRaisesRegex(RuntimeError, "no pending shared verify"):
                            runtime.finish_speculative_forward(Vector([0]), Vector([1]), Vector([accepted]))
                for ids, bases, counts in (([9], [1], [2]), ([0], [2], [2]),
                                           ([0], [1], [-1]), ([0], [1], [5]), ([0], [1], [])):
                    runtime.dispatch(inputs())
                    with self.assertRaises(ValueError):
                        runtime.finish_speculative_forward(Vector(ids), Vector(bases), Vector(counts))
                    self.assertIsNone(runtime.pending_idle_verify)
                current = inputs()
                runtime.dispatch(current)
                # The token snapshots values rather than retaining reused
                # framework host-buffer storage across finalization.
                current.request_id.values[0] = 9
                current.attention_inputs.prefix_lengths.values[0] = 2
                runtime.finish_speculative_forward(Vector([0]), Vector([1]), Vector([2]))
                runtime.dispatch(inputs())
                runtime.abort_speculative_forward()
                self.assertIsNone(runtime.pending_idle_verify)
                runtime.abort_speculative_forward()
                with self.assertRaisesRegex(RuntimeError, "no pending shared verify"):
                    runtime.finish_speculative_forward(Vector([0]), Vector([1]), Vector([2]))
                # Skipping a real/non-verify call never grants permission to
                # ignore a later missing business transaction callback.
                for current in (inputs(fake=False), inputs(verify=False)):
                    runtime.dispatch(current)
                    self.assertIsNone(runtime.pending_idle_verify)
                    with self.assertRaisesRegex(RuntimeError, "no pending shared verify"):
                        runtime.finish_speculative_forward(Vector([0]), Vector([1]), Vector([2]))
            self.assertEqual(runtime.calls, 0)
            self.assertIsNone(runtime.pending_verify)
            records = [json.loads(path.read_text()) for path in Path(directory).glob("target-*.json")]
            self.assertEqual(len(records), 10)
            self.assertEqual(len({item["no_business_call"] for item in records}), 10)
            self.assertTrue(all(item["idle"] and item["native_rows_published"] == 0 for item in records))
            self.assertEqual(sum(item["event"] == "abort" for item in records), 2)
            # Business events retain their original report call number and
            # cannot collide with the first no-business token's own serial.
            runtime.calls = 1
            runtime.event("accept", {"call": 1, "idle": False})
            self.assertTrue((Path(directory) / "target-rank0-idle00001-accept.json").is_file())
            business = Path(directory) / "target-rank0-call00001-accept.json"
            self.assertEqual(json.loads(business.read_text())["call"], 1)


    def test_hc_pre_preserves_leading_token_layout(self):
        # Execute the adapter, including its unchanged state setter/post/head.
        # These shape-only tensors carry ordered storage, not HC arithmetic.
        import itertools
        import math
        import weakref

        class ShapeView:
            def __init__(self, shape, storage=None):
                self.shape = tuple(shape)
                self.storage = (
                    tuple(range(math.prod(shape))) if storage is None else storage
                )
                if math.prod(self.shape) != len(self.storage):
                    raise ValueError("shape/view element count mismatch")

            def reshape(self, *shape):
                return ShapeView(shape, self.storage)

            def at(self, *indices):
                offset = 0
                for size, index in zip(self.shape, indices):
                    offset = offset * size + index
                return self.storage[offset]

        @dataclass(frozen=True)
        class StepMix:
            hidden: object = None
            post_mix: object = None
            residual_mix: object = None
            pre_mix: object = None

        tree = ast.parse((PPU / "shared_modules.py").read_text())
        classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"PPUSharedHCUnit", "PPUSharedHCHead"}
        ]
        for node in classes:
            node.body = [
                item for item in node.body
                if not isinstance(item, ast.FunctionDef) or item.name != "__init__"
            ]
        namespace = {
            "HCUnitBase": object, "nn": SimpleNamespace(Module=object),
            "PPUStepMix": StepMix, "replace": replace, "weakref": weakref,
        }
        selected = ast.Module(body=classes, type_ignores=[])
        exec(compile(selected, str(PPU / "shared_modules.py"), "exec"), namespace)
        unit_class = namespace["PPUSharedHCUnit"]
        head_class = namespace["PPUSharedHCHead"]

        for leading in ((1, 3), (1, 4), (2, 3), (5,), (2, 1)):
            for flat_provider in (True, False):
                with self.subTest(leading=leading, flat_provider=flat_provider):
                    residual = ShapeView((*leading, 4, 8))
                    calls, produced = [], []

                    def mhc(x, branch, *, phase, state=None, parameters=None):
                        self.assertIs(x, residual)
                        if state is not None:
                            self.assertEqual(state.pre_mix.shape, (*leading, 4))
                        calls.append((phase, state))
                        if phase == "pre":
                            self.assertIsNone(branch)
                            self.assertEqual(set(parameters), {"fn", "base", "scale"})
                            axes = (math.prod(leading),) if flat_provider else leading
                            result = StepMix(
                                hidden=ShapeView((*leading, 8)),
                                pre_mix=ShapeView((*axes, 4)),
                                post_mix=ShapeView((*axes, 4, 1)),
                                residual_mix=ShapeView((*axes, 4, 4)),
                            )
                            produced.append(result)
                            return result
                        if phase == "post":
                            self.assertEqual(branch.shape, (*leading, 8))
                            self.assertEqual(state.post_mix.shape, (*leading, 4, 1))
                            self.assertEqual(state.residual_mix.shape, (*leading, 4, 4))
                            return StepMix(hidden=residual)
                        self.assertEqual(phase, "head")
                        self.assertIsNone(branch)
                        return StepMix(hidden=ShapeView((*leading, 8)))

                    previous = None
                    for _ in range(2):
                        unit = unit_class.__new__(unit_class)
                        unit.state = None
                        unit.hc_mult = 4
                        unit.fn, unit.base, unit.scale = object(), object(), object()
                        unit.ops = SimpleNamespace(mhc=mhc)
                        unit.pre_mix_out = None
                        unit.set_previous(previous)
                        hidden, post, comb = unit._pre_impl(residual)
                        native = produced[-1]
                        self.assertIs(
                            calls[-1][1], None if previous is None else previous.state
                        )
                        self.assertIs(hidden, native.hidden)
                        self.assertEqual(hidden.shape, (*leading, 8))
                        self.assertIs(post, unit.state.post_mix)
                        self.assertIs(comb, unit.state.residual_mix)
                        self.assertIs(unit.pre_mix_out, unit.state.pre_mix)
                        for name, tail in (
                            ("pre_mix", (4,)), ("post_mix", (4, 1)),
                            ("residual_mix", (4, 4)),
                        ):
                            actual = getattr(unit.state, name)
                            original = getattr(native, name)
                            self.assertEqual(actual.shape, (*leading, *tail))
                            self.assertIs(actual.storage, original.storage)
                            tokens = itertools.product(*(range(n) for n in leading))
                            for row, indices in enumerate(tokens):
                                lanes = itertools.product(*(range(n) for n in tail))
                                for lane in lanes:
                                    old_indices = (
                                        (row, *lane)
                                        if flat_provider else (*indices, *lane)
                                    )
                                    self.assertEqual(
                                        actual.at(*indices, *lane),
                                        original.at(*old_indices),
                                    )
                        self.assertIs(
                            unit._post_impl(hidden, residual, post, comb), residual
                        )
                        self.assertIs(calls[-1][1].post_mix, post)
                        self.assertIs(calls[-1][1].residual_mix, comb)
                        previous = unit
                    head = head_class.__new__(head_class)
                    head.ops = SimpleNamespace(mhc=mhc)
                    head._previous_ref = weakref.ref(previous)
                    self.assertEqual(head.head(residual).shape, (*leading, 8))
                    self.assertIs(calls[-1][1], previous.state)
                    self.assertEqual(
                        [phase for phase, _ in calls],
                        ["pre", "post", "pre", "post", "head"],
                    )


if __name__ == "__main__":
    unittest.main()
