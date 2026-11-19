"""CPU regression for layout selection using the production adapter function.

Extract only ``enabled`` from its source so this topology regression requires
neither an ATOM import nor a GPU/native extension. The ROCm integration target
also runs these checks alongside the real kernels and graph replay cases.
"""

import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


def patched_adapter_tree():
    path = Path("atom/plugin/rtpllm/utils/linear_double_buffer.py")
    patch = Path(__file__).resolve().with_name("atom.patch")
    with tempfile.TemporaryDirectory() as root:
        subprocess.run(
            ["git", "apply", f"--include={path}", str(patch)], cwd=root, check=True
        )
        return ast.parse((Path(root) / path).read_text(), filename=str(path))


ADAPTER = patched_adapter_tree()


def load_functions(names, namespace=None):
    functions = [
        node
        for node in ADAPTER.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {} if namespace is None else namespace
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "linear_double_buffer.py", "exec"),
        namespace,
    )
    return namespace


enabled = load_functions({"enabled"})["enabled"]
SPLIT_TAGS = ("linear", "linear_ssm", "linear_replay")


class CacheBindings:
    def __init__(self, tags, bindings, *, layout=3, state_slots=2, replay_slots=1):
        self.group_tags = tags
        self.bindings = bindings
        self.visits = []
        self.layout = layout
        self.state_slots = state_slots
        self.replay_slots = replay_slots

    def get_layer_cache_groups(self, layer):
        self.visits.append(layer)
        return [SimpleNamespace(tag=tag) for tag in self.bindings[layer]]

    def get_linear_state_layout(self, tag):
        assert tag == "linear_ssm"
        return self.layout

    def get_request_state_slots(self, tag):
        return self.state_slots if tag == "linear_ssm" else self.replay_slots


def runtime(cache, layers):
    return SimpleNamespace(
        kv_cache=cache,
        _rtp_layer_maps=({layer: object() for layer in layers}, {}, {}),
    )


class RuntimeAdapterBindingsTest(unittest.TestCase):
    def test_initialization_installs_hooks_before_any_forward(self):
        functions = load_functions(
            {"enabled", "initialize_runtime", "_install", "_clear", "_save",
             "_activate", "_prepare", "_finish", "_finalize", "_requires_fail_stop"},
            {"torch": SimpleNamespace(version=SimpleNamespace(hip="test")),
             "MethodType": MethodType,
             "LinearDoubleBufferTransaction": lambda **kwargs: SimpleNamespace(**kwargs)},
        )
        target = runtime(CacheBindings(SPLIT_TAGS, {0: SPLIT_TAGS}, layout=5, state_slots=1), [0])
        target._rtp_model_config = SimpleNamespace(atom_qwen4_exp_mtp_target=True)
        target.parallelism_config = SimpleNamespace(role_type="DECODE")
        target._hybrid_enabled = True
        target._hybrid_reducer = object()
        functions["initialize_runtime"](target)
        self.assertTrue(target.requires_speculative_target_commit_fail_stop())
        transaction = target._linear_db_transaction
        functions["initialize_runtime"](target)
        self.assertIs(target._linear_db_transaction, transaction)
        draft = runtime(CacheBindings(SPLIT_TAGS, {}), [])
        functions["initialize_runtime"](draft)
        self.assertFalse(hasattr(draft, "_linear_db_transaction"))

    def test_fail_stop_hook_tracks_single_state_transaction(self):
        functions = load_functions(
            {
                "_install",
                "_clear",
                "_save",
                "_activate",
                "_prepare",
                "_finish",
                "_finalize",
                "_requires_fail_stop",
            },
            {
                "torch": SimpleNamespace(version=SimpleNamespace(hip="test")),
                "MethodType": MethodType,
                "LinearDoubleBufferTransaction": lambda **kwargs: SimpleNamespace(
                    **kwargs
                ),
            },
        )
        for slots in (1, 2):
            target = SimpleNamespace(
                _linear_db_state_slots=slots,
                _rtp_model_config=SimpleNamespace(atom_qwen4_exp_mtp_target=True),
                parallelism_config=SimpleNamespace(role_type="DECODE"),
                _hybrid_enabled=True,
                _hybrid_reducer=object(),
            )
            functions["_install"](target)
            self.assertIs(
                target.requires_speculative_target_commit_fail_stop(), slots == 1
            )
            self.assertFalse(target._hybrid_enabled)
            self.assertIsNotNone(target._hybrid_reducer)
            functions["_install"](target)  # Same mode rebind is idempotent.
            target._linear_db_state_slots = 3 - slots
            with self.assertRaisesRegex(RuntimeError, "commit mode"):
                functions["_install"](target)

    def test_single_q1_metadata_commits_directly_without_speculative_stage(self):
        class PaddedTable:
            ndim = 2
            shape = (2, 17)

            def __getitem__(self, index):
                self.selected_columns = index[1].stop
                return self

        state_table, replay_table = PaddedTable(), PaddedTable()
        calls = []
        clear_calls = []
        buffers = (object(), object(), object(), object())
        target = SimpleNamespace(
            _linear_db_state_slots=1,
            _linear_db_indices={(2, 1): buffers},
            _linear_db_transaction=SimpleNamespace(
                clear=lambda: clear_calls.append(True)
            ),
            _linear_db_pools=(
                (SimpleNamespace(shape=(5, 2, 128, 128)),),
                (SimpleNamespace(shape=(5, 3, 2, 3, 128)),),
                None,
                None,
            ),
        )
        metadata = SimpleNamespace(
            num_prefills=0, num_spec_decodes=0, num_decodes=2, num_actual_tokens=2
        )
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(sequence_lengths_plus_1_device=object()),
            attention_inputs_by_tag={
                "linear_ssm": SimpleNamespace(kv_cache_block_id_device=state_table),
                "linear_replay": SimpleNamespace(kv_cache_block_id_device=replay_table),
            },
        )
        prepare = load_functions(
            {"prepare_metadata"},
            {
                "resolve_pool_indices": lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
            },
        )["prepare_metadata"]
        prepare(target, metadata, inputs)
        self.assertIs(metadata.rtp_ssm_non_spec_indices, buffers[0])
        self.assertFalse(hasattr(metadata, "rtp_hybrid_record_only"))
        self.assertEqual(clear_calls, [True])
        self.assertEqual(
            (state_table.selected_columns, replay_table.selected_columns), (1, 1)
        )
        self.assertEqual(calls[0][1], {"minimum_length": 1, "state_slots": 1})

    def test_full_only_draft_ignores_compatibility_split_groups(self):
        # mergeMTPModule preserves these global groups without binding them to
        # the full-attention-only draft layer. Selecting split here rejects MTP.
        cache = CacheBindings((*SPLIT_TAGS, "full"), {0: ("full",)})
        draft = runtime(cache, ())
        self.assertFalse(enabled(draft))
        self.assertFalse(enabled(draft))
        self.assertEqual(cache.visits, [])

    def test_target_binds_every_linear_layer(self):
        cache = CacheBindings(
            (*SPLIT_TAGS, "full"),
            {0: SPLIT_TAGS, 1: ("full",), 2: SPLIT_TAGS},
        )
        target = runtime(cache, (0, 2))
        self.assertTrue(enabled(target))
        self.assertEqual(cache.visits, [0, 2])
        self.assertTrue(enabled(target))
        self.assertEqual(cache.visits, [0, 2])

    def test_target_partial_global_layout_is_rejected(self):
        for missing in ("linear_ssm", "linear_replay"):
            with self.subTest(missing=missing):
                tags = tuple(tag for tag in SPLIT_TAGS if tag != missing)
                target = runtime(CacheBindings(tags, {0: tags}), (0,))
                with self.assertRaisesRegex(ValueError, "split cache layout"):
                    enabled(target)
                self.assertFalse(hasattr(target, "_linear_db_layout_cache"))

    def test_target_partial_layer_binding_is_rejected(self):
        for missing in SPLIT_TAGS:
            with self.subTest(missing=missing):
                bound = tuple(tag for tag in SPLIT_TAGS if tag != missing)
                cache = CacheBindings(SPLIT_TAGS, {0: SPLIT_TAGS, 2: bound})
                target = runtime(cache, (0, 2))
                with self.assertRaisesRegex(ValueError, "bindings for layer 2"):
                    enabled(target)
                self.assertFalse(hasattr(target, "_linear_db_layout_cache"))
        # Both split groups can exist globally while a target GDN layer is
        # entirely unsplit. Do not silently fall back to the combined parser.
        cache = CacheBindings(SPLIT_TAGS, {0: ("linear",)})
        with self.assertRaisesRegex(ValueError, "bindings for layer 0"):
            enabled(runtime(cache, (0,)))

    def test_legacy_target_remains_disabled(self):
        cache = CacheBindings(("linear", "full"), {0: ("linear",)})
        self.assertFalse(enabled(runtime(cache, (0,))))
        self.assertEqual(cache.visits, [])

    def test_cache_rebinding_revalidates_layout(self):
        first = CacheBindings(SPLIT_TAGS, {0: SPLIT_TAGS})
        target = runtime(first, (0,))
        self.assertTrue(enabled(target))
        second = CacheBindings(("linear",), {0: ("linear",)})
        target.kv_cache = second
        self.assertFalse(enabled(target))
        third = CacheBindings(SPLIT_TAGS, {0: ("linear", "linear_ssm")})
        target.kv_cache = third
        with self.assertRaisesRegex(ValueError, "bindings for layer 0"):
            enabled(target)
        self.assertEqual(third.visits, [0])
        target.kv_cache = first
        self.assertTrue(enabled(target))
        self.assertEqual(first.visits, [0, 0])

    def test_backend_binding_failure_propagates(self):
        cache = CacheBindings(SPLIT_TAGS, {})
        target = runtime(cache, (3,))
        with self.assertRaises(KeyError):
            enabled(target)
        self.assertFalse(hasattr(target, "_linear_db_layout_cache"))

    def test_single_layout_uses_published_slots_and_rebinding(self):
        cache = CacheBindings(SPLIT_TAGS, {0: SPLIT_TAGS}, layout=5, state_slots=1)
        target = runtime(cache, (0,))
        self.assertTrue(enabled(target))
        self.assertEqual(target._linear_db_state_slots, 1)
        target.kv_cache = CacheBindings(SPLIT_TAGS, {0: SPLIT_TAGS})
        self.assertTrue(enabled(target))
        self.assertEqual(target._linear_db_state_slots, 2)

    def test_unknown_or_inconsistent_slot_contract_is_rejected(self):
        for layout, slots, replay in ((3, 1, 1), (5, 2, 1), (7, 1, 1), (5, 1, 2)):
            with self.subTest(layout=layout, slots=slots, replay=replay):
                cache = CacheBindings(
                    SPLIT_TAGS,
                    {0: SPLIT_TAGS},
                    layout=layout,
                    state_slots=slots,
                    replay_slots=replay,
                )
                target = runtime(cache, (0,))
                with self.assertRaises(ValueError):
                    enabled(target)
                self.assertFalse(hasattr(target, "_linear_db_layout_cache"))


if __name__ == "__main__":
    unittest.main()
