"""CPU-only tests for RL weight updates, QSA freezing, and VMM lifecycle."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader import rl_vmm
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.rl_weight_update import RlWeightUpdate
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import W


class _DummyWeight:
    def __init__(self, name: str, resident_device=None):
        self.name = name
        self.resident_device = resident_device


class _FakeLoader:
    def __init__(self, static_table_keys, static_meta_keys, emit_items):
        self._static_table_keys = set(static_table_keys)
        self._static_meta_keys = set(static_meta_keys)
        self._emit_items = emit_items

    def get_rl_static_target_keys(self, weights):
        return set(self._static_table_keys), set(self._static_meta_keys)

    def get_rl_frozen_qsa_target_keys(self, frozen_source_names):
        return set()

    def prepare_weights_from_tensors(
        self,
        named_tensors,
        device,
        *,
        max_buffer_bytes,
        borrow_singleton_sources=True,
        frozen_qsa_source_names=(),
    ):
        received = dict(named_tensors)
        for layer, target_name, source_name in self._emit_items:
            yield layer, target_name, received[source_name]


class _FakeCudaDriver:
    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemLocationType = SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=2)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_NONE=0)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

    class CUmemAllocationProp:
        def __init__(self):
            self.location = SimpleNamespace()

    class CUmemAccessDesc:
        def __init__(self):
            self.location = SimpleNamespace()

    def __init__(self):
        self.calls = []
        self.next_handle = 1
        self.next_ptr = 0x100000
        self.fail_set_access = False

    def cuInit(self, flags):
        self.calls.append(("init", flags))
        return (0,)

    def cuMemCreate(self, size, prop, flags):
        handle = self.next_handle
        self.next_handle += 1
        self.calls.append(
            (
                "create",
                size,
                handle,
                prop.type,
                prop.location.id,
                prop.requestedHandleTypes,
            )
        )
        return (0, handle)

    def cuMemAddressReserve(self, size, alignment, addr, flags):
        ptr = self.next_ptr
        self.next_ptr += size
        self.calls.append(("reserve", ptr, size))
        return (0, ptr)

    def cuMemMap(self, ptr, size, offset, handle, flags):
        self.calls.append(("map", ptr, size, handle))
        return (0,)

    def cuMemSetAccess(self, ptr, size, descs, count):
        self.calls.append(("access", ptr, size, descs[0].flags))
        return (1,) if self.fail_set_access else (0,)

    def cuMemUnmap(self, ptr, size):
        self.calls.append(("unmap", ptr, size))
        return (0,)

    def cuMemRelease(self, handle):
        self.calls.append(("release", handle))
        return (0,)

    def cuMemAddressFree(self, ptr, size):
        self.calls.append(("free_address", ptr, size))
        return (0,)


class _FakeQsaComponent:
    def __init__(self, name, source_name):
        self.name = name
        self.source_name = source_name

    def get_tensor_names(self, layer_id, load_config):
        return [self.source_name]


class _FakeQsaModule:
    def __init__(self, components):
        self.components = components

    def get_components(self):
        return self.components


class _FakeStreamWeight:
    def __init__(self, name, source_name):
        self.name = name
        self.source_name = source_name

    def get_tensor_names(self, layer_id, load_config):
        return [self.source_name]

    def load(self, collector, layer_id, device, load_config):
        return {self.name: collector.tensors[self.source_name]}


class _FakeTensorCollector:
    def __init__(self, names, database, *, stacked_experts=None):
        self.names = set(names)
        self.tensors = {}

    def store_tensor(self, name, tensor):
        self.tensors[name] = tensor
        return self.names <= self.tensors.keys()

    def clear(self):
        self.tensors.clear()


def _fake_qsa_loader(layer_ids=(3, 7), *, missing_target=None):
    target_names = (
        W.qwen4_indexer_qk_proj_w,
        W.qwen4_indexer_q_ln_gamma,
        W.qwen4_indexer_k_ln_gamma,
    )
    layer_weights = {}
    source_names = {}
    for layer_id in layer_ids:
        components = []
        for target in target_names:
            if (layer_id, target) == missing_target:
                continue
            source = f"model.layers.{layer_id}.indexer.{target.rsplit('.', 2)[-2]}"
            source_names[layer_id, target] = source
            components.append(_FakeQsaComponent(target, source))
        layer_weights[layer_id] = [_FakeQsaModule(components)]
    loader = ModelLoader.__new__(ModelLoader)
    loader.model_config = SimpleNamespace(model_type="qwen4_exp")
    loader._load_config = SimpleNamespace(layer_range=list(layer_ids), database=object())
    loader._model_weights_info = SimpleNamespace(layer_weights=layer_weights)
    return loader, source_names, target_names


class RlWeightUpdateUnitTest(unittest.TestCase):
    def test_converted_weight_lifetime_does_not_depend_on_cyclic_gc(self):
        weight = AtomicWeight("nested", [])
        weight._load_raw_tensor = lambda *_: torch.ones(1)
        weight._split = lambda tensor, _: tensor
        weight._postprocess = lambda tensor, *_: {"nested": {"leaf": tensor}}
        config = SimpleNamespace(merge_lora=False)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for convert in (
                lambda: weight.load(None, None, "cpu", config),
                lambda: weight.update(torch.ones(1), "cpu", config),
            ):
                outputs = convert()
                tensor_ref = weakref.ref(outputs["leaf"])
                del outputs
                self.assertIsNone(tensor_ref())
        finally:
            if was_enabled:
                gc.enable()

    def test_frozen_ple_and_int64_metadata_excluded_while_ple_dense_kept(self):
        ple_shard = _DummyWeight(W.qwen4_ple_ngram_shards, resident_device="cpu")
        hash_mult = _DummyWeight(W.qwen4_ple_multipliers)
        hash_mod = _DummyWeight(W.qwen4_ple_ngram_offsets)
        hash_tok = _DummyWeight(W.qwen4_ple_ngram_vocab_sizes)
        cpu_only = _DummyWeight("custom_cpu_static", resident_device="cpu")
        ple_dense = _DummyWeight("ple_key_proj")

        self.assertTrue(ModelLoader._is_rl_static_component(ple_shard))
        self.assertTrue(ModelLoader._is_rl_static_component(hash_mult))
        self.assertTrue(ModelLoader._is_rl_static_component(hash_mod))
        self.assertTrue(ModelLoader._is_rl_static_component(hash_tok))
        self.assertTrue(ModelLoader._is_rl_static_component(cpu_only))
        self.assertFalse(ModelLoader._is_rl_static_component(ple_dense))

    def test_bf16_hc_paused_inplace_unlock_live_refusal_and_tail_check(self):
        # Simulate qwen4_exp merged HC buffer: [16, 8] BF16,
        # active rows [0..3], zero-padded tail [4..15].
        merged = torch.zeros((16, 8), dtype=torch.bfloat16)
        mix_down = merged[0:2]
        inject = merged[2:4]
        for view in (mix_down, inject):
            view._rtp_requires_restart_for_update = True
            view._rtp_allow_paused_inplace_update = True
            view._rtp_merged_parent = merged
            view._rtp_merged_tail_start = 4

        mw = ModelWeights(num_layers=1, device="cpu", dtype=torch.bfloat16)
        mw.set_layer_weight(0, "hc.mix_down", mix_down)
        mw.set_layer_weight(0, "hc.inject", inject)
        # Static INT64 metadata + CPU PLE shard must stay untouched.
        meta_tensor = torch.tensor([11, 22, 33], dtype=torch.int64)
        ple_shard = torch.zeros((4, 8), dtype=torch.bfloat16)
        mw.set_layer_weight(0, W.qwen4_ple_multipliers, meta_tensor)
        mw.set_layer_weight(0, f"{W.qwen4_ple_ngram_shards}.0", ple_shard)

        new_mix = torch.full((2, 8), 3.0, dtype=torch.bfloat16)
        new_inject = torch.full((2, 8), 5.0, dtype=torch.bfloat16)

        # Live update (default allow_paused_inplace=False) MUST still refuse.
        with self.assertRaisesRegex(PermissionError, "requires a service restart"):
            mw.update_layer_weight(0, "hc.mix_down", new_mix)

        # Full paused RlWeightUpdate transaction succeeds in-place and preserves pointers + zero tail.
        loader = _FakeLoader(
            static_table_keys={(0, f"{W.qwen4_ple_ngram_shards}.0")},
            static_meta_keys={(0, W.qwen4_ple_multipliers)},
            emit_items=[
                (0, "hc.mix_down", "model.layers.0.hc.down"),
                (0, "hc.inject", "model.layers.0.hc.inject"),
            ],
        )
        before_mix_ptr = mix_down.data_ptr()
        before_inj_ptr = inject.data_ptr()
        tx = RlWeightUpdate(
            loader=loader,
            weights=mw,
            device="cpu",
            source_shapes={
                "model.layers.0.hc.down": (2, 8),
                "model.layers.0.hc.inject": (2, 8),
            },
            max_buffer_bytes=1024 * 1024,
            source_dtypes={
                "model.layers.0.hc.down": torch.bfloat16,
                "model.layers.0.hc.inject": torch.bfloat16,
            },
        )
        tx.push("model.layers.0.hc.down", new_mix)
        tx.push("model.layers.0.hc.inject", new_inject)
        receipt = tx.finish()
        self.assertEqual(receipt["hf_sources"], 2)
        self.assertEqual(receipt["internal_tensors"], 2)
        self.assertEqual(receipt["static_table_tensors"], 1)
        self.assertEqual(receipt["static_meta_tensors"], 1)
        self.assertEqual(mix_down.data_ptr(), before_mix_ptr)
        self.assertEqual(inject.data_ptr(), before_inj_ptr)
        self.assertTrue(torch.equal(merged[0:2], new_mix))
        self.assertTrue(torch.equal(merged[2:4], new_inject))
        self.assertTrue(torch.equal(merged[4:], torch.zeros((12, 8), dtype=torch.bfloat16)))

        # Corrupting the zero-padded tail before commit MUST fail transaction validation.
        tx2 = RlWeightUpdate(
            loader=loader,
            weights=mw,
            device="cpu",
            source_shapes={
                "model.layers.0.hc.down": (2, 8),
                "model.layers.0.hc.inject": (2, 8),
            },
            max_buffer_bytes=1024 * 1024,
        )
        tx2.push("model.layers.0.hc.down", new_mix)
        merged[4, 0] = 1.0
        tx2.push("model.layers.0.hc.inject", new_inject)
        with self.assertRaisesRegex(RuntimeError, "trailing padding rows became non-zero"):
            tx2.finish()

    def test_restart_only_target_without_paused_allow_flag_rejected_pre_write(self):
        t = torch.zeros((2, 8), dtype=torch.bfloat16)
        t._rtp_requires_restart_for_update = True
        mw = ModelWeights(num_layers=1, device="cpu", dtype=torch.bfloat16)
        mw.set_layer_weight(0, "fp8_repacked", t)
        loader = _FakeLoader(set(), set(), [(0, "fp8_repacked", "src.w")])
        with self.assertRaisesRegex(PermissionError, "requires a service restart"):
            RlWeightUpdate(
                loader=loader,
                weights=mw,
                device="cpu",
                source_shapes={"src.w": (2, 8)},
                max_buffer_bytes=1024,
            )


class RlVmmWeightContractTest(unittest.TestCase):
    def test_qwen4_initialize_runs_vmm_owner_check_once(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel

        class _Manager:
            def __init__(self):
                self.calls = []

            def finish_after_initialize(self, weights, model):
                self.calls.append((weights, model))
                return {"inventory_sha256": "startup"}

        model = object.__new__(Qwen4ExpModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(is_mtp=False)
        model.weight = object()
        manager = _Manager()
        recorded = []
        model._rl_vmm_weight_manager = manager
        model._rl_vmm_record_startup = recorded.append
        init_resource = SimpleNamespace(kv_cache=None, is_speculative=False)

        self.assertTrue(model.initialize(init_resource))
        self.assertEqual(manager.calls, [(model.weight, model)])
        self.assertEqual(recorded, [{"inventory_sha256": "startup"}])
        self.assertFalse(hasattr(model, "_rl_vmm_weight_manager"))
        self.assertFalse(hasattr(model, "_rl_vmm_record_startup"))
        self.assertTrue(model.initialize(init_resource))
        self.assertEqual(len(manager.calls), 1)

    def test_vmm_manager_keeps_only_a_weak_model_reference(self):
        class _Model:
            pass

        manager = object.__new__(rl_vmm.RlVmmWeightManager)
        model = _Model()
        model_ref = weakref.ref(model)
        manager.bind_model(model)

        del model
        gc.collect()
        self.assertIsNone(model_ref())

    def test_vmm_sleep_remaps_same_va_and_is_idempotent(self):
        driver = _FakeCudaDriver()
        with patch.object(rl_vmm, "_driver", return_value=driver), patch.object(
            rl_vmm, "round_up_to_granularity", side_effect=lambda size, _: size
        ):
            allocation = rl_vmm.VmmAllocation(4096, 0, "weights")
            original_ptr = allocation.ptr
            self.assertFalse(allocation.remap())
            self.assertEqual(allocation.unmap_for_sleep(), 4096)
            self.assertFalse(allocation.mapped)
            self.assertEqual(allocation.unmap_for_sleep(), 0)
            self.assertTrue(allocation.remap())
            self.assertEqual(allocation.ptr, original_ptr)
            self.assertTrue(allocation.mapped)
            self.assertFalse(allocation.remap())
            allocation.close()

        calls = [call[0] for call in driver.calls]
        self.assertEqual(
            calls,
            [
                "init",
                "create",
                "reserve",
                "map",
                "access",
                "unmap",
                "release",
                "create",
                "map",
                "access",
                "unmap",
                "release",
                "free_address",
            ],
        )
        map_calls = [call for call in driver.calls if call[0] == "map"]
        self.assertEqual([call[1] for call in map_calls], [original_ptr, original_ptr])
        access_calls = [call for call in driver.calls if call[0] == "access"]
        self.assertEqual(
            [call[3] for call in access_calls],
            [driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE] * 2,
        )
        create_calls = [call for call in driver.calls if call[0] == "create"]
        self.assertEqual(
            [(call[3], call[4], call[5]) for call in create_calls],
            [
                (
                    driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
                    0,
                    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE,
                )
            ]
            * 2,
        )

    def test_vmm_failed_access_rolls_back_handle_and_allows_retry(self):
        driver = _FakeCudaDriver()
        with patch.object(rl_vmm, "_driver", return_value=driver), patch.object(
            rl_vmm, "round_up_to_granularity", side_effect=lambda size, _: size
        ):
            allocation = rl_vmm.VmmAllocation(8192, 0, "weights")
            original_ptr = allocation.ptr
            allocation.unmap_for_sleep()
            driver.fail_set_access = True
            with self.assertRaisesRegex(RuntimeError, "cuMemSetAccess failed"):
                allocation.remap()
            self.assertFalse(allocation.mapped)
            self.assertIsNone(allocation.handle)
            self.assertEqual(allocation.ptr, original_ptr)
            driver.fail_set_access = False
            self.assertTrue(allocation.remap())
            allocation.close()

        rollback = [call[0] for call in driver.calls]
        self.assertIn("unmap", rollback)
        self.assertEqual(rollback.count("release"), 3)
        self.assertEqual(rollback.count("reserve"), 1)

    def test_qsa_freeze_requires_complete_local_layer_triplets(self):
        loader, _source_names, target_names = _fake_qsa_loader()
        groups = loader.get_rl_qsa_source_groups()
        frozen_sources = groups[3][0]
        required_sources = set().union(*(sources for sources, _ in groups.values()))
        required_sources.add("model.embed_tokens.weight")
        expected = required_sources - set(frozen_sources)

        self.assertEqual(
            loader.validate_rl_frozen_qsa_source_names(required_sources, expected),
            frozen_sources,
        )
        self.assertEqual(
            loader.get_rl_frozen_qsa_target_keys(frozen_sources),
            {(3, name) for name in target_names},
        )
        with self.assertRaisesRegex(ValueError, "complete per-layer source triplet"):
            loader.validate_rl_frozen_qsa_source_names(
                required_sources, expected | {next(iter(frozen_sources))}
            )
        with self.assertRaisesRegex(ValueError, "unsupported missing sources"):
            loader.validate_rl_frozen_qsa_source_names(
                required_sources, expected - {"model.embed_tokens.weight"}
            )
        with self.assertRaisesRegex(ValueError, "unknown sources"):
            loader.validate_rl_frozen_qsa_source_names(
                required_sources, expected | {"unexpected.source"}
            )
        with self.assertRaisesRegex(ValueError, "contains duplicates"):
            loader.validate_rl_frozen_qsa_source_names(
                required_sources, list(expected) + [next(iter(expected))]
            )

    def test_qsa_freeze_skips_only_omitted_sources_from_stream(self):
        loader, source_names, _target_names = _fake_qsa_loader()
        load_config = loader._load_config
        load_config.is_ft_style_weight = False
        load_config.merge_lora = False
        all_groups = []
        for (layer_id, target), source in source_names.items():
            all_groups.append(
                ModelLoader.WeightInfo(
                    _FakeStreamWeight(target, source), layer_id, None
                )
            )
        shared_source = "model.embed_tokens.weight"
        all_groups.append(
            ModelLoader.WeightInfo(
                _FakeStreamWeight("embedding", shared_source), None, None
            )
        )
        frozen_sources = loader.get_rl_qsa_source_groups()[3][0]
        received_names = (
            set(source_names.values()) - set(frozen_sources) | {shared_source}
        )
        named_tensors = [(name, torch.ones(1)) for name in sorted(received_names)]

        with patch.object(loader, "_rl_mutable_groups", return_value=(all_groups, [])), patch.object(
            loader,
            "_get_rl_group_source_names",
            side_effect=lambda group: {group.weight.source_name},
        ), patch("rtp_llm.model_loader.loader.TensorCollector", _FakeTensorCollector):
            converted = list(
                loader.prepare_weights_from_tensors(
                    named_tensors,
                    "cpu",
                    max_buffer_bytes=1024,
                    frozen_qsa_source_names=frozen_sources,
                )
            )

        converted_targets = {(layer_id, name) for layer_id, name, _ in converted}
        expected_targets = {
            (layer_id, target)
            for (layer_id, target), source in source_names.items()
            if source not in frozen_sources
        } | {(None, "embedding")}
        self.assertEqual(converted_targets, expected_targets)


if __name__ == "__main__":
    unittest.main()
