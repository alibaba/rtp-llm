"""Real base/PEFT files; local parameter transitions, not request-level serving."""

import json
import tempfile
import threading
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.lora.lora_manager import LoraManager
from rtp_llm.models_py.lora_checkpoint import (
    RankLocalCheckpointAdapters,
    merge_startup_lora_checkpoint,
    prepare_lora_checkpoint_factors,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.runtime_weight_update import LoRAUpdate, RuntimeLoRAManager


class Projection(RtpModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 3), requires_grad=False)

    def prepare_lora_factors(self, name, a, b):
        if name != "weight":
            raise ValueError(name)
        return prepare_lora_checkpoint_factors(a, b, 3, 2)


@register_model("_direct_lora_boundary")
class Owner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.proj = Projection()

    def owns_lora_checkpoint_target(self, name):
        return name != "remote.weight"


class DirectAdapterTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        save_file({"proj.weight": self.base}, str(self.root / "model.safetensors"))
        self.loader = NewModelLoader(
            SimpleNamespace(model_type="_direct_lora_boundary"),
            NewLoaderConfig(
                device="cpu", load_method="scratch", compute_dtype=torch.float32
            ),
            model_path=str(self.root),
        )
        self.model = self.loader.load()
        self.path = self.adapter("one", ["proj", "remote"])
        self.expected_delta = torch.tensor(
            [[2, 4, 6], [-2, -4, -6]], dtype=torch.float32
        )
        self.manager = RankLocalCheckpointAdapters(self.model, self.loader)

    def adapter(self, directory, projections):
        dest = self.root / directory
        dest.mkdir()
        (dest / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 2}))
        tensors = {}
        for prefix in projections:
            tensors[f"base_model.model.{prefix}.lora_A.weight"] = torch.tensor(
                [[1.0, 2.0, 3.0]]
            )
            tensors[f"base_model.model.{prefix}.lora_B.weight"] = torch.tensor(
                [[1.0], [-1.0]]
            )
        save_file(tensors, str(dest / "adapter_model.safetensors"))
        return str(dest)

    def test_named_checkpoint_cannot_pair_factors_from_different_adapters(self):
        path = Path(self.path) / "adapter_model.safetensors"
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0], [-1.0]])

        def factors(left, right, target="proj"):
            return {
                f"{target}.lora_A{left}.weight": a.clone(),
                f"{target}.lora_B{right}.weight": b.clone(),
            }

        save_file(factors(".red", ".red"), str(path))
        self.manager.load("serving-alias", self.path)
        self.manager.merge("serving-alias", pause_engine=nullcontext)
        weight = self.model.proj.weight
        before, ptr = weight.clone(), weight.data_ptr()
        self.assertTrue(torch.equal(before, self.base + self.expected_delta))
        for invalid in (
            factors(".red", ".blue"),
            factors("", ".red"),
            factors(".red", ""),
            {**factors(".red", ".red"), **factors(".blue", ".blue", "remote")},
        ):
            with self.subTest(keys=tuple(invalid)):
                save_file(invalid, str(path))
                with self.assertRaisesRegex(ValueError, "Mixed adapter names"):
                    self.manager.load("invalid", self.path)
                self.assertEqual(self.manager.loaded_adapters, ("serving-alias",))
                self.assertEqual(self.manager.active_adapters, ("serving-alias",))
                self.assertEqual(weight.data_ptr(), ptr)
                self.assertTrue(torch.equal(weight, before))

    def test_overflow_never_installs_partial_weights_or_changes_active_adapters(self):
        weights = {k: torch.zeros(1, 1, dtype=torch.float16) for k in ("good", "bad")}
        manager = RuntimeLoRAManager(weights)
        for adapter in ("a", "b"):
            manager.load(
                adapter,
                [
                    LoRAUpdate(k, torch.tensor([[v]]), torch.tensor([[v]]))
                    for k, v in (("good", 1.0), ("bad", 200.0))
                ],
            )
        manager.merge("a")
        before = {k: v.clone() for k, v in weights.items()}
        ptrs = {k: v.data_ptr() for k, v in weights.items()}
        self.assertEqual(before["bad"].item(), 40000)
        # First overflow the sum of two finite deltas, then a new base + one delta.
        for operation in (
            lambda: manager.merge(["a", "b"]),
            lambda: manager.update_base_weights(before),
        ):
            with self.assertRaisesRegex(ValueError, "Non-finite merged LoRA weight.*bad"):
                operation()
            self.assertEqual(manager.active_adapters, ("a",))
            self.assertEqual(manager.loaded_adapters, ("a", "b"))
            for k, v in weights.items():
                self.assertEqual(v.data_ptr(), ptrs[k])
                self.assertTrue(torch.equal(v, before[k]))
                self.assertEqual(manager.base_weight(k).item(), 0)
        manager.unmerge()
        self.assertEqual(manager.active_adapters, ())
        self.assertTrue(all(v.item() == 0 for v in weights.values()))

    def test_load_merge_unmerge_unload_preserve_storage(self):
        manager = self.manager
        tensor = self.model.proj.weight
        ptr = tensor.data_ptr()
        manager.load("a", self.path)
        self.assertEqual(manager.loaded_adapters, ("a",))
        torch.testing.assert_close(tensor, self.base)
        with self.assertRaises(TypeError):
            manager.merge("a", pause_engine=None)
        entered = []

        @contextmanager
        def paused():
            entered.append(True)
            yield

        manager.merge("a", pause_engine=paused)
        torch.testing.assert_close(tensor, self.base + self.expected_delta)
        with self.assertRaisesRegex(RuntimeError, "merged LoRA"):
            with manager.base_update_guard():
                self.fail("Writer must not run")
        manager.unmerge(pause_engine=paused)
        torch.testing.assert_close(tensor, self.base)
        # A subsequent merge takes the current base, not the old snapshot.
        with manager.base_update_guard(), torch.inference_mode():
            tensor.add_(2)
        manager.merge("a", pause_engine=paused)
        torch.testing.assert_close(tensor, self.base + 2 + self.expected_delta)
        manager.unload("a", pause_engine=paused)
        torch.testing.assert_close(tensor, self.base + 2)
        self.assertEqual(manager.loaded_adapters, ())
        self.assertEqual(entered, [True] * 4)
        self.assertEqual(tensor.data_ptr(), ptr)

    def test_startup_merge_and_failure_before_mutation(self):
        malformed = self.adapter("bad", ["proj", "unknown"])
        with self.assertRaisesRegex(NotImplementedError, "unknown"):
            merge_startup_lora_checkpoint(self.model, self.loader, "bad", malformed)
        torch.testing.assert_close(self.model.proj.weight, self.base)
        with self.assertRaisesRegex(NotImplementedError, "unknown"):
            self.manager.load("bad", malformed)
        self.assertEqual(self.manager.loaded_adapters, ())
        merge_startup_lora_checkpoint(self.model, self.loader, "a", self.path)
        torch.testing.assert_close(
            self.model.proj.weight, self.base + self.expected_delta
        )
        manager = RankLocalCheckpointAdapters(
            self.model, self.loader, startup_adapters=("a",)
        )
        with self.assertRaisesRegex(ValueError, "startup base"):
            manager.load("a", self.path)
        with self.assertRaisesRegex(RuntimeError, "merged LoRA"):
            with manager.base_update_guard():
                self.fail("Cannot overwrite startup merge")

    def test_nonowner_stage_and_replaced_parameter(self):
        path = self.adapter("other", ["remote"])
        merge_startup_lora_checkpoint(self.model, self.loader, "x", path)
        self.manager.load("x", path)
        self.manager.merge("x", pause_engine=nullcontext)
        self.manager.unload("x", pause_engine=nullcontext)
        torch.testing.assert_close(self.model.proj.weight, self.base)
        self.manager.load("x", self.path)
        self.model.proj.weight = torch.nn.Parameter(
            self.base.clone(), requires_grad=False
        )
        with self.assertRaisesRegex(RuntimeError, "bindings changed"):
            self.manager.merge("x", pause_engine=nullcontext)
        self.assertEqual(self.manager.active_adapters, ())
        torch.testing.assert_close(self.model.proj.weight, self.base)

    def test_excludes_base_writer_until_transition_has_committed(self):
        manager = RankLocalCheckpointAdapters(
            self.model, self.loader, admin_lock=threading.RLock()
        )
        manager.load("a", self.path)
        attempting, done = threading.Event(), threading.Event()
        errors = []

        def activate():
            try:
                attempting.set()
                manager.merge("a", pause_engine=nullcontext)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        with manager.base_update_guard(), torch.inference_mode():
            thread = threading.Thread(target=activate)
            thread.start()
            self.assertTrue(attempting.wait(3))
            self.assertFalse(done.wait(0.05))
            self.model.proj.weight.add_(1)
        thread.join(5)
        self.assertTrue(done.is_set())
        self.assertEqual(errors, [])
        torch.testing.assert_close(
            self.model.proj.weight, self.base + 1 + self.expected_delta
        )

    def test_missing_request_backend_is_not_global_merge(self):
        engine = SimpleNamespace(
            model=SimpleNamespace(model_weights_loader=self.loader)
        )
        with self.assertRaisesRegex(
            NotImplementedError, "cannot select adapters per request"
        ):
            LoraManager(engine)
        torch.testing.assert_close(self.model.proj.weight, self.base)

    def test_excluded_fp4_owner_merges_actual_rank_local_rows(self):
        from rtp_llm.models_py.layers.linear import ColumnParallelLinear
        from rtp_llm.models_py.quant_methods.base import QuantizationConfig

        @register_model("_direct_excluded_fp4")
        class ExcludedFP4(RtpModule):
            def __init__(self, config, load):
                super().__init__()
                self.proj = ColumnParallelLinear(
                    3,
                    2,
                    tp_size=2,
                    tp_rank=config.checked_rank,
                    quant_config=QuantizationConfig(
                        "modelopt_fp4", ignored_layers=["proj"]
                    ),
                    prefix="proj",
                    params_dtype=torch.bfloat16,
                )

        path = self.adapter("excluded", ["proj"])
        for method in ("scratch", "fastsafetensors"):
            for rank in range(2):
                with self.subTest(rank=rank, method=method):
                    config = SimpleNamespace(
                        model_type="_direct_excluded_fp4", checked_rank=rank
                    )
                    loader = NewModelLoader(
                        config,
                        NewLoaderConfig(
                            device="cpu",
                            load_method=method,
                            compute_dtype=torch.bfloat16,
                        ),
                        model_path=str(self.root),
                    )
                    model = loader.load()
                    self.assertEqual(loader.loaded_method.value, method)
                    with torch.inference_mode():
                        premerge = model.proj.weight.clone()
                    torch.testing.assert_close(
                        premerge, self.base[rank : rank + 1].bfloat16()
                    )
                    merge_startup_lora_checkpoint(model, loader, "a", path)
                    torch.testing.assert_close(
                        model.proj.weight,
                        premerge + self.expected_delta[rank : rank + 1].bfloat16(),
                    )

    def test_request_bridge_registers_only_after_success(self):
        class Wrapper:
            fail = True

            def add_lora(self, *args):
                if self.fail:
                    raise RuntimeError("registration failed")

            def remove_lora(self, *args):
                if self.fail:
                    raise RuntimeError("removal failed")

        wrapper = Wrapper()
        weights = SimpleNamespace(lora_a_weights=[], lora_b_weights=[])
        reader = SimpleNamespace(load_lora_weights=lambda *args: weights)
        model = SimpleNamespace(
            model_weights_loader=reader, model_config=SimpleNamespace(lora_infos={})
        )
        manager = LoraManager(
            SimpleNamespace(model=model, rtp_llm_op_=SimpleNamespace(ft_op=wrapper))
        )
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            manager.add_lora("a", self.path)
        self.assertEqual(manager.lora_infos_, {})
        wrapper.fail = False
        manager.add_lora("a", self.path)
        wrapper.fail = True
        with self.assertRaisesRegex(RuntimeError, "removal failed"):
            manager.remove_lora("a")
        self.assertEqual(manager.lora_infos_, {"a": self.path})


if __name__ == "__main__":
    unittest.main()
