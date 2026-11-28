"""Real CPU safetensors/SHM regression; not a linked gRPC or CUDA test."""

import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from multiprocessing import shared_memory
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.model_loader.tipc import SharedMemoryIPCHelper
from rtp_llm.models_py.layers.embedding import VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import QKVParallelLinear, RowParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_weight_manager import NewLoaderWeightManager
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_main_update_compat")
class _Owner(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, config, load):
        super().__init__()
        tp, rank, dtype = load.tp_size, load.tp_rank, load.compute_dtype
        self.embed_tokens = VocabParallelEmbedding(8, 4, tp, rank, dtype)
        layer = RtpModule()
        layer.input_layernorm = RMSNorm(4, params_dtype=dtype)
        layer.self_attn = RtpModule()
        layer.self_attn.qkv_proj = QKVParallelLinear(
            4, 2, 1, 2, tp, rank, params_dtype=dtype
        )
        layer.self_attn.o_proj = RowParallelLinear(
            4, 4, tp_size=tp, tp_rank=rank, params_dtype=dtype
        )
        self.layers = torch.nn.ModuleDict({"0": layer})

    def checkpoint_weight_name_filter(self):
        return lambda name: not name.startswith("model.layers.1.")

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def runtime_weight_view(self):
        return {"embedding": self.embed_tokens.weight}


class MainUpdateCompatTest(unittest.TestCase):
    def test_resident_mode_rejects_updates_before_transport_or_write(self):
        model, loader = self.load(0)
        manager = NewLoaderWeightManager(model, loader, resident_weights_enabled=True)
        before = model.embed_tokens.weight.clone()
        with self.assertRaisesRegex(RuntimeError, "resident weights"):
            manager.update({"method": "shm", "name": "model.embed_tokens.weight", "desc": "invalid"})
        with self.assertRaisesRegex(RuntimeError, "resident weights"):
            manager.update_tensors({"model.embed_tokens.weight": torch.ones_like(before)})
        self.assertEqual(manager.version, 0)
        torch.testing.assert_close(model.embed_tokens.weight, before)

    def test_retry_busy_rank_after_peer_already_accepted_update(self):
        # The public call updates a single rank. A busy peer cannot undo an
        # earlier successful response; keep inference paused until it catches up.
        managers = [NewLoaderWeightManager(*self.load(rank)) for rank in (0, 1)]
        entered, release = threading.Event(), threading.Event()

        def still_in_forward():
            with managers[1]._model._weight_update_lock:
                entered.set()
                release.wait(10)

        worker = threading.Thread(target=still_in_forward)
        worker.start()
        name = "model.embed_tokens.weight"
        values = torch.arange(32).reshape(8, 4).bfloat16()
        try:
            self.assertTrue(entered.wait(5))
            managers[0].update_tensors({name: values})
            with self.assertRaises(BlockingIOError):
                managers[1].update_tensors({name: values})
            self.assertEqual([m.version for m in managers], [1, 0])
            torch.testing.assert_close(managers[0]._model.embed_tokens.weight, values[:4])
            torch.testing.assert_close(managers[1]._model.embed_tokens.weight, torch.zeros_like(values[4:]))
        finally:
            release.set()
            worker.join()
        managers[1].update_tensors({name: values})
        torch.testing.assert_close(managers[1]._model.embed_tokens.weight, values[4:])
        self.assertEqual([m.version for m in managers], [1, 1])

    def test_busy_forward_rejects_update_without_writing_and_allows_retry(self):
        model, loader = self.load()
        manager = NewLoaderWeightManager(model, loader)
        name = "model.layers.0.input_layernorm.weight"
        weight = model.layers["0"].input_layernorm.weight
        original = weight.detach().clone()
        replacement = torch.full_like(original, 3)
        entered, release = threading.Event(), threading.Event()

        def forward():
            with model._weight_update_lock:
                entered.set()
                release.wait(10)

        worker = threading.Thread(target=forward)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaises(BlockingIOError):
                manager.update_tensors({name: replacement})
            self.assertEqual(manager.version, 0)
            torch.testing.assert_close(weight, original)
        finally:
            release.set()
            worker.join()
        manager.update_tensors({name: replacement})
        self.assertEqual(manager.version, 1)
        torch.testing.assert_close(weight, replacement)

    def load(self, rank=0):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        cfg = SimpleNamespace(model_type="_main_update_compat", num_layers=2)
        data = {
            "model.embed_tokens.weight": torch.zeros(8, 4),
            "model.layers.0.input_layernorm.weight": torch.ones(4),
            "model.layers.0.self_attn.qkv_proj.weight": torch.zeros(8, 4),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        }
        save_file(
            {k: v.bfloat16() for k, v in data.items()},
            os.path.join(root.name, "model.safetensors"),
        )
        loader = NewModelLoader(
            cfg,
            NewLoaderConfig(
                device="cpu",
                load_method="scratch",
                tp_size=2,
                tp_rank=rank,
                compute_dtype=torch.bfloat16,
            ),
            model_path=root.name,
        )
        model = loader.load()
        return model, loader

    @staticmethod
    def send(manager, name, value):
        memory = shared_memory.SharedMemory(
            create=True, size=value.numel() * value.element_size()
        )
        try:
            desc = SharedMemoryIPCHelper().build_tensor_meta(value, memory).encode()
            manager.update({"name": name, "method": "shm", "desc": desc})
        finally:
            memory.close()
            memory.unlink()

    def test_real_bf16_shm_fused_and_square_gemm_layout_per_tp_rank(self):
        for rank in (0, 1):
            model, loader = self.load(rank)
            manager = NewLoaderWeightManager(model, loader)
            pointers = {n: t.data_ptr() for n, t in model.named_parameters()}
            qkv = torch.arange(32).reshape(8, 4).bfloat16()
            out = torch.arange(16).reshape(4, 4).bfloat16()
            self.send(
                manager,
                "decoder.layers.0.self_attention.linear_qkv.weight",
                qkv.t().contiguous(),
            )
            self.send(
                manager, "model.layers.0.self_attn.o_proj.weight", out.t().contiguous()
            )
            self.send(manager, "model.embed_tokens.weight", qkv)
            attn = model.layers["0"].self_attn
            torch.testing.assert_close(
                attn.qkv_proj.weight, torch.cat([qkv[rank * 2 : rank * 2 + 2], qkv[4:]])
            )
            torch.testing.assert_close(
                attn.o_proj.weight, out[:, rank * 2 : rank * 2 + 2]
            )
            torch.testing.assert_close(
                model.embed_tokens.weight, qkv[rank * 4 : rank * 4 + 4]
            )
            self.assertEqual(
                pointers, {n: t.data_ptr() for n, t in model.named_parameters()}
            )
            self.assertEqual(manager.version, 3)
            # Existing clients may address an exported global directly. Resolve
            # it back to the same parameter without legacy descriptors.
            self.send(manager, "embedding", qkv + 1)
            torch.testing.assert_close(
                model.embed_tokens.weight, (qkv + 1)[rank * 4 : rank * 4 + 4]
            )

    def test_foreign_pp_noop_malformed_fails_no_partial_write(self):
        model, loader = self.load()
        manager = NewLoaderWeightManager(model, loader)
        norm = model.layers["0"].input_layernorm.weight
        before = norm.clone()
        self.send(
            manager, "model.layers.1.input_layernorm.weight", torch.arange(4).bfloat16()
        )
        with self.assertRaises(ValueError):
            self.send(
                manager,
                "model.layers.2.input_layernorm.weight",
                torch.arange(4).bfloat16(),
            )
        with self.assertRaises((RuntimeError, ValueError, NotImplementedError)):
            manager.update_tensors(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros_like(before),
                    "bad": torch.ones(3),
                }
            )
        torch.testing.assert_close(before, norm)
        self.assertEqual(manager.version, 0)

    def test_shm_transport_has_no_old_descriptor_import(self):
        # Import transport before any legacy module; this suite uses nothing
        # but direct leaves. Bazel/pytest harnesses may import other suites.
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-c",
                "from rtp_llm.model_loader.tipc import SharedMemoryIPCHelper; import sys; "
                "assert not {'rtp_llm.model_loader.loader', 'rtp_llm.model_loader.weight_module'} & sys.modules.keys()",
            ],
            check=True,
        )

    def test_borrowed_parameter_dtype_and_shared_restart_contract(self):
        model, loader = self.load()
        manager = NewLoaderWeightManager(model, loader)
        norm = model.layers["0"].input_layernorm
        with self.assertRaises(TypeError):
            self.send(
                manager,
                "model.layers.0.input_layernorm.weight",
                torch.zeros(4, dtype=torch.float32),
            )
        norm.weight._rtp_requires_restart_for_update = True
        with self.assertRaisesRegex(PermissionError, "restart"):
            self.send(
                manager,
                "model.layers.0.input_layernorm.weight",
                torch.zeros(4, dtype=torch.bfloat16),
            )
        model.embed_tokens._rtp_borrowed_parameter_names = frozenset({"weight"})
        with self.assertRaisesRegex(PermissionError, "non-owning"):
            self.send(
                manager,
                "model.embed_tokens.weight",
                torch.zeros(8, 4, dtype=torch.bfloat16),
            )
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
        self.assertEqual(manager.version, 0)

    def test_merged_adapter_prevents_writes(self):
        class Active:
            @contextmanager
            def base_update_guard(self):
                raise PermissionError("active adapters")
                yield

        model, loader = self.load()
        manager = NewLoaderWeightManager(model, loader, adapters=Active())
        with self.assertRaisesRegex(PermissionError, "active adapters"):
            manager.update_tensors({"model.embed_tokens.weight": torch.ones(8, 4)})
        self.assertEqual(manager.version, 0)


if __name__ == "__main__":
    unittest.main()
