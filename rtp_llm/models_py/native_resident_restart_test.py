"""Fresh process replay: the checkpoint shard is absent in the second worker."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model
from rtp_llm.resident_weights import ResidentWeights, store


class _Expert(RtpModule):
    def __init__(self):
        super().__init__()
        self.routed_w13 = torch.nn.Parameter(torch.empty(2), requires_grad=False)

    def take_executor_weight(self, name):
        value = self._parameters[name]
        self._parameters[name] = None
        return value

    def initialize(self, hot):
        from rtp_llm.models_py.modules.fp8_fp4_weight_source import OwnerWeightSource, resident_init_tensors

        source = OwnerWeightSource(self)
        def build():
            assert not hot, "replay reran a consuming transform"
            return (source.take("routed_w13") * 2,)
        self.packed, = resident_init_tensors(
            source, transform_key="pack", output_names=("packed",),
            consumed_names=("routed_w13",), builder=build,
        )


@register_model("_resident_checkpoint_restart")
class _Owner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 3), requires_grad=False)
        self.tied = self.weight
        self.expert = _Expert()

    @classmethod
    def from_ft_checkpoint_header(cls, config, load, checkpoint):
        assert "global.weight" in checkpoint.entries
        assert "global.other" not in checkpoint.entries
        assert checkpoint._shape(*checkpoint.entries["global.weight"]) == (2, 3)
        assert checkpoint.read_rank_slice("global.weight", checkpoint.topology.tp_rank) == (2, 3)
        return cls(config, load)

    def load_ft_checkpoint(self, checkpoint):
        self.load_weights((name, checkpoint.read("global." + name)) for name in ("weight", "expert.routed_w13"))

    def resident_constructor_state(self):
        return {}


@register_model("_resident_draft_restart")
class _Draft(RtpModule):
    preserved_ft_parameter_loans = {"head.weight": "global.weight"}

    def __init__(self, config, load):
        super().__init__()
        self.head = RtpModule()
        self.head.weight = torch.nn.Parameter(torch.empty(2, 3), requires_grad=False)
        self.expert = _Expert()

    def prepare_borrowed_parameters(self, bindings):
        assert set(bindings) == {"head.weight"}

    def checkpoint_weight_name_filter(self):
        return lambda key: key == "expert.routed_w13"

    def load_ft_checkpoint(self, checkpoint):
        self.load_weights({"expert.routed_w13": checkpoint.read("global.expert.routed_w13")})

    def resident_constructor_state(self):
        return {}


@register_model("_resident_final_owner")
class _FinalOwner(RtpModule):
    resident_after_post_load = True

    def __init__(self, config, load):
        super().__init__()
        self.proj = _Projection()

    def resident_constructor_state(self):
        return {}


class _Projection(RtpModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 3), requires_grad=False)

    def prepare_lora_factors(self, name, a, b):
        from rtp_llm.models_py.lora_checkpoint import prepare_lora_checkpoint_factors

        assert name == "weight"
        return prepare_lora_checkpoint_factors(a, b, 3, 2)

    def process_weights_after_loading(self):
        self.weight.mul_(2)


def _final_run(root, hot):
    manager = ResidentWeights("target", 0, {"owner_layout": "final-lora-v1"}, False)
    record = manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (record is not None) == hot
    model = NewModelLoader(
        SimpleNamespace(model_type="_resident_final_owner"),
        NewLoaderConfig(device="cuda:0", load_method="scratch", compute_dtype=torch.float32),
        model_path=root,
    ).load(resident=manager, restored=record, startup_lora=("one", str(Path(root, "one"))))
    expected = torch.arange(6).reshape(2, 3) * 2 + torch.tensor([[2, 4, 6], [-2, -4, -6]])
    torch.testing.assert_close(model.proj.weight.cpu(), expected.float(), rtol=0, atol=0)
    assert manager.allocation_of(model.proj.weight) is not None
    manager.finalize()


def _run(root, hot, method):
    manager = ResidentWeights("target", 0, {"owner_layout": "tiny-native-v1"}, False)
    restored = manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (restored is not None) == hot
    loader = NewModelLoader(
        SimpleNamespace(model_type="_resident_checkpoint_restart"),
        NewLoaderConfig(device="cuda:0", load_method=method, compute_dtype=torch.float32),
        model_path=root,
    )
    model = loader.load(resident=manager, restored=restored)
    assert loader._selected_load_method.value == method
    assert model.weight is model.tied
    assert manager.allocation_of(model.weight) is not None
    model.expert.initialize(hot)
    assert model.expert.routed_w13 is None
    torch.testing.assert_close(model.expert.packed.cpu(), torch.tensor([2., 4.]), rtol=0, atol=0)
    assert manager.allocation_of(model.expert.packed) is not None
    logits = torch.tensor([[1., 2., 3.]], device="cuda:0") @ model.weight.T
    torch.testing.assert_close(logits.cpu(), torch.tensor([[8., 26.]]), rtol=0, atol=0)
    manager.finalize()

    from rtp_llm.models_py.borrowed_parameters import BorrowedParameter

    draft_manager = ResidentWeights("draft", 0, {"owner_layout": "tiny-draft-v1"}, False)
    draft_record = draft_manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (draft_record is not None) == hot
    if hot:
        assert not any(entry.name == "head.weight" for entry in draft_record)
    draft = NewModelLoader(
        SimpleNamespace(model_type="_resident_draft_restart"), loader.load_config,
        model_path=root, parameter_aliases={"head.weight": BorrowedParameter(model, "weight")},
    ).load(resident=draft_manager, restored=draft_record)
    assert draft.head.weight is model.weight
    assert draft_manager.allocation_of(draft.head.weight) is None
    draft.expert.initialize(hot)
    draft_manager.finalize()


class NativeResidentRestartTest(unittest.TestCase):
    def test_fresh_worker_without_hf_checkpoint(self):
        self._restart(False)

    def test_fresh_worker_without_ft_checkpoint(self):
        self._restart(True)

    def test_fresh_worker_with_final_layout_and_merged_adapter(self):
        self._restart(False, final=True)

    def test_fst_fresh_worker_without_hf_checkpoint(self):
        self._restart(False, method="fastsafetensors")

    def test_fst_fresh_worker_without_ft_checkpoint(self):
        self._restart(True, method="fastsafetensors")

    def _restart(self, ft, final=False, method="scratch"):
        name = "rtp_llm_native_owner_test." + uuid.uuid4().hex
        env = dict(os.environ, RTP_LLM_RESIDENT_STORE=name, RESTART_TEST_METHOD=method)
        with tempfile.TemporaryDirectory() as root:
            shard = Path(root, "model.safetensors")
            values = {"weight": torch.arange(6).float().reshape(2, 3), "expert.routed_w13": torch.tensor([1., 2.])}
            index = shard.with_name("model.safetensors.index.json")
            if ft:
                values = {"rank_0_0_0.global." + k: v for k, v in values.items()}
                index.write_text(json.dumps({"is_ft_style_weight": True, "weight_map": dict.fromkeys(values, shard.name)}))
            if final:
                values = {"proj.weight": torch.arange(6).float().reshape(2, 3)}
                adapter = Path(root, "one")
                adapter.mkdir()
                (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 2}))
                save_file({
                    "proj.lora_A.weight": torch.tensor([[1., 2., 3.]]),
                    "proj.lora_B.weight": torch.tensor([[1.], [-1.]]),
                }, str(adapter / "adapter_model.safetensors"))
            save_file(values, str(shard))
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                for phase in ("cold", "hot"):
                    if phase == "hot":
                        shard.unlink()
                        if ft:
                            index.unlink()
                        if final:
                            (adapter / "adapter_config.json").unlink()
                            (adapter / "adapter_model.safetensors").unlink()
                    subprocess.run([sys.executable, __file__, root, "final_" + phase if final else phase], env=env, check=True, timeout=90)
            finally:
                store.stop_store(name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[-1] in ("cold", "hot"):
        _run(sys.argv[1], sys.argv[2] == "hot", os.environ["RESTART_TEST_METHOD"])
    elif len(sys.argv) == 3 and sys.argv[-1] in ("final_cold", "final_hot"):
        _final_run(sys.argv[1], sys.argv[2] == "final_hot")
    else:
        unittest.main()
