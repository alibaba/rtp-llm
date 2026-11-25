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

    def load_ft_checkpoint(self, checkpoint):
        self.load_weights((name, checkpoint.read("global." + name)) for name in ("weight", "expert.routed_w13"))

    def resident_constructor_state(self):
        return {}


def _run(root, hot):
    manager = ResidentWeights("target", 0, {"owner_layout": "tiny-native-v1"}, False)
    restored = manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (restored is not None) == hot
    loader = NewModelLoader(
        SimpleNamespace(model_type="_resident_checkpoint_restart"),
        NewLoaderConfig(device="cuda:0", load_method="scratch", compute_dtype=torch.float32),
        model_path=root,
    )
    model = loader.load(resident=manager, restored=restored)
    assert model.weight is model.tied
    assert manager.allocation_of(model.weight) is not None
    model.expert.initialize(hot)
    assert model.expert.routed_w13 is None
    torch.testing.assert_close(model.expert.packed.cpu(), torch.tensor([2., 4.]), rtol=0, atol=0)
    assert manager.allocation_of(model.expert.packed) is not None
    logits = torch.tensor([[1., 2., 3.]], device="cuda:0") @ model.weight.T
    torch.testing.assert_close(logits.cpu(), torch.tensor([[8., 26.]]), rtol=0, atol=0)
    manager.finalize()


class NativeResidentRestartTest(unittest.TestCase):
    def test_fresh_worker_without_hf_checkpoint(self):
        self._restart(False)

    def test_fresh_worker_without_ft_checkpoint(self):
        self._restart(True)

    def _restart(self, ft):
        name = "rtp_llm_native_owner_test." + uuid.uuid4().hex
        env = dict(os.environ, RTP_LLM_RESIDENT_STORE=name)
        with tempfile.TemporaryDirectory() as root:
            shard = Path(root, "model.safetensors")
            values = {"weight": torch.arange(6).float().reshape(2, 3), "expert.routed_w13": torch.tensor([1., 2.])}
            index = shard.with_name("model.safetensors.index.json")
            if ft:
                values = {"rank_0_0_0.global." + k: v for k, v in values.items()}
                index.write_text(json.dumps({"is_ft_style_weight": True, "weight_map": dict.fromkeys(values, shard.name)}))
            save_file(values, str(shard))
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                for phase in ("cold", "hot"):
                    if phase == "hot":
                        shard.unlink()
                        if ft:
                            index.unlink()
                    subprocess.run([sys.executable, __file__, root, phase], env=env, check=True, timeout=90)
            finally:
                store.stop_store(name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[-1] in ("cold", "hot"):
        _run(sys.argv[1], sys.argv[2] == "hot")
    else:
        unittest.main()
