"""Cold record and fresh-worker replay of the native V4 indexer's projection."""

import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import torch

from rtp_llm.models_py import resident_checkpoint
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.modules.dsv4.fp8.test import dsv4_indexer_checkpoint_test as fixture
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.resident_weights import GLOBAL_SCOPE, ResidentWeights, store


def _worker(shard, warm):
    manager = ResidentWeights("target", 0, {"owner": "v4_indexer_csa"}, verify=False)
    restored = manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (restored is not None) == warm
    layer = fixture.Dsv4IndexerCheckpointTest()._layer().cuda()
    parameter = layer.weights_proj
    if warm:
        assert not os.path.exists(shard)
        method = manager.init_scalar(GLOBAL_SCOPE, "native.backend", resident_checkpoint._replay_only)
        assert method == NewLoaderLoadMethod.SCRATCH.value
        resident_checkpoint.restore(layer, manager, restored, "cuda:0")
    else:
        layer.load_weights(get_all_weights([shard]))
        resident_checkpoint.record(layer, manager, NewLoaderLoadMethod.SCRATCH)

    raw = layer._resident_bank.global_weights["weights_proj"]
    assert manager.allocation_of(raw) is not None
    expected_raw = torch.arange(256).reshape(2, 128).to(torch.bfloat16)
    expected = expected_raw * (128**-0.5 * 2**-0.5)
    for _ in range(2):
        layer.process_weights_after_loading()
        assert layer.weights_proj is parameter
        assert raw.data_ptr() != parameter.data_ptr()
        torch.testing.assert_close(raw.cpu(), expected_raw, rtol=0, atol=0)
        torch.testing.assert_close(parameter.cpu(), expected, rtol=0, atol=0)
    manager.finalize()


class ResidentIndexerTest(unittest.TestCase):
    def test_raw_record_survives_cold_fold_and_warm_replay(self):
        name = "rtp_v4_indexer." + uuid.uuid4().hex
        env = dict(os.environ, RTP_LLM_RESIDENT_STORE=name)
        with tempfile.TemporaryDirectory() as root:
            shard = Path(root, "pytorch_model.bin")
            torch.save(fixture.Dsv4IndexerCheckpointTest()._weights(), shard)
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                for phase in ("cold", "warm"):
                    if phase == "warm":
                        shard.unlink()
                    subprocess.run([sys.executable, __file__, str(shard), phase], env=env, check=True, timeout=120)
            finally:
                store.stop_store(name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] in ("cold", "warm"):
        _worker(sys.argv[1], sys.argv[2] == "warm")
    else:
        unittest.main()
