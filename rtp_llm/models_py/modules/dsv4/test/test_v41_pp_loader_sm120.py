"""Stage-filtered named checkpoint bytes and cold maintained CEP4PP2 engine."""

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.models.deepseek_v41 import DeepSeekV41Weight
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
    save_checkpoint,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_engine_sm120 import (
    run_rank,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import (
    wide_fixture,
)
from rtp_llm.ops import HWKernelConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def make_pp_loader(config, kv_config, rank):
    parallelism = parallelism_for(rank)
    hw = HWKernelConfig()
    hw.enable_cuda_graph = hw.enable_native_cuda_graph = False
    descriptor = DeepSeekV41Weight(config, parallelism, hw, kv_config)
    with patch.object(sys, "argv", [sys.argv[0]]):
        loader = ModelLoader(
            config, descriptor, [], CkptDatabase(config.ckpt_path), LoadMethod.SCRATCH
        )
    assert loader.get_load_config().tp_size == 1
    assert loader.get_load_config().ffn_tp_size == 1
    assert loader.get_load_config().lm_head_tp_size == 4
    assert list(loader._layer_ids) == (
        list(range(21)) if rank < 4 else list(range(21, 40))
    )
    return parallelism, hw, loader


def make_named_worker(rank, metadata):
    config, kv_config = load_configuration(Path(metadata["checkpoint_path"]))
    parallelism, hw, loader = make_pp_loader(config, kv_config, rank)
    names, _ = loader._generate_weight_info()
    owned = set(range(21)) if rank < 4 else set(range(21, 40))
    assert all("engram" not in name for name in names)
    assert ("embed.weight" in names) == (rank < 4)
    assert ("head.weight" in names) == (rank >= 4)
    for name in names:
        if name.startswith("layers."):
            assert int(name.split(".")[1]) in owned, name
        if ".ffn.experts." in name:
            assert int(name.split(".ffn.experts.")[1].split(".")[0]) == rank % 4
    loaded = loader.load_weights(f"cuda:{rank}")
    torch.manual_seed(41701)
    args, expected = wide_fixture()
    assert len(loaded.weights) == 40
    for layer_id, layer in enumerate(loaded.weights):
        assert bool(layer) == (layer_id in owned), layer_id
        if not layer:
            continue
        expected_keys = set(expected.weights[layer_id])
        # The synthetic full fixture includes an unused gate for ratio-1
        # sources. The maintained descriptor loads gates only for ratio 2.
        if args.compress_ratios[layer_id] != 2:
            expected_keys.discard(W.v4_compressor_wgate)
        assert set(layer) == expected_keys, (
            layer_id,
            set(layer) - expected_keys,
            expected_keys - set(layer),
        )
        for key, value in layer.items():
            oracle = expected.weights[layer_id][key]
            if key.startswith("v4.routed."):
                oracle = oracle[rank % 4 : rank % 4 + 1]
            if key in (W.v4_hc_attn_scale, W.v4_hc_ffn_scale):
                oracle = oracle.reshape(3, 1)
            torch.testing.assert_close(
                value.view(torch.uint8),
                oracle.view(torch.uint8),
                rtol=0,
                atol=0,
                msg=f"rank={rank} layer={layer_id} key={key}",
            )
    globals_owned = (
        {W.embedding} if rank < 4 else {W.final_ln_gamma, W.final_ln_beta, W.lm_head}
    )
    assert set(loaded.global_weights) == globals_owned | {W.rope_cos_sin_cache}, (
        rank,
        set(loaded.global_weights),
        globals_owned,
    )
    for key, value in loaded.global_weights.items():
        if key == W.rope_cos_sin_cache:
            # Inherited runtime weight generation is independent of checkpoint
            # global ownership. The maintained V4.1 attention owns its RoPE.
            assert value.shape == (
                config.max_seq_len,
                config.attn_config.rope_config.dim,
            )
            assert value.dtype == torch.float32 and value.is_contiguous()
            assert torch.isfinite(value).all()
            continue
        if key == W.final_ln_beta:
            oracle = torch.zeros_like(value)
        else:
            oracle = expected.global_weights[key]
            if key == W.lm_head:
                width = oracle.size(0) // 4
                oracle = oracle.narrow(0, rank % 4 * width, width)
        torch.testing.assert_close(value, oracle, rtol=0, atol=0, msg=f"global {key}")
    print(
        "DSV41_PP_LOADER "
        + json.dumps(
            {
                "rank": rank,
                "stage": rank // 4,
                "layers": sorted(owned),
                "expert": rank % 4,
                "checkpoint_names": len(names),
                "globals": sorted(globals_owned),
                "derived_globals": [W.rope_cos_sin_cache],
                "head_already_sharded": rank >= 4,
                "all_loaded_bytes_exact": True,
            }
        ),
        flush=True,
    )
    worker = DeepSeekV41Model(
        config, parallelism, loaded, None, 2, py_hw_kernel_config=hw
    )
    assert not worker._materialized
    return worker


class V41PPLoaderSM120Test(unittest.TestCase):
    def test_named_stage_loader_and_cold_background_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(41701)
            args, weights = wide_fixture()
            save_checkpoint(root, args, weights)
            del weights
            metadata = json.loads((root / "config.json").read_text())["text_config"]
            (root / "cache").mkdir()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(
                run_rank,
                args=(port, metadata, make_named_worker, True),
                nprocs=8,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
