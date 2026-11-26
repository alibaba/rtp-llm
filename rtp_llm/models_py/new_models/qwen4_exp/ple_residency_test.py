"""Real HF and historical rank-local FT bytes -> fresh fixed-host PLE owner."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewModelLoader,
    NewLoaderConfig,
    NewLoaderLoadMethod,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.ple_weights import Qwen4PLEHostTable
from rtp_llm.models_py.registry import register_model
from rtp_llm.utils.model_weight import W


@register_model("_q4_ple_host_io")
class _Stage(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.table = Qwen4PLEHostTable(
            4,
            3,
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            source_fp8=True,
            preserve_fp8=config.preserve,
            dtype=torch.bfloat16,
        )
        self.aux = torch.nn.Parameter(torch.empty(3), requires_grad=False)

    def checkpoint_weight_name_filter(self):
        return lambda name: (
            self.table.owns_checkpoint_name(name[6:])
            if name.startswith("table.")
            else True
        )

    def load_ft_checkpoint(self, checkpoint):
        used = self.table.load_ft_shards(checkpoint, 0)
        if set(checkpoint.entries) != used | {"layers.0.auxiliary"}:
            raise ValueError("Unclaimed checkpoint tensor in PLE stage")
        self.load_weights({"aux": checkpoint.read("layers.0.auxiliary")})


@register_model("_fst_rank_divergent_files")
class _RankNamed(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.rank = load.tp_rank
        for key in ("common", f"mixed_{self.rank}", f"separate_{self.rank}"):
            self.register_parameter(
                key, torch.nn.Parameter(torch.empty(2), requires_grad=False)
            )

    def checkpoint_weight_name_filter(self):
        return self._parameters.__contains__


def _rank_filter_worker(rank, directory, invalid):
    root = Path(directory)
    torch.distributed.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = NewModelLoader(
            {"model_type": "_fst_rank_divergent_files"},
            NewLoaderConfig(
                device="cpu", tp_size=2, tp_rank=rank, load_method="fastsafetensors"
            ),
            model_path=directory,
        )
        if invalid:
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError, r"distributed prerequisite failure: rank 1: ValueError: Shape mismatch for _RankNamed\.mixed_1"
            ):
                model.load()
        else:
            owned = model.load()
            for name, value in (
                ("common", 5),
                (f"mixed_{rank}", 10 + rank),
                (f"separate_{rank}", 20 + rank),
            ):
                torch.testing.assert_close(
                    owned.get_parameter(name), torch.full((2,), float(value))
                )
            assert model._selected_load_method == NewLoaderLoadMethod.FASTSAFETENSORS
    finally:
        torch.distributed.destroy_process_group()


def check_local_ple(test, method, *, exported, preserve=True):
    dtype = torch.float8_e4m3fn if preserve or not exported else torch.bfloat16
    values = {i: (torch.arange(6).view(2, 3) + i).to(dtype) for i in range(4)}
    # HF release stores BF16 [1]; legacy FT exports the normalized FP32 scalar.
    scale = (
        torch.tensor(0.25, dtype=torch.float32)
        if exported
        else torch.tensor([0.00019931793212890625], dtype=torch.bfloat16)
    )
    aux = torch.tensor([3.0, 7.0, 9.0])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if exported:
            prefix = "rank_01_00_00.layers.0."
            data = {
                prefix + W.qwen4_ple_ngram_shards + f".{i}": values[i] for i in (2, 3)
            }
            if preserve:
                data[prefix + W.qwen4_ple_ngram_scale] = scale
            data[prefix + "auxiliary"] = aux
        else:
            data = {f"table.shard_{i}.weight": values[i] for i in range(4)}
            if method == "scratch":
                # An unfiltered load would try to validate these foreign tensors.
                for i in (0, 1):
                    data[f"table.shard_{i}.weight"] = torch.ones(9, dtype=torch.int32)
            data["table.weight_scale"] = scale
            data["aux"] = aux
        # Separate files make off-rank exclusion and ordinary auxiliary loading
        # independent of checkpoint serialization order.
        files = {}
        for i, (key, value) in enumerate(data.items()):
            filename = f"part-{i}.safetensors"
            save_file({key: value}, str(root / filename))
            files[key] = filename
        index = {"metadata": {}, "weight_map": files}
        if exported:
            index.update(
                is_ft_style_weight=True,
                __env__params__={"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 1},
            )
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
        if method == "fastsafetensors" and not exported:
            torch.multiprocessing.spawn(_fst_worker, args=(directory,), nprocs=2)
        else:
            _verify(test, root, method, 1, preserve, values, aux, scale)


def _fst_worker(rank, directory):
    root = Path(directory)
    torch.distributed.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        values = {
            i: (torch.arange(6).view(2, 3) + i).to(torch.float8_e4m3fn)
            for i in range(4)
        }
        _verify(
            unittest.TestCase(),
            root,
            "fastsafetensors",
            rank,
            True,
            values,
            torch.tensor([3.0, 7.0, 9.0]),
            torch.tensor(0.00019931793212890625),
        )
    finally:
        torch.distributed.destroy_process_group()


def _verify(test, root, method, rank, preserve, values, aux, scale):
    loader = NewModelLoader(
        SimpleNamespace(model_type="_q4_ple_host_io", num_layers=1, preserve=preserve),
        NewLoaderConfig(device="cpu", tp_size=2, tp_rank=rank, load_method=method),
        model_path=str(root),
    )
    model = loader.load()
    test.assertEqual(loader._selected_load_method, NewLoaderLoadMethod(method))
    indices = (2 * rank, 2 * rank + 1)
    test.assertEqual(model.table.shard_indices, indices)
    torch.testing.assert_close(model.aux, aux)
    pointers = []
    for i, tensor in zip(indices, model.table.runtime_shards()):
        test.assertEqual(tensor.device.type, "cpu")
        test.assertEqual(tensor.dtype, values[i].dtype)
        test.assertTrue(
            torch.equal(tensor.view(torch.uint8), values[i].view(torch.uint8))
        )
        pointers.append(tensor.data_ptr())
    if preserve:
        torch.testing.assert_close(model.table.weight_scale, scale.float().reshape(()))
        test.assertEqual(model.table.weight_scale.device.type, "cpu")
    model.to("meta")
    test.assertTrue(model.aux.is_meta)
    test.assertEqual(pointers, [p.data_ptr() for p in model.table.runtime_shards()])


class PLEStorageTest(unittest.TestCase):
    def test_varying_collective_key_and_file_schedules(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                files = {}
                for name, entries in {
                    "shared": {"common": torch.full((2,), 5.0)},
                    "mixed": {
                        "mixed_0": torch.full((2,), 10.0),
                        "mixed_1": torch.full((3 if invalid else 2,), 11.0),
                    },
                    "rank-zero": {"separate_0": torch.full((2,), 20.0)},
                    "rank-one": {"separate_1": torch.full((2,), 21.0)},
                }.items():
                    f = name + ".safetensors"
                    save_file(entries, str(root / f))
                    files.update({key: f for key in entries})
                (root / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": files})
                )
                torch.multiprocessing.spawn(
                    _rank_filter_worker, args=(d, invalid), nprocs=2
                )

    def test_host_checkpoint_formats(self):
        for method in ("scratch", "fastsafetensors"):
            for exported, preserve in ((False, True), (True, True), (True, False)):
                with self.subTest(method=method, exported=exported, preserve=preserve):
                    check_local_ple(self, method, exported=exported, preserve=preserve)


if __name__ == "__main__":
    unittest.main()
