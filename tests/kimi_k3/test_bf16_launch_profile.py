import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "k3_launch_bf16",
    Path(__file__).resolve().parents[2] / "example/k3/main_migration/launch_bf16.py",
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def arguments(tmp_path, role="DECODE", layers=93):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({"num_hidden_layers": layers}))
    draft = tmp_path / "draft"
    draft.mkdir()
    server = tmp_path / "server"
    server.touch()
    return SimpleNamespace(
        checkpoint=str(target),
        draft_checkpoint=str(draft),
        role=role,
        peer_ip="192.0.2.1",
        start_port=31000,
        peer_port=32000,
        server=str(server),
    )


@pytest.mark.parametrize("role", ["PREFILL", "DECODE"])
def test_fixed_full_profile(tmp_path, role):
    environment, command = launcher.launch_config(arguments(tmp_path, role))
    options = dict(zip(command[1::2], command[2::2]))
    assert environment["ACT_TYPE"] == environment["SP_ACT_TYPE"] == "BF16"
    assert environment["LOAD_METHOD"] == "fastsafetensors"
    assert environment["GEN_NUM_PER_CIRCLE"] == "3"
    for key in ("FP8_GEMM", "FP8_MLA", "FP8_KV_CACHE"):
        assert environment[key] == "0"
    for key in ("tp_size", "ep_size", "ffn_sp_size"):
        assert options["--" + key] == "8"
    assert options["--dp_size"] == options["--prefill_cp_size"] == "1"
    assert options["--cache_store_rdma_mode"] == "1"
    assert options["--enable_cuda_graph"] == str(int(role == "DECODE"))
    assert options["--concurrency_limit"] == "16"
    assert int(options["--max_seq_len"]) > 2 * 65536


def test_reject_four_layer_as_formal_profile(tmp_path):
    with pytest.raises(ValueError, match="93"):
        launcher.launch_config(arguments(tmp_path, layers=4))


@pytest.mark.parametrize("field", ["start_port", "peer_port"])
def test_reject_rank_port_overflow(tmp_path, field):
    args = arguments(tmp_path)
    setattr(args, field, 65500)
    with pytest.raises(ValueError, match="port"):
        launcher.launch_config(args)
