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
    assert "CUBLAS_WORKSPACE_CONFIG" not in environment
    assert "CUBLASLT_WORKSPACE_SIZE" not in environment
    assert environment["GEN_NUM_PER_CIRCLE"] == "3"
    for key in ("FP8_GEMM", "FP8_MLA", "FP8_KV_CACHE"):
        assert environment[key] == "0"
    for key in ("tp_size", "ep_size", "ffn_sp_size"):
        assert options["--" + key] == "8"
    assert options["--dp_size"] == options["--prefill_cp_size"] == "1"
    assert options["--cache_store_rdma_mode"] == "1"
    assert options["--decode_retry_times"] == "0"
    assert options["--prefill_retry_times"] == "0"
    assert "--int8_kv_cache" not in options
    assert options["--enable_cuda_graph"] == str(int(role == "DECODE"))
    assert options["--concurrency_limit"] == "16"
    assert int(options["--max_seq_len"]) > 2 * 65536
    assert options["--reserver_runtime_mem_mb"] == "14336"


def test_reject_four_layer_as_formal_profile(tmp_path):
    with pytest.raises(ValueError, match="93"):
        launcher.launch_config(arguments(tmp_path, layers=4))


def test_existing_fp8_switches_control_both_roles_without_changing_compute_dtype(tmp_path):
    args = arguments(tmp_path)
    args.fp8_gemm = True
    args.fp8_mla = True
    args.fp8_kv_cache = True
    environment, command = launcher.launch_config(args)
    options = dict(zip(command[1::2], command[2::2]))
    assert environment["FP8_GEMM"] == environment["FP8_MLA"] == environment["FP8_KV_CACHE"] == "1"
    assert options["--fp8_kv_cache"] == "1"
    assert environment["ACT_TYPE"] == environment["SP_ACT_TYPE"] == "BF16"


def test_fp8_mla_rejects_mismatched_cache_switch(tmp_path):
    args = arguments(tmp_path)
    args.fp8_mla = True
    with pytest.raises(ValueError, match="FP8_MLA.*FP8_KV_CACHE"):
        launcher.launch_config(args)


@pytest.mark.parametrize("field", ["start_port", "peer_port"])
def test_reject_rank_port_overflow(tmp_path, field):
    args = arguments(tmp_path)
    setattr(args, field, 65500)
    with pytest.raises(ValueError, match="port"):
        launcher.launch_config(args)


@pytest.mark.parametrize(
    "path", ["/data2/user/run", "/data7/user/run", "/ssd/5/user/run"]
)
def test_selected_host_data_roots_are_allowed_after_mount_check(monkeypatch, path):
    monkeypatch.setattr(Path, "resolve", lambda self, **kwargs: self)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **kw: "ext4\n")
    launcher.require_local(path)


@pytest.mark.parametrize("path", ["/database/run", "/data-old/run", "/tmp/run"])
def test_unrelated_path_prefixes_are_not_local_data_roots(monkeypatch, path):
    monkeypatch.setattr(Path, "resolve", lambda self, **kwargs: self)
    with pytest.raises(ValueError, match="local data destination"):
        launcher.require_local(path)


def test_allowed_prefix_does_not_exempt_network_mount(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self, **kwargs: self)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **kw: "nfs4\n")
    with pytest.raises(ValueError, match="filesystem"):
        launcher.require_local("/data7/user/run")


def test_cpu_tp_socket_fits_actual_full_model_run_directory():
    run = "/data7/luohaocheng.lhc/k3-main-decode-precheck-20260922-02"
    environment = launcher.cpu_tp_socket_environment(run)
    path = Path(environment["RTP_LLM_CPU_TP_BROADCASTER_DIR"]) / (
        "rtp_llm_tp_" + environment["RTP_LLM_CPU_TP_BROADCASTER_ID"] + "_dp0_0.sock"
    )
    assert len(bytes(path)) < 108
    assert path.parent.parent == Path(run)


def test_cpu_tp_socket_rejects_oversize_run_path():
    with pytest.raises(ValueError, match="too long"):
        launcher.cpu_tp_socket_environment("/data7/" + "x" * 100)


@pytest.mark.parametrize("output,code", [
    ("No IB devices found\n", 1),
    ("0 HCAs found:\n", 0),
    ("state: PORT_DOWN (1)\n", 0),
    ("state: PORT_ACTIVE (4)\n", 1),
])
def test_rdma_probe_rejects_unusable_container(monkeypatch, tmp_path, output, code):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=output, returncode=code))
    with pytest.raises(RuntimeError, match="No active RDMA"):
        launcher.require_rdma_device(tmp_path)
    assert (tmp_path / "rdma-preflight.txt").read_text() == output


def test_rdma_probe_records_active_port(monkeypatch, tmp_path):
    output = "hca_id: mlx5_bond_0\n\tstate: PORT_ACTIVE (4)\n"
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=output, returncode=0))
    launcher.require_rdma_device(tmp_path)
    assert (tmp_path / "rdma-preflight.txt").read_text() == output


def test_explicit_bond_hcas_exclude_other_visible_devices():
    devices = 'mlx5_2 guid\nmlx5_bond_0 guid\nmlx5_bond_1 guid\n'
    links = ('link mlx5_bond_0/1 state ACTIVE physical_state LINK_UP netdev rdma0\n'
             'link mlx5_bond_1/1 state ACTIVE physical_state LINK_UP netdev rdma1\n')
    assert launcher.validate_rdma_hcas('mlx5_bond_0,mlx5_bond_1', devices, links) == 'mlx5_bond_0,mlx5_bond_1'
    for invalid in ('', 'mlx5_bond_0,mlx5_bond_0', 'missing', 'mlx5_2'):
        with pytest.raises(ValueError):
            launcher.validate_rdma_hcas(invalid, devices, links)
    with pytest.raises(ValueError):
        launcher.validate_rdma_hcas('mlx5_bond_0', devices, links.replace('ACTIVE', 'DOWN'))


@pytest.mark.parametrize("role", ["PREFILL", "DECODE"])
def test_explicit_four_layer_keeps_execution_contract(tmp_path, role):
    args = arguments(tmp_path, role, layers=4)
    args.debug_four_layer = True
    path = Path(args.checkpoint) / "config.json"
    config = {"num_hidden_layers": 4, "attn_res_block_size": 12,
              "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}}
    path.write_text(json.dumps(config))
    environment, command = launcher.launch_config(args)
    options = dict(zip(command[1::2], command[2::2]))
    assert environment["GEN_NUM_PER_CIRCLE"] == "3"
    assert environment["ACT_TYPE"] == environment["SP_ACT_TYPE"] == "BF16"
    for key in ("tp_size", "ep_size", "ffn_sp_size"):
        assert options["--" + key] == "8"
    assert options["--enable_cuda_graph"] == str(int(role == "DECODE"))
    assert options["--cache_store_rdma_mode"] == "1"
    config["attn_res_block_size"] = 4
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="AttnRes"):
        launcher.launch_config(args)
    config["attn_res_block_size"] = 12
    config["linear_attn_config"]["full_attn_layers"] = []
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="MLA"):
        launcher.launch_config(args)

def test_four_layer_flag_rejects_full_checkpoint(tmp_path):
    args = arguments(tmp_path)
    args.debug_four_layer = True
    with pytest.raises(ValueError, match="4 target layers"):
        launcher.launch_config(args)
