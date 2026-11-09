"""Verify dispatch and rollback coverage for the FP32 paged KDA kernel."""

import importlib
from unittest.mock import patch

import pytest
import torch

kda = importlib.import_module(
    "rtp_llm.models_py.triton_kernels.kimi_kda.fused_recurrent"
)


class _LaunchRecorder:
    def __getitem__(self, grid):
        def launch(**kwargs):
            self.kwargs = kwargs

        return launch


@pytest.mark.parametrize("tokens", range(1, 8))
@pytest.mark.parametrize("batch", [1, 15, 16, 24, 32, 47, 48, 49, 64, 65, 96])
@pytest.mark.parametrize("capability", [(8, 9), (10, 0), (10, 3)])
@pytest.mark.parametrize("fast", [False, True])
def test_verify_geometry_and_fallback(tokens, batch, capability, fast):
    q = torch.empty(batch, tokens, 64, 128, device="meta", dtype=torch.bfloat16)
    state = torch.empty(2, 64, 128, 128, device="meta", dtype=torch.float32)
    recorder = _LaunchRecorder()
    with (
        patch.object(kda, "fused_recurrent_kda_fwd_kernel", recorder),
        patch.object(kda, "_FAST_VERIFY_ENABLED", fast),
        patch.object(kda, "_WIDE_VERIFY_ENABLED", False),
        patch.object(kda, "_HOIST_GATE_ENABLED", True),
        patch.object(kda, "_VERIFY_LAUNCH_CONFIG", (16, 1, 3)),
        patch("torch.cuda.get_device_capability", return_value=capability),
    ):
        kda.fused_recurrent_kda_fwd(
            q,
            q,
            q,
            q,
            torch.empty(batch, tokens, 64, device="meta"),
            128**-0.5,
            state,
            block_map=torch.empty(batch, 660, device="meta", dtype=torch.int32),
            sequence_lengths=torch.empty(batch, device="meta", dtype=torch.int32),
            seq_size_per_block=128,
            decode_low_warps=True,
        )
    expected = (32, 4, 2)
    if tokens == 1 and 16 <= batch <= 64 and capability in ((10, 0), (10, 3)):
        expected = (16, 1, 2)
    if tokens == 4 and batch <= 64 and capability == (10, 0):
        expected = (16, 1, 3 if batch == 48 else 2)
    fast_verify_shape = (batch == 48 and 2 <= tokens <= 6) or (
        tokens == 5 and 16 <= batch <= 64
    )
    if fast and fast_verify_shape and capability == (10, 0):
        expected = (16, 1, 3)
    actual = recorder.kwargs
    assert (actual["BV"], actual["num_warps"], actual["num_stages"]) == expected
    assert actual["HOIST_GATE"] == (
        batch == 48 and (tokens == 4 or (fast and 2 <= tokens <= 6))
    )


def test_verify_config_default_override_and_disable(monkeypatch):
    monkeypatch.delenv("GLM53_KDA_VERIFY_CONFIG", raising=False)
    assert kda._verify_launch_config() == (16, 1, 3)
    for disabled in ("", "0"):
        monkeypatch.setenv("GLM53_KDA_VERIFY_CONFIG", disabled)
        assert kda._verify_launch_config() is None
    monkeypatch.setenv("GLM53_KDA_VERIFY_CONFIG", "32,4,2")
    assert kda._verify_launch_config() == (32, 4, 2)
    monkeypatch.setenv("GLM53_KDA_VERIFY_CONFIG", "16,0,3")
    with pytest.raises(ValueError):
        kda._verify_launch_config()


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_precision_and_hoist_switches_at_import(monkeypatch, flag):
    import importlib.util
    import sys

    for env in (
        "GLM53_KDA_FAST_VERIFY",
        "GLM53_KDA_WIDE_VERIFY",
        "GLM53_KDA_HOIST_GATE",
    ):
        monkeypatch.delenv(env, raising=False)
        if flag is not None:
            monkeypatch.setenv(env, flag)
    spec = importlib.util.spec_from_file_location("isolated_kda_controls", kda.__file__)
    isolated = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, isolated)
    spec.loader.exec_module(isolated)
    assert isolated._FAST_VERIFY_ENABLED == (flag != "0")
    assert isolated._WIDE_VERIFY_ENABLED == (flag != "0")
    assert isolated._HOIST_GATE_ENABLED == (flag != "0")


@pytest.mark.parametrize(
    "unsupported",
    [
        "heads",
        "width",
        "bf16_state",
        "state_layout",
        "variable_lengths",
        "no_low_warps",
    ],
)
def test_unvalidated_state_or_model_geometry_keeps_fallback(unsupported):
    batch, tokens = 48, 5
    heads = 32 if unsupported == "heads" else 64
    width = 64 if unsupported == "width" else 128
    q = torch.empty(batch, tokens, heads, width, device="meta", dtype=torch.bfloat16)
    state = torch.empty(
        2,
        heads,
        width,
        width,
        device="meta",
        dtype=torch.bfloat16 if unsupported == "bf16_state" else torch.float32,
    )
    recorder = _LaunchRecorder()
    with (
        patch.object(kda, "fused_recurrent_kda_fwd_kernel", recorder),
        patch.object(kda, "_FAST_VERIFY_ENABLED", True),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        kda.fused_recurrent_kda_fwd(
            q,
            q,
            q,
            q,
            torch.empty(batch, tokens, heads, device="meta"),
            width**-0.5,
            state,
            block_map=torch.empty(batch, 660, device="meta", dtype=torch.int32),
            sequence_lengths=torch.empty(batch, device="meta", dtype=torch.int32),
            cu_seqlens=(
                torch.empty(batch + 1, device="meta", dtype=torch.int64)
                if unsupported == "variable_lengths"
                else None
            ),
            state_v_first=unsupported == "state_layout",
            seq_size_per_block=128,
            decode_low_warps=unsupported != "no_low_warps",
        )
    assert (
        recorder.kwargs["BV"],
        recorder.kwargs["num_warps"],
        recorder.kwargs["num_stages"],
    ) == (32, 4, 2)


@pytest.mark.parametrize(
    "batch,tokens,override,expected",
    [
        (8, 2, {}, (16, 1, 3)),
        (32, 6, {}, (16, 1, 3)),
        (48, 8, {}, (16, 1, 3)),
        (64, 8, {}, (16, 1, 3)),
        (7, 2, {}, (32, 4, 2)),
        (65, 8, {}, (32, 4, 2)),
        (32, 7, {}, (32, 4, 2)),
        (32, 6, {"wide": False}, (32, 4, 2)),
        (48, 6, {"wide": False}, (16, 1, 3)),
        (32, 5, {"wide": False}, (16, 1, 3)),
        (32, 6, {"fast": False}, (32, 4, 2)),
        (32, 6, {"capability": (10, 3)}, (32, 4, 2)),
        (32, 6, {"page": 64}, (32, 4, 2)),
        (32, 6, {"store": False}, (32, 4, 2)),
        (32, 6, {"inplace": False}, (32, 4, 2)),
        (32, 6, {"v_first": True}, (32, 4, 2)),
        (32, 6, {"dtype": torch.bfloat16}, (32, 4, 2)),
    ],
)
def test_wide_verify_respects_rollback_and_paged_state_contract(
    batch, tokens, override, expected
):
    q = torch.empty(batch, tokens, 64, 128, device="meta", dtype=torch.bfloat16)
    state = torch.empty(
        2, 64, 128, 128, device="meta", dtype=override.get("dtype", torch.float32)
    )
    recorder = _LaunchRecorder()
    with (
        patch.object(kda, "fused_recurrent_kda_fwd_kernel", recorder),
        patch.object(kda, "_FAST_VERIFY_ENABLED", override.get("fast", True)),
        patch.object(kda, "_WIDE_VERIFY_ENABLED", override.get("wide", True)),
        patch.object(kda, "_VERIFY_LAUNCH_CONFIG", (16, 1, 3)),
        patch(
            "torch.cuda.get_device_capability",
            return_value=override.get("capability", (10, 0)),
        ),
    ):
        _, updated = kda.fused_recurrent_kda_fwd(
            q,
            q,
            q,
            q,
            torch.empty(batch, tokens, 64, device="meta"),
            128**-0.5,
            state,
            block_map=torch.empty(batch, 1050, device="meta", dtype=torch.int32),
            sequence_lengths=torch.empty(batch, device="meta", dtype=torch.int32),
            seq_size_per_block=override.get("page", 128),
            decode_low_warps=True,
            inplace_final_state=override.get("inplace", True),
            store_states=override.get("store", True),
            state_v_first=override.get("v_first", False),
        )
    actual = recorder.kwargs
    assert (actual["BV"], actual["num_warps"], actual["num_stages"]) == expected
    assert actual["h0"] is state
    assert actual["ht"] is updated
    assert (updated is state) == override.get("inplace", True)
    assert actual["STORE_STATES"] == override.get("store", True)
