import sys
from types import SimpleNamespace

import pytest

from rtp_llm.test.cuda13_preflight import prepare_cuda13_runtime


@pytest.mark.parametrize(
    "runtime,devices,capability,required",
    [
        (None, 2, (10, 0), 1),
        ("12.9", 2, (10, 0), 1),
        ("13.0", 1, (10, 0), 2),
        ("13.0", 2, (9, 0), 1),
        ("13.0", 2, (10, 0), 0),
    ],
)
def test_preflight_rejects_wrong_runtime_or_hardware(
    monkeypatch, runtime, devices, capability, required
):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda=runtime),
            cuda=SimpleNamespace(
                device_count=lambda: devices,
                get_device_capability=lambda index: capability,
            ),
        ),
    )
    with pytest.raises(RuntimeError):
        prepare_cuda13_runtime(required)


def test_preflight_checks_every_requested_device_and_preserves_cache(
    monkeypatch, tmp_path
):
    checked = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(
                device_count=lambda: 2,
                get_device_capability=lambda index: checked.append(index) or (10, 3),
            ),
        ),
    )
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path / "configured"))
    prepare_cuda13_runtime(2)
    assert checked == [0, 1]
