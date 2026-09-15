"""Execute the original ROCm BeamSearch GTests from native pytest."""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from rtp_llm.test.amd_coverage import AMD_GTEST_CASES

pytestmark = [pytest.mark.gpu(type="MI308X")]
_BINARY = Path(__file__).resolve().parents[5] / "libs/test/rocm_beam_search_op_test"


@pytest.mark.parametrize("gtest_case", AMD_GTEST_CASES, ids=AMD_GTEST_CASES)
def test_rocm_beam_search_gtest(gtest_case, tmp_path):
    assert _BINARY.is_file(), f"Missing native test binary: {_BINARY}; run build_ext"
    report = tmp_path / "gtest.xml"
    # A standalone executable cannot reuse the libraries loaded into Python.
    env = os.environ.copy()
    library_paths = [
        str(Path(torch.__file__).resolve().parent / "lib"),
        str(_BINARY.parent.parent),
    ]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    result = subprocess.run(
        [str(_BINARY), f"--gtest_filter={gtest_case}", f"--gtest_output=xml:{report}"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert report.is_file(), output
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(cases) == 1, output
    case = cases[0]
    assert f"{case.get('classname')}.{case.get('name')}" == gtest_case, output
    assert case.get("status") == "run", output
    assert case.get("result") == "completed", output
    assert not any(
        case.find(tag) is not None for tag in ("failure", "error", "skipped")
    ), output
