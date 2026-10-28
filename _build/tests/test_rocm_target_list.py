"""Test target.lst setup on ordinary temporary files, without accessing /opt/rocm."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from _build.rocm_target_list import initialize_target_list


@pytest.mark.parametrize("initial", [None, "", "\n", "gfx942\n", "\ngfx942\n\n"])
def test_initialization_is_idempotent(tmp_path, initial):
    path = tmp_path / "target.lst"
    if initial is not None:
        path.write_text(initial)
    initialize_target_list(path)
    first = path.read_bytes()
    initialize_target_list(path)
    assert path.read_bytes() == first
    assert [line.strip() for line in path.read_text().splitlines() if line.strip()] == [
        "gfx942"
    ]


@pytest.mark.parametrize(
    "initial", ["gfx90a\n", "gfx942\ngfx90a\n", "gfx942\ngfx942\n", "gfx942-extra\n"]
)
def test_invalid_existing_architectures_are_rejected_without_overwrite(
    tmp_path, initial
):
    path = tmp_path / "target.lst"
    path.write_text(initial)
    with pytest.raises(RuntimeError, match="expected only 'gfx942'"):
        initialize_target_list(path)
    assert path.read_text() == initial


def test_concurrent_initialization_writes_one_target(tmp_path):
    path = tmp_path / "target.lst"
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(initialize_target_list, [path] * 8))
    assert path.read_text() == "gfx942\n"


def test_missing_container_bin_directory_remains_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        initialize_target_list(tmp_path / "missing-bin" / "target.lst")
