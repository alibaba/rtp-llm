"""Locate smoke response artifacts without changing task/golden input paths."""

from pathlib import Path


def smoke_actual_path(
    taskinfo_path: str, query_idx: int, output_dir: str, package_smoke_dir: str
) -> Path:
    """Return a confined artifact path for an OSS or internal smoke task.

    Absolute paths originate from the ordinary smoke runner. Map them against
    the actual package and its two supported internal-source layouts, keeping
    the complete repository-relative identity. Legacy relative paths retain
    their existing artifact layout.
    """
    task = Path(taskinfo_path)
    if not task.name or ".." in task.parts:
        raise ValueError(f"Invalid task artifact path: {taskinfo_path!r}")
    if isinstance(query_idx, bool) or not isinstance(query_idx, int) or query_idx < 0:
        raise ValueError(f"Invalid query index: {query_idx!r}")

    relative = task
    if task.is_absolute():
        smoke_dir = Path(package_smoke_dir).resolve()
        source = task.resolve()
        # Internal sources may be nested/symlinked from the OSS workspace or
        # be its sibling in the ordinary paired checkout.
        roots = (
            (
                smoke_dir.parents[2] / "internal_source/rtp_llm/test/smoke",
                Path("internal_source/rtp_llm/test/smoke"),
            ),
            (
                smoke_dir.parents[3] / "internal_source/rtp_llm/test/smoke",
                Path("internal_source/rtp_llm/test/smoke"),
            ),
            (smoke_dir, Path("rtp_llm/test/smoke")),
        )
        for root, prefix in roots:
            try:
                relative = prefix / source.relative_to(root.resolve())
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"Task path is outside smoke source roots: {taskinfo_path!r}"
            )

    relative = relative.with_suffix(f".query_{query_idx}.json")
    output_root = Path(output_dir).resolve()
    artifact_root = output_root / "smoke_actual"
    destination = artifact_root / relative
    # Check only this destination's components, including a pre-existing file.
    # An artifact symlink must neither escape the root nor alias another task.
    component = output_root
    for part in ("smoke_actual", *relative.parts):
        component /= part
        if component.is_symlink():
            raise ValueError(f"Artifact path contains a symlink: {component}")
    destination.resolve().relative_to(artifact_root)
    return destination
