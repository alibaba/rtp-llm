"""Content identities and startup registration for JIT cache producers.

This module uses only the standard library: runtime bootstrap may register
producers before importing GPU libraries or the model execution framework.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path

ComponentIdentityProvider = Callable[[Mapping[str, str]], tuple[str, ...] | None]
_COMPONENT_IDENTITY_PROVIDERS: dict[str, ComponentIdentityProvider] = {}


def register_component_identity(name: str, provider: ComponentIdentityProvider) -> None:
    """Register a runtime's producer before the first cache scope is resolved.

    A provider includes the relevant ABI scopes and all selected compiler/source
    identities. Returning None uses the component's default package identity.
    """
    _COMPONENT_IDENTITY_PROVIDERS[name] = provider


def component_identity(name: str, scopes: Mapping[str, str]) -> tuple[str, ...] | None:
    provider = _COMPONENT_IDENTITY_PROVIDERS.get(name)
    return provider(scopes) if provider is not None else None


def isolated_package_identity(
    root_value: str,
    package: Path,
    distribution: str,
    *,
    companion_record_prefixes: tuple[str, ...] = (),
    include_compiler_binaries: bool = False,
) -> str:
    """Identify an explicitly selected wheel without consulting global dist-info."""
    root = Path(root_value).resolve(strict=True)
    selected = (root / package).resolve()
    if not selected.is_dir():
        raise ValueError(f"JIT producer is not a package directory: {selected}")
    normalized = distribution.replace("-", "_").lower()
    candidates = (
        *root.glob("*.dist-info/RECORD"),
        *(root / distribution.replace("-", "_")).glob("*.dist-info/RECORD"),
    )
    prefix = normalized + "_"
    records = [
        path
        for path in candidates
        if (name := path.parent.name.lower().replace("-", "_")).startswith(prefix)
        and name[len(prefix) : len(prefix) + 1].isdigit()
    ]
    if len(records) > 1:
        raise ValueError(f"ambiguous isolated {distribution} wheel at {root}")
    # Editable and manually unpacked installs may not carry dist-info. Hash
    # the selected code as well as RECORD so local operator patches move the
    # scope even when wheel metadata was left unchanged.
    # Remote snapshots are shared by installations with the same contents,
    # even when their package directories differ between hosts.
    digest = sha256()
    if records:
        digest.update(records[0].read_bytes())
    # Producers may also consume companion header/library wheels.
    for record in sorted(
        path
        for path in candidates
        if any(
            path.parent.name.lower().replace("-", "_").startswith(prefix)
            for prefix in companion_record_prefixes
        )
    ):
        digest.update(record.parent.name.lower().encode() + b"\0")
        digest.update(record.read_bytes())
    code = sorted(
        path
        for path in selected.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".py", ".so", ".cu", ".cuh", ".h", ".inc", ".cpp"}
            or (include_compiler_binaries and "bin" in path.relative_to(selected).parts)
        )
        and "__pycache__" not in path.relative_to(selected).parts
    )
    if not code:
        raise ValueError(f"cannot identify isolated {distribution} code at {selected}")
    for path in code:
        digest.update(path.relative_to(selected).as_posix().encode() + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def selected_package_identity(
    module: str, distribution: str, *, include_compiler_binaries: bool = False
) -> str:
    """Fingerprint the package Python will import, including local source edits."""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise ValueError(f"{module} is not importable")
    package = Path(spec.origin).resolve().parent
    return isolated_package_identity(
        str(package.parent),
        Path(package.name),
        distribution,
        include_compiler_binaries=include_compiler_binaries,
    )
