#!/usr/bin/env python3
"""Build a source-only, side-by-side FlashInfer 0.7.1 derivative.

This is a build-time namespace relocation, never an import hook. No dependency
is imported, installed, or compiled. The input directory is an extracted wheel
or an installed source tree; binary artifacts are deliberately not reused.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


VERSION = "0.7.1"
WHEEL_SHA256 = "69302ed6e91314a85a4392324becc194594a782ba181646d25d5c305e1b7e8c4"
NAMESPACE = "flashinfer_opensource"
REVISION = "rtp-sm90-moe-side-by-side-v1"
BINARY_SUFFIXES = {".so", ".pyc", ".pyo", ".pyd", ".dll", ".a", ".o", ".cubin", ".fatbin"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rename(text):
    # Apply to both source contents and resource paths. In particular C++
    # includes/macros/namespaces, torch op schemas, Python imports, env names,
    # logger names and optional artifact-provider package discovery all agree.
    return text.replace("flashinfer", NAMESPACE).replace(
        "FLASHINFER", "FLASHINFER_OPENSOURCE"
    )


def lazy_init(source, extra=None):
    """Keep relative re-exports but avoid eagerly importing unrelated backends."""
    tree = ast.parse(source)
    exports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    raise ValueError("Wildcard relative re-export needs explicit audit")
                exports[alias.asname or alias.name] = (module, alias.name)
    exports.update(extra or {})
    doc = ast.get_docstring(tree) or "FlashInfer source derivative."
    return (
        repr(doc)
        + "\n# Build-time RTP relocation: lazy imports; SM90 CUTLASS MoE + norm only.\n"
        + "from importlib import import_module as _import_module\n"
        + "_EXPORTS = " + repr(exports) + "\n"
        + "def __getattr__(name):\n"
        + "    if name not in _EXPORTS:\n"
        + "        raise AttributeError(name)\n"
        + "    module_name, symbol = _EXPORTS[name]\n"
        + "    if module_name.strip('.'):\n"
        + "        value = getattr(_import_module(module_name, __name__), symbol)\n"
        + "    else:\n"
        + "        value = _import_module(module_name + symbol, __name__)\n"
        + "    globals()[name] = value\n"
        + "    return value\n"
    )


def replace_once(text, old, new, path):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected {path}: expected one audited site {old!r}")
    return text.replace(old, new, 1)


def relocate(source_site, output_site, source_wheel_sha256=None):
    source_site = source_site.resolve()
    output_site = output_site.resolve()
    if output_site == source_site or source_site in output_site.parents:
        raise ValueError("Output must be separate from the source tree")
    if output_site.exists() and any(output_site.iterdir()):
        raise ValueError("Output must be absent or empty; refusing to overwrite")
    metadata_path = source_site / "flashinfer_python-0.7.1.dist-info" / "METADATA"
    metadata = metadata_path.read_text()
    if "\nVersion: 0.7.1\n" not in metadata:
        raise ValueError("Only audited FlashInfer 0.7.1 is supported")
    if source_wheel_sha256 is not None and source_wheel_sha256 != WHEEL_SHA256:
        raise ValueError("Unexpected official wheel hash")
    source_root = source_site / "flashinfer"
    if not source_root.is_dir():
        raise ValueError("Input has no flashinfer package")
    source_records, output_records, skipped = {}, {}, []
    output_site.mkdir(parents=True, exist_ok=True)

    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_site)
        if "__pycache__" in path.parts or path.suffix in BINARY_SUFFIXES:
            skipped.append(str(relative))
            continue
        # Installed packages can contain links to source resources. Resolve and
        # materialize their bytes, never preserve links into an external env.
        raw = path.read_bytes()
        source_records[str(relative)] = digest(raw)
        out_relative = Path(rename(str(relative)))
        output = output_site / out_relative
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            transformed = raw
        else:
            preserve_license = path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
            transformed_source = source if preserve_license else rename(source)
            package_relative = str(path.relative_to(source_root))
            if package_relative in {
                "__init__.py", "jit/__init__.py", "fused_moe/__init__.py",
                "quantization/__init__.py", "jit/gemm/__init__.py",
            }:
                transformed_source = lazy_init(transformed_source)
            if package_relative == "jit/env.py":
                # No binary-provider discovery, including renamed providers.
                # AOT/CUBIN binaries were built with original namespaces.
                parsed = ast.parse(transformed_source)
                lines = transformed_source.splitlines(keepends=True)
                for node in reversed(parsed.body):
                    if isinstance(node, ast.FunctionDef) and node.name in {
                        "has_flashinfer_opensource_jit_cache", "has_flashinfer_opensource_cubin"
                    }:
                        lines[node.lineno - 1:node.end_lineno] = [
                            f"def {node.name}():\n",
                            "    # Source-only namespace derivative: never reuse upstream binaries.\n",
                            "    return False\n",
                        ]
                transformed_source = "".join(lines)
            if package_relative == "jit/core.py":
                transformed_source = replace_once(
                    transformed_source,
                    "    check_cuda_arch()\n",
                    '    name = "flashinfer_opensource_" + name\n    check_cuda_arch()\n',
                    package_relative,
                )
            if package_relative.endswith("flashinfer_cutlass_fused_moe_binding.cu"):
                transformed_source = replace_once(
                    transformed_source, 'return "fused_moe_runner";',
                    'return "flashinfer_opensource_fused_moe_runner";', package_relative,
                )
            if package_relative == "norm/__init__.py":
                transformed_source = replace_once(
                    transformed_source,
                    'os.environ.get("FLASHINFER_OPENSOURCE_USE_CUDA_NORM", "0")',
                    'os.environ.get("FLASHINFER_OPENSOURCE_USE_CUDA_NORM", "1")',
                    package_relative,
                )
            if path.suffix == ".py":
                ast.parse(transformed_source, filename=str(out_relative))
            transformed = transformed_source.encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(transformed)
        output_records[str(out_relative)] = digest(transformed)

    # Keep upstream provenance and all license files without pretending this is
    # an installed replacement of the original distribution.
    dist_info = metadata_path.parent
    license_root = output_site / NAMESPACE / "_upstream_dist_info"
    for path in sorted(dist_info.rglob("*")):
        if path.is_file() and path.name not in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}:
            relative = path.relative_to(dist_info)
            output = license_root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            raw = path.read_bytes()
            output.write_bytes(raw)
            output_records[str(output.relative_to(output_site))] = digest(raw)

    manifest = {
        "relocation_revision": REVISION,
        "namespace": NAMESPACE,
        "upstream_version": VERSION,
        "official_wheel_sha256": source_wheel_sha256,
        "source_tree_sha256": digest(json.dumps(source_records, sort_keys=True).encode()),
        "output_tree_sha256": digest(json.dumps(output_records, sort_keys=True).encode()),
        "source_files": source_records,
        "output_files": output_records,
        "skipped_binary_or_bytecode": skipped,
        "supported_scope": "SM90 CUTLASS MoE and CUDA norm; source-only JIT; not all FlashInfer APIs",
    }
    (output_site / NAMESPACE / "_relocation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: manifest[key] for key in (
        "relocation_revision", "namespace", "upstream_version", "source_tree_sha256", "output_tree_sha256"
    )}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-site", required=True, type=Path)
    parser.add_argument("--output-site", required=True, type=Path)
    parser.add_argument("--source-wheel-sha256")
    args = parser.parse_args()
    relocate(args.source_site, args.output_site, args.source_wheel_sha256)


if __name__ == "__main__":
    main()
