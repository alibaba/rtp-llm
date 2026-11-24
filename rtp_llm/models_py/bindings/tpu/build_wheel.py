"""Build a Linux BTC wheel inside the target Qwen TorchTPU development environment.

First build the scheduler DSO in this RTP checkout:
  bazel build --config=cpu --dynamic_mode=off //rtp_llm/cpp/cache/block_tree_cache:libbtc_tree.so

The rank build uses the Qwen3.5 TorchTPU SDK and Bazel 8.8.2. Materialize
the transitive CcInfo.compilation_context.headers for XLA's
//xla/pjrt/c_api_client:pjrt_c_api_client and //xla/pjrt:pjrt_layout.
Put their includes, quote_includes, system_includes, external_includes, the
execution root and its configured bazel-bin in bazel-includes.txt, one
absolute directory per line. The file must include LLVM/MLIR generated
headers and the SDK roots for Eigen, ml_dtypes, Riegeli, FarmHash and
HighwayHash when those dependencies are separate from the Bazel execroot.

Example using the prepared SDK (SDK and TPU_SYNC are checkout paths):
  CXX=g++-13 MAX_JOBS=3 python rtp_llm/models_py/bindings/tpu/build_wheel.py \\
    --xla-source "$SDK/pinned-xla-git" --include "$SDK/protobuf-33.6/src" \\
    --include "$SDK/abseil-cpp-20260526.0" --include-file "$SDK/bazel-includes.txt" \\
    --future-sfinae-patch "$TPU_SYNC/third_party/xla/future_sfinae.patch"

Additional SDK roots can be supplied with --include. The future patch can be
omitted when the checkout already contains the matching TPU Sync fix.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parents[4]
BTC = Path("rtp_llm/cpp/cache/block_tree_cache")
TPU = Path("rtp_llm/models_py/bindings/tpu")


def stage_rank_headers(overlay, xla, absl_options, future_patch):
    # These options affect the C++ ABI of libxla_base.so. A -D does not
    # override Abseil's unconditional definition in options.h.
    options = overlay / "absl/base/options.h"
    options.parent.mkdir(parents=True)
    contents, count = re.subn(
        r"^#define ABSL_OPTION_USE_STD_SOURCE_LOCATION [02]$",
        "#define ABSL_OPTION_USE_STD_SOURCE_LOCATION 0",
        absl_options.read_text(),
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(
            f"unexpected Abseil source_location ABI option in {absl_options}"
        )
    options.write_text(contents)

    relative = Path("xla/tsl/concurrency/future.h")
    future = overlay / relative
    future.parent.mkdir(parents=True)
    shutil.copyfile(xla / relative, future)
    # Hash of ce107528's header with TPU Sync's GCC SFINAE patch applied.
    patched = "0e3e885c431248aa9e276e7e5d57d81bff02d4e669e693584b8f38932a1e5984"
    if hashlib.sha256(future.read_bytes()).hexdigest() != patched:
        if future_patch is None:
            raise ValueError(
                "supply --future-sfinae-patch from the target TPU Sync checkout"
            )
        patch = future_patch.resolve(strict=True)
        upstream_patch = (
            "c628cdcd797f636fef92c72a87fb3ab929404e8936aaac226ed5cc7553cd41c7"
        )
        if hashlib.sha256(patch.read_bytes()).hexdigest() != upstream_patch:
            raise ValueError(f"incorrect target TPU Sync future_sfinae.patch: {patch}")
        subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=overlay,
            check=True,
        )
        if hashlib.sha256(future.read_bytes()).hexdigest() != patched:
            raise ValueError("future.h does not match the patched TorchTPU SDK")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--xla-source", required=True, type=Path)
    parser.add_argument("--future-sfinae-patch", type=Path)
    parser.add_argument(
        "--include-file",
        type=Path,
        help="ordered absolute SDK include roots, one per line",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=Path,
        help="additional include directory from the matching XLA SDK build (repeatable)",
    )
    parser.add_argument(
        "--tree-library", type=Path, default=ROOT / "bazel-bin" / BTC / "libbtc_tree.so"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("a Linux TorchTPU builder is required")

    real_libtpu = Path(distribution("libtpu").locate_file("libtpu/libtpu.so")).resolve(
        strict=True
    )
    os.environ["TPU_LIBRARY_PATH"] = str(real_libtpu)
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    sdk_dist = distribution("torch-tpu")
    sdk = Path(sdk_dist.locate_file("torch_tpu")).resolve()
    include = sdk / "include"
    revision = (include / "XLA_COMMIT").read_text().strip()
    xla = args.xla_source.resolve(strict=True)
    if (
        not re.fullmatch("[0-9a-f]{40}", revision)
        or revision
        != subprocess.check_output(
            ["git", "-C", str(xla), "rev-parse", "HEAD"], text=True
        ).strip()
    ):
        parser.error(f"--xla-source must be the TorchTPU XLA checkout {revision}")
    torch_release = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch.__version__)
    if not torch_release:
        parser.error(f"unknown Torch ABI {torch.__version__}")
    abi = "_".join(torch_release.groups())
    common = sdk / "csrc/common"
    glue = common / f"glue_{abi}"
    glue_name = f"pywrap_{abi}_common"
    includes = [
        ROOT,
        include,
        xla,
        xla / "third_party/tsl",
        *(p.resolve(strict=True) for p in args.include),
    ]
    if args.include_file:
        for line in args.include_file.read_text().splitlines():
            if not line.strip():
                continue
            root = Path(line.strip())
            if not root.is_absolute():
                parser.error(f"--include-file needs absolute paths; got {root}")
            if root.is_dir():
                includes.append(root)
    includes = list(dict.fromkeys(includes))
    headers = {}
    for header in [
        "csrc/api/tensor_buffer.h",
        "xla/pjrt/c_api_client/pjrt_c_api_client.h",
        "xla/xla_data.pb.h",
        "absl/base/options.h",
        "absl/base/config.h",
        "mlir/IR/BuiltinOps.h",
        "mlir/IR/BuiltinOps.h.inc",
        "google/protobuf/runtime_version.h",
    ]:
        headers[header] = next(
            (d / header for d in includes if (d / header).is_file()), None
        )
        if headers[header] is None:
            parser.error(
                f"missing {header}; supply the matching SDK header tree with --include"
            )
    # Validate the first matching SDK header, which is what the compiler sees.
    proto_version = headers["google/protobuf/runtime_version.h"]
    if (
        "#define PROTOBUF_OSS_VERSION 6033006"
        not in proto_version.read_text().splitlines()
    ):
        parser.error(
            f"expected Protobuf 33.6 ahead of Torch headers; found {proto_version}"
        )
    absl_options = headers["absl/base/options.h"]
    absl_config = headers["absl/base/config.h"].read_text().splitlines()
    if not {
        "#define ABSL_LTS_RELEASE_VERSION 20260526",
        "#define ABSL_LTS_RELEASE_PATCH_LEVEL 0",
    }.issubset(absl_config):
        parser.error(f"expected Abseil 20260526.0 in {absl_options.parent}")
    for lib in [common / "libxla_base.so", glue / f"lib{glue_name}.so"]:
        if not lib.is_file():
            parser.error(f"missing TorchTPU library {lib}")

    tree = args.tree_library.resolve(strict=True)
    symbols = set(
        subprocess.check_output(
            ["nm", "-D", "--defined-only", str(tree)], text=True
        ).split()
    )
    required = {
        "btc_tree_create",
        "btc_tree_match_whole_pages_with_join",
        "btc_tree_settle_remote_write",
        "btc_tree_configure_remote_host",
        "btc_tree_take_remote_writes",
        "btc_tree_check_remote_close",
    }
    if required - symbols:
        parser.error(
            f"rebuild the scheduler library in this checkout; missing {sorted(required - symbols)}"
        )
    out = args.output_dir.resolve()
    source_files = [
        BTC / "block_pool" / (name + ".cc")
        for name in [
            "IBlockPool",
            "DiskBlockIO",
            "DiskBlockPool",
            "DiskIOExecutor",
            "AlignedHostMemory",
            "HostStagingBlockPool",
        ]
    ]
    source_files.extend(
        BTC / "transfer" / (name + ".cc")
        for name in ["TransferBatchAsyncContext", "TransferWorkLoop", "tpu/TpuPageDma"]
    )
    source_files.append(TPU / "BtcTpuTransferBinding.cc")
    extension = CppExtension(
        "_btc_tpu_transfer",
        sources=[str(p) for p in source_files],
        include_dirs=[str(p) for p in includes],
        define_macros=[
            (key, "1")
            for key in [
                "RTP_BTC_STANDALONE",
                "RTP_BTC_RANK_STANDALONE",
                "RTP_BTC_TPU_RANK",
                "TORCH_TPU_USER_UNSUPPORTED_PRIVATE_HEADERS",
                "LLVM_ENABLE_THREADS",
            ]
        ],
        libraries=[glue_name, "xla_base"],
        library_dirs=[str(glue), str(common)],
        runtime_library_dirs=[
            "$ORIGIN/torch/lib",
            "$ORIGIN/torch_tpu/csrc/common",
            f"$ORIGIN/torch_tpu/csrc/common/glue_{abi}",
        ],
        extra_compile_args=["-std=c++20", "-fvisibility=hidden", "-pthread"],
        extra_link_args=["-pthread"],
    )

    class BuildRankExtension(BuildExtension):
        def build_extension(self, ext):
            super().build_extension(ext)
            binary = self.get_ext_fullpath(ext.name)
            # Some target SDK wheels still declare the old, unversioned SONAME.
            generic = "libpywrap_torch_tpu_common.so"
            actual = f"lib{glue_name}.so"
            libs = subprocess.check_output(
                ["patchelf", "--print-needed", binary], text=True
            ).splitlines()
            if generic in libs:
                subprocess.run(
                    ["patchelf", "--replace-needed", generic, actual, binary],
                    check=True,
                )
            elif actual not in libs:
                raise ValueError(
                    f"BTC rank extension did not link the installed {actual}"
                )
            # The extension is in a build directory; its installed
            # $ORIGIN/torch* paths only exist after installing the wheel.
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [
                    str(glue),
                    str(common),
                    str(Path(torch.__file__).parent / "lib"),
                    *filter(None, env.get("LD_LIBRARY_PATH", "").split(os.pathsep)),
                ]
            )
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import importlib.util, sys, torch, torch_tpu; "
                    "s = importlib.util.spec_from_file_location('_btc_tpu_transfer', sys.argv[1]); "
                    "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.RankTransfer",
                    str(Path(binary).resolve()),
                ],
                env=env,
                check=True,
            )

    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="btc-tpu-wheel-") as tmp:
        stage = Path(tmp)
        overlay = stage / "rank-headers"
        stage_rank_headers(overlay, xla, absl_options, args.future_sfinae_patch)
        # Place the staged headers and XLA SDK ahead of every Torch include
        # root, including the old Protobuf bundled with torch/include.
        extension.include_dirs = list(
            dict.fromkeys(
                [str(overlay), *(str(p) for p in includes), *extension.include_dirs]
            )
        )
        (stage / "rtp_llm").symlink_to(ROOT / "rtp_llm", target_is_directory=True)
        shutil.copyfile(ROOT / BTC / "external/btc_tree.py", stage / "btc_tree.py")
        shutil.copyfile(
            ROOT / BTC / "storage_backend/v6d/btc_v6d_backend.py",
            stage / "btc_v6d_backend.py",
        )
        package = stage / "_btc_tpu_lib"
        package.mkdir()
        (package / "__init__.py").touch()
        bundled_tree = package / "libbtc_tree.so"
        shutil.copyfile(tree, bundled_tree)
        # The scheduler DSO is a C ABI shared by different Python/Torch builds.
        # Requiring only the Linux/C++ runtime makes that independence testable.
        subprocess.run(
            ["patchelf", "--remove-rpath", str(bundled_tree)],
            check=True,
        )
        needed = subprocess.check_output(
            ["patchelf", "--print-needed", str(bundled_tree)], text=True
        ).splitlines()
        runtime = {
            "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0",
            "libstdc++.so.6", "libgcc_s.so.1", "librt.so.1", "libatomic.so.1",
            "libgomp.so.1",
        }
        extra = [
            dep for dep in needed
            if dep not in runtime and not re.fullmatch(r"ld-linux[\w-]*\.so\.\d+", dep)
        ]
        if extra:
            parser.error(f"rebuild the CPU libbtc_tree.so; unexpected runtime libraries: {extra}")
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"}
        }
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import ctypes, os, sys; "
                "ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | os.RTLD_LOCAL)",
                str(bundled_tree),
            ],
            check=True,
            env=clean_env,
        )
        try:
            os.chdir(stage)
            setup(
                name="btc-tpu-transfer",
                version="0.1.0",
                py_modules=["btc_tree", "btc_v6d_backend"],
                packages=["_btc_tpu_lib"],
                package_data={"_btc_tpu_lib": ["libbtc_tree.so"]},
                ext_modules=[extension],
                cmdclass={"build_ext": BuildRankExtension},
                install_requires=[
                    f"torch=={torch.__version__}",
                    f"torch-tpu=={sdk_dist.version}",
                ],
                extras_require={
                    "tair": [
                        "v6d==0.5.0+gea5c25eb",
                        "kvs-connector==0.1.4+v033.migration2",
                    ]
                },
                zip_safe=False,
                script_name="setup.py",
                script_args=["bdist_wheel", "--dist-dir", str(out)],
            )
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
