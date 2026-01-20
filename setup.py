#!/usr/bin/env python3
"""
RTP-LLM Setup Script

This script integrates Bazel C++ extension building with setuptools.
The C++ extensions are built using Bazel and then copied to the package.

Wheel naming follows vllm convention:
  rtp_llm-{VERSION}+{PLATFORM}-cp{PY}-cp{PY}-{PLATFORM_TAG}.whl
  
Examples:
  rtp_llm-0.1.0+ppu1.5.2-cp310-cp310-linux_x86_64.whl
  rtp_llm-0.1.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
  rtp_llm-0.1.0+rocm62-cp310-cp310-linux_x86_64.whl
"""
import os
import sys
import shutil
import subprocess
import platform
import datetime
import json
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    bdist_wheel = None

# Base version from pyproject.toml
BASE_VERSION = "0.1.0"

# Platform to version suffix mapping (for wheel naming)
PLATFORM_CONFIG_VERSIONS = {
    "ppu": "ppu1.5.2",
    "cuda12_6": "cu126",
    "cuda12_9": "cu129",
    "cuda12_9_arm": "cu129",
    "rocm": "rocm62",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.absolute()


def get_version_with_platform() -> str:
    """Get version string with platform suffix.
    
    Returns version like: 0.1.0+ppu1.5.2 or 0.1.0+cu126
    """
    detected = detect_build_config()
    return f"{BASE_VERSION}+{PLATFORM_CONFIG_VERSIONS[detected]}"



def get_platform_tag() -> str:
    """Get wheel platform tag.
    
    Returns platform tag like: linux_x86_64, manylinux_2_28_x86_64
    """
    machine = platform.machine()
    
    # Extract platform from RTP_BAZEL_CONFIG
    build_config = detect_build_config()
    
    # ARM builds
    if "arm" in build_config.lower() or machine == "aarch64":
        return f"manylinux_2_28_{machine}"
    
    # Standard Linux x86_64
    return f"linux_{machine}"


def generate_proto_files(project_root: Path = None) -> None:
    """Generate Python files from .proto definitions.
    
    This function compiles all .proto files in the project to Python modules.
    Must be called before building wheel or running tests.
    """
    from grpc_tools import protoc
    import grpc_tools

    # grpcio-tools vendors well-known protos under grpc_tools/_proto,
    # e.g. google/protobuf/wrappers.proto. We must add it to proto_path.
    grpc_tools_proto_root = (Path(grpc_tools.__file__).resolve().parent / "_proto")

    # Proto file locations
    proto_files = [
        "rtp_llm/proto/model_rpc_service.proto",
        "rtp_llm/proto/all_embedding_rpc_service.proto",
    ]
    
    print(f"Generating protobuf files...")
    
    for proto_file in proto_files:
        proto_path = project_root / proto_file
        if not proto_path.exists():
            raise FileNotFoundError(f"Proto file not found: {proto_path}")

        print(f"  Compiling {proto_file}")
        args = [
            "grpc_tools.protoc",
            f"-I{project_root}",
            f"-I{grpc_tools_proto_root}",
            f"--python_out={project_root}",
            f"--grpc_python_out={project_root}",
            str(proto_path),
        ]
        code = protoc.main(args)
        if code != 0:
            raise RuntimeError(f"Failed to compile {proto_file}, exit code: {code}, args={args}")

    print("Proto file generation complete.")


# ============================================================================
# Bazel Configuration - Unified via RTP_BAZEL_CONFIG
# ============================================================================
# 
# RTP_BAZEL_CONFIG is passed directly to bazelisk as command-line arguments.
# RTP_BAZEL_APPEND_CONFIG appends args to the auto-detected default config.
# Use native Bazel argument format.
#
# Examples:
#   export RTP_BAZEL_CONFIG="--config=ppu"
#   export RTP_BAZEL_CONFIG="--config=cuda12_6 --config=sm9x"
#   export RTP_BAZEL_CONFIG="--config=rocm --jobs=200"
#   export RTP_BAZEL_CONFIG="--config=cuda12_6 --config=sm9x --verbose_failures"
#   export RTP_BAZEL_APPEND_CONFIG="--config=sm9x --jobs=200"
#
# Common configs:
#   --config=ppu          PPU platform
#   --config=cuda12_6     CUDA 12.6
#   --config=cuda12_9     CUDA 12.9
#   --config=rocm         ROCm/AMD
#   --config=sm8x         CUDA SM 8.x (Ampere)
#   --config=sm9x         CUDA SM 9.x (Hopper)
#
# ============================================================================


# ============================================================================
# Bazel Configuration Functions
# ============================================================================
# 统一的配置获取函数。
# 注意: setup.py 不能依赖 rtp_llm.config_utils，因为 rtp_llm 可能未安装。
# 这些函数与 rtp_llm/config_utils.py 中的函数保持一致。

# PPU 检测路径和环境变量
_PPU_SDK_PATHS = [Path("/usr/local/PPU_SDK"), Path("/opt/ppu")]
_PPU_ENV_VARS = ["PPU_SDK_PATH", "PPU_HOME"]


def _get_bazel_config() -> str:
    """获取 RTP_BAZEL_CONFIG 环境变量的原始值。"""
    return os.environ.get("RTP_BAZEL_CONFIG", "")


def _get_bazel_append_config() -> str:
    """获取 RTP_BAZEL_APPEND_CONFIG 环境变量的原始值。"""
    return os.environ.get("RTP_BAZEL_APPEND_CONFIG", "")


def parse_bazel_config(default_config: str = "") -> list:
    """
    Parse RTP_BAZEL_CONFIG/RTP_BAZEL_APPEND_CONFIG and return bazel arguments.
    
    Returns:
        list: Bazel arguments parsed from environment config(s)
        
    Examples:
        "--config=ppu" -> ["--config=ppu"]
        "--config=cuda12_6 --config=sm9x" -> ["--config=cuda12_6", "--config=sm9x"]
    """
    bazel_config = _get_bazel_config()
    append_config = _get_bazel_append_config()
    args = []

    if bazel_config:
        print(f"Using RTP_BAZEL_CONFIG: {bazel_config}")
        args.extend(bazel_config.split())
    elif default_config:
        args.append(f"--config={default_config}")

    if append_config:
        print(f"Using RTP_BAZEL_APPEND_CONFIG: {append_config}")
        args.extend(append_config.split())

    return args


def extract_platform_from_config() -> str:
    """
    Extract platform name from RTP_BAZEL_CONFIG for wheel naming.
    
    Returns:
        str: Platform name (ppu, cuda12_6, rocm, etc.) or empty string
    """
    bazel_config = _get_bazel_config()
    # Look for --config=<platform> pattern
    for arg in bazel_config.split():
        if arg.startswith("--config="):
            config_value = arg.split("=", 1)[1]
            # Skip SM configs, we want platform config
            if config_value in PLATFORM_CONFIG_VERSIONS:
                return config_value
    return ""


def _detect_ppu() -> bool:
    """检测文件系统中是否存在 PPU SDK。"""
    if any(p.exists() for p in _PPU_SDK_PATHS):
        return True
    if any(os.environ.get(var) for var in _PPU_ENV_VARS):
        return True
    return False


def _get_cuda_version_from_json() -> str:
    """从 /usr/local/cuda/version.json 读取 CUDA 版本。
    
    Returns:
        str: CUDA 版本字符串，例如 "12.9.20250531"，如果文件不存在或解析失败则返回 None
    """
    version_json_path = Path("/usr/local/cuda/version.json")
    if not version_json_path.exists():
        return None
    
    try:
        with open(version_json_path, 'r') as f:
            data = json.load(f)
            version = data.get("cuda", {}).get("version")
            return version
    except (json.JSONDecodeError, KeyError, IOError) as e:
        return None


def _get_cuda_config_from_version(version_str: str) -> str:
    """根据 CUDA 版本字符串确定配置名称。
    
    Args:
        version_str: CUDA 版本字符串，例如 "12.9.20250531" 或 "12.6.0"
    
    Returns:
        str: 配置名称，例如 "cuda12_6" 或 "cuda12_9"
    """
    if not version_str:
        return "cuda12_6"  # 默认值
    
    # 解析版本号，例如 "12.9.20250531" -> 主版本号 12.9
    parts = version_str.split('.')
    if len(parts) >= 2:
        major = parts[0]
        minor = parts[1]
        version_key = f"{major}.{minor}"
        
        # 根据版本选择配置
        if version_key >= "12.9":
            return "cuda12_9"
        elif version_key >= "12.6":
            return "cuda12_6"
        else:
            return "cuda12_6"  # 默认使用 12.6
    
    return "cuda12_6"  # 默认值


def _detect_cuda() -> bool:
    """检测文件系统中是否存在 CUDA。"""
    return Path("/usr/local/cuda").exists() or bool(shutil.which("nvcc"))


def _detect_rocm() -> bool:
    """检测文件系统中是否存在 ROCm。"""
    return Path("/opt/rocm").exists()


def _ensure_rocm_target_list() -> None:
    """Validate ROCm target list is gfx942 only."""
    target_list_path = Path("/opt/rocm/bin/target.lst")
    if not target_list_path.exists():
        raise RuntimeError(
            "ROCm target list not found: /opt/rocm/bin/target.lst. "
            "Please follow the ROCm pre-setup step in "
            "github-opensource/internal_source/ci/new_test.sh to add gfx942."
        )

    try:
        with open(target_list_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
    except OSError as e:
        raise RuntimeError(
            f"Failed to read ROCm target list: {target_list_path}. "
            f"Error: {e}"
        ) from e

    targets = [line for line in lines if line]
    if targets != ["gfx942"]:
        raise RuntimeError(
            "ROCm target list mismatch in /opt/rocm/bin/target.lst: "
            f"expected only 'gfx942', got {targets}. "
            "Please follow the ROCm pre-setup step in "
            "github-opensource/internal_source/ci/new_test.sh to set gfx942."
        )


def detect_build_config(verbose: bool = True) -> str:
    """Detect appropriate Bazel config based on environment.
    
    Returns platform name for auto-detection when RTP_BAZEL_CONFIG is not set.
    """
    # 1. Check RTP_BAZEL_CONFIG first
    build_config = extract_platform_from_config()
    if build_config:
        if build_config == "rocm":
            _ensure_rocm_target_list()
        return build_config
    
    # 2. 文件系统检测
    if _detect_ppu():
        if verbose:
            print("Detected PPU environment")
        return "ppu"
    
    if _detect_cuda():
        # 从 version.json 读取 CUDA 版本
        cuda_version = _get_cuda_version_from_json()
        if cuda_version:
            cuda_config = _get_cuda_config_from_version(cuda_version)
            if verbose:
                print(f"Detected CUDA environment (version: {cuda_version}, config: {cuda_config})")

            if platform.machine() == "aarch64" and cuda_config == "cuda12_9":
                return "cuda12_9_arm"

            return cuda_config

    
    if _detect_rocm():
        if verbose:
            print("Detected ROCm environment")
        _ensure_rocm_target_list()
        return "rocm"
    
    # Fallback - raise error
    raise RuntimeError(
        "Cannot detect build configuration. Please set RTP_BAZEL_CONFIG environment variable. "
        "Example: export RTP_BAZEL_CONFIG='--config=ppu' or '--config=cuda12_6 --config=sm9x'"
    )


def should_skip_bazel_build() -> bool:
    """Check if Bazel build should be skipped.
    
    Returns:
        bool: True if RTP_SKIP_BAZEL_BUILD environment variable is set, False otherwise.
    """
    return bool(os.environ.get("RTP_SKIP_BAZEL_BUILD"))


def rewrite_torch_root() -> None:
    """Rewrite .torch_bazelrc to replace $(TORCH_ROOT) with actual torch installation path.
    
    This function detects the torch installation path from the current Python environment
    and updates .torch_bazelrc to use the actual path instead of the environment variable.
    """

    import torch
    # Get torch root directory (parent of torch package directory)
    # torch.__path__[0] gives us the torch package directory
    torch_package_dir = Path(torch.__path__[0])
    torch_root = str(torch_package_dir.absolute())
    print(f"Detected TORCH_ROOT: {torch_root}")
    
    content = '''
build --action_env=TORCH_ROOT=$(TORCH_ROOT)
build --host_action_env=TORCH_ROOT=$(TORCH_ROOT)
'''

    project_root = get_project_root()
    torch_bazelrc_path = project_root / ".torch_bazelrc"

    with open(torch_bazelrc_path, 'w+') as f:
        content = content.replace("$(TORCH_ROOT)", torch_root)
        f.write(content)
    
    # Only write if content changed
    print(f"Updated {torch_bazelrc_path} with TORCH_ROOT={torch_root}")


def build_bazel_extensions(build_config: str) -> None:
    """Build C++ extensions using Bazel.
    
    Args:
        build_config: Platform name (ppu, cuda12_6, rocm) used for logging.
                      Actual bazel args come from RTP_BAZEL_CONFIG.
    """
    # Rewrite .torch_bazelrc with actual torch installation path
    rewrite_torch_root()
    
    project_root = get_project_root()
    
    # Create log directory
    log_dir = project_root / "build_logs"
    log_dir.mkdir(exist_ok=True)
    
    # Log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bazel_build_{build_config}_{timestamp}.log"
    
    print(f"=" * 60)
    print(f"Building C++ extensions with config: {build_config}")
    print(f"Project root: {project_root}")
    print(f"Build log: {log_file}")
    print(f"=" * 60)
    
    # Bazel targets to build
    targets = [
        "//:th_transformer",
        "//:rtp_compute_ops",
        "//:th_transformer_config",
    ]
    
    # Auto-set output_user_root for platform-specific cache isolation
    # This ensures PPU/CUDA/ROCm builds don't share cache, maximizing cache reuse
    # Note: --output_user_root must be before the 'build' command
    bazel_args = parse_bazel_config(default_config=build_config)
    has_output_root = any("--output_user_root" in arg for arg in bazel_args)
    
    # Build command - bazelisk [startup_options] build [build_options] targets
    cmd = ["bazelisk"]
    
    # Add output_user_root as startup option (before 'build')
    if not has_output_root:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", f"bazel_{build_config}")
        cmd.append(f"--output_user_root={cache_dir}")
        print(f"Using platform-specific cache: {cache_dir}")
    
    # Add 'build' command and targets
    cmd.extend(["build", *targets])
    
    # Add RTP_BAZEL_CONFIG/RTP_BAZEL_APPEND_CONFIG args directly (native bazel format)
    if bazel_args:
        # Filter out output_user_root from bazel_args (it's a startup option)
        build_args = [arg for arg in bazel_args if "--output_user_root" not in arg]
        cmd.extend(build_args)

    # Note: We don't add --jobs by default, let Bazel decide based on system resources.
    # Users can add --jobs=N via RTP_BAZEL_CONFIG if needed.

    # Remove uv/build from PATH in build isolation env, make sure bazel cache reuse
    # path_env = os.environ["PATH"].split(":")
    # path_env = [path for path in path_env if 'uv/build' not in path]
    # new_path = ":".join(path_env)
    # os.environ['PATH'] = new_path

    print(f"PATH: {os.environ['PATH']}")
    print(f"Running: {' '.join(cmd)}")
    print(f"This may take a while... Check {log_file} for progress.")

    # Run bazel with output redirected to log file
    try:
        with open(log_file, 'w') as f:
            # Write command info to log
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {project_root}\n")
            f.write(f"Build config: {build_config}\n")
            f.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            # Run subprocess with output to both file and terminal (tee-like behavior)
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Stream output to both file and stdout
            for line in process.stdout:
                f.write(line)
                f.flush()
                # Print progress indicators to terminal
                if any(x in line for x in ['[', 'INFO:', 'ERROR:', 'WARNING:', 'FAILED']):
                    print(line.rstrip())
            
            process.wait()
            
            # Write completion info
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Finished at: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Exit code: {process.returncode}\n")
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
    
    except subprocess.CalledProcessError as e:
        print(f"\n" + "=" * 60)
        print(f"ERROR: Bazel build failed with exit code {e.returncode}")
        print(f"Check the full log at: {log_file}")
        print(f"Last 50 lines of log:")
        print("=" * 60)
        
        # Print last 50 lines of log
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        except Exception:
            pass
        
        raise RuntimeError(
            f"Bazel build failed. Check log file: {log_file}"
        ) from e
    
    print(f"\nBazel build completed successfully!")
    print(f"Full log available at: {log_file}")
    
    # Copy built .so files to package
    copy_extensions(project_root, build_config)


def copy_core_so_files(bazel_bin: Path, target_dir: Path) -> None:
    """Copy core .so files (always required)."""
    core_so_files = [
        "libth_transformer.so",
        "librtp_compute_ops.so",
        "libth_transformer_config.so",
    ]
    
    for so_file in core_so_files:
        src = bazel_bin / so_file
        if src.exists():
            dst = target_dir / so_file
            print(f"  {src} → {dst}")
            shutil.copy2(src, dst)
        else:
            # Try to find in subdirectories
            found = list(bazel_bin.glob(f"**/{so_file}"))
            if found:
                src = found[0]
                dst = target_dir / so_file
                print(f"  {src} → {dst}")
                shutil.copy2(src, dst)
            else:
                print(f"  Warning: {so_file} not found in {bazel_bin}")


def copy_kernel_so_files(bazel_bin: Path, target_dir: Path) -> int:
    """Copy kernel .so files (generated by copy_so rules in bazel).
    
    Returns:
        int: Number of files copied
    """
    kernel_so_patterns = [
        "libmmha*.so",
        "libdmmha*.so",
        "libfa.so",
        "libflashinfer*.so",
        "libfpA_intB.so",
        "libint8_gemm.so",
        "libmoe*.so",
        "libacext*.so",
        "libdeepgemm*.so",
        "libdeep_ep*.so",
        
        "libaccl_ep.so",
        "libdeep_ep_rocm.so",

        # 3fs
        "libhf3fs_api_shared.so",
        "libboost_context.so.1.71.0",
        "libboost_filesystem.so.1.71.0",
        "libboost_program_options.so.1.71.0",
        "libboost_regex.so.1.71.0",
        "libboost_system.so.1.71.0",
        "libboost_thread.so.1.71.0",
        "libboost_atomic.so.1.71.0",
        "libdouble-conversion.so.3",
        "libgflags.so.2.2",
        "libglog.so.0",
        "libevent-2.1.so.7",
        "libdwarf.so.1",
        "libicui18n.so.66",
        "libicuuc.so.66",
        "libicudata.so.66",
        "libunwind.so.8",
        "libssl.so.1.1",
        "libcrypto.so.1.1",
        
        # aiter_copy_files patterns
        "module_aiter_enum.so",
        "module_custom_all_reduce.so",
        "module_quick_all_reduce.so",
        "module_moe_sorting.so",
        "module_moe_asm.so",
        "module_gemm_a8w8_bpreshuffle.so",
        "module_gemm_a8w8_blockscale.so",
        "module_quant.so",
        "module_smoothquant.so",
        "module_pa.so",
        "module_activation.so",
        "module_attention_asm.so",
        "module_mha_fwd.so",
        "module_fmha_v3_varlen_fwd.so",
        "module_norm.so",
        "module_rmsnorm.so",
        "module_gemm_a8w8.so",
        "module_moe_ck2stages.so",
        "module_deepgemm.so",
    
        # aiter_src_copy files 
        "libmodule_aiter_enum.so",
        "libmodule_custom_all_reduce.so",
        "libmodule_quick_all_reduce.so",
        "libmodule_moe_sorting.so",
        "libmodule_moe_asm.so",
        "libmodule_gemm_a8w8_bpreshuffle.so",
        "libmodule_gemm_a8w8_blockscale.so",
        "libmodule_quant.so",
        "libmodule_smoothquant.so",
        "libmodule_pa.so",
        "libmodule_activation.so",
        "libmodule_deepgemm.so",
        "libmodule_attention_asm.so",
        "libmodule_mha_fwd.so",
        "libmodule_fmha_v3_varlen_fwd.so",
        "libmodule_norm.so",
        "libmodule_rmsnorm.so",
        "libmodule_gemm_a8w8.so",
        "libmodule_moe_ck2stages.so",
    ]
    
    print(f"\nCopying kernel extensions:")
    
    # First, collect all matching files and deduplicate by filename
    # Use a dict to store unique files: key is filename, value is source path
    unique_files = {}
    
    for pattern in kernel_so_patterns:
        # Search in bazel-bin root first
        found_files = list(bazel_bin.glob(pattern))
        # Also search in subdirectories
        found_files.extend(list(bazel_bin.glob(f"**/{pattern}")))
        
        for src in found_files:
            if src.is_file():
                filename = src.name
                # Only keep the first occurrence of each filename
                if filename not in unique_files:
                    unique_files[filename] = src
    
    # Now copy all unique files
    copied_count = 0
    for filename, src in sorted(unique_files.items()):
        dst = target_dir / filename
        print(f"  {src} → {dst}")
        shutil.copy2(src, dst)
        copied_count += 1
    
    print(f"  Copied {copied_count} kernel .so files")
    return copied_count

def copy_extensions(project_root: Path, build_config: str) -> None:
    """Copy built .so files and proto files from bazel-bin to package.
    
    This function orchestrates the copying of all extension files by calling
    specialized sub-functions for different file categories.
    """
    bazel_bin = project_root / "bazel-bin"
    target_dir = project_root / "rtp_llm" / "libs"
    
    # Delete and recreate libs directory to ensure clean state
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying extensions to {target_dir}:")
    
    # Copy core .so files
    copy_core_so_files(bazel_bin, target_dir)
    
    # Copy kernel .so files
    copy_kernel_so_files(bazel_bin, target_dir)


def build_all():
    project_root = get_project_root()
    
    # Skip Bazel build only if explicitly disabled
    if not should_skip_bazel_build():
        generate_proto_files(project_root)
        # Auto-detect platform if RTP_BAZEL_CONFIG not set
        build_config = detect_build_config()
        # build_bazel_extensions will call copy_extensions, which copies proto files
        build_bazel_extensions(build_config)
    else:
        print("Skipping Bazel build (RTP_SKIP_BAZEL_BUILD is set)")

# a placeholder to make setuptools happy
class BazelExtension(Extension):
    def __init__(self):
        super().__init__('BazelExtension', sources=[])

    def build(self):
        build_all()


class BuildBazelExtension(build_ext):
    def run(self):

        for ext in self.extensions:
            if isinstance(ext, BazelExtension):
                ext.build()
        super().run() # 运行原始的 run，以处理非 Bazel 扩展 (如果有的话)

    def copy_extensions_to_source(self):
        """Override to skip copying BazelExtension .so files.
        
        BazelExtension is just a placeholder. The actual .so files are built
        by Bazel and copied to rtp_llm/libs/ by build_all(), so we don't need
        to copy them here. This prevents errors in editable mode when the
        placeholder .so file doesn't exist.
        """
        # Filter out BazelExtension from extensions to copy
        original_extensions = self.extensions
        self.extensions = [ext for ext in self.extensions if not isinstance(ext, BazelExtension)]
        
        try:
            # Only copy non-Bazel extensions if any exist
            if self.extensions:
                super().copy_extensions_to_source()
            # If no extensions left, skip the copy step (this is fine for BazelExtension)
        finally:
            # Restore original extensions list
            self.extensions = original_extensions

if bdist_wheel is not None:
    class BdistWheelWithPlatform(bdist_wheel):
        """Custom bdist_wheel that sets platform-specific wheel name."""
        
        def finalize_options(self):
            super().finalize_options()
            # Force platform-specific wheel (not "any")
            self.root_is_pure = False
            self.plat_name_supplied = True
            self.plat_name = get_platform_tag()
        
        def get_tag(self):
            # Get Python version
            py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
            
            # Return (python_tag, abi_tag, platform_tag)
            return (py_version, py_version, get_platform_tag())
else:
    BdistWheelWithPlatform = None


def _load_toml():
    """Load tomllib or tomli module."""
    try:
        import tomllib
        return tomllib
    except ImportError:
        try:
            import tomli as tomllib
            return tomllib
        except ImportError:
            return None


def get_base_dependencies() -> list:
    """Get base dependencies from pyproject.toml [project.base-dependencies].
    
    Since we use dynamic dependencies, we store base deps in a custom key.
    """
    tomllib = _load_toml()
    if not tomllib:
        print("Warning: tomllib/tomli not available")
        return []
    
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        return []
    
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        
        # Get base dependencies from custom key
        base_deps = pyproject.get("tool", {}).get("rtp-llm", {}).get("base-dependencies", [])
        return base_deps
    except Exception as e:
        print(f"Warning: Could not read base dependencies: {e}")
        return []


def get_platform_dependencies() -> list:
    """Get platform-specific dependencies based on RTP_BAZEL_CONFIG.
    
    This merges the optional dependencies into the main dependencies
    so the wheel has the correct platform-specific packages.
    """
    # Extract platform from RTP_BAZEL_CONFIG
    build_config = detect_build_config()
    if not build_config:
        return []
    
    tomllib = _load_toml()
    if not tomllib:
        print("Warning: tomllib/tomli not available")
        return []
    
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        return []
    
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        
        optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
        
        # Map build config to optional dependency key
        config_to_key = {
            "ppu": "ppu",
            "cuda12": "cuda12",
            "cuda12_6": "cuda12",
            "cuda12_9": "cuda12_9",
            "rocm": "rocm",
            "cuda12_9_arm": "cuda12_arm",
        }
        
        dep_key = config_to_key.get(build_config)
        if dep_key and dep_key in optional_deps:
            deps = optional_deps[dep_key]
            print(f"Adding platform dependencies for {build_config}: {len(deps)} packages")
            return deps
    except Exception as e:
        print(f"Warning: Could not read platform dependencies: {e}")
    
    return []


def get_all_dependencies() -> list:
    """Get all dependencies (base + platform-specific)."""
    base_deps = get_base_dependencies()
    platform_deps = get_platform_dependencies()
    
    # Merge: platform deps override base deps for same package
    all_deps = list(base_deps)
    
    # Get package names from platform deps to check for conflicts
    platform_packages = set()
    for dep in platform_deps:
        # Extract package name (before @ or version specifier)
        if "@" in dep:
            pkg = dep.split("@")[0].strip()
        elif ">=" in dep:
            pkg = dep.split(">=")[0].strip()
        elif "==" in dep:
            pkg = dep.split("==")[0].strip()
        elif "<" in dep:
            pkg = dep.split("<")[0].strip()
        else:
            pkg = dep.strip()
        platform_packages.add(pkg.lower().replace("-", "_").replace(".", "_"))
    
    # Filter out base deps that conflict with platform deps
    filtered_base = []
    for dep in all_deps:
        if "@" in dep:
            pkg = dep.split("@")[0].strip()
        elif ">=" in dep:
            pkg = dep.split(">=")[0].strip()
        elif "==" in dep:
            pkg = dep.split("==")[0].strip()
        elif "<" in dep:
            pkg = dep.split("<")[0].strip()
        else:
            pkg = dep.strip()
        
        normalized = pkg.lower().replace("-", "_").replace(".", "_")
        if normalized not in platform_packages:
            filtered_base.append(dep)
    
    # Combine
    result = filtered_base + platform_deps
    print(f"Total dependencies: {len(result)} (base: {len(filtered_base)}, platform: {len(platform_deps)})")
    return result


if __name__ == "__main__":
    # Get all dependencies (base + platform-specific)
    all_deps = get_all_dependencies()
    
    # Get dynamic version
    version = get_version_with_platform()
    print(f"Building rtp-llm version: {version}")
    
    cmdclass = {}
    if BdistWheelWithPlatform is not None:
        cmdclass["bdist_wheel"] = BdistWheelWithPlatform
    
    setup(
        version=version,
        install_requires=all_deps if all_deps else None,

        ext_modules=[BazelExtension()],
        cmdclass={'build_ext': BuildBazelExtension},
    )

