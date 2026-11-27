"""Build the Python binding and its CMake library from the same source tree."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class NativeSourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        package = Path(__file__).resolve().parent
        output = Path(base_dir)
        inputs = runpy.run_path(str(package / "native_sources.bzl"))
        for name in inputs["BTC_NATIVE_SOURCES"] + inputs["BTC_NATIVE_HEADERS"]:
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(package / name, dest)
        support = package / "support"
        source = support if support.is_dir() else package.parents[3]
        for name in inputs["BTC_SUPPORT_HEADERS"]:
            dest = output / "support" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dest)


class BuildBinding(build_py):
    def run(self):
        super().run()
        native = Path(self.get_finalized_command("build").build_temp).resolve() / "btc"
        cmake = os.environ.get("CMAKE", "cmake")
        subprocess.run([
            cmake, "-S", str(Path(__file__).resolve().parent), "-B", str(native),
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={native / 'lib'}",
        ], check=True)
        subprocess.run([cmake, "--build", str(native), "--target", "block_tree_cache",
                        "--config", "Release", "--parallel"], check=True)
        # Install the DSO once; the CMake install also provides SONAME symlinks
        # for native consumers, which a wheel would store as additional copies.
        library = "libblock_tree_cache.dylib" if sys.platform == "darwin" else "libblock_tree_cache.so"
        shutil.copy2(native / "lib" / library, Path(self.build_lib) / "block_tree_cache" / library)


class BindingWheel(bdist_wheel):
    def get_tag(self):
        # ctypes loads the C ABI; only the OS, architecture and native runtime matter.
        return "py3", "none", super().get_tag()[2]


setup(distclass=NativeDistribution,
      cmdclass={"bdist_wheel": BindingWheel, "build_py": BuildBinding, "sdist": NativeSourceDistribution})
