import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser

desc = """
Triton ahead-of-time compiler in Bazel
"""


def get_triton_tools_path(imports):
    py_path = []
    for path in imports:
        new_path = os.path.join("external", path)
        sys.path.append(new_path)
        py_path.append(new_path)
    import triton.tools

    # return path like ~/.local/lib/python3.10/site-packages/triton/__init__.py
    triton_tools_path = os.path.dirname(triton.tools.__file__)
    return triton_tools_path, ":".join(py_path)


def compile_kernel(
    interpreter,
    script_name,
    kernel_name,
    output_dir,
    output_name,
    spec,
    num_warps,
    grid,
    imports,
):
    triton_tools_path, py_path = get_triton_tools_path(imports)
    with open(f"{triton_tools_path}/compile.py") as f:
        code = f.read()
    pattern = r"triton\.compile\(([^,]+),\s*options=([^)]+)\)"
    replacement = r'triton.compile(\1, target=GPUTarget("cuda", 80, 32), options=\2)'
    modified_code = re.sub(pattern, replacement, code)
    os.makedirs(f"{output_dir}/tools", exist_ok=True)
    for name in ["compile.c", "compile.h"]:
        shutil.copy(f"{triton_tools_path}/{name}", f"{output_dir}/tools/{name}")
    with open(f"{output_dir}/tools/compile.py", "w") as f:
        f.write("from triton.backends.compiler import GPUTarget\n")
        f.write(modified_code)
    compile_command = [
        interpreter,
        f"{output_dir}/tools/compile.py",
        script_name,
        "-n",
        kernel_name,
        "-o",
        f"{output_dir}/{output_name}",
        "-on",
        output_name,
        "-w",
        f"{num_warps}",
        # "-ns", "1",
        "-s",
        spec,
        "-g",
        grid,
    ]
    env = os.environ.copy()
    PPU_SDK = env["PPU_SDK"]
    CUDA_HOME = env["CUDA_HOME"]
    env["PYTHONPATH"] = py_path
    PATH = env["PATH"]
    PATH = f"{PPU_SDK}/bin:{CUDA_HOME}/bin:{PATH}"
    env["PATH"] = PATH
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["TRITON_TARGET_PPU"] = "1.3.0"
    subprocess.run(compile_command, check=True, env=env)


def parse_int_list(arg):
    return [int(x) for x in arg.split(",")]


def rename_generated_files(output_dir, output_name, bazel_id):
    hdr = None
    src = None
    for entry in os.listdir(output_dir):
        if entry == "tools":
            continue
        if entry.endswith(".h"):
            if hdr is not None:
                raise Exception("duplicate header file", entry)
            hdr = entry
            continue
        if entry.endswith(".c"):
            if src is not None:
                raise Exception("duplicate source file", entry)
            src = entry
            continue
        raise Exception("bad file", entry)

    def rename_file(old_file, new_file):
        old_file_path = os.path.join(output_dir, old_file)
        new_file_path = os.path.join(output_dir, new_file)
        ret = os.rename(old_file_path, new_file_path)

    rename_file(hdr, f"{output_name}__{bazel_id}.h")
    rename_file(src, f"{output_name}__{bazel_id}.c")


def main():
    parser = ArgumentParser(description=desc)
    parser.add_argument("--interpreter", type=str, required=True)
    parser.add_argument("--script_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--kernel_name", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--num_warps", type=int, required=True)
    parser.add_argument("--bazel_id", type=str, required=True)
    parser.add_argument("--imports", nargs="+", required=True)
    args = parser.parse_args()
    script_name = args.script_name
    kernel_name = args.kernel_name
    interpreter = args.interpreter
    spec = args.spec
    num_warps = args.num_warps
    bazel_id = args.bazel_id
    output_dir = f"{args.output_dir}/aot/{bazel_id}"
    output_name = args.output_name
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    compile_kernel(
        interpreter,
        script_name,
        kernel_name,
        output_dir,
        output_name,
        spec,
        num_warps,
        args.grid,
        args.imports,
    )
    rename_generated_files(output_dir, output_name, bazel_id)


if __name__ == "__main__":
    main()
