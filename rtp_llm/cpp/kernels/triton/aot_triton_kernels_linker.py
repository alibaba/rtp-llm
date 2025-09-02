import os
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


def main():
    parser = ArgumentParser(description=desc)
    parser.add_argument("--interpreter", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--hdrs", nargs="+", type=str, required=True)
    parser.add_argument("--imports", nargs="+", required=True)
    args = parser.parse_args()
    interpreter = args.interpreter
    output_dir = f"{args.output_dir}/aot/"
    os.makedirs(output_dir, exist_ok=True)
    output_name = args.output_name
    hdrs = args.hdrs

    triton_tools_path, py_path = get_triton_tools_path(args.imports)
    link_command = (
        [interpreter, f"{triton_tools_path}/link.py"]
        + hdrs
        + ["-o", f"{output_dir}/{output_name}"]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = py_path
    subprocess.run(link_command, check=True, env=env)


if __name__ == "__main__":
    main()
