import os
import sys

test_srcdir = os.environ.get("TEST_SRCDIR", "")
if test_srcdir:
    # list all directories in TEST_SRCDIR
    sub_dirs = os.listdir(test_srcdir)
    for sub_dir in sub_dirs:
        print(f"sub TEST_SRCDIR: {sub_dir}")
        if sub_dir.startswith("pip_"):
            sys.path.append(f"{test_srcdir}/{sub_dir}/site-packages")

for path in sys.path:
    print(f"sys.path: {path}")

from subs.test_sub_module import TestSubModule


def set_trace_on_tty():
    """
    启动一个连接到当前终端的 PDB 会话。
    在 Unix-like 系统上工作。
    """
    try:
        import pdb

        # 尝试打开 /dev/tty。如果失败（例如在非交互式会话中），则什么也不做。
        tty_r = open("/dev/tty", "r")
        tty_w = open("/dev/tty", "w")
        pdb.Pdb(stdin=tty_r, stdout=tty_w).set_trace()
    except OSError as e:
        # 在无法打开 tty 的环境中（如CI/CD），优雅地跳过调试
        print(f"Warning: Could not open /dev/tty: {e}. Skipping pdb.")
        import traceback

        traceback.print_exc()


# TODO(wangyin): how to import python modules (like torch) from c++?
# TODO(wangyin): how to pass parameters from c++ to python?
# TODO(wangyin): how to log properly in python? (inherits binary starter python logger config)


class GptModel:
    def __init__(self) -> None:
        print("Fake GPT model initialized")
        # raise NotImplementedError("This is a fake GPT model for testing purposes, not implemented yet.")
        for key, value in os.environ.items():
            print(f"export {key}={value}")

        self.test_sub_module = TestSubModule()

    def forward(self, hidden):
        self.test_sub_module.run()

        set_trace_on_tty()

        print("Fake GPT model forward called")

        # raise NotImplementedError("This is a fake GPT model for testing purposes, not implemented yet.")
        return hidden
