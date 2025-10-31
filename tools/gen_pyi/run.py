import os
import re
import sys

# 先导入依赖模块
try:
    import libth_transformer_config  # type: ignore # noqa

    print(f"✓ Imported libth_transformer_config")
except Exception as e:
    print(f"✗ Failed to import libth_transformer_config: {e}")
try:
    import libth_transformer  # type: ignore # noqa

    print(f"✓ Imported libth_transformer")
except Exception as e:
    print(f"✗ Failed to import libth_transformer: {e}")
try:
    import torch  # noqa

    print(f"✓ Imported torch")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")

from pybind11_stubgen.__init__ import main


def cleanup_and_exit(exit_code: int) -> None:
    """清理并退出，使用 os._exit 避免析构函数导致的内存问题"""
    # 强制清理一些可能的内存
    import gc

    gc.collect()
    # 使用 os._exit 直接退出，跳过所有析构函数
    # 这样可以避免 PyTorch 析构函数中的内存损坏问题
    os._exit(exit_code)


if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    try:
        main()
        # 如果成功，直接退出，不运行析构函数
        cleanup_and_exit(0)
    except SystemExit as e:
        # 处理 sys.exit() 调用
        cleanup_and_exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        cleanup_and_exit(1)
