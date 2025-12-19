import os
import sys
import traceback
import logging
import gc
from pybind11_stubgen.__init__ import main

from rtp_llm.config.log_config import setup_logging
setup_logging()
import rtp_llm.ops

def cleanup_and_exit(exit_code: int) -> None:
    """清理并退出，使用 os._exit 避免析构函数导致的内存问题"""
    # 强制清理一些可能的内存
    gc.collect()
    # 使用 os._exit 直接退出，跳过所有析构函数
    # 这样可以避免 PyTorch 析构函数中的内存损坏问题
    os._exit(exit_code)


if __name__ == "__main__":
    import libth_transformer_config
    import libth_transformer
    import librtp_compute_ops

    sos = ['libth_transformer_config', 'libth_transformer', 'librtp_compute_ops']
    # Get the absolute real path of the current file
    current_file_path = os.path.realpath(__file__)
    # Get the directory containing the current file
    current_dir = os.path.dirname(current_file_path)
    # Navigate to rtp_llm/ops relative to the current file location
    # From tools/gen_pyi/ to rtp_llm/ops/ is ../../rtp_llm/ops
    output_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'rtp_llm', 'ops'))
    logging.info(f"output directory: {output_dir}")
    try:
        for so in sos:
            main([so, '-o', output_dir])
        # 如果成功，直接退出，不运行析构函数
        cleanup_and_exit(0)
    except SystemExit as e:
        # 处理 sys.exit() 调用
        cleanup_and_exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

        traceback.print_exc()
        cleanup_and_exit(1)
