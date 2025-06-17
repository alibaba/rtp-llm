
import sys
import os

import torch

def set_trace_on_tty():
    """
    启动一个连接到当前终端的 PDB 会话。
    在 Unix-like 系统上工作。
    """
    try:
        import pdb
        # 尝试打开 /dev/tty。如果失败（例如在非交互式会话中），则什么也不做。
        tty_r = open('/dev/tty', 'r')
        tty_w = open('/dev/tty', 'w')
        pdb.Pdb(stdin=tty_r, stdout=tty_w).set_trace()
    except OSError as e:
        # 在无法打开 tty 的环境中（如CI/CD），优雅地跳过调试
        print(f"Warning: Could not open /dev/tty: {e}. Skipping pdb.")
        import traceback
        traceback.print_exc()
        pass

class GptModelBase:
    def __init__(self) -> None:
        print("model base initialized")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

