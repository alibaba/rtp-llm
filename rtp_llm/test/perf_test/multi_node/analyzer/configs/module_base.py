from typing import Callable, Optional

from pydantic import BaseModel


class TestSetUp(BaseModel):
    dp_size: int = 1
    tp_size: int = 1
    batch_size: int = 1
    seq_len: int = 1


class ModuleConf(BaseModel):
    name_pattern: Optional[str] = None
    num_flop_calc_func: Optional[Callable[[TestSetUp], int]] = None
    mem_io_calc_func: Optional[Callable[[TestSetUp], int]] = None


def calc_gemm_num_flop(m: int, n: int, k: int) -> int:
    return m * n * k * 2


def calc_gemm_mem_io(
    m: int, n: int, k: int, n_input_bytes: int, n_output_bytes: int
) -> int:
    return n_input_bytes * (m * k + m * n) + n_output_bytes * (n * k)


# 16k seq len, bs=1,
# 激活： A35B 20G 激活
# kv cache: 16384 / 64 * 2640 = 660 MB
# 20G / 8000 GB/s / 0.7 = 4 ms
