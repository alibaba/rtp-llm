"""Optional module matching and model-estimation contracts for trace analysis."""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TestSetUp:
    dp_size: int = 1
    tp_size: int = 1
    batch_size: int = 1
    seq_len: int = 1
    mode: Optional[str] = None
    steps: Optional[int] = None
    estimation_error: Optional[str] = None


@dataclass
class ModuleConf:
    name_pattern: Optional[str] = None
    num_flop_calc_func: Optional[Callable[[TestSetUp], int]] = None
    mem_io_calc_func: Optional[Callable[[TestSetUp], int]] = None
    mode: Optional[str] = None
    tp_size: Optional[int] = None
    formula_per_kernel: bool = False


def normalize_module_config(config):
    if not isinstance(config, dict) or not config:
        raise ValueError("module_config must be a nonempty mapping")
    result = {}
    for name, value in config.items():
        spec = ModuleConf(name_pattern=value) if isinstance(value, str) else value
        if not isinstance(name, str) or not name or not isinstance(spec, ModuleConf):
            raise ValueError(
                "module_config requires named string patterns or ModuleConf values"
            )
        if not isinstance(spec.name_pattern, str) or not spec.name_pattern:
            raise ValueError(f"module {name} requires a nonempty name_pattern")
        if any(
            fn is not None and not callable(fn)
            for fn in (spec.num_flop_calc_func, spec.mem_io_calc_func)
        ):
            raise ValueError(f"module {name} formula must be callable")
        if spec.mode not in (None, "prefill", "decode") or (
            spec.tp_size is not None
            and (type(spec.tp_size) is not int or spec.tp_size <= 0)
        ):
            raise ValueError(f"module {name} has invalid mode/TP constraints")
        if type(spec.formula_per_kernel) is not bool:
            raise ValueError(f"module {name} formula_per_kernel must be boolean")
        result[name] = spec
    return result


def calc_gemm_num_flop(m: int, n: int, k: int) -> int:
    return m * n * k * 2


def calc_gemm_mem_io(
    m: int, n: int, k: int, n_input_bytes: int, n_output_bytes: int
) -> int:
    """Bytes for weights W[m,n], inputs X[n,k], and output Y[m,k]."""
    return n_input_bytes * (m * n + n * k) + n_output_bytes * (m * k)
