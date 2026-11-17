"""Ascend torch_npu based profiling for rtp-llm.

This module ports the vllm-ascend torch_npu profiler recipe into rtp-llm.
It is instantiated and driven from C++ (``TorchProfile``) via pybind11:
the C++ side acquires the GIL on the engine loop thread and calls
``AscendTorchNpuProfiler(output_dir, trace_name).start()`` / ``.stop()``.

The profiler must be active across the Python ``forward_micro_batch`` call
(which issues the npu ops), and since the engine loop thread is the same
thread that runs the C++ ``tick()`` and the python forward, acquiring the
GIL in start()/stop() is deadlock free.

On ``stop()`` torch_npu invokes the ``on_trace_ready`` callback
(``tensorboard_trace_handler``) which writes the ``*_ascend_pt`` dump.
Run ``torch_npu.profiler.profiler.analyse(<dump_dir>)`` afterwards to get
``trace_view.json`` / ``kernel_details.csv`` etc.
"""

import os
from typing import Optional


class AscendTorchNpuProfiler:
    """Thin wrapper around ``torch_npu.profiler.profile``.

    Mirrors ``vllm_ascend/profiler/torch_npu_profiler.py`` but is decoupled
    from the vLLM ``WorkerProfiler`` abstraction: rtp-llm's C++
    ``StepWindowProfiler`` owns the window control (start_step / num_steps),
    this class only owns the data collection.
    """

    def __init__(self,
                 output_dir: str,
                 trace_name: str,
                 with_stack: bool = False,
                 with_memory: bool = False) -> None:
        # Lazy import: keeps this module importable on non-ascend hosts so
        # that accidental imports never crash the process.
        import torch_npu  # noqa: F401  (registers NPU dispatch)

        if not output_dir:
            raise RuntimeError(
                "Ascend profiling requires a non-empty output_dir "
                "(set via torch_cuda_profiler_dir).")
        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.trace_name = trace_name or "ascend_profile"

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        # NOTE: with_modules in torch_npu is equivalent to torch with_stack;
        # it introduces significant overhead, keep it off by default.
        self._profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=with_stack,
            profile_memory=with_memory,
            with_modules=with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir,
                worker_name=self.trace_name,
            ),
        )

    def start(self) -> None:
        self._profiler.start()

    def stop(self) -> None:
        # on_trace_ready fires here -> writes <output_dir>/<host>_<trace_name>_ascend_pt/
        self._profiler.stop()
