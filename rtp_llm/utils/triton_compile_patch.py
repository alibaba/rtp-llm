# -*- coding: utf-8 -*-
"""
Triton Compile Monitor - Enhanced Version

Features:
1. Monitor Triton kernel compilation time
2. Extract cache-related information (signature, constants, specialization, etc.)
3. Provide compilation statistics

Usage:
    from rtp_llm.utils.triton_compile_patch import enable_compile_monitor
    enable_compile_monitor()  # Must be called before defining @triton.jit kernels

Environment Variables:
    RTP_LLM_TRITON_DEBUG=1  Enable detailed debug logging

Log Output:
    - INFO level: Basic compilation information (always output)
      Example: Compiled _causal_conv1d_fwd_kernel: 600.15ms

    - INFO level (DEBUG mode only): Detailed information in JSON format (prevents log interleaving in multi-process scenarios)
      Includes: kernel_name, compile_time_ms, module, triton_ops, target, options,
                signature, constants, fn_cache_key, hash, attrs, attrs_hash, timestamp, etc.
"""

import functools
import json
import logging
import os
import time
from typing import Any, Dict


class TritonCompileMonitor:
    """Monitor for Triton kernel compilation."""

    def __init__(self):
        self.debug_mode = os.environ.get("RTP_LLM_TRITON_DEBUG", "0") == "1"

    def __call__(self, original_compile):
        """
        Decorator for the compile function.

        Args:
            original_compile: Original compile function to be wrapped

        Returns:
            Wrapped compile function with monitoring capabilities
        """

        @functools.wraps(original_compile)
        def wrapper(src, target=None, options=None):
            try:
                kernel_info = self._extract_kernel_info(src, target, options)
            except Exception as e:
                logging.error(f"Failed to extract kernel info: {str(e)}")
                kernel_info = {}
            kernel_name = kernel_info.get("kernel_name", "unknown")

            start_time = time.perf_counter()
            result = original_compile(src, target=target, options=options)
            compile_time = time.perf_counter() - start_time
            compile_time_ms = round(compile_time * 1000, 2)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Compiled {kernel_name}: {compile_time_ms}ms")

            if self.debug_mode:
                self._log_detailed_info(
                    kernel_name, compile_time_ms, kernel_info, timestamp
                )
            return result

        return wrapper

    def _extract_kernel_info(
        self, src: Any, target: Any = None, options: Any = None
    ) -> Dict[str, Any]:
        """
        Extract kernel information from src object, including cache-related metadata.

        Args:
            src: Source object containing kernel information
            target: Compilation target
            options: Compilation options

        Returns:
            Dictionary containing extracted kernel information
        """
        info = {
            "kernel_name": "unknown",
            "module": "unknown",
            "triton_ops": [],
            "target": str(target) if target else "None",
            "options": str(options) if options else "None",
        }

        if hasattr(src, "fn"):
            if hasattr(src.fn, "__name__"):
                info["kernel_name"] = src.fn.__name__
            if hasattr(src.fn, "__module__"):
                info["module"] = src.fn.__module__

            if hasattr(src.fn, "src"):
                source_code = src.fn.src
                ops = ["tl.load", "tl.store", "tl.dot", "tl.sum", "tl.atomic_add"]
                op_counts = []
                for op in ops:
                    count = source_code.count(op)
                    if count > 0:
                        op_counts.append(f"{op}({count})")
                info["triton_ops"] = op_counts

        if hasattr(src, "signature"):
            info["signature"] = str(src.signature)

        if hasattr(src, "constants"):
            info["constants"] = str(src.constants)

        if hasattr(src, "attrs"):
            info["attrs"] = str(src.attrs)
            if hasattr(src.attrs, "hash") and callable(src.attrs.hash):
                try:
                    info["attrs_hash"] = str(src.attrs.hash())
                except Exception as attrs_hash_error:
                    info["attrs_hash_error"] = str(attrs_hash_error)

        if hasattr(src, "specialization"):
            info["specialization"] = str(src.specialization)

        if hasattr(src, "fn") and hasattr(src.fn, "cache_key"):
            info["fn_cache_key"] = str(src.fn.cache_key)

        if hasattr(src, "hash") and callable(src.hash):
            try:
                info["hash"] = str(src.hash())
            except Exception as hash_error:
                info["hash_error"] = str(hash_error)

        if hasattr(src, "cache_key"):
            info["cache_key"] = str(src.cache_key)

        return info

    def _log_detailed_info(self, kernel_name, compile_time_ms, kernel_info, timestamp):
        """
        Log detailed kernel compilation information in JSON format.

        Args:
            kernel_name: Name of the kernel
            compile_time_ms: Compilation time in milliseconds
            kernel_info: Dictionary containing kernel metadata
            timestamp: Compilation timestamp
        """
        detail_info = {
            "kernel_name": kernel_name,
            "compile_time_ms": compile_time_ms,
            "module": kernel_info.get("module", "unknown"),
            "triton_ops": kernel_info.get("triton_ops", []),
            "target": str(kernel_info.get("target", "None")),
            "options": kernel_info.get("options", {}),
            "timestamp": timestamp,
        }

        if "signature" in kernel_info:
            detail_info["signature"] = kernel_info["signature"]
        if "constants" in kernel_info:
            detail_info["constants"] = kernel_info["constants"]
        if "specialization" in kernel_info:
            detail_info["specialization"] = kernel_info["specialization"]
        if "fn_cache_key" in kernel_info:
            detail_info["fn_cache_key"] = kernel_info["fn_cache_key"]
        if "hash" in kernel_info:
            detail_info["hash"] = kernel_info["hash"]
        if "hash_error" in kernel_info:
            detail_info["hash_error"] = kernel_info["hash_error"]
        if "cache_key" in kernel_info:
            detail_info["cache_key"] = kernel_info["cache_key"]
        if "attrs" in kernel_info:
            detail_info["attrs"] = kernel_info["attrs"]
        if "attrs_hash" in kernel_info:
            detail_info["attrs_hash"] = kernel_info["attrs_hash"]
        if "attrs_hash_error" in kernel_info:
            detail_info["attrs_hash_error"] = kernel_info["attrs_hash_error"]

        json_output = json.dumps(detail_info, indent=2, ensure_ascii=False)
        logging.info(f"\n{json_output}")


_monitor = None


def enable_compile_monitor():
    """
    Enable Triton compilation monitoring.

    Must be called before defining @triton.jit kernels!

    Environment Variables:
        RTP_LLM_TRITON_DEBUG=1  Enable detailed debug logging (shows signature, constants, and other detailed information)
    """
    global _monitor

    try:
        from triton.compiler import compiler

        if not hasattr(compiler, "_original_compile"):
            compiler._original_compile = compiler.compile
            _monitor = TritonCompileMonitor()
            compiler.compile = _monitor(compiler._original_compile)

            from triton.runtime.jit import JITFunction

            if not hasattr(JITFunction, "_original_create_binder"):
                original_create_binder = JITFunction.create_binder

                def patched_create_binder(self, *args, **kwargs):
                    original_create_binder(self, *args, **kwargs)
                    from triton.compiler import compiler

                    self.compile = compiler.compile

                JITFunction._original_create_binder = original_create_binder
                JITFunction.create_binder = patched_create_binder

                debug_status = (
                    "DEBUG MODE ON"
                    if _monitor.debug_mode
                    else "DEBUG MODE OFF (set RTP_LLM_TRITON_DEBUG=1 to enable)"
                )
                print(
                    f"[RTP-LLM TRITON COMPILE MONITOR] enable triton compile monitor ({debug_status})"
                )
        else:
            print(
                "[RTP-LLM TRITON COMPILE MONITOR] triton compile monitor already enabled"
            )

    except ImportError as e:
        print(f"[RTP-LLM TRITON COMPILE MONITOR] failed to import triton: {e}")
