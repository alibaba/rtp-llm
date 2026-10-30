#!/usr/bin/env python3
"""Bounded, model-free reproduction of MM/default-stream CUDA coupling.

Run this with the service's CUDA/PyTorch environment on a spare GPU.
Exact pre-fix Tensor.to(pinned_cpu_options, non_blocking=false) cycle:
  python3 multimodal_cuda_stream_repro.py --mode legacy --copy-api to \\
      --non-blocking false --submission-handshake --expect-blocked
Pinned-copy counterexample (the two submissions make progress normally):
  python3 multimodal_cuda_stream_repro.py --mode legacy --copy-api copy \\
      --non-blocking false --submission-handshake
Isolated stream, explicit pinned output, event-completion fix:
  python3 multimodal_cuda_stream_repro.py --mode isolated --copy-api copy \\
      --non-blocking true --cold-pinned --cold-stream --submission-handshake

This causal probe uses real GPU writes and D2H copies. The C++ regression beside
it exercises the actual public MultimodalProcessor entry point. This script is
not a replacement for that regression.
"""

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time


def worker(args):
    import torch

    torch.cuda.set_device(0)
    torch.cuda.init()
    driver = ctypes.CDLL("libcuda.so.1")

    def cuda(name, argtypes, *values):
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        status = function(*values)
        if status:
            raise RuntimeError(f"{name} returned CUDA status {status}")

    void_p = ctypes.c_void_p
    size_t = ctypes.c_size_t
    u32 = ctypes.c_uint32
    u64 = ctypes.c_uint64
    host = void_p()
    device = u64()
    # DEVICEMAP: the independent watchdog releases the GPU wait using a host
    # store, so cleanup does not need a driver API that may be lock-blocked.
    cuda(
        "cuMemHostAlloc",
        [ctypes.POINTER(void_p), size_t, ctypes.c_uint],
        ctypes.byref(host),
        4,
        2,
    )
    cuda(
        "cuMemHostGetDevicePointer_v2",
        [ctypes.POINTER(u64), void_p, ctypes.c_uint],
        ctypes.byref(device),
        host,
        0,
    )
    flag = u32.from_address(host.value)
    flag.value = 0
    fresh_stream = void_p()

    elements = 16 * 1024 * 1024 if args.cold_pinned else 1024
    source = torch.empty(elements, device="cuda", dtype=torch.float32)
    copy_stream = torch.cuda.Stream()
    non_blocking = (
        args.mode == "isolated"
        if args.non_blocking == "auto"
        else args.non_blocking == "true"
    )

    def to_cpu(tensor):
        # This is the real ATen operator called by C++ Tensor.to(TensorOptions,
        # non_blocking). Python Tensor.to() does not expose the pin_memory
        # TensorOptions argument, so invoking _to_copy preserves that argument.
        return torch.ops.aten._to_copy.default(
            tensor,
            device=torch.device("cpu"),
            pin_memory=True,
            non_blocking=non_blocking,
        )

    # Warm GPU kernels before closing the unrelated model stream's gate.
    with torch.cuda.stream(copy_stream):
        source.fill_(7)
    copy_stream.synchronize()
    destination = None
    if args.copy_api == "copy" and not args.cold_pinned:
        destination = torch.empty(elements, pin_memory=True)
        with torch.cuda.stream(copy_stream):
            destination.copy_(source, non_blocking=False)
    if args.copy_api == "to":
        with torch.cuda.stream(copy_stream):
            warm_output = to_cpu(source[:1024] if args.cold_pinned else source)
        copy_stream.synchronize()
        del warm_output
    torch.cuda.synchronize()

    released = threading.Event()
    complete = threading.Event()
    d2h_started = threading.Event()
    submission_ready = threading.Event()
    submission_returned = threading.Event()
    submission_result = {}
    submission_thread = None

    if args.submission_handshake:
        # Keep allocation, first-use CUDA thread setup, and copy kernel setup
        # outside the dependency cycle. Only a pinned H2D API remains inside it.
        h2d_source = torch.full((1024,), 31.0, pin_memory=True)
        h2d_destination = torch.empty(1024, device="cuda", dtype=torch.float32)

        def submit_then_release():
            try:
                torch.cuda.set_device(0)
                with torch.cuda.stream(torch.cuda.default_stream()):
                    h2d_destination.copy_(h2d_source, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    submission_ready.set()
                    d2h_started.wait()
                    # Give the announced D2H call time to enter the driver. This
                    # is scheduling coordination, not a simulated driver lock.
                    time.sleep(0.2)
                    submission_result["h2d_submission_started"] = True
                    h2d_destination.copy_(h2d_source, non_blocking=True)
                    submission_result["h2d_returned_before_watchdog"] = (
                        not released.is_set()
                    )
                    # This CPU store is the only normal path releasing the
                    # model's pending GPU dependency. If D2H holds the driver
                    # lock, the H2D submission cannot return and release it.
                    flag.value = 1
                    submission_returned.set()
            except BaseException as error:
                submission_result["h2d_error"] = repr(error)
                submission_ready.set()
                submission_returned.set()

        submission_thread = threading.Thread(target=submit_then_release, daemon=True)
        submission_thread.start()
        if not submission_ready.wait(15):
            raise RuntimeError("H2D submission thread did not finish pre-gate warmup")
        if "h2d_error" in submission_result:
            raise RuntimeError(submission_result["h2d_error"])

    def release_gate():
        if not complete.wait(8):
            released.set()
            flag.value = 1

    watchdog = threading.Thread(target=release_gate, daemon=True)
    original = torch.cuda.current_stream().cuda_stream
    wait_name = (
        "cuStreamWaitValue32_v2"
        if hasattr(driver, "cuStreamWaitValue32_v2")
        else "cuStreamWaitValue32"
    )
    cuda(
        wait_name, [void_p, u64, u32, ctypes.c_uint], None, device, 1, 1
    )  # WAIT_VALUE_EQ
    watchdog.start()
    start = time.monotonic()
    try:
        if args.cold_stream:
            print(
                json.dumps({"phase": "nonblocking_stream_creation_start"}), flush=True
            )
            cuda(
                "cuStreamCreate",
                [ctypes.POINTER(void_p), ctypes.c_uint],
                ctypes.byref(fresh_stream),
                1,
            )
            copy_stream = torch.cuda.ExternalStream(fresh_stream.value, device=0)
            print(
                json.dumps({"phase": "nonblocking_stream_creation_complete"}),
                flush=True,
            )
        selected = (
            copy_stream if args.mode == "isolated" else torch.cuda.default_stream()
        )
        print(
            json.dumps(
                {
                    "phase": "gate_closed",
                    "consumer_stream": selected.cuda_stream,
                    "mode": args.mode,
                }
            ),
            flush=True,
        )
        with torch.cuda.stream(selected):
            if destination is None and args.copy_api == "copy":
                # Fresh process, large unique host allocation: also exposes host
                # pin allocation's implicit device synchronization if present.
                print(json.dumps({"phase": "cold_pinned_allocation_start"}), flush=True)
                destination = torch.empty(elements, pin_memory=True)
                print(
                    json.dumps({"phase": "cold_pinned_allocation_complete"}), flush=True
                )
            source.fill_(17)
            print(json.dumps({"phase": "gpu_fill_submitted"}), flush=True)
            d2h_started.set()
            if args.copy_api == "to":
                destination = to_cpu(source)
            else:
                destination.copy_(source, non_blocking=non_blocking)
            print(
                json.dumps(
                    {
                        "phase": "d2h_api_returned",
                        "output_is_pinned": destination.is_pinned(),
                    }
                ),
                flush=True,
            )
            if non_blocking:
                ready_event = torch.cuda.Event()
                ready_event.record(selected)
                ready_event.synchronize()
        closed_at_completion = flag.value == 0 and not released.is_set()
        d2h_returned_before_watchdog = not released.is_set()
        if submission_thread is not None:
            submission_returned.wait(2)
        # CPU-only direct reads, with no hidden device synchronization.
        ready = destination[0].item() == 17 and destination[-1].item() == 17
        restored = torch.cuda.current_stream().cuda_stream == original
        result = {
            "mode": args.mode,
            "cold_pinned": args.cold_pinned,
            "cold_stream": args.cold_stream,
            "copy_api": args.copy_api,
            "non_blocking": non_blocking,
            "output_is_pinned": destination.is_pinned(),
            "seconds": round(time.monotonic() - start, 3),
            "completed_with_default_gate_closed": closed_at_completion,
            "host_data_ready": ready,
            "caller_stream_restored": restored,
            "watchdog_released_gate": released.is_set(),
            "submission_handshake": args.submission_handshake,
            "d2h_returned_before_watchdog": d2h_returned_before_watchdog,
            **submission_result,
        }
        print(json.dumps(result), flush=True)
        expected = (
            not closed_at_completion if args.expect_blocked else closed_at_completion
        )
        if args.submission_handshake:
            # The H2D thread may release the gate before an isolated large D2H
            # finishes; the property here is natural progress without watchdog.
            expected = d2h_returned_before_watchdog == (not args.expect_blocked)
        submission_ok = not args.submission_handshake or (
            submission_result.get("h2d_returned_before_watchdog")
            == (not args.expect_blocked)
            and "h2d_error" not in submission_result
        )
        return 0 if expected and ready and restored and submission_ok else 1
    finally:
        complete.set()
        d2h_started.set()
        flag.value = 1
        watchdog.join(timeout=2)
        if submission_thread is not None:
            submission_thread.join(timeout=2)
        torch.cuda.synchronize()
        # A raw ExternalStream is non-owning. Keep its CUDA handle alive for
        # this child context's lifetime: PyTorch caching allocators can retain
        # that allocation/copy stream and record events when Python tensor
        # locals are destroyed after this finally block. Destroying it here
        # creates a use-after-destroy during ordinary interpreter shutdown.
        # The CUDA context reclaims this one bounded stream at process exit,
        # mirroring the production helper's per-device stream lifetime.
        cuda("cuMemFreeHost", [void_p], host)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "isolated"), required=True)
    parser.add_argument("--expect-blocked", action="store_true")
    parser.add_argument("--cold-pinned", action="store_true")
    parser.add_argument("--cold-stream", action="store_true")
    parser.add_argument("--submission-handshake", action="store_true")
    parser.add_argument("--copy-api", choices=("copy", "to"), default="copy")
    parser.add_argument(
        "--non-blocking", choices=("auto", "true", "false"), default="auto"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_stream and args.mode != "isolated":
        parser.error("--cold-stream is only meaningful with --mode isolated")
    if args.worker:
        return worker(args)
    command = [sys.executable, __file__, *sys.argv[1:], "--worker"]
    try:
        return subprocess.run(command, timeout=40, check=False).returncode
    except subprocess.TimeoutExpired:
        print("CUDA probe exceeded 40 seconds; child was killed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
