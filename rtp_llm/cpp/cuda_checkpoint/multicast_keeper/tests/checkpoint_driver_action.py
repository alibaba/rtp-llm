#!/usr/bin/env python3

import argparse

from rtp_llm.utils.checkpoint_controller import (
    CUDA_SUCCESS,
    DRIVER_STATE_CHECKPOINTED,
    DRIVER_STATE_LOCKED,
    DRIVER_STATE_RUNNING,
    LibCudaCheckpointDriver,
)

EXPECTED_STATES = {
    "lock": DRIVER_STATE_LOCKED,
    "checkpoint": DRIVER_STATE_CHECKPOINTED,
    "restore": DRIVER_STATE_LOCKED,
    "unlock": DRIVER_STATE_RUNNING,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=("lock", "checkpoint", "restore", "unlock"), required=True
    )
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--lock-timeout-ms", type=int, default=60000)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if not 0 <= args.lock_timeout_ms <= 0xFFFFFFFF:
        parser.error("--lock-timeout-ms must fit uint32")

    driver = LibCudaCheckpointDriver()
    if args.action == "lock":
        result = driver.lock(args.pid, args.lock_timeout_ms)
    else:
        result = getattr(driver, args.action)(args.pid)
    if result != CUDA_SUCCESS:
        raise RuntimeError(
            f"{args.action} pid={args.pid} failed: {driver.error_string(result)}"
        )
    state = driver.get_state(args.pid)
    expected = EXPECTED_STATES[args.action]
    if state != expected:
        raise RuntimeError(
            f"{args.action} pid={args.pid} reached state={state}, expected={expected}"
        )
    print(f"CUDA_CHECKPOINT_DRIVER action={args.action} pid={args.pid} state={state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
