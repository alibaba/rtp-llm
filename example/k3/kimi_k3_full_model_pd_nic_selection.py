#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from typing import Optional


DEFAULT_BOND_HCAS = tuple(f"mlx5_bond_{index}" for index in range(8))


class HcaValidationError(ValueError):
    pass


def parse_allowlist(value: str) -> tuple[str, ...]:
    nics = tuple(value.split(","))
    if not nics or any(not nic for nic in nics):
        raise HcaValidationError(
            "SMOKE_ACCL_USE_NICS must be a non-empty comma-separated list"
        )
    if len(set(nics)) != len(nics):
        raise HcaValidationError("SMOKE_ACCL_USE_NICS contains duplicate HCAs")
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", nic) is None for nic in nics):
        raise HcaValidationError("SMOKE_ACCL_USE_NICS contains an invalid HCA name")
    return nics


def validate_allowlist(
    value: str, devices_output: str, links_output: str
) -> tuple[str, ...]:
    nics = parse_allowlist(value)
    devices = {
        fields[0]
        for line in devices_output.splitlines()
        if (fields := line.split()) and fields[0].startswith("mlx5_")
    }
    missing = [nic for nic in nics if nic not in devices]
    if missing:
        raise HcaValidationError(
            f"SMOKE_ACCL_USE_NICS HCAs are absent from ibv_devices: {missing}"
        )

    for nic in nics:
        pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(nic)}(?:/|:|\s)")
        lines = [line for line in links_output.splitlines() if pattern.search(line)]
        if not lines:
            raise HcaValidationError(
                f"SMOKE_ACCL_USE_NICS HCA has no rdma link: {nic}"
            )
        if not any(
            re.search(r"\bstate\s+ACTIVE\b", line)
            and re.search(r"\bphysical_state\s+LINK_UP\b", line)
            for line in lines
        ):
            raise HcaValidationError(
                f"SMOKE_ACCL_USE_NICS HCA is not ACTIVE/LINK_UP: {nic}"
            )
    return nics


def select_default_allowlist(
    devices_output: str, links_output: str
) -> tuple[Optional[str], Optional[str]]:
    value = ",".join(DEFAULT_BOND_HCAS)
    try:
        validate_allowlist(value, devices_output, links_output)
    except HcaValidationError as error:
        return None, str(error)
    return value, None


def read_rdma_inventory() -> tuple[str, str]:
    try:
        devices_output = subprocess.check_output(
            ["ibv_devices"], text=True, stderr=subprocess.STDOUT
        )
        links_output = subprocess.check_output(
            ["rdma", "link", "show"], text=True, stderr=subprocess.STDOUT
        )
    except FileNotFoundError as error:
        raise HcaValidationError(
            f"{error.filename} is unavailable for RDMA HCA discovery"
        ) from error
    except subprocess.CalledProcessError as error:
        output = (error.stdout or "").strip()
        detail = f": {output}" if output else ""
        raise HcaValidationError(
            f"{' '.join(error.cmd)} failed with exit code {error.returncode}{detail}"
        ) from error
    return devices_output, links_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit")
    args = parser.parse_args()

    try:
        devices_output, links_output = read_rdma_inventory()
        if args.explicit is not None:
            nics = validate_allowlist(args.explicit, devices_output, links_output)
            print(",".join(nics))
            return 0

        selected, reason = select_default_allowlist(devices_output, links_output)
        if selected is None:
            print(
                "warning: default bond HCA allowlist is unavailable; "
                f"leaving ACCL_USE_NICS unset for Barex auto-discovery: {reason}",
                file=sys.stderr,
            )
            return 0
        print(selected)
        return 0
    except HcaValidationError as error:
        if args.explicit is not None:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(
            "warning: RDMA HCA inventory could not be read; "
            f"leaving ACCL_USE_NICS unset for Barex auto-discovery: {error}",
            file=sys.stderr,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
