#!/usr/bin/env bash
# Check the final shared libraries, not a statically linked C++ test fixture.
# Also usable against an extracted wheel: pass compute .so, then engine .so.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <librtp_compute_ops.so> <libth_transformer.so>" >&2
    exit 2
fi

compute_symbols=$(nm -D -C --defined-only "$1")
engine_symbols=$(nm -D -C --defined-only "$2")

for method in initDevices getRegistrationMap getCurrentDevices registerDevice; do
    symbol="rtp_llm::DeviceFactory::${method}("
    if ! grep -Fq "$symbol" <<< "$compute_symbols"; then
        echo "FAIL: compute library does not define ${symbol}" >&2
        exit 1
    fi
    if grep -Fq "$symbol" <<< "$engine_symbols"; then
        echo "FAIL: engine library duplicates ${symbol}; depend on //:rtp_compute_ops instead of static devices libraries" >&2
        exit 1
    fi
done

echo "PASS: DeviceFactory implementation is owned by the compute shared library"
