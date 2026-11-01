# Native EGM / ATen bridge

`engram_native.cc` uses CUDA Driver VMM to create HOST_NUMA + FABRIC memory,
import it on a consumer GPU and expose uint8 ATen views without a full table
copy. This development extension has a hard ceiling of 128 GiB per allocation
and 64 MiB per `Owner.load` call. The Python checkpoint/bootstrap policy keeps
the original 1 GiB per-allocation default; bootstrap also defaults to 8 GiB total.
Larger mappings require explicit local configuration described in
[capacity policy](../CAPACITY.md). A hard ceiling is not a capacity guarantee.
It is not the production Bazel extension.

## Build

Use a compatible Linux CUDA 13 development container, C++ compiler, CUDA Driver
headers and torch `2.11.0+cu130`. Historical testing used ARM64, driver
580.95.05, toolkit 13.2.86 and Triton 3.6.0; other combinations need validation.

From the repository root:

```bash
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=1
export ENGRAM_NATIVE_BUILD_DIR="$HOME/.cache/rtp_llm/engram_native"
python tools/engram_sidecar/native/load_native.py --verbose
```

Set `CUDA_DRIVER_LINK_DIR` if link-time `libcuda.so` is not under
`$CUDA_HOME/lib64/stubs`. Stubs are for linking only: never put them in runtime
`LD_LIBRARY_PATH` or RPATH. Runtime must resolve the actual container driver.
The loader preserves torch's ABI settings. Import alone does not initialize CUDA.
Rebuild the bridge on both sides when changing this capacity ceiling; consumers
reject allocations larger than their loaded bridge advertises before import.

## API and ownership

```python
from load_native import load_native
native = load_native()
owner = native.Owner(size=4194304, numa=0, device=0)
owner.load(data=b"example bytes", offset=0)
info = owner.info()
handle = owner.export_handle()  # Private capability; never log this value.
consumer = native.Consumer(handle, info["allocation_bytes"], info["granularity"], 0)
view = consumer.tensor_view(shape=[13], offset=0)
```

`numa` identifies a NUMA node on the owner's local host; `device` is a local
visible GPU ordinal. Resolve both from verified platform topology, not a remote
host's numbering. Export seals the owner against further writes. Imports verify
allocation properties and grant READ access only to the selected consumer GPU.

`tensor_view` uses `from_blob` with an explicit target device. The tensor storage
retains a mapping lease; aliases keep that lease alive. `close()` refuses live
views, drains the selected primary context and then unmaps/releases in order.
The caller must first prevent new GPU submissions. `info()` and
`resource_counters()` describe this module's resources, not all driver memory.

CUDA Graph does not automatically keep Python tensor leases alive. Use the
storage/Graph guard in `engram_storage.py`, drain and reset every graph before
releasing views. A failed drain never permits speculative unmapping. The owner
has no distributed consumer reference counter: keep it alive until every
consumer has completed cleanup. Owner/host loss recovery is not implemented.

## Tests

CPU contract test, with torch/numpy installed:

```bash
python tools/engram_sidecar/native/test_embedding_contract.py --repo "$PWD"
python tools/engram_sidecar/test_build_paths.py
```

After reserving one suitable idle GPU, explicitly restrict visibility to its
UUID. Then run the native tests with a bounded external timeout:

```bash
CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_RESERVED_UUID timeout --kill-after=10s 300s \
  python tools/engram_sidecar/native/test_native.py --device 0
```

Native tests cover property/bounds checks, sealed writes, constructor rollback,
alias leases and repeated attach/view/drain/close. Fault-injection exceptions
exercise resource ownership; they are not GPU/driver failure recovery tests.

## Sharded mapping check

`check_shard_mapping.py` verifies that N fabric allocations can be stitched
into the two VA spans the lookup kernel expects. `shard_owner.py` publishes one
shard from another node and keeps it alive until `STOP`; pass its private
handle file to the consumer with `--remote-handles`. Each mapped range needs
`cuMemSetAccess` before the consumer GPU can read it.


References: [CUDA EGM](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#multi-node-multi-gpu),
[CUDA VMM](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html),
[ATen from_blob](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/ops/from_blob.h).
