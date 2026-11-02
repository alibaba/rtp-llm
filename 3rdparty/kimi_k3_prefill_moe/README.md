# K3 routed-expert Prefill kernel

This optional package supplies the pinned routed-only SiTU kernel. It does not
replace the installed `deep_gemm`, K3 modeling, or Decode's BF16 shared-expert
kernel. K3 selects it during Prefill initialization only after verifying its
source, patch, Torch ABI, isolation flags, and binary receipt. Without the
package, the standard backend remains available.

Build on each destination machine with its actual RTP Torch environment, CUDA
13, GCC 13 or later, and the elfutils development headers. Use a task-owned
output directory and the approved personal build container:

```bash
python3 3rdparty/kimi_k3_prefill_moe/build.py \
  --archives "$TASK_ROOT/original-distributions" \
  --work-dir "$TASK_ROOT/prefill-moe-source" \
  --output-dir "$TASK_ROOT/prefill-moe-package" \
  --download --cxx "$CXX" --extra-include "$ELFUTILS_INCLUDE"
```

Add the output directory to the launcher's `PYTHONPATH`. Preserve the emitted
`BUILD_INFO.json` and record the actual import and shared-library paths. Do not
copy build outputs between machines or overwrite another model's runtime.

`UPSTREAM.json` pins DeepGEMM, CUTLASS, DeepJIT, and the two historical patches.
The vLLM-derived build helper adds `-fvisibility=hidden` and
`-fno-gnu-unique`: exposing either library's private JIT globals can corrupt the
other version's include registry. These flags change symbol binding and leave
the CUDA algorithm source intact.

The adapter maps the pinned kernel's `activation_alpha` to gate beta, and its
`activation_beta` to optional up beta. It never infers that mapping from names
on an arbitrary installed version. Its symmetric buffer retains the original
1920-token capacity alignment; memory estimates use that ABI as well.

DeepGEMM retains its MIT license. Keep each dependency’s upstream LICENSE
and header notices; the builder copies them into the private package. This
directory contains the build recipe and patches, not native binaries or weights.

The pinned DeepJIT revision does not include a standalone LICENSE file. The build records that fact and its exact commit in `licenses/DeepJIT/UPSTREAM.json`; it does not assign DeepGEMM’s MIT license to DeepJIT. Upstream header notices remain intact.
