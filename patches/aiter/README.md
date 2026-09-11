# ROCm GDN decode padding backport

This patch targets the AITER wheel pinned in `deps/http.bzl`. It backports
kernel-side padding output initialization using that wheel's GTensor API.
Negative read/write indices write zero output without touching the state pool.

## Build and install

```bash
bazelisk build //rtp_llm:rtp_llm --config=rocm
python -m pip install --no-deps --force-reinstall \
  bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl
```

`deps/http.bzl` applies the patch to the checksummed external AITER archive.
The ROCm-only dependency in `models_py/triton_kernels/BUILD` then packages the
patched kernel as `fla/_aiter_gdr_decode_padding.py` inside the RTP wheel.
No manual patching of installed AITER is needed. NVIDIA builds do not include
this generated source through the FLA target.

At runtime, `aiter_gdn_padding_backport.py` verifies the installed AITER
wrapper/kernel hashes and the packaged patched-source hash. A private wrapper
uses the patched factory without modifying the original AITER module or files.
Only this verified backend uses uninitialized output allocation in RTP. Missing
or mismatched sources retain the original backend with RTP output `zeros`.
The checks are cached at backend initialization; no GPU-to-CPU synchronization
is added to replay.

Verify from outside the source checkout using the Python environment where the
new RTP wheel was installed:

```python
from rtp_llm.models_py.triton_kernels.fla.aiter_gdn_padding_backport import (
    padding_safe_backend,
)

backend = padding_safe_backend()
assert backend is not None
factory = backend.__globals__["create_vk_gdr_decode_kernel"]
print(factory.__wrapped__.__code__.co_filename)
```

The path must point to RTP's packaged `_aiter_gdr_decode_padding.py`. Merely
installing the original AITER wheel, or running an unbuilt source checkout,
does not enable the backport. Review/remove this patch and update the hash
contract when changing the pinned AITER version.

Regression coverage is in `test_aiter_flydsl_gdn_decode.py`: poisoned padding
output, one-sided negative indices, untouched padding state, source mismatch
fallback, numerical equivalence and Graph replay.
