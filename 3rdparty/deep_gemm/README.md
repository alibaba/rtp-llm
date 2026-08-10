# DeepGEMM dependency maintenance

RTP-LLM resolves the runtime DeepGEMM wheel through the platform dependency
set, not from an example directory.  For CUDA13/B300, the authoritative URL is
declared in `internal_source/deps/requirements_torch_gpu_cuda13.txt` and its
SHA256 is pinned in `requirements_lock_torch_gpu_cuda13.txt`.

`build_cuda13_b300_wheel.sh` is the reproducible source recipe for the Kimi K3
wheel.  It checks out DeepGEMM commit
`f5a76426fa084087169693fd0cd815223576d6e9` and applies
`0003-k3-cuda13-float-nttp.patch`.  Run it only on L20-dev-115, inside
`lhc_GPU` as `luohaocheng.lhc`:

```bash
DEEP_GEMM_WHEEL_OUTPUT=/data1/luohaocheng.lhc/artifacts/deep_gemm \
  ./3rdparty/deep_gemm/build_cuda13_b300_wheel.sh
```

The output directory must be outside the checkout.  Publish the resulting
wheel to the artifact store, then update the dependency URL and lock hash; do
not commit wheel binaries into this repository.
