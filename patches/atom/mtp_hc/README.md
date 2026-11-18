# Default-on ROCm Qwen4 MTP hyper-connection fusion

This patch enables ATOM's existing fused HC path inside the MTP drafter,
materializes the final pending residual before returning its hidden states,
and uses the fused HC head-input mixer. It does not change the target model,
GDN state handling, KV layout, acceptance algorithm or sampling configuration.
The switch defaults on only on the ROCm RTP-ATOM path. Non-ROCm and
non-RTP paths retain their original behavior. Set ATOM_RTP_MTP_HC_FUSION=0
to explicitly disable it.

## Install after the ATOM wheel and existing dependency patches

Requires ATOM `0.1.8+g9af739efc` (wheel SHA256 in `manifest.json`), the
existing RTP/ATOM adapters, amd-aiter `0.1.29+rocm7.2.0.git97d2389` and FlyDSL
0.4.2. Do not apply to the older 0.1.1 wheel. The ROCm image installs and verifies this post-wheel patch by default.
Use the Python interpreter that actually runs the service:

```sh
python3 patches/atom/mtp_hc/apply.py
python3 patches/atom/mtp_hc/apply.py --verify-only
```

For the internal copy use `internal_source/dockerfile/mtp_hc/apply.py`.
The installer checks version and source/patch/output hashes, validates in a
temporary directory, refuses unknown local edits, and is idempotent. Restart
workers after applying. There is no C++ change required by this patch.
