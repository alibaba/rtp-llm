# RTP FlyDSL GDN MTP adapter

RTP consumes the existing `atom.utils.envs.ATOM_ENABLE_REPLAYSSM` setting.
On supported ROCm RTP targets, unset/`1` selects FlyDSL with hybrid replay;
`0` selects plain FlyDSL. No RTP-specific ReplaySSM switch is needed.
Retired RTP-specific switches are ignored. Unsupported hybrid-consensus
topologies retain FlyDSL. Standalone and SGLang keep their native semantics.


The chain kernel is extended with independent initial read slots; each verify
candidate writes directly to the RTP state pool slot. Physical VK storage is
selected before any state writes. Gating and Q/K normalization are fused.
Read indices are packed once per forward, not once per GDN layer. Mixed batches
retain the original path. This adapter is for fixed-length candidate chains,
not speculative trees.

For the RTP production profile, install ATOM `0.1.8+g9af739efc`,
amd-aiter `0.1.29+rocm7.2.0.git97d2389` and FlyDSL `0.4.2` first.
Run the same post-install command used by the internal ROCm image on the
freshly installed packages, supplying the Python interpreter for those packages
and the pinned FlyDSL wheel:

```sh
bash patches/apply_qwen38_rocm.sh /opt/conda310/bin/python /path/to/flydsl-0.4.2-cp310-cp310-manylinux_2_27_x86_64.whl
```

The base ATOM patch includes layer-wise P/D cache publication. The installer
recognizes the production wheel with this base patch already applied.
The ATOM `0.1.1.dev1+g0189b3506` profile uses its version-specific adapter
in `patches/atom/main_0189b3506/`.

GPU checks cover indexed MTP reads and graph replay with partial acceptance:
```sh
python patches/atom/tests/test_flydsl_gdn_mtp.py
python patches/atom/tests/test_hybrid_gdn_graph.py
```
