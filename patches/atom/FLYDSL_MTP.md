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

Apply `patches/atom/qwen38_replayssm.patch` to ATOM
0.1.8+g9af739efc and `patches/aiter/qwen38_flydsl_gdn_mtp.patch` to
amd-aiter 0.1.29+rocm7.2.0.git97d2389. FlyDSL is 0.4.2.
Also apply `patches/aiter/qwen38_python310_compat.patch`: it moves
heuristics dictionaries outside decorators so Python 3.10 source inspection
can find the Triton function. Numerical function bodies are unchanged.
The internal installer pins source, patch and output hashes and applies these patches
after wheel installation; it rejects unreviewed changes. No C++ RTP rebuild
is required for this Python/kernel-generator adapter.

GPU checks cover indexed MTP reads and graph replay with partial acceptance:
```sh
python patches/atom/tests/test_flydsl_gdn_mtp.py
python patches/atom/tests/test_hybrid_gdn_graph.py
```
