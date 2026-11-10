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

GPU regressions:
`python patches/atom/tests/test_flydsl_gdn_mtp.py` and
`python patches/atom/tests/test_hybrid_gdn_graph.py`
It compares outputs and all candidate states against Triton, then continues
from each possible accepted prefix with reused destination slots. Tested B=2/4,
width=2/3/4, BF16 activations, FP32 states; tolerance atol=.005, rtol=.01.

Interface selection (TP-local heads 8/24, dimensions128, verify width3):

| Batch | chain RTP us | SGLang core us | SGLang plus RTP state scatter us |
|---:|---:|---:|---:|
|2|15.77|15.05|43.93|
|4|20.94|20.87|65.40|
|8|26.52|25.43|105.04|
|16|41.27|40.20|194.58|

Seven CUDA graph timing samples of 100 iterations, median reported. Outputs
and candidate states were identical between these interfaces. SGLang's core
is marginally faster but its snapshot-buffer output requires conversion to
RTP per-candidate pool slots. The tested index_copy adapter is much slower.
A future fused state writer could change this conclusion.

End-to-end tests on 88_amd, xjn_112_test, GPUs6/7, TP=EP=2, MTP2, prefix cache:
all three modes passed fixed greedy-output and eight prefix/MTP checks. Two
matched seeds per scenario show 7/8 TPOT means improved versus Replay off
(max8.01%), and 6/8 throughput means improved (max5.39%). 10K C2/C4 throughput
regressed .32%/.95%. These small samples do not demonstrate universal gains.
Logs, commands and results: /data/home/xingjunna.xjn/xjn_116_debug/flydsl_mtp_dev/.
