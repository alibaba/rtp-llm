# Managed NIXL reference from Qwen3.6 disaggregated experiments

This directory preserves the exact source snapshots used by the November 2026
Qwen3.6 experiments. They are opt-in research references, not an enabled
backend or a claim that Qwen3.8-2.4T training has passed. The existing
`disaggregated/verl_nixl_guarded.py` is intentionally unchanged. The file of
that name in this directory is the qualified partial-lease variant; its
content hash is `b14be8fa0b69bc8c33dff4ae0694beed5e34bc0849517f9a4ca6464e48191406`.

The managed context, manager gate, bootstrap and external entry preserve the
server PULL context across generations, guard per-generation PUSH contexts,
bind participant and engine identities, and hold uncertain failures for
identity-scoped external handling. The shutdown reference illustrates the
corresponding rLLM actor lifecycle. The gateway reference preserves the lower
of the client and session token limits. The snapshots and their accompanying
tests retain their original Q36-specific environment variables, pinned source
hashes, paths and imports; they require a qualified deployment and must not be
treated as a drop-in or as a generic 2.4T configuration. In particular, the
bootstrap expects the parent directory's `full_hf_relay.py` and the fixed verl
source revisions described in the experiment evidence.

Evidence supports a Qwen3.6 4-train/4-inference on-policy RL run with two
nonzero updates; an 8-train/4-inference EP8-to-EP4 three-generation test with
two nonzero updates to a small selected parameter set; and three generations
through real Ray, NIXL and EP4 native workers using a controlled source.
The latter two tests are not full 8+4 RL. Several subsequent 8+4 on-policy
runs completed rollout and parameter synchronization but had zero reward
variance and no effective update. The bounded resampling experiment was
unfinished and is deliberately excluded from this reference.

The current transfer reconstructs complete HF tensors and broadcasts them;
it is not expert-shard direct transfer. Qwen3.8-2.4T tensor layout, memory
budget, 64-train/32-inference scale, throughput, and fault recovery remain
unverified. A HOLD retains software ownership; it does not establish that NIC
traffic has stopped. The per-source CUDA synchronization is a correctness
boundary with unmeasured 2.4T performance cost. No tests were rerun when these
snapshots were saved because the user requested that testing stop.
