# Experimental guarded NIXL checkpoint engine

This adapter extends the unmodified verl NIXL checkpoint engine at commit
`7c760b079215a674db1d759fe761e2aa1312baf1`. It checks the upstream source
SHA256 `5ba0aa3ecb41ef71849f9919d2e291defa2b78be7e29a3a640ddc25602493a71`
before initialization. A mismatch or a symlink is rejected. The package is not
vendored here: install the complete matching verl and its NIXL/CuPy/Ray
dependencies in each checkpoint-engine worker environment.

Set `actor_rollout_ref.rollout.checkpoint_engine.backend=rtp_nixl_guarded`
and `custom_backend_module=verl_nixl_guarded`. Add the absolute path of this
directory to `PYTHONPATH` on **every training and rollout CE worker** before
worker startup. The plugin imports `full_hf_relay` from the same directory.
The outer `update_weights_bucket_megabytes` supplies `bucket_size`; do not
duplicate it in `engine_kwargs.rtp_nixl_guarded`. Configure the per-tensor
bound and timeouts there where appropriate. Verify actual worker environments;
setting a driver-only environment does not establish worker propagation.

The local NIXL 1.5.2 environment needed `UCX_MODULES=^efa` to avoid an
exit-time library conflict. Other UCX network settings are deployment-specific;
verify them with the real transport and lane evidence.

This is an experimental full-HF correctness baseline, not target-sharded
redistribution. Two NIXL buckets and, when necessary, one whole reassembled
source tensor are resident; budget device memory explicitly. Native IPC
consumers must implement the explicit all-rank ACK lease interface. Failed or
timed-out transfers retain resources until identity-bound process cleanup;
never assume an in-flight NIXL release stopped RDMA. The CPU tests are mocks
and do not qualify NIXL, CUDA, native IPC, or a model training run.
