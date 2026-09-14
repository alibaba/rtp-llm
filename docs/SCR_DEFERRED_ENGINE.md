# Deferred engine startup for SCR templates

A pre-service template initializes model weights, warmup, KV cache, executor and scheduler before the checkpoint barrier. The normal engine computation loop starts only after template release, through LocalRpcServer::startDeferredServices. This applies to every rank, including ranks without a TP broadcaster.

Deferring network listeners alone is insufficient for Decode with DP greater than one: the background scheduler can submit dummy streams and GPU kernels even when no client request is accepted. A CUDA synchronization at barrier arrival does not prevent subsequent submissions.

Normal startup keeps its existing immediate loop start. Repeated release must not create another thread, and aborting a prepared template must safely stop an engine whose loop never started.

Validation target: //rtp_llm/cpp/normal_engine/test:deferred_engine_start_test (CUDA). It checks ordinary immediate startup, stop-before-start, one-time release followed by generation, and RPC release for a leader without peers and a TP follower without a broadcaster. The RPC role tests use a single-GPU mock engine; they do not create a multi-rank communicator.

The Python targets `//rtp_llm/utils/test:scr_runtime_fixup_test`, `scr_restore_env_file_test`, `scr_advertise_ip_test`, `scr_pd_advertisement_test` and `scr_native_logger_test` cover barrier failures, fresh environment input, explicit endpoint precedence and repeated identity repair before release. The Logger test compiles the real Logger with alog/autil test doubles.

The live acceptance must additionally verify Decode DP2 dump/restore and a real PD request with correlated KV transfer, including cross-node restore. Unit tests and compilation do not establish that acceptance.
