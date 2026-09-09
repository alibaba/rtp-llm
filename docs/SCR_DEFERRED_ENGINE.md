# Deferred engine startup for SCR templates

A pre-service template initializes model weights, warmup, KV cache, executor and scheduler before the checkpoint barrier. The normal engine computation loop starts only after template release, through LocalRpcServer::startDeferredServices. This applies to every rank, including ranks without a TP broadcaster.

Deferring network listeners alone is insufficient for Decode with DP greater than one: the background scheduler can submit dummy streams and GPU kernels even when no client request is accepted. A CUDA synchronization at barrier arrival does not prevent subsequent submissions.

Normal startup keeps its existing immediate loop start. Repeated release must not create another thread, and aborting a prepared template must safely stop an engine whose loop never started.

Validation target: //rtp_llm/cpp/normal_engine/test:deferred_engine_start_test (CUDA). It checks stop-before-start and one-time release followed by generation. The live acceptance must additionally verify Decode DP2 dump/restore and a real PD request with correlated KV transfer, including cross-node restore. Unit tests and compilation do not establish that acceptance.
