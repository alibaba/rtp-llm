# First Response Buffering

`DASH_SC_GRPC_PROXY_BUFFER_FIRST_CHUNK` is read when the proxy starts.
The default is `1`: hold the first response until the second response arrives,
preserving the existing behavior used to smooth PD handoff latency.

Set it to `0` and restart the frontend to forward each response immediately.
Other values retain buffering. The startup log includes `buffer_first_chunk`,
and forward access records report `buffered_stage=disabled` when disabled.
Terminal frames still end forwarding and close the downstream RPC.
This also applies when the first frame is already terminal and buffering is enabled.

For the Qwen3.6 Whale proxy deployment, set the variable explicitly to `0` in
the deployment configuration. Rebuilding the image alone preserves the default
buffering behavior. Verify the startup log and `buffered_stage=disabled` in an
actual forward access record after all frontend workers restart.

This setting affects internal first-response latency. It does not change
non-streaming response aggregation, gateway total-duration limits, model
execution time, or the number of requests admitted by the model backend.

Cancellation while waiting for a response must close the downstream call
without first flushing a buffered frame to the cancelled caller.
