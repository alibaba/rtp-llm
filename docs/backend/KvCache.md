# KVCM cache-state Subscriber

RTP-LLM can supervise an external KVCM Subscriber alongside its backend. RTP
only resolves runtime metadata and owns the child-process lifecycle; polling
`GetCacheStatus`, snapshot diffing, retry, and KVCM `ReportEvent` calls are
implemented by the `tair-kvcache/subscriber` package.

Set `KVCM_SUBSCRIBER_CONFIG` to an existing Subscriber YAML file to enable the
process. When DP is 1, RTP derives the cache gRPC endpoint from the resolved
server configuration: rank 0 uses `START_PORT + 1`. The KVCM host identity uses
the inference HTTP endpoint at `START_PORT`.

```bash
export KVCM_SUBSCRIBER_CONFIG=/path/to/subscriber.yaml
export KVCM_SUBSCRIBER_COMMAND="/path/to/venv/bin/subscriber"
python -m rtp_llm.start_server
```

Optional overrides:

- `RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS`: comma-separated cache gRPC endpoints.
  This is required for DP greater than 1 because remote DP addresses cannot be
  inferred safely from one local process. Supply exactly one unique endpoint
  for each DP rank.
- `KVCM_HOST_IP_PORT`: stable inference-service identity reported to KVCM.
- `KVCM_SUBSCRIBER_WORLD_RANK`: world rank that owns the Subscriber; defaults
  to `0`, preventing duplicate reporters in a distributed launch.
- `KVCM_SUBSCRIBER_COMMAND`: executable and optional arguments; defaults to
  `subscriber`.
- `KVCM_SUBSCRIBER_REQUIRED`: set to `1`, `true`, `yes`, or `on` to make
  Subscriber startup and runtime failures fail the RTP service. It defaults to
  the optional mode.

Subscriber reporting is fail-open by default: invalid configuration disables
reporting without taking down inference, while a missing executable or runtime
exit is retried by the dedicated supervisor. The supervisor is managed through
RTP's unchanged process lifecycle and stops the whole Subscriber process group
when RTP exits. Set `KVCM_SUBSCRIBER_REQUIRED=1` where reporting is mandatory;
in that mode the same failures shut down the service group instead of retrying
in the background.
