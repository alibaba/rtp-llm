package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

/**
 * Operator-facing tuning surface for the dispatcher. Pure POJO — loading and validation live in
 * {@link DispatcherConfiguration#dispatchConfig()}, mirroring how {@code ConfigService} loads
 * {@code FlexlbConfig}. Every timeout/safety knob that "no one actually tunes" lives as a
 * constant inside {@code DispatcherConfiguration} / {@code PassthroughClient} / {@code FeClient}
 * (see those classes for FE_CONNECT_TIMEOUT_MS / FE_PENDING_ACQUIRE_TIMEOUT_MS /
 * STREAM_TIMEOUT_MS / MAX_RESPONSE_BYTES).
 *
 * <p>Loading order: defaults → JSON from {@code DISPATCH_CONFIG} env → per-field env overrides
 * (e.g. {@code DISPATCH_BATCH_TIMEOUT_MS}, {@code DISPATCH_PROBE_PATH}). The per-field env wins,
 * matching the {@code FLEXLB_CONFIG} contract operators already know.
 *
 * <p>Unknown JSON properties are ignored so a stale {@code DISPATCH_CONFIG} carrying old field
 * names (subBatchSize, feRequestTimeoutMs, …) still boots — they just have no effect.
 */
@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class DispatchConfig {

    /**
     * Chunk splitting DSL. {@code count:N} → exactly N chunks (default). {@code size:N} →
     * each chunk holds at most N items. Bare integer is shorthand for {@code size:N}.
     * Parsed eagerly during loading so a malformed value fails fast at boot.
     */
    private String subBatch = "count:5";

    /**
     * Service-discovery name for the FE pool. Presence of {@code DISPATCH_FE_POOL_SERVICE_ID}
     * env (or this field non-blank in {@code DISPATCH_CONFIG} JSON) is the dispatcher's enable
     * signal — every dispatcher bean is gated on
     * {@code @ConditionalOnProperty("dispatch.fe-pool-service-id")} so a blank value means
     * the dispatcher subsystem never loads and {@code /dispatcher/**} routes are not registered.
     */
    private String fePoolServiceId = "";

    /**
     * Per batch sub-call: how long to wait for the FE to start responding (first byte). Stops once
     * the response header arrives — body read is bounded by infra defaults, not this. Same idea as
     * ft_proxy's {@code -t}, but only covers the header-wait window in reactor-netty's model.
     */
    private int batchTimeoutMs = 5000;

    /**
     * Max concurrent TCP connections <strong>per FE host</strong> (not total across the pool).
     * Reactor-netty's {@code ConnectionProvider} pools per remote address, so with N FE hosts the
     * effective ceiling is {@code feMaxConnectionsPerHost × N}. Tune from target QPS × avg request
     * time / FE count, with safety margin.
     */
    private int feMaxConnectionsPerHost = 200;

    /**
     * Max pending acquires <strong>per FE host</strong> when the connection pool is exhausted.
     * Acts as a backpressure ring buffer; exceeding it makes the dispatcher fail fast instead of
     * piling up an unbounded queue under overload.
     */
    private int feMaxPendingAcquirePerHost = 1000;

    /**
     * Path the {@link FeHealthChecker} probes via {@code GET <feUrl><probePath>} every 1s. Default
     * matches rtp_llm FE's {@code /frontend_health} endpoint; switch to {@code /health} for vLLM
     * deployments or any other backend that exposes a different liveness path. The 2-fail-then-dead,
     * 1-success-resets, optimistic-default semantics in {@link FeHealthChecker} are unchanged
     * regardless of path — only the URL suffix moves.
     */
    private String probePath = "/frontend_health";

    /**
     * BE pre-assignment toggle. When {@code true}, the dispatcher resolves N BE targets via
     * master's {@code /rtp_llm/batch_schedule} before fanout and appends each target into
     * the chunk's {@code generate_config.role_addrs} (matching Python
     * {@code rtp_llm.config.generate_config.RoleAddr}: {@code {role, ip, http_port, grpc_port}})
     * so the receiving FE skips its own master round-trip.
     *
     * <p>Defaults to {@code true} because the stamping target —
     * {@code generate_config.role_addrs} — is a field FE has supported in production for a
     * long time (see {@code rtp_llm.server.backend_rpc_server_visitor.route_ips}: when
     * {@code role_addrs} is non-empty the FE skips master entirely; the same mechanism powers
     * PD-disagg's prefill→decode handoff). No FE-side change is required for this toggle to
     * take effect end-to-end.
     *
     * <p>Operators can flip {@code DISPATCH_PRE_ASSIGN_BE=false} (or set
     * {@code preAssignBe: false} in {@code DISPATCH_CONFIG}) to opt out for diagnostics or
     * staged rollback.
     *
     * <p>If the dispatcher's call to {@code /batch_schedule} fails, the WARN is emitted once
     * and the request degrades silently to the no-stamp path — never block traffic on a
     * routing optimization.
     */
    private boolean preAssignBe = true;

    /** Parsed sub-batch spec; populated by {@link DispatcherConfiguration} during loading. */
    private transient SubBatchSpec subBatchSpec;

    public SubBatchSpec subBatchSpec() {
        return subBatchSpec;
    }
}
