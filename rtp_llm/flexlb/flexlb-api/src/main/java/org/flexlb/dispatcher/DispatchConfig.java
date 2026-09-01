package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnore;
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
     * the response header arrives; the body read is separately capped by {@link FeClient}'s
     * whole-call timeout ({@code batchTimeoutMs + }{@link #bodyReadMarginMs}). Same idea as
     * ft_proxy's {@code -t}, but only covers the header-wait window in reactor-netty's model.
     *
     * <p>Note for non-streaming generation endpoints ({@code /batch_infer} etc.): FE sends the
     * response headers only after the whole chunk finishes generating, so this must cover the
     * full generation time of one chunk — not a network-level header latency. Tune down for
     * embedding-only deployments where sub-second responses are the norm.
     */
    private int batchTimeoutMs = 30_000;

    /**
     * Extra budget past {@link #batchTimeoutMs} for reading the response body. {@code batchTimeoutMs}
     * only bounds time-to-headers; without a whole-call cap an FE that sends headers and then stalls
     * mid-body (half-open connection, GC-wedged process) would pin the request and its pooled
     * connection forever. Headers arrive after generation completes, so the body is just
     * bytes-on-the-wire — the default covers a full 16MB response with wide margin. Raise it for FE
     * fleets on slow links.
     */
    private long bodyReadMarginMs = 30_000;

    /**
     * Path the {@link FeHealthChecker} probes via {@code GET <feUrl><probePath>} every 1s. Default
     * matches rtp_llm FE's {@code /frontend_health} endpoint; switch to {@code /health} for vLLM
     * deployments or any other backend that exposes a different liveness path. The 2-fail-then-dead,
     * 1-success-resets, optimistic-default semantics in {@link FeHealthChecker} are unchanged
     * regardless of path — only the URL suffix moves.
     */
    private String probePath = "/frontend_health";

    /**
     * Source of each chunk's FE assignment. {@code master} (default) uses the elected master's
     * single cursor for even, attributable fleet-wide distribution. {@code local} is the explicit
     * availability escape hatch: it bypasses master FE assignment and uses this dispatcher's own
     * health-filtered {@link FePool}. Set {@code DISPATCH_FE_ALLOCATION=local} during a master
     * outage or on a multi-role deployment whose master cannot serve {@code /batch_schedule};
     * restore {@code master} after the incident to regain the global cursor.
     *
     * <p>This is a string at the external-config boundary so the shared env-reflection helper does
     * not need to change enum parsing semantics. {@link DispatcherConfiguration} validates and
     * normalizes it once at startup through {@link FeAllocationMode#parse(String)}.
     */
    private String feAllocation = FeAllocationMode.MASTER.configValue();

    /**
     * BE pre-assignment toggle. When {@code true}, the dispatcher resolves N BE targets via
     * master's {@code /rtp_llm/batch_schedule} before fanout and appends each target into
     * the chunk's {@code generate_config.role_addrs} (matching Python
     * {@code rtp_llm.config.generate_config.RoleAddr}: {@code {role, ip, http_port, grpc_port}})
     * so the receiving FE skips its own master round-trip.
     *
     * <p>Defaults to {@code false}: this optimization crosses an HTTP/JSON version boundary, and
     * mixed-version FE fleets must keep serving correctly on first rollout. Enable it explicitly
     * only after every FE build satisfies the version precondition below.
     *
     * <p><strong>FE version precondition when enabled:</strong> the FE build must include
     * {@code RoleAddr.validate_role} (the {@code @field_validator("role", mode="before")} in
     * {@code rtp_llm/config/generate_config.py}, on main since commit {@code 53dc319bd}).
     * Older FE builds leave {@code role_addrs} as {@code list[dict]} and fail every stamped
     * request with HTTP 500 at {@code model_rpc_client}'s {@code addr.role} access — the
     * dispatcher is the first caller to deliver {@code role_addrs} via the HTTP body, so the
     * latent FE bug only fires with this toggle on.
     *
     * <p>Operators can keep/flip {@code DISPATCH_PRE_ASSIGN_BE=false} (or set
     * {@code preAssignBe: false} in {@code DISPATCH_CONFIG}) to opt out for diagnostics or
     * staged rollback.
     *
     * <p>Disabling this flag does not disable master FE allocation. It changes a master request to
     * FE-only, so no unused BE target is selected and no BE round-robin cursor is advanced.
     */
    private boolean preAssignBe = false;

    /**
     * Parsed sub-batch spec; populated by {@link DispatcherConfiguration} during loading.
     *
     * <p>{@code @JsonIgnore}, not {@code transient}: Jackson does not honour the {@code transient}
     * keyword by default, and Lombok exposes this as a bean property — so a {@code subBatchSpec}
     * key appearing in {@code DISPATCH_CONFIG} would be bound onto a derived field instead of
     * being ignored. It is derived from {@link #subBatch} and never part of the wire contract.
     */
    @JsonIgnore
    private SubBatchSpec subBatchSpec;
}
