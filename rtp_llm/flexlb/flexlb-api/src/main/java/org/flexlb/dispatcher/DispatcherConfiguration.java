package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import org.flexlb.config.EnvConfigOverrides;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Dispatcher infrastructure beans (config, connection provider, shared WebClient, route table).
 * Domain beans (FePool, FeHealthChecker, FeClient, …) are individually annotated
 * {@code @Component @ConditionalOnProperty} so the entire dispatcher subsystem is gated on
 * {@code dispatch.fe-pool-service-id} being a non-blank value (via Spring's relaxed binding
 * the {@code DISPATCH_FE_POOL_SERVICE_ID} env maps to this property automatically). No
 * second "enabled" flag — presence-of-FE-pool-name IS the enable signal: a deployment that
 * names a FE pool obviously means to run the dispatcher, and one that doesn't has nothing
 * for the dispatcher to call anyway.
 */
@Configuration
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatcherConfiguration {

    /**
     * TCP three-way-handshake timeout for every dispatcher → FE connection (passthrough and
     * batch fanout — {@link FeClient} reads it too). Aligned with the codebase's
     * {@code HttpNettyConfig.syncNettyClient}'s value — same deployment class
     * (same-cluster HTTP). Hardcoded rather than exposed as config because connect timeout is
     * almost never an operator-tuned knob; if a deployment ever needs different here it's a
     * deployment-topology change, not a runtime config.
     */
    static final int FE_CONNECT_TIMEOUT_MS = 1000;

    /**
     * How long a request waits for an available connection from the FE pool before failing. Fires
     * when {@link #FE_MAX_CONNECTIONS_PER_HOST} is exhausted AND {@link #FE_MAX_PENDING_ACQUIRE_PER_HOST}
     * queue has room. Hardcoded because operators tune capacity, not patience — the right
     * pendingAcquire timeout follows mechanically from capacity sizing.
     */
    private static final int FE_PENDING_ACQUIRE_TIMEOUT_MS = 3000;

    /**
     * Max concurrent TCP connections <strong>per FE host</strong> (not total across the pool).
     * Reactor-netty's {@code ConnectionProvider} pools per remote address, so with N FE hosts the
     * effective ceiling is {@code FE_MAX_CONNECTIONS_PER_HOST × N}. Sized for the workloads we run
     * today (target QPS × avg request time / FE count, with safety margin); change here if the
     * deployment topology shifts.
     */
    private static final int FE_MAX_CONNECTIONS_PER_HOST = 200;

    /**
     * Max pending acquires <strong>per FE host</strong> when the connection pool is exhausted.
     * Acts as a backpressure ring buffer; exceeding it makes the dispatcher fail fast instead of
     * piling up an unbounded queue under overload.
     */
    private static final int FE_MAX_PENDING_ACQUIRE_PER_HOST = 1000;

    @Bean
    public DispatchConfig dispatchConfig() {
        return loadAndValidate(System.getenv());
    }

    /**
     * Exposes the dispatcher's batch endpoint table as a Spring bean for {@link DispatchRouter}
     * to autowire. Single source of truth — every batch endpoint the dispatcher serves is one
     * row in {@link BatchEndpointSpec#SPECS}.
     */
    @Bean
    public List<BatchEndpointSpec> batchEndpointSpecs() {
        return BatchEndpointSpec.SPECS;
    }

    /**
     * Load + validate the dispatcher config from the given env map. Mirrors how
     * {@link org.flexlb.config.ConfigService} loads {@code FlexlbConfig}: defaults → JSON from
     * the {@code DISPATCH_CONFIG} env → per-field {@code DISPATCH_*} overrides. Validates and
     * eagerly parses the sub-batch DSL so a bad value fails fast at boot rather than on the
     * first request. Package-private so {@code DispatchConfigTest} can call it directly with
     * a stubbed env.
     */
    static DispatchConfig loadAndValidate(Map<String, String> env) {
        String json = env.get("DISPATCH_CONFIG");
        DispatchConfig c = (json == null || json.isBlank())
                ? new DispatchConfig()
                : JsonUtils.toObject(json, DispatchConfig.class);
        EnvConfigOverrides.apply(c, "DISPATCH_", env);
        validate(c);
        return c;
    }

    private static void validate(DispatchConfig c) {
        // Spring's @ConditionalOnProperty has already filtered "env unset" / literal "false" by the
        // time this @Bean is wired. This defensive check catches the edge where the env is set to
        // pure whitespace — OnPropertyCondition treats that as "not false" and activates the bean,
        // but downstream lookups against ServiceDiscovery would silently return zero hosts.
        if (c.getFePoolServiceId() == null || c.getFePoolServiceId().isBlank()) {
            throw new IllegalArgumentException(
                    "DISPATCH_FE_POOL_SERVICE_ID (or DISPATCH_CONFIG.fePoolServiceId) must be a "
                            + "non-blank vipserver/discovery name when the dispatcher is loaded");
        }
        if (c.getBatchTimeoutMs() <= 0) {
            throw new IllegalArgumentException("batchTimeoutMs must be > 0, got " + c.getBatchTimeoutMs());
        }
        if (c.getProbePath() == null || c.getProbePath().isBlank()) {
            throw new IllegalArgumentException(
                    "probePath must not be blank — set DISPATCH_PROBE_PATH=/frontend_health (rtp_llm) "
                            + "or /health (vLLM) etc.; got '" + c.getProbePath() + "'");
        }
        // SubBatchSpec.parse throws IllegalArgumentException with a precise message on bad DSL.
        c.setSubBatchSpec(SubBatchSpec.parse(c.getSubBatch()));
    }

    /**
     * Dedicated named connection provider so dispatcher fanout cannot starve
     * {@code GeneralHttpNettyService}'s master connections. Reactor-netty pools per remote
     * address, so the effective ceiling is {@code FE_MAX_CONNECTIONS_PER_HOST × N FE hosts}.
     */
    @Bean("dispatcherFeConnectionProvider")
    public ConnectionProvider dispatcherFeConnectionProvider() {
        return ConnectionProvider.builder("dispatcher-fe")
                .maxConnections(FE_MAX_CONNECTIONS_PER_HOST)
                .pendingAcquireTimeout(Duration.ofMillis(FE_PENDING_ACQUIRE_TIMEOUT_MS))
                .pendingAcquireMaxCount(FE_MAX_PENDING_ACQUIRE_PER_HOST)
                .build();
    }

    /**
     * Shared WebClient for passthrough and health-probe traffic. No {@code responseTimeout} —
     * mid-stream silence is normal for SSE, and {@link PassthroughClient} caps the body Flux
     * with its own {@code STREAM_TIMEOUT_MS} as a safety net.
     */
    @Bean("dispatcherPassthroughWebClient")
    public WebClient dispatcherPassthroughWebClient(WebClient.Builder builder,
            @Qualifier("dispatcherFeConnectionProvider") ConnectionProvider provider) {
        HttpClient passthroughHttp = HttpClient.create(provider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS);
        return builder.clone()
                .clientConnector(new ReactorClientHttpConnector(passthroughHttp))
                .build();
    }

    /**
     * Health probes get their own tiny connection pool, isolated from the fanout/passthrough
     * data plane: when {@link #FE_MAX_CONNECTIONS_PER_HOST} is saturated under load, a probe
     * queued behind data traffic exceeds its 500ms budget and marks a healthy FE dead —
     * exactly when accurate liveness matters most. Two connections per host cover the
     * one-probe-per-interval cadence with headroom.
     */
    @Bean("dispatcherProbeWebClient")
    public WebClient dispatcherProbeWebClient(WebClient.Builder builder) {
        ConnectionProvider probeProvider = ConnectionProvider.builder("dispatcher-fe-probe")
                .maxConnections(2)
                .pendingAcquireTimeout(Duration.ofMillis(FE_PENDING_ACQUIRE_TIMEOUT_MS))
                .build();
        HttpClient probeHttp = HttpClient.create(probeProvider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS);
        return builder.clone()
                .clientConnector(new ReactorClientHttpConnector(probeHttp))
                .build();
    }

    /**
     * Dispatcher routes on the SHARED 7001 listener. All dispatcher routes are anchored under
     * {@code /dispatcher/**}, disjoint from the Master's {@code /rtp_llm/*}, so no relative
     * ordering between the RouterFunction beans is needed.
     */
    @Bean
    public RouterFunction<ServerResponse> dispatcherRoutes(DispatchRouter router) {
        return router.routes();
    }

    /**
     * Emits the boot WARN line surfacing dispatcher footprint. WARN so the line survives default
     * {@code LOG_LEVEL=null} gating — operators need this exact line to verify which FE pool,
     * batchSpecs count, and timeouts the dispatcher came up with.
     */
    @Bean
    SmartInitializingSingleton dispatcherBootLog(DispatchConfig cfg, DispatcherFePoolRefresher refresher) {
        return () -> Logger.warn(
                "dispatcher enabled: fePoolServiceId={}, seedHosts={}, subBatch={}, batchSpecs={}, "
                        + "batchTimeoutMs={}, probePath={}, preAssignBe={}",
                cfg.getFePoolServiceId(), refresher.currentSize(), cfg.getSubBatch(),
                BatchEndpointSpec.SPECS.size(),
                cfg.getBatchTimeoutMs(), cfg.getProbePath(), cfg.isPreAssignBe());
    }
}
