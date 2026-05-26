package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.netty.channel.ChannelOption;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

@Configuration
public class DispatcherConfiguration {

    /**
     * TCP three-way-handshake timeout for dispatcher → FE connections. Aligned with the
     * codebase's {@code HttpNettyConfig.syncNettyClient}'s value — same deployment class
     * (same-cluster HTTP). Hardcoded rather than exposed as config because connect timeout is
     * almost never an operator-tuned knob; if a deployment ever needs different here it's a
     * deployment-topology change, not a runtime config.
     */
    private static final int FE_CONNECT_TIMEOUT_MS = 1000;

    /**
     * How long a request waits for an available connection from the FE pool before failing. Fires
     * when {@code feMaxConnectionsPerHost} is exhausted AND {@code feMaxPendingAcquirePerHost}
     * queue has room. Hardcoded because operators tune capacity ({@code feMaxConnectionsPerHost}),
     * not patience — the right pendingAcquire timeout follows mechanically from capacity sizing.
     */
    private static final int FE_PENDING_ACQUIRE_TIMEOUT_MS = 3000;

    @Bean
    public DispatchConfig dispatchConfig() {
        return DispatchConfig.fromJson(System.getenv("DISPATCH_CONFIG"));
    }

    /**
     * Dispatcher routes on the SHARED 7001 listener, ordered last so the catch-all never shadows
     * the Master's /rtp_llm/* (which are @Order(0) — see {@code HttpLoadBalanceServer}). Returns
     * null when disabled — Spring will not register a null bean, so disabled deployments add
     * nothing to the route table.
     *
     * <p>FE pool membership is sourced from {@link ServiceDiscovery}: the bean injected here is
     * the auto-configured one — NoOp in open-source / dev (reads {@code DOMAIN_ADDRESS:<id>} env
     * var) and VipServer-backed in the internal profile (push subscription). The dispatcher
     * subscribes via {@code listen()} and mirrors every host-change callback into an
     * {@link AtomicReference} that {@link FePool} reads on every {@code next()}.
     *
     * <p>Batch-aware routes are registered per {@link BatchEndpointSpec} supplied by
     * {@link BatchEndpointRegistry}; everything else under {@code /dispatcher/**} falls through to
     * {@link WebClientPassthroughClient}.
     *
     * <p>The dispatcher shares its JVM, listener, and heap with the master. Three things keep the
     * two from starving each other: the shared Jackson {@link ObjectMapper} bean (no drifting
     * config), a dedicated named {@link ConnectionProvider} ("dispatcher-fe") so fanout cannot
     * starve {@code GeneralHttpNettyService}'s master connections, and the per-response byte cap
     * in {@link WebClientFeClient} so a misbehaving FE cannot swamp the shared heap.
     *
     * <p>Two {@link HttpClient} instances share that one connection provider so the pool stays
     * unified but the timeout semantics split by role:
     * <ul>
     *   <li>FE batch traffic gets {@code responseTimeout=batchTimeoutMs} — chunks are
     *       JSON-tiny and complete in milliseconds, so a stalled read is a real fault.</li>
     *   <li>Passthrough streaming gets no {@code responseTimeout} — mid-stream silence is normal
     *       for SSE. Its only cap is {@link WebClientPassthroughClient}'s body-Flux
     *       {@code .timeout(STREAM_TIMEOUT_MS)} hardcoded constant.</li>
     * </ul>
     * Both share {@link ChannelOption#CONNECT_TIMEOUT_MILLIS} so dead FEs fast-fail on connect.
     */
    @Bean
    @Order(Ordered.LOWEST_PRECEDENCE)
    public RouterFunction<ServerResponse> dispatcherRoutes(DispatchConfig cfg,
                                                           ObjectMapper mapper,
                                                           WebClient.Builder webClientBuilder,
                                                           ServiceDiscovery serviceDiscovery,
                                                           List<BatchEndpointSpec> specs) {
        if (!cfg.isEnabled()) {
            return null;
        }
        ConnectionProvider feConnections = ConnectionProvider.builder("dispatcher-fe")
                .maxConnections(cfg.getFeMaxConnectionsPerHost())
                .pendingAcquireTimeout(Duration.ofMillis(FE_PENDING_ACQUIRE_TIMEOUT_MS))
                .pendingAcquireMaxCount(cfg.getFeMaxPendingAcquirePerHost())
                .build();
        HttpClient feBatchHttp = HttpClient.create(feConnections)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS)
                .responseTimeout(Duration.ofMillis(cfg.getBatchTimeoutMs()));
        HttpClient passthroughHttp = HttpClient.create(feConnections)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS);
        WebClient.Builder feBatchBuilder = webClientBuilder.clone()
                .clientConnector(new ReactorClientHttpConnector(feBatchHttp));
        WebClient passthroughWebClient = webClientBuilder.clone()
                .clientConnector(new ReactorClientHttpConnector(passthroughHttp))
                .build();

        String serviceId = cfg.getFePoolServiceId();
        AtomicReference<List<String>> fePoolUrls = new AtomicReference<>(
                toUrls(serviceDiscovery.getHosts(serviceId)));
        serviceDiscovery.listen(serviceId, hosts -> {
            List<String> urls = toUrls(hosts);
            fePoolUrls.set(urls);
            // WARN level (always-on) — FE pool topology changes are operationally important
            // and infrequent, matching the codebase convention used by ZK election / engine sync.
            Logger.warn("dispatcher FE pool updated: serviceId={}, hosts={}", serviceId, urls.size());
        });
        FeHealthChecker healthChecker = new FeHealthChecker(
                fePoolUrls::get, passthroughWebClient, cfg.getProbePath());
        healthChecker.start();
        FePool pool = new FePool(fePoolUrls::get, healthChecker::isAlive);

        FeClient feClient = new WebClientFeClient(feBatchBuilder);
        FanoutService fanout = new FanoutService(feClient, pool);
        GenericBatchHandler batchHandler = new GenericBatchHandler(fanout, mapper, cfg.subBatchSpec());

        PassthroughClient passthrough = new WebClientPassthroughClient(passthroughWebClient, pool);
        DispatchHandler handler = new DispatchHandler(passthrough);
        // WARN so the boot footprint survives default LOG_LEVEL=null gating — operators
        // need this exact line to verify which FE pool, batchSpecs count, and timeouts the
        // dispatcher came up with.
        Logger.warn("dispatcher enabled: fePoolServiceId={}, seedHosts={}, subBatch={}, batchSpecs={}, "
                        + "batchTimeoutMs={}, feMaxConnectionsPerHost={}, feMaxPendingAcquirePerHost={}, "
                        + "probePath={}",
                serviceId, fePoolUrls.get().size(), cfg.getSubBatch(), specs.size(),
                cfg.getBatchTimeoutMs(), cfg.getFeMaxConnectionsPerHost(),
                cfg.getFeMaxPendingAcquirePerHost(), cfg.getProbePath());
        return new DispatchRouter(batchHandler, handler, specs).routes();
    }

    private static List<String> toUrls(List<WorkerHost> hosts) {
        return hosts.stream().map(h -> "http://" + h.getIpPort()).collect(Collectors.toList());
    }
}
