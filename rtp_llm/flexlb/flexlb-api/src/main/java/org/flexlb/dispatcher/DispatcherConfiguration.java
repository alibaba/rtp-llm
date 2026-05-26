package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.netty.channel.ChannelOption;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectProvider;
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
     * Singleton FE-pool refresher, created only when the dispatcher is enabled. Returning
     * {@code null} when disabled keeps Spring from registering the bean — and from
     * subscribing a listener or parking a {@code @Scheduled} method on the
     * {@code task-scheduler} pool — in deployments that do not use the dispatcher at all.
     *
     * <p>The refresher mirrors sync's {@code EngineAddressNameResolver}: a listener for the
     * push fast path plus a {@code @Scheduled} poll on the shared {@code task-scheduler}
     * pool as the always-on safety net. Operators see one staleness model and one threading
     * model across the whole flexlb process.
     */
    @Bean
    public DispatcherFePoolRefresher dispatcherFePoolRefresher(DispatchConfig cfg,
                                                               ServiceDiscovery serviceDiscovery) {
        if (!cfg.isEnabled()) {
            return null;
        }
        return new DispatcherFePoolRefresher(serviceDiscovery, cfg.getFePoolServiceId());
    }

    /**
     * Dispatcher routes on the SHARED 7001 listener, ordered last so the catch-all never shadows
     * the Master's /rtp_llm/* (which are @Order(0) — see {@code HttpLoadBalanceServer}). Returns
     * null when disabled — Spring will not register a null bean, so disabled deployments add
     * nothing to the route table.
     *
     * <p>FE pool membership is sourced from {@link DispatcherFePoolRefresher}, which polls
     * {@link ServiceDiscovery} on the shared {@code task-scheduler} pool. {@link FePool}
     * reads the latest snapshot through {@link DispatcherFePoolRefresher#source()} on every
     * {@code next()}.
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
                                                           ObjectProvider<DispatcherFePoolRefresher> fePoolRefresherProvider,
                                                           BatchScheduleCoordinator batchScheduleCoordinator,
                                                           List<BatchEndpointSpec> specs) {
        if (!cfg.isEnabled()) {
            return null;
        }
        // Disabled deployments do not register the refresher bean; the ObjectProvider here
        // tolerates that without forcing dispatcherRoutes to share the same instantiation
        // policy. When enabled, the bean must exist — getObject() throws loudly if not.
        DispatcherFePoolRefresher fePoolRefresher = fePoolRefresherProvider.getObject();
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

        FeHealthChecker healthChecker = new FeHealthChecker(
                fePoolRefresher.source(), passthroughWebClient, cfg.getProbePath());
        healthChecker.start();
        FePool pool = new FePool(fePoolRefresher.source(), healthChecker::isAlive);

        FeClient feClient = new WebClientFeClient(feBatchBuilder);
        FanoutService fanout = new FanoutService(feClient, pool);
        BatchScheduleClient batchScheduleClient = new LocalBatchScheduleClient(batchScheduleCoordinator);
        GenericBatchHandler batchHandler = new GenericBatchHandler(
                fanout, mapper, cfg.subBatchSpec(), batchScheduleClient, cfg.isPreAssignBe());

        PassthroughClient passthrough = new WebClientPassthroughClient(passthroughWebClient, pool);
        DispatchHandler handler = new DispatchHandler(passthrough);
        // WARN so the boot footprint survives default LOG_LEVEL=null gating — operators
        // need this exact line to verify which FE pool, batchSpecs count, and timeouts the
        // dispatcher came up with.
        Logger.warn("dispatcher enabled: fePoolServiceId={}, seedHosts={}, subBatch={}, batchSpecs={}, "
                        + "batchTimeoutMs={}, feMaxConnectionsPerHost={}, feMaxPendingAcquirePerHost={}, "
                        + "probePath={}, preAssignBe={}",
                cfg.getFePoolServiceId(), fePoolRefresher.currentSize(), cfg.getSubBatch(),
                specs.size(), cfg.getBatchTimeoutMs(), cfg.getFeMaxConnectionsPerHost(),
                cfg.getFeMaxPendingAcquirePerHost(), cfg.getProbePath(), cfg.isPreAssignBe());
        return new DispatchRouter(batchHandler, handler, specs).routes();
    }
}
