package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
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

@Configuration
public class DispatcherConfiguration {

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
     * <p>Same-JVM resource isolation (decision F4 / F4b):
     * <ul>
     *   <li>Reuses the auto-configured Spring {@link ObjectMapper} bean — no second copy with
     *       drifting Jackson config.</li>
     *   <li>Uses a dedicated, named {@link ConnectionProvider} ("dispatcher-fe") so dispatcher
     *       fanout cannot starve {@code GeneralHttpNettyService}'s connections to the master
     *       (which the Master's slave→master forward uses).</li>
     *   <li>{@link WebClientFeClient} caps the per-response in-memory size at
     *       {@code feMaxResponseBytes} so a misbehaving FE cannot swamp the shared heap.</li>
     * </ul>
     */
    @Bean
    @Order(Ordered.LOWEST_PRECEDENCE)
    public RouterFunction<ServerResponse> dispatcherRoutes(DispatchConfig cfg,
                                                           ObjectMapper mapper,
                                                           WebClient.Builder webClientBuilder) {
        if (!cfg.isEnabled()) {
            return null;
        }
        ConnectionProvider feConnections = ConnectionProvider.builder("dispatcher-fe")
                .maxConnections(cfg.getFeMaxConnections())
                .pendingAcquireTimeout(Duration.ofMillis(cfg.getFeRequestTimeoutMs()))
                .pendingAcquireMaxCount(cfg.getFeMaxPendingAcquire())
                .build();
        WebClient.Builder feBuilder = webClientBuilder.clone()
                .clientConnector(new ReactorClientHttpConnector(HttpClient.create(feConnections)));
        FePool pool = new FePool(cfg.getFePoolAddresses());
        FeClient feClient = new WebClientFeClient(feBuilder, cfg.getFeRequestTimeoutMs(), cfg.getFeMaxResponseBytes());
        FanoutService fanout = new FanoutService(feClient, pool, mapper, cfg.getSubBatchSize());
        PassthroughClient passthrough = new WebClientPassthroughClient(feBuilder.build(), pool, cfg.getFeRequestTimeoutMs());
        DispatchHandler handler = new DispatchHandler(fanout, passthrough, mapper);
        return new DispatchRouter(handler).routes();
    }
}
