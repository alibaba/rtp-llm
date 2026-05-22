package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
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

@Slf4j
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
     * <p>FE pool membership is sourced from {@link ServiceDiscovery}: the bean injected here is
     * the auto-configured one — NoOp in open-source / dev (reads {@code DOMAIN_ADDRESS:<id>} env
     * var) and VipServer-backed in the internal profile (push subscription). The dispatcher
     * subscribes via {@code listen()} and mirrors every host-change callback into an
     * {@link AtomicReference} that {@link FePool} reads on every {@code next()}.
     *
     * <p>The dispatcher shares its JVM, listener, and heap with the master. Three things keep the
     * two from starving each other: the shared Jackson {@link ObjectMapper} bean (no drifting
     * config), a dedicated named {@link ConnectionProvider} ("dispatcher-fe") so fanout cannot
     * starve {@code GeneralHttpNettyService}'s master connections, and the per-response byte cap
     * in {@link WebClientFeClient} so a misbehaving FE cannot swamp the shared heap.
     */
    @Bean
    @Order(Ordered.LOWEST_PRECEDENCE)
    public RouterFunction<ServerResponse> dispatcherRoutes(DispatchConfig cfg,
                                                           ObjectMapper mapper,
                                                           WebClient.Builder webClientBuilder,
                                                           ServiceDiscovery serviceDiscovery) {
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

        String serviceId = cfg.getFePoolServiceId();
        AtomicReference<List<String>> fePoolUrls = new AtomicReference<>(
                toUrls(serviceDiscovery.getHosts(serviceId)));
        serviceDiscovery.listen(serviceId, hosts -> {
            List<String> urls = toUrls(hosts);
            fePoolUrls.set(urls);
            log.info("dispatcher FE pool updated: serviceId={}, hosts={}", serviceId, urls.size());
        });
        FePool pool = new FePool(fePoolUrls::get);

        FeClient feClient = new WebClientFeClient(feBuilder, cfg.getFeRequestTimeoutMs(), cfg.getFeMaxResponseBytes());
        FanoutService fanout = new FanoutService(feClient, pool, mapper, cfg.getSubBatchSize());
        PassthroughClient passthrough = new WebClientPassthroughClient(feBuilder.build(), pool, cfg.getFeRequestTimeoutMs());
        DispatchHandler handler = new DispatchHandler(fanout, passthrough, mapper);
        log.info("dispatcher enabled: fePoolServiceId={}, seedHosts={}, subBatchSize={}",
                serviceId, fePoolUrls.get().size(), cfg.getSubBatchSize());
        return new DispatchRouter(handler).routes();
    }

    private static List<String> toUrls(List<WorkerHost> hosts) {
        return hosts.stream().map(h -> "http://" + h.getIpPort()).collect(Collectors.toList());
    }
}
