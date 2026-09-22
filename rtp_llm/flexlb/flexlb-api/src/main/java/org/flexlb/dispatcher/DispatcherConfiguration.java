package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Callable;

/** Lazy HTTP ingress resources, enabled only through FLEXLB_CONFIG.httpDispatcher.enabled. */
@Configuration
@Lazy
public class DispatcherConfiguration {

    static final int FE_CONNECT_TIMEOUT_MS = 1000;

    private static final int FE_PENDING_ACQUIRE_TIMEOUT_MS = 3000;

    private static final int FE_MAX_CONNECTIONS_PER_HOST = 200;

    private static final int FE_MAX_PENDING_ACQUIRE_PER_HOST = 1000;

    @Bean
    public DispatchConfig dispatchConfig(ConfigurableEnvironment environment) {
        return loadAndValidate(environment);
    }

    @Bean(name = "dispatcherCpuScheduler", destroyMethod = "dispose")
    public Scheduler dispatcherCpuScheduler() {
        int threads = Math.max(2, Math.min(Runtime.getRuntime().availableProcessors(), 8));
        return Schedulers.newParallel("dispatcher-cpu", threads);
    }

    static DispatchConfig loadAndValidate(ConfigurableEnvironment environment) {
        DispatchConfig config = new DispatchConfig();
        // Binder throws on conversion errors; BindResult represents binding presence, not deferred failures.
        Binder.get(environment).bind("dispatch", Bindable.ofInstance(config));
        config.setTrustedRoutingToken(System.getenv().getOrDefault("DISPATCH_ROUTING_TOKEN", "").trim());
        validate(config);
        return config;
    }

    private static void validate(DispatchConfig c) {
        Assert.hasText(c.getProbePath(), "dispatch.probe-path must not be blank");
        URI probe = URI.create(c.getProbePath());
        Assert.isTrue(c.getProbePath().startsWith("/") && probe.getRawAuthority() == null && probe.getRawFragment() == null,
                "dispatch.probe-path must be an absolute HTTP path without authority or fragment");
        Assert.isTrue(c.getBatchTimeoutMs() > 0, "dispatch.batch-timeout-ms must be > 0");
        Assert.isTrue(!c.isPreAssignBe() || c.getFePoolServiceId().isBlank(),
                "BE preassignment uses colocated worker HTTP endpoints; set DISPATCH_PRE_ASSIGN_BE=false for an FE pool override");
        Assert.isTrue(!c.isPreAssignBe() || !c.getTrustedRoutingToken().isBlank(),
                "DISPATCH_ROUTING_TOKEN must be non-blank when preAssignBe is enabled");
        c.setSubBatchSpec(SubBatchSpec.parse(c.getSubBatch()));
    }

    @Bean(name = "dispatcherFeConnectionProvider", destroyMethod = "dispose")
    public ConnectionProvider dispatcherFeConnectionProvider() {
        return ConnectionProvider.builder("dispatcher-fe")
                .maxConnections(FE_MAX_CONNECTIONS_PER_HOST)
                .pendingAcquireTimeout(Duration.ofMillis(FE_PENDING_ACQUIRE_TIMEOUT_MS))
                .pendingAcquireMaxCount(FE_MAX_PENDING_ACQUIRE_PER_HOST)
                .build();
    }

    @Bean("dispatcherPassthroughWebClient")
    public WebClient dispatcherPassthroughWebClient(WebClient.Builder builder,
            @Qualifier("dispatcherFeConnectionProvider") ConnectionProvider provider) {
        HttpClient passthroughHttp = HttpClient.create(provider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS);
        return builder.clone()
                .clientConnector(new ReactorClientHttpConnector(passthroughHttp))
                .build();
    }

    // Separate from inference; Reactor Netty pools by remote address, so the limit below is per FE endpoint.
    @Bean(name = "dispatcherProbeConnectionProvider", destroyMethod = "dispose")
    public ConnectionProvider dispatcherProbeConnectionProvider() {
        return ConnectionProvider.builder("dispatcher-fe-probe")
                .maxConnections(2)
                .pendingAcquireTimeout(Duration.ofMillis(FE_PENDING_ACQUIRE_TIMEOUT_MS))
                .build();
    }

    @Bean("dispatcherProbeWebClient")
    public WebClient dispatcherProbeWebClient(WebClient.Builder builder,
            @Qualifier("dispatcherProbeConnectionProvider") ConnectionProvider probeProvider) {
        HttpClient probeHttp = HttpClient.create(probeProvider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS);
        return builder.clone()
                .clientConnector(new ReactorClientHttpConnector(probeHttp))
                .build();
    }

    @Bean
    public RouterFunction<ServerResponse> dispatcherRoutes(ConfigService configService,
                                                           ObjectProvider<DispatchRouter> router) {
        // Inspect the typed config before creating HTTP clients, discovery tasks or validating credentials.
        return configService.loadBalanceConfig().getHttpDispatcher().isEnabled()
                ? router.getObject().routes() : request -> Mono.empty();
    }

    @Bean
    public FePool fePool(DispatchConfig cfg, ServiceDiscovery discovery,
                         WorkerAddressService workerAddresses, ModelMetaConfig model,
                         @Qualifier("dispatcherProbeWebClient") WebClient probeClient,
                         DispatcherMetricsReporter metrics) {
        Callable<List<WorkerHost>> lookup = () -> discovery.getHosts(cfg.getFePoolServiceId());
        if (cfg.getFePoolServiceId().isBlank()) {
            List<RoleType> roles = model.requiredRoles().stream().filter(role -> role != RoleType.VIT).toList();
            Assert.isTrue(roles.size() == 1,
                    "set DISPATCH_FE_POOL_SERVICE_ID when the worker HTTP ingress is ambiguous");
            RoleType role = roles.getFirst();
            lookup = () -> workerAddresses.getEngineWorkerList(model.modelName(), role);
            Logger.info("dispatcher FE discovery reuses worker HTTP endpoints: model={}, role={}", model.modelName(), role);
        }
        Logger.info("dispatcher enabled: {}", JsonUtils.toString(cfg));
        return new FePool(discovery, probeClient, cfg, metrics, lookup);
    }
}
