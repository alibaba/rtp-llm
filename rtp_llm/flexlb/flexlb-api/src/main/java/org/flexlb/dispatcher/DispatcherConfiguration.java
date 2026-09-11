package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;

/** Dispatcher configuration and isolated HTTP connection pools; enabled by the FE discovery name. */
@Configuration
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
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
        Binder.get(environment).bind("dispatch", Bindable.ofInstance(config));
        config.setTrustedRoutingToken(environment.getProperty("DISPATCH_ROUTING_TOKEN", "").trim());
        validate(config);
        return config;
    }

    private static void validate(DispatchConfig c) {
        Assert.hasText(c.getFePoolServiceId(), "DISPATCH_FE_POOL_SERVICE_ID must name the FE discovery service");
        Assert.hasText(c.getProbePath(), "dispatch.probe-path must not be blank");
        Assert.notNull(c.getFeAllocation(), "dispatch.fe-allocation must be master or local");
        Assert.isTrue(c.getBatchTimeoutMs() > 0, "dispatch.batch-timeout-ms must be > 0");
        Assert.isTrue(c.getBodyReadMarginMs() >= 0, "dispatch.body-read-margin-ms must be >= 0");
        Assert.isTrue(c.getMaxAggregateResponseBytes() > 0, "dispatch.max-aggregate-response-bytes must be > 0");
        Assert.isTrue(c.getMaxAggregateRequestBytes() > 0, "dispatch.max-aggregate-request-bytes must be > 0");
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

    // Probe traffic must not queue behind inference requests and falsely mark busy FEs dead.
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
    public RouterFunction<ServerResponse> dispatcherRoutes(DispatchRouter router) {
        return router.routes();
    }

    @Bean
    SmartInitializingSingleton dispatcherBootLog(DispatchConfig cfg) {
        return () -> Logger.warn("dispatcher enabled: {}", JsonUtils.toString(cfg));
    }
}
