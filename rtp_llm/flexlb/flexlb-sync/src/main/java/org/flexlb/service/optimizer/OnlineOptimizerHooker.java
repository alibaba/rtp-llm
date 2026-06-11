package org.flexlb.service.optimizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.OnlineOptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.listener.AppShutDownHooker;
import org.flexlb.transport.GeneralHttpNettyService;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class OnlineOptimizerHooker implements AppOnlineHooker, AppShutDownHooker {

    private final GeneralHttpNettyService httpService;
    private final ServiceDiscovery serviceDiscovery;
    private final OnlineOptimizerConfig optimizerConfig;

    @Getter
    private final boolean enabled;

    @Getter
    private volatile OnlineOptimizerClient client;

    // Guard against duplicate afterStartUp invocations (e.g. sidecar retry on /hook/after_start)
    // to prevent duplicate discovery refresh and registration schedulers.
    private final Object lifecycleLock = new Object();
    private boolean started;
    private boolean stopped;

    public OnlineOptimizerHooker(
            GeneralHttpNettyService httpService,
            ServiceDiscovery serviceDiscovery,
            ModelMetaConfig modelMetaConfig) {

        this.httpService = httpService;
        this.serviceDiscovery = serviceDiscovery;
        this.optimizerConfig = resolveOnlineOptimizerConfig(modelMetaConfig);
        this.enabled = optimizerConfig != null;
        if (!enabled) {
            log.info("OnlineOptimizer disabled by MODEL_SERVICE_CONFIG");
            return;
        }

        Endpoint endpoint = optimizerConfig.toEndpoint();
        log.info("OnlineOptimizer enabled: instanceGroup={}, instanceId={}, address={}",
                optimizerConfig.getInstanceGroup(), optimizerConfig.getInstanceId(),
                endpoint.getDiscovery().getType().getValue() + ":" + endpoint.getAddress());
    }

    @Override
    public void afterStartUp() {
        if (!enabled) return;
        synchronized (lifecycleLock) {
            if (stopped) {
                log.info("OnlineOptimizer already stopped, skip late afterStartUp invocation");
                return;
            }
            if (started) {
                log.info("OnlineOptimizer afterStartUp already executed, skip duplicate invocation");
                return;
            }
            started = true;

            OptimizerInstanceParams params = optimizerConfig.toInstanceParams();

            // Resolver startup and all discovery I/O run on the client's registration scheduler.
            OptimizerAddressResolver resolver = createAddressResolver();
            OnlineOptimizerClient newClient =
                    new OnlineOptimizerClient(
                            httpService,
                            resolver,
                            optimizerConfig.getInstanceGroup(),
                            optimizerConfig.getPath(),
                            optimizerConfig.getRegisterTimeoutMs());
            this.client = newClient;

            log.info("OnlineOptimizer registration params: blockSize={}, linearStep={}, "
                            + "optimizerStateInfo={}, locationSpecInfos={}, locationSpecGroups={}",
                    params.getBlockSize(), params.getLinearStep(), params.getOptimizerStateInfo(),
                    params.getLocationSpecInfos(), params.getLocationSpecGroups());
            newClient.startRegistrationAsync(optimizerConfig.getInstanceId(), params);
            log.info("OnlineOptimizer registration submitted (async): instanceId={}",
                    optimizerConfig.getInstanceId());
        }
    }

    @Override
    public void beforeShutdown() {
        OnlineOptimizerClient currentClient;
        synchronized (lifecycleLock) {
            stopped = true;
            currentClient = client;
        }
        if (currentClient == null) {
            log.info("OnlineOptimizer beforeShutdown: client not initialized, nothing to shutdown");
            return;
        }
        log.info("OnlineOptimizer shutting down: instanceId={}", optimizerConfig.getInstanceId());
        currentClient.shutdown();
        log.info("OnlineOptimizer shutdown completed");
    }

    @Override
    public int priority() {
        return 0;
    }

    private OptimizerAddressResolver createAddressResolver() {
        // Defer start() to the client's async retry; afterStartUp never waits for discovery I/O.
        Endpoint endpoint = optimizerConfig.toEndpoint();
        log.info("OnlineOptimizer using ServiceDiscoveryAddressResolver, type={}, address={}",
                endpoint.getDiscovery().getType().getValue(), endpoint.getAddress());
        return new ServiceDiscoveryAddressResolver(serviceDiscovery, endpoint);
    }

    private static OnlineOptimizerConfig resolveOnlineOptimizerConfig(ModelMetaConfig modelMetaConfig) {
        for (ServiceRoute route : modelMetaConfig.getServiceRoutes()) {
            if (route != null && route.isOnlineOptimizerEnabled()) {
                return route.getOnlineOptimizer();
            }
        }
        return null;
    }
}
