package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.OptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.util.JsonUtils;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

/**
 * Loads and validates the model routing configuration once during startup.
 */
@Slf4j
@Configuration
public class ModelServiceConfiguration {

    private static final String MODEL_SERVICE_CONFIG = "MODEL_SERVICE_CONFIG";

    @Bean
    public ModelMetaConfig modelMetaConfig(Environment environment, RoutingServiceDiscovery serviceDiscovery) {
        String modelConfigJson = environment.getProperty(MODEL_SERVICE_CONFIG);
        if (StringUtils.isBlank(modelConfigJson)) {
            throw new IllegalStateException(MODEL_SERVICE_CONFIG + " must not be blank");
        }

        ServiceRoute serviceRoute = JsonUtils.toObject(modelConfigJson, ServiceRoute.class);
        validateServiceRoute(serviceRoute, serviceDiscovery);

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        log.info("Loaded model service route: serviceId={}, endpoints={}",
                serviceRoute.getServiceId(), serviceRoute.getAllEndpoints().size());
        return modelMetaConfig;
    }

    private void validateServiceRoute(ServiceRoute serviceRoute, RoutingServiceDiscovery serviceDiscovery) {
        if (StringUtils.isBlank(serviceRoute.getServiceId())) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG service_id must not be blank");
        }
        if (CollectionUtils.isEmpty(serviceRoute.getRoleEndpoints())) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG role_endpoints must not be empty");
        }

        var endpoints = serviceRoute.getAllEndpoints();
        if (CollectionUtils.isEmpty(endpoints)) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG must contain at least one role endpoint");
        }
        for (Endpoint endpoint : endpoints) {
            serviceDiscovery.validate(endpoint);
        }

        validateKvcm(serviceRoute.getKvcm(), serviceDiscovery);
        validateOptimizer(serviceRoute.getOptimizer(), serviceDiscovery);
    }

    private void validateKvcm(KvcmConfig kvcm, RoutingServiceDiscovery serviceDiscovery) {
        if (kvcm == null || !kvcm.isEnabled()) {
            return;
        }
        if (kvcm.getRequestTimeoutMs() <= 0 || kvcm.getLeaderRefreshIntervalMs() <= 0) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG kvcm timeouts must be greater than zero");
        }
        if (kvcm.getHeartbeatFailureThreshold() <= 0
                || kvcm.getQueryFailureThreshold() <= 0
                || kvcm.getRecoverySuccessThreshold() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG kvcm health thresholds must be greater than zero");
        }
        validateLocalStandby(kvcm.getLocalStandby());
        serviceDiscovery.validate(kvcm.toEndpoint());
    }

    private void validateLocalStandby(LocalStandbyConfig localStandby) {
        if (localStandby == null) {
            return;
        }
        if (localStandby.getBlockSize() < 0
                || localStandby.getBlockSize() > Integer.MAX_VALUE
                || localStandby.getTtlMs() <= 0
                || localStandby.getMinimumTtlMs() <= 0
                || localStandby.getMinimumTtlMs() > localStandby.getTtlMs()
                || !Double.isFinite(localStandby.getTtlReductionStartRatio())
                || localStandby.getTtlReductionStartRatio() <= 0
                || localStandby.getTtlReductionStartRatio() >= 1
                || localStandby.getMaximumEntries() <= 0
                || !Double.isFinite(localStandby.getCapacityMultiplier())
                || localStandby.getCapacityMultiplier() < 1.0
                || localStandby.getAsyncQueueCapacity() <= 0
                || localStandby.getHashThreadCount() <= 0
                || localStandby.getHashQueueCapacity() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG kvcm.local_standby contains invalid values");
        }
    }

    private void validateOptimizer(
            OptimizerConfig optimizer,
            RoutingServiceDiscovery serviceDiscovery) {
        if (optimizer == null || !optimizer.isEnabled()) {
            return;
        }
        if (StringUtils.isBlank(optimizer.getPath())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.path must not be blank");
        }
        if (!optimizer.getPath().startsWith("/")) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.path must start with '/'");
        }
        if (optimizer.getPath().indexOf('?') >= 0 || optimizer.getPath().indexOf('#') >= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.path must not contain query or fragment");
        }
        if (optimizer.getDiscovery() != null
                && optimizer.getDiscovery().getPollIntervalMs() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.discovery.poll_interval_ms "
                            + "must be greater than zero");
        }
        serviceDiscovery.validate(optimizer.toEndpoint());
    }
}
