package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.OptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Loads and validates the model routing configuration once during startup.
 */
@Slf4j
@Configuration
public class ModelServiceConfiguration {

    private static final String MODEL_SERVICE_CONFIG = "MODEL_SERVICE_CONFIG";

    @Bean
    public ModelMetaConfig modelMetaConfig(ConfigService configService, RoutingServiceDiscovery serviceDiscovery) {
        ServiceRoute serviceRoute = configService.loadModelServiceConfig();
        if (serviceRoute == null) {
            throw new IllegalStateException(MODEL_SERVICE_CONFIG + " must not be blank");
        }

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
        if (kvcm == null) {
            return;
        }
        serviceDiscovery.validate(kvcm.toEndpoint());
    }

    private void validateOptimizer(
            OptimizerConfig optimizer,
            RoutingServiceDiscovery serviceDiscovery) {
        if (optimizer == null) {
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
        serviceDiscovery.validate(optimizer.toEndpoint());
    }
}
