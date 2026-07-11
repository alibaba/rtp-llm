package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.Endpoint;
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
    public ModelMetaConfig modelMetaConfig(
            Environment environment,
            RoutingServiceDiscovery serviceDiscovery) {
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

    private void validateServiceRoute(
            ServiceRoute serviceRoute,
            RoutingServiceDiscovery serviceDiscovery) {
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
    }
}
