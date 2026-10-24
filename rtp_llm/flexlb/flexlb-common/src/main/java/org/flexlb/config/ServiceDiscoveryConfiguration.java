package org.flexlb.config;

import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.LocalServiceDiscovery;
import org.flexlb.discovery.ServiceDiscovery;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static com.google.common.base.Preconditions.checkArgument;

@Configuration
public class ServiceDiscoveryConfiguration {

    @Bean
    @ConditionalOnMissingBean(ServiceDiscovery.class)
    public static ServiceDiscovery serviceDiscovery(ModelMetaConfig modelConfig) {
        ServiceRoute route = modelConfig.getServiceRoute();
        String file = route.getDiscoveryFile();
        if (file == null || file.isBlank()) {
            return new LocalServiceDiscovery(route.getHosts());
        }
        checkArgument(route.getHosts().isEmpty(), "MODEL_SERVICE_CONFIG must use either hosts or discovery_file");
        return new LocalServiceDiscovery(file);
    }
}
