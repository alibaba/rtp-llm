package org.flexlb.config;

import lombok.Getter;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class ModelMetaConfig {

    /**
     * Model metadata configuration
     */
    private static final ConcurrentHashMap<String/*serviceId*/, ServiceRoute> modelServiceRoute = new ConcurrentHashMap<>();

    @Getter
    private static final Set<String> loadBalanceSyncModels = new HashSet<>();

    public static void putServiceRoute(String serviceId, ServiceRoute serviceRoute) {
        modelServiceRoute.put(serviceId, serviceRoute);
        if (Boolean.TRUE.equals(serviceRoute.getLoadBalance())) {
            String modelName = IdUtils.getModelNameByServiceId(serviceRoute.getServiceId());
            loadBalanceSyncModels.add(modelName);
        }
    }

    public ServiceRoute getServiceRoute(String serviceId) {
        return modelServiceRoute.get(serviceId);

    }
}
