package org.flexlb.config;

import lombok.Getter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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

    /**
     * Union of role types declared by all registered service routes. Unlike the
     * runtime view in ModelWorkerStatus, this reflects deployment configuration and
     * stays stable when a role's workers are temporarily down or not yet synced.
     */
    public List<RoleType> getConfiguredRoleTypes() {
        Set<RoleType> roleTypes = new HashSet<>();
        for (ServiceRoute serviceRoute : modelServiceRoute.values()) {
            roleTypes.addAll(serviceRoute.getAllRoleTypes());
        }
        return new ArrayList<>(roleTypes);
    }
}
