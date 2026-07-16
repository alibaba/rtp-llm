package org.flexlb.config;

import lombok.Getter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

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

    /**
     * Memoized {@link #getConfiguredRoleTypes()} result. The route table only changes through
     * {@link #putServiceRoute}/{@link #removeServiceRoute} (startup and tests), while the union
     * is read per {@code /batch_schedule} request — recomputing the set walk per request is
     * pure waste. Cleared on every table mutation.
     */
    private static volatile List<RoleType> configuredRoleTypesCache;

    public static void putServiceRoute(String serviceId, ServiceRoute serviceRoute) {
        modelServiceRoute.put(serviceId, serviceRoute);
        configuredRoleTypesCache = null;
        if (Boolean.TRUE.equals(serviceRoute.getLoadBalance())) {
            String modelName = IdUtils.getModelNameByServiceId(serviceRoute.getServiceId());
            loadBalanceSyncModels.add(modelName);
        }
    }

    /** Removes a registered route. Lets tests undo a {@link #putServiceRoute} so the
     *  process-wide route table stays free of cross-test residue. */
    public static void removeServiceRoute(String serviceId) {
        ServiceRoute removed = modelServiceRoute.remove(serviceId);
        configuredRoleTypesCache = null;
        if (removed != null && Boolean.TRUE.equals(removed.getLoadBalance())) {
            loadBalanceSyncModels.remove(IdUtils.getModelNameByServiceId(removed.getServiceId()));
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
        List<RoleType> cached = configuredRoleTypesCache;
        if (cached != null) {
            return cached;
        }
        Set<RoleType> roleTypes = new HashSet<>();
        for (ServiceRoute serviceRoute : modelServiceRoute.values()) {
            roleTypes.addAll(serviceRoute.getAllRoleTypes());
        }
        List<RoleType> computed = List.copyOf(roleTypes);
        configuredRoleTypesCache = computed;
        return computed;
    }
}
