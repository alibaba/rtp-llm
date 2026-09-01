package org.flexlb.config;

import lombok.Getter;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class ModelMetaConfig {

    /**
     * Model metadata configuration
     */
    private static final ConcurrentHashMap<String/*serviceId*/, ServiceRoute> modelServiceRoute = new ConcurrentHashMap<>();

    /** Concurrent: mutated through the route-table entry points while readers iterate it. */
    @Getter
    private static final Set<String> loadBalanceSyncModels = ConcurrentHashMap.newKeySet();

    /**
     * Memoized {@link #getConfiguredRoleTypes()} result, tagged with the route-table version it
     * was computed against. The route table only changes through
     * {@link #putServiceRoute}/{@link #removeServiceRoute} (startup and tests), while the union
     * is read per {@code /batch_schedule} request — recomputing the set walk per request is
     * pure waste. The version tag keeps a computation that raced a table mutation from being
     * published: it is only valid while the version it snapshotted is still current.
     */
    private static volatile VersionedRoleTypes configuredRoleTypesCache;

    /** Bumped on every route-table mutation; see {@link #configuredRoleTypesCache}. */
    private static final AtomicLong routeTableVersion = new AtomicLong();

    private record VersionedRoleTypes(long version, List<RoleType> roleTypes) {
    }

    public static void putServiceRoute(String serviceId, ServiceRoute serviceRoute) {
        modelServiceRoute.put(serviceId, serviceRoute);
        routeTableVersion.incrementAndGet();
        if (Boolean.TRUE.equals(serviceRoute.getLoadBalance())) {
            String modelName = IdUtils.getModelNameByServiceId(serviceRoute.getServiceId());
            loadBalanceSyncModels.add(modelName);
        }
    }

    /** Removes a registered route. Lets tests undo a {@link #putServiceRoute} so the
     *  process-wide route table stays free of cross-test residue. */
    public static void removeServiceRoute(String serviceId) {
        ServiceRoute removed = modelServiceRoute.remove(serviceId);
        routeTableVersion.incrementAndGet();
        if (removed != null && Boolean.TRUE.equals(removed.getLoadBalance())) {
            String modelName = IdUtils.getModelNameByServiceId(removed.getServiceId());
            // Several serviceIds can map to one modelName, so the model stays in the sync set until
            // the last load-balanced route referencing it is gone — dropping it on the first removal
            // would stop syncing a model other live routes still serve.
            boolean stillReferenced = modelServiceRoute.values().stream()
                    .filter(route -> Boolean.TRUE.equals(route.getLoadBalance()))
                    .anyMatch(route -> modelName.equals(IdUtils.getModelNameByServiceId(route.getServiceId())));
            if (!stillReferenced) {
                loadBalanceSyncModels.remove(modelName);
            }
        }
    }

    public ServiceRoute getServiceRoute(String serviceId) {
        return modelServiceRoute.get(serviceId);

    }

    /**
     * Unique service-discovery addresses referenced by the registered route table. This is a
     * diagnostics view, sorted so error messages and tests remain deterministic.
     */
    public List<String> getConfiguredDiscoveryAddresses() {
        Set<String> addresses = new TreeSet<>();
        for (ServiceRoute serviceRoute : modelServiceRoute.values()) {
            for (Endpoint endpoint : serviceRoute.getAllEndpoints()) {
                if (endpoint != null && StringUtils.isNotBlank(endpoint.getAddress())) {
                    addresses.add(endpoint.getAddress());
                }
            }
        }
        return List.copyOf(addresses);
    }

    /**
     * Union of role types declared by all registered service routes. Unlike the
     * runtime view in ModelWorkerStatus, this reflects deployment configuration and
     * stays stable when a role's workers are temporarily down or not yet synced.
     */
    public List<RoleType> getConfiguredRoleTypes() {
        long version = routeTableVersion.get();
        VersionedRoleTypes cached = configuredRoleTypesCache;
        if (cached != null && cached.version() == version) {
            return cached.roleTypes();
        }
        Set<RoleType> roleTypes = new HashSet<>();
        for (ServiceRoute serviceRoute : modelServiceRoute.values()) {
            roleTypes.addAll(serviceRoute.getAllRoleTypes());
        }
        List<RoleType> computed = List.copyOf(roleTypes);
        if (routeTableVersion.get() == version) {
            configuredRoleTypesCache = new VersionedRoleTypes(version, computed);
        }
        return computed;
    }
}
