package org.flexlb.config;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.constant.CommonConstants;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.IdUtils;

import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class ModelMetaConfig {

    private static final List<RoleType> ROUTING_ORDER = List.of(
            RoleType.PDFUSION,
            RoleType.DECODE,
            RoleType.PREFILL,
            RoleType.VIT);

    private final ConcurrentHashMap<String, ServiceRoute> modelServiceRoute =
            new ConcurrentHashMap<>();

    private volatile ServiceRoute primaryRoute;
    private volatile String modelName = "";
    private volatile List<RoleType> requiredRoles = List.of();

    public void putServiceRoute(String serviceId, ServiceRoute serviceRoute) {
        if (serviceId == null || serviceId.isBlank()) {
            throw new IllegalArgumentException("serviceId must not be blank");
        }
        if (serviceRoute == null) {
            throw new IllegalArgumentException("serviceRoute must not be null");
        }
        modelServiceRoute.put(serviceId, serviceRoute);
        if (primaryRoute == null || serviceId.equals(primaryRoute.getServiceId())) {
            setPrimaryRoute(serviceRoute);
        }
    }

    public ServiceRoute getServiceRoute(String serviceId) {
        return modelServiceRoute.get(serviceId);
    }

    public Collection<ServiceRoute> getServiceRoutes() {
        return List.copyOf(modelServiceRoute.values());
    }

    private void setPrimaryRoute(ServiceRoute serviceRoute) {
        this.primaryRoute = serviceRoute;
        this.modelName = modelNameFromServiceId(serviceRoute.getServiceId());
        this.requiredRoles = resolveRequiredRoles(serviceRoute);
    }

    private static List<RoleType> resolveRequiredRoles(ServiceRoute serviceRoute) {
        if (serviceRoute == null) {
            return List.of();
        }
        List<RoleType> parsedRoles = serviceRoute.getAllRoleTypes();
        Set<RoleType> configured = parsedRoles.isEmpty()
                ? EnumSet.noneOf(RoleType.class)
                : EnumSet.copyOf(parsedRoles);
        return ROUTING_ORDER.stream()
                .filter(configured::contains)
                .toList();
    }

    private static String modelNameFromServiceId(String serviceId) {
        if (serviceId == null) {
            return "";
        }
        String servicePrefix = CommonConstants.FUNCTION + ".";
        if (serviceId.startsWith(servicePrefix)
                && serviceId.length() > servicePrefix.length()) {
            return IdUtils.getModelNameByServiceId(serviceId);
        }
        return serviceId;
    }

    /** Immutable request topology; live endpoint occupancy never changes it. */
    public List<RoleType> requiredRoles() {
        return requiredRoles;
    }

    public String modelName() {
        return modelName;
    }

    /** Return a fresh structural list for the single configured service. */
    public List<Pair<String, Endpoint>> endpointsWithGroup(
            String requestedModelName,
            RoleType role) {
        if (requestedModelName == null || role == null) {
            return Collections.emptyList();
        }
        for (Map.Entry<String, ServiceRoute> entry : modelServiceRoute.entrySet()) {
            ServiceRoute route = entry.getValue();
            if (route == null) {
                continue;
            }
            String routeServiceId = route.getServiceId();
            String routeModelName = modelNameFromServiceId(routeServiceId);
            if ((requestedModelName.equals(routeModelName)
                    || requestedModelName.equals(routeServiceId)
                    || requestedModelName.equals(entry.getKey()))
                    && route.getAllRoleTypes().contains(role)) {
                return route.getAllEndpointsWithGroup(role);
            }
        }
        return Collections.emptyList();
    }
}
