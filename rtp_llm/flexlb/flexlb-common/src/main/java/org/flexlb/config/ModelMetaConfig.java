package org.flexlb.config;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.constant.CommonConstants;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import java.util.EnumSet;
import java.util.List;
import java.util.Set;

@Component
public class ModelMetaConfig {

    private static final List<RoleType> ROUTING_ORDER = List.of(
            RoleType.PDFUSION,
            RoleType.DECODE,
            RoleType.PREFILL,
            RoleType.VIT);

    private final ServiceRoute serviceRoute;
    private final String modelName;
    private final List<RoleType> requiredRoles;

    public ModelMetaConfig() {
        String document = System.getenv("MODEL_SERVICE_CONFIG");
        if (document == null || document.isBlank()) {
            throw new IllegalStateException(
                    "master load balancer env MODEL_SERVICE_CONFIG is empty");
        }
        ServiceRoute parsed = JsonUtils.toObject(document, ServiceRoute.class);
        if (parsed.getServiceId() == null || parsed.getServiceId().isBlank()) {
            throw new IllegalStateException(
                    "MODEL_SERVICE_CONFIG must declare service_id");
        }
        String servicePrefix = CommonConstants.FUNCTION + ".";
        if (!parsed.getServiceId().startsWith(servicePrefix)
                || parsed.getServiceId().length() == servicePrefix.length()) {
            throw new IllegalStateException(
                    "MODEL_SERVICE_CONFIG service_id must identify one model");
        }
        List<RoleType> parsedRoles = parsed.getAllRoleTypes();
        Set<RoleType> configured = parsedRoles.isEmpty()
                ? EnumSet.noneOf(RoleType.class)
                : EnumSet.copyOf(parsedRoles);
        List<RoleType> roles = ROUTING_ORDER.stream()
                .filter(configured::contains)
                .toList();
        if (roles.isEmpty()) {
            throw new IllegalStateException(
                    "MODEL_SERVICE_CONFIG must declare at least one routable role");
        }
        this.serviceRoute = parsed;
        this.modelName = IdUtils.getModelNameByServiceId(parsed.getServiceId());
        this.requiredRoles = roles;
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
        if (!modelName.equals(requestedModelName)
                || !requiredRoles.contains(role)) {
            return List.of();
        }
        return serviceRoute.getAllEndpointsWithGroup(role);
    }
}
