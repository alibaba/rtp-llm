package org.flexlb.engine.grpc.client;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Resolves KVCM namespaces from explicit configuration or engine deployment metadata.
 */
@Slf4j
@Component
public class KvcmNamespaceResolver {

    private final ServiceRoute serviceRoute;
    private final String configuredNamespace;
    private final RoutingServiceDiscovery serviceDiscovery;
    private final ConcurrentMap<String, ConcurrentMap<RoleType, String>> namespaceByGroupAndRole =
            new ConcurrentHashMap<>();

    public KvcmNamespaceResolver(
            ModelMetaConfig modelMetaConfig,
            RoutingServiceDiscovery serviceDiscovery) {
        this.serviceRoute = modelMetaConfig.getServiceRoutes().stream().findFirst().orElse(null);
        KvcmConfig config = serviceRoute != null ? serviceRoute.getKvcm() : null;
        this.configuredNamespace = config != null && config.isEnabled()
                ? StringUtils.trimToNull(config.getNamespace())
                : null;
        this.serviceDiscovery = serviceDiscovery;
    }

    public String resolve(RoleType roleType, String group) {
        if (configuredNamespace != null) {
            return configuredNamespace;
        }
        ConcurrentMap<RoleType, String> namespaceByRole =
                namespaceByGroupAndRole.get(StringUtils.defaultString(group));
        return namespaceByRole == null ? null : namespaceByRole.get(roleType);
    }

    public boolean usesConfiguredNamespace() {
        return configuredNamespace != null;
    }

    public void refresh() {
        if (serviceRoute == null || configuredNamespace != null) {
            return;
        }

        boolean changed = false;
        for (GroupRoleEndPoint roleEndpoints : serviceRoute.getRoleEndpoints()) {
            String group = StringUtils.defaultString(roleEndpoints.getGroup());
            for (RoleType roleType : RoleType.values()) {
                Endpoint endpoint = roleEndpoints.getRoleEndpoint(roleType);
                if (endpoint == null) {
                    continue;
                }
                List<WorkerHost> hosts = serviceDiscovery.getHosts(endpoint);
                if (hosts.isEmpty()) {
                    continue;
                }

                String namespace = hosts.getFirst().getDeploymentName();
                if (StringUtils.isBlank(namespace)) {
                    continue;
                }

                ConcurrentMap<RoleType, String> namespaceByRole =
                        namespaceByGroupAndRole.computeIfAbsent(group, ignored -> new ConcurrentHashMap<>());
                String currentNamespace = namespaceByRole.get(roleType);
                if (!namespace.equals(currentNamespace)) {
                    namespaceByRole.put(roleType, namespace);
                    changed = true;
                }
            }
        }

        if (changed) {
            log.info("Updated KVCM role namespaces: {}", namespaceByGroupAndRole);
        }
    }
}
