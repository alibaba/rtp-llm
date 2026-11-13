package org.flexlb.config.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.cache.KvcmConfig;
import org.flexlb.util.discovery.DiscoveryEndpoint;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Groups a model service's role endpoints and exposes their discovery topology.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class ServiceRoute {

    @JsonProperty("service_id")
    private String serviceId;

    private List<GroupRoleEndPoint> roleEndpoints = new ArrayList<>();

    @JsonProperty("kvcm")
    private KvcmConfig kvcm;

    @JsonProperty("optimizer")
    private OptimizerConfig optimizer;

    @JsonProperty("hosts")
    private Map<String, List<String>> hosts = Map.of();

    @JsonProperty("discovery_file")
    private String discoveryFile;

    @JsonProperty("load_balance")
    private Boolean loadBalance;

    @JsonProperty("role_endpoints")
    public void setRoleEndpoints(List<GroupRoleEndPoint> roleEndpoints) {
        this.roleEndpoints = roleEndpoints == null ? new ArrayList<>() : roleEndpoints;
        for (GroupRoleEndPoint roleEndpoint : this.roleEndpoints) {
            if (roleEndpoint != null) {
                roleEndpoint.applyGroupToEndpoints();
            }
        }
    }

    public List<Pair<String, DiscoveryEndpoint>> getAllEndpointsWithGroup(RoleType roleType) {
        List<Pair<String, DiscoveryEndpoint>> endpoints = new ArrayList<>();
        for (GroupRoleEndPoint groupRoleEndPoint : roleEndpoints) {
            DiscoveryEndpoint endpoint = groupRoleEndPoint.getRoleEndpoint(roleType);
            endpoints.add(Pair.of(groupRoleEndPoint.getGroup(), endpoint));
        }
        return endpoints;
    }

    public List<DiscoveryEndpoint> getAllEndpoints() {
        List<DiscoveryEndpoint> allEndpoints = new ArrayList<>();
        for (GroupRoleEndPoint groupRoleEndPoint : roleEndpoints) {
            List<DiscoveryEndpoint> endpointList = groupRoleEndPoint.getAllEndpoints();
            allEndpoints.addAll(endpointList);
        }
        return allEndpoints;
    }

    public List<RoleType> getAllRoleTypes() {
        Set<RoleType> roleTypes = new HashSet<>();
        roleEndpoints.stream()
                .map(GroupRoleEndPoint::getRoleTypes)
                .flatMap(List::stream)
                .forEach(roleTypes::add);
        return new ArrayList<>(roleTypes);
    }
}
