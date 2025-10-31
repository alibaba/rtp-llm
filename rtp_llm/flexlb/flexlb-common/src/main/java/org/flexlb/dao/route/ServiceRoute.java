package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class ServiceRoute {

    @JsonProperty("service_id")
    private String serviceId;

    @JsonProperty("role_endpoints")
    private List<GroupRoleEndPoint> roleEndpoints = new ArrayList<>();

    @JsonProperty("load_balance")
    private Boolean loadBalance;

    public List<Pair<String, Endpoint>> getAllEndpointsWithGroup(RoleType roleType) {
        List<Pair<String, Endpoint>> endpoints = new ArrayList<>();
        for (GroupRoleEndPoint groupRoleEndPoint : roleEndpoints) {
            Endpoint endpoint = groupRoleEndPoint.getRoleEndpoint(roleType);
            endpoints.add(Pair.of(groupRoleEndPoint.getGroup(), endpoint));
        }
        return endpoints;
    }

    public List<Endpoint> getAllEndpoints() {
        List<Endpoint> allEndpoints = new ArrayList<>();
        for (GroupRoleEndPoint groupRoleEndPoint : roleEndpoints) {
            List<Endpoint> endpointList = groupRoleEndPoint.getAllEndpoints();
            allEndpoints.addAll(endpointList);
        }
        return allEndpoints;
    }

    public List<Endpoint> getRoleEndpoints(RoleType roleType) {
        List<Endpoint> returnRoleEndpoints = new ArrayList<>();
        roleEndpoints.stream()
                .map(groupRoleEndPoint -> groupRoleEndPoint.getRoleEndpoint(roleType))
                .filter(Objects::nonNull)
                .forEach(returnRoleEndpoints::add);
        return returnRoleEndpoints;
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
