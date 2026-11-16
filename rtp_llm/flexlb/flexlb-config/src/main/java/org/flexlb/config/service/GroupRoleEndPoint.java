package org.flexlb.config.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * Associates a discovery endpoint with its routing group and worker role.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class GroupRoleEndPoint {
    @JsonProperty("group")
    private String group;

    @JsonProperty("prefill_endpoint")
    private WorkerDiscoveryEndpoint prefillEndpoint;

    @JsonProperty("decode_endpoint")
    private WorkerDiscoveryEndpoint decodeEndpoint;

    @JsonProperty("vit_endpoint")
    private WorkerDiscoveryEndpoint vitEndpoint;

    @JsonProperty("encoder_endpoint")
    private WorkerDiscoveryEndpoint encoderEndpoint;

    @JsonProperty("pd_fusion_endpoint")
    private WorkerDiscoveryEndpoint pdFusionEndpoint;

    public WorkerDiscoveryEndpoint getRoleEndpoint(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoint;
        } else if (roleType == RoleType.DECODE) {
            return decodeEndpoint;
        } else if (roleType == RoleType.VIT) {
            return vitEndpoint;
        } else if (roleType == RoleType.ENCODER) {
            return encoderEndpoint;
        } else if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoint;
        }
        return null;
    }

    void applyGroupToEndpoints() {
        String endpointGroup = group == null ? "" : group;
        for (WorkerDiscoveryEndpoint endpoint : getAllEndpoints()) {
            endpoint.setGroup(endpointGroup);
        }
    }

    List<WorkerDiscoveryEndpoint> getAllEndpoints() {
        List<WorkerDiscoveryEndpoint> endpoints = new ArrayList<>();
        if (prefillEndpoint != null) {
            endpoints.add(prefillEndpoint);
        }
        if (decodeEndpoint != null) {
            endpoints.add(decodeEndpoint);
        }
        if (vitEndpoint != null) {
            endpoints.add(vitEndpoint);
        }
        if (encoderEndpoint != null) {
            endpoints.add(encoderEndpoint);
        }
        if (pdFusionEndpoint != null) {
            endpoints.add(pdFusionEndpoint);
        }
        return endpoints;
    }

    List<RoleType> getRoleTypes() {
        List<RoleType> roleTypes = new ArrayList<>();
        if (prefillEndpoint != null) {
            roleTypes.add(RoleType.PREFILL);
        }
        if (decodeEndpoint != null) {
            roleTypes.add(RoleType.DECODE);
        }
        if (vitEndpoint != null) {
            roleTypes.add(RoleType.VIT);
        }
        if (encoderEndpoint != null) {
            roleTypes.add(RoleType.ENCODER);
        }
        if (pdFusionEndpoint != null) {
            roleTypes.add(RoleType.PDFUSION);
        }
        return roleTypes;
    }
}
