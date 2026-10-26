package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class GroupRoleEndPoint {
    @JsonProperty("group")
    private String group;

    @JsonProperty("prefill_endpoint")
    private Endpoint prefillEndpoint;

    @JsonProperty("decode_endpoint")
    private Endpoint decodeEndpoint;

    @JsonProperty("vit_endpoint")
    private Endpoint vitEndpoint;

    @JsonProperty("pd_fusion_endpoint")
    private Endpoint pdFusionEndpoint;

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
        if (pdFusionEndpoint != null) {
            roleTypes.add(RoleType.PDFUSION);
        }
        return roleTypes;
    }

    List<Endpoint> getAllEndpoints() {
        List<Endpoint> endpoints = new ArrayList<>();
        if (prefillEndpoint != null) {
            endpoints.add(prefillEndpoint);
        }
        if (decodeEndpoint != null) {
            endpoints.add(decodeEndpoint);
        }
        if (vitEndpoint != null) {
            endpoints.add(vitEndpoint);
        }
        if (pdFusionEndpoint != null) {
            endpoints.add(pdFusionEndpoint);
        }
        return endpoints;
    }

    Endpoint getRoleEndpoint(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoint;
        } else if (roleType == RoleType.DECODE) {
            return decodeEndpoint;
        } else if (roleType == RoleType.VIT) {
            return vitEndpoint;
        } else if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoint;
        }
        return null;
    }
}
