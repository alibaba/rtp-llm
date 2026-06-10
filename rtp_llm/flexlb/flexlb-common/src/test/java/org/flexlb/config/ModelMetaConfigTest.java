package org.flexlb.config;

import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelMetaConfigTest {

    private ServiceRoute routeWithRoles(String serviceId, boolean prefill, boolean decode, boolean pdFusion) {
        GroupRoleEndPoint group = new GroupRoleEndPoint();
        group.setGroup("g1");
        if (prefill) {
            group.setPrefillEndpoint(new Endpoint());
        }
        if (decode) {
            group.setDecodeEndpoint(new Endpoint());
        }
        if (pdFusion) {
            group.setPdFusionEndpoint(new Endpoint());
        }
        ServiceRoute route = new ServiceRoute();
        route.setServiceId(serviceId);
        route.setRoleEndpoints(List.of(group));
        return route;
    }

    @Test
    void getConfiguredRoleTypes_returns_union_of_registered_routes() {
        ModelMetaConfig.putServiceRoute("model_meta_cfg_test.pd.service",
                routeWithRoles("model_meta_cfg_test.pd.service", true, true, false));

        ModelMetaConfig config = new ModelMetaConfig();
        List<RoleType> roles = config.getConfiguredRoleTypes();

        assertTrue(roles.contains(RoleType.PREFILL), "configured PREFILL must be reported: " + roles);
        assertTrue(roles.contains(RoleType.DECODE), "configured DECODE must be reported: " + roles);
    }
}
