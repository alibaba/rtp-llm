package org.flexlb.config;

import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelMetaConfigTest {

    private static final String TEST_SERVICE_ID = "model_meta_cfg_test.pd.service";

    @AfterEach
    void tearDown() {
        ModelMetaConfig.removeServiceRoute(TEST_SERVICE_ID);
    }

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
        ModelMetaConfig.putServiceRoute(TEST_SERVICE_ID,
                routeWithRoles(TEST_SERVICE_ID, true, true, false));

        ModelMetaConfig config = new ModelMetaConfig();
        List<RoleType> roles = config.getConfiguredRoleTypes();

        assertTrue(roles.contains(RoleType.PREFILL), "configured PREFILL must be reported: " + roles);
        assertTrue(roles.contains(RoleType.DECODE), "configured DECODE must be reported: " + roles);
    }
}
