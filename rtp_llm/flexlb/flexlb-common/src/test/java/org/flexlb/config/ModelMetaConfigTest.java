package org.flexlb.config;

import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
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

    @Test
    void getConfiguredRoleTypes_sees_a_route_replacement_after_a_cached_read() {
        ModelMetaConfig config = new ModelMetaConfig();
        ModelMetaConfig.putServiceRoute(TEST_SERVICE_ID,
                routeWithRoles(TEST_SERVICE_ID, true, false, false));
        assertTrue(config.getConfiguredRoleTypes().contains(RoleType.PREFILL));

        // The first read populated the cache; replacing the route must invalidate it.
        ModelMetaConfig.putServiceRoute(TEST_SERVICE_ID,
                routeWithRoles(TEST_SERVICE_ID, false, true, false));
        List<RoleType> roles = config.getConfiguredRoleTypes();

        assertTrue(roles.contains(RoleType.DECODE), "the replacement's roles must be visible: " + roles);
        assertFalse(roles.contains(RoleType.PREFILL),
                "the replaced route's roles must not survive in the cached union: " + roles);
    }

    @Test
    void getConfiguredRoleTypes_sees_a_removal_after_a_cached_read() {
        ModelMetaConfig config = new ModelMetaConfig();
        ModelMetaConfig.putServiceRoute(TEST_SERVICE_ID,
                routeWithRoles(TEST_SERVICE_ID, false, false, true));
        assertTrue(config.getConfiguredRoleTypes().contains(RoleType.PDFUSION));

        ModelMetaConfig.removeServiceRoute(TEST_SERVICE_ID);

        assertFalse(config.getConfiguredRoleTypes().contains(RoleType.PDFUSION),
                "a removed route's roles must not survive in the cached union");
    }
}
