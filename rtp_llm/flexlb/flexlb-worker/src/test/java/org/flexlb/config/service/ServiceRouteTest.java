package org.flexlb.config.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.cache.KvcmConfig;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.util.discovery.DiscoverySettings;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

/**
 * Checks model service route configuration and identity.
 */
class ServiceRouteTest {
    private static final String ROUTE_JSON =
            """
            {
              "service_id": "aigc.text-generation.generation.engine_service",
              "prefill_endpoint": {
                "address": "com.aicheng.whale.pre.deepseek_dp_tp_test",
                "protocol": "http",
                "path": "/"
              },
              "decode_endpoint": {
                "address": "com.aicheng.whale.pre.test_pd_gang2.decode",
                "protocol": "http",
                "path": "/"
              },
              "role_endpoints": [{
                "group": "ea119_PPU_ZW810E_16TP_decode64",
                "prefill_endpoint": {
                  "address": "com.aicheng.whale.pre.deepseek_dp_tp_test",
                  "protocol": "http",
                  "path": "/"
                },
                "decode_endpoint": {
                  "address": "com.aicheng.whale.pre.test_pd_gang2.decode",
                  "protocol": "http",
                  "path": "/"
                }
              }]
            }
            """;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void testConfigLoader() throws Exception {
        assertStandardRoute(objectMapper.readValue(ROUTE_JSON, ServiceRoute.class));
    }

    @Test
    void serializesRoleEndpointsUnderItsWireName() throws Exception {
        ServiceRoute route = objectMapper.readValue(ROUTE_JSON, ServiceRoute.class);

        var json = objectMapper.valueToTree(route);

        Assertions.assertTrue(json.has("role_endpoints"));
        Assertions.assertFalse(json.has("roleEndpoints"));
        assertStandardRoute(objectMapper.treeToValue(json, ServiceRoute.class));
    }

    @Test
    void testConfigLoaderList() {
        ServiceRoute[] serviceRoutes =
                JsonUtils.toObject("[" + ROUTE_JSON + "]", ServiceRoute[].class);
        assertStandardRoute(serviceRoutes[0]);
    }

    @Test
    void should_load_pd_fusion_and_vit_endpoints() throws Exception {
        String json =
                """
                {
                  "service_id": "test.service",
                  "role_endpoints": [{
                    "group": "g1",
                    "pd_fusion_endpoint": {"address": "pd", "protocol": "http", "path": "/"},
                    "vit_endpoint": {"address": "vit", "protocol": "http", "path": "/"}
                  }]
                }
                """;

        ServiceRoute serviceRoute = objectMapper.readValue(json, ServiceRoute.class);

        Assertions.assertTrue(
                serviceRoute
                        .getAllRoleTypes()
                        .containsAll(List.of(RoleType.PDFUSION, RoleType.VIT)));
        Assertions.assertEquals(
                "pd",
                serviceRoute
                        .getAllEndpointsWithGroup(RoleType.PDFUSION)
                        .getFirst()
                        .getRight()
                        .getAddress());
        Assertions.assertEquals(
                "vit",
                serviceRoute
                        .getAllEndpointsWithGroup(RoleType.VIT)
                        .getFirst()
                        .getRight()
                        .getAddress());
    }

    @Test
    void should_load_encoder_endpoint_as_independent_role() throws Exception {
        String json =
                """
                {
                  "service_id": "test.service",
                  "role_endpoints": [{
                    "group": "g1",
                    "encoder_endpoint": {"address": "encoder", "protocol": "http", "path": "/"},
                    "vit_endpoint": {"address": "vit", "protocol": "http", "path": "/"}
                  }]
                }
                """;

        ServiceRoute route = objectMapper.readValue(json, ServiceRoute.class);
        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);

        Assertions.assertEquals(List.of(RoleType.VIT, RoleType.ENCODER), config.requiredRoles());
        Assertions.assertEquals(
                "encoder",
                route.getAllEndpointsWithGroup(RoleType.ENCODER)
                        .getFirst()
                        .getRight()
                        .getAddress());
        Assertions.assertEquals(
                "g1",
                route.getAllEndpointsWithGroup(RoleType.ENCODER).getFirst().getRight().getGroup());
        Assertions.assertEquals(
                "vit",
                route.getAllEndpointsWithGroup(RoleType.VIT).getFirst().getRight().getAddress());
        Assertions.assertEquals(RoleType.ENCODER, RoleType.fromString("ENCODER"));
        Assertions.assertEquals(
                StrategyErrorType.NO_ENCODER_WORKER,
                org.flexlb.worker.endpoint.StrategyErrorType.forRole(RoleType.ENCODER));
    }

    @Test
    void usesDefaultDashScopeBaseUrlWhenOmitted() throws Exception {
        String json =
                """
                {
                  "address": "v-test",
                  "discovery": {"type": "dashscope"}
                }
                """;

        DiscoveryEndpoint endpoint = objectMapper.readValue(json, DiscoveryEndpoint.class);

        Assertions.assertEquals(
                DiscoverySettings.DEFAULT_DASHSCOPE_BASE_URL, endpoint.getDiscovery().getBaseUrl());
    }

    @Test
    void loadsKvcmConfigurationWithCodeDefaults() throws Exception {
        String json =
                """
                {
                  "service_id": "test-service",
                  "kvcm": {
                    "address": "v-kvcm",
                    "namespace": "vllm-test-0",
                    "discovery": {"type": "dashscope"}
                  },
                  "role_endpoints": []
                }
                """;

        ServiceRoute serviceRoute = objectMapper.readValue(json, ServiceRoute.class);

        Assertions.assertNotNull(serviceRoute.getKvcm());
        Assertions.assertEquals("v-kvcm", serviceRoute.getKvcm().getAddress());
        Assertions.assertEquals("vllm-test-0", serviceRoute.getKvcm().getNamespace());
        Assertions.assertEquals("grpc", serviceRoute.getKvcm().toEndpoint().getProtocol());
        Assertions.assertEquals(
                KvcmConfig.DEFAULT_BOOTSTRAP_PORT, serviceRoute.getKvcm().getPort());
    }

    private static void assertStandardRoute(ServiceRoute serviceRoute) {
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints().size());
        GroupRoleEndPoint roleEndpoint = serviceRoute.getRoleEndpoints().getFirst();
        Assertions.assertEquals("ea119_PPU_ZW810E_16TP_decode64", roleEndpoint.getGroup());
        DiscoveryEndpoint prefillEndpoint = new DiscoveryEndpoint();
        prefillEndpoint.setAddress("com.aicheng.whale.pre.deepseek_dp_tp_test");
        prefillEndpoint.setProtocol("http");
        prefillEndpoint.setPath("/");
        prefillEndpoint.setGroup("ea119_PPU_ZW810E_16TP_decode64");
        Assertions.assertEquals(prefillEndpoint, roleEndpoint.getPrefillEndpoint());
        DiscoveryEndpoint decodeEndpoint = new DiscoveryEndpoint();
        decodeEndpoint.setAddress("com.aicheng.whale.pre.test_pd_gang2.decode");
        decodeEndpoint.setProtocol("http");
        decodeEndpoint.setPath("/");
        decodeEndpoint.setGroup("ea119_PPU_ZW810E_16TP_decode64");
        Assertions.assertEquals(decodeEndpoint, roleEndpoint.getDecodeEndpoint());
    }
}
