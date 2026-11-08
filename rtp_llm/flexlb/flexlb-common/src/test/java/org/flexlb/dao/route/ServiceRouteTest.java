package org.flexlb.dao.route;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.service.config.parser.ModelServiceConfigParser;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ServiceRouteTest {
    private static final String ROUTE_JSON = """
            {
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

        assertTrue(json.has("role_endpoints"));
        assertFalse(json.has("roleEndpoints"));
        assertStandardRoute(objectMapper.treeToValue(json, ServiceRoute.class));
    }

    @Test
    void ignoresLegacyServiceIdWithoutSerializingIt() {
        ServiceRoute route =
                ModelServiceConfigParser.parse(ROUTE_JSON.replaceFirst("\\{", "{\"service_id\":\"legacy-service\","));

        assertStandardRoute(route);
        assertFalse(JsonUtils.strictValueToTree(route).has("service_id"));
    }

    private static void assertStandardRoute(ServiceRoute serviceRoute) {
        assertEquals(1, serviceRoute.getRoleEndpoints().size());
        GroupRoleEndPoint roleEndpoint = serviceRoute.getRoleEndpoints().getFirst();
        assertEquals("ea119_PPU_ZW810E_16TP_decode64", roleEndpoint.getGroup());
        Endpoint prefillEndpoint = new Endpoint();
        prefillEndpoint.setAddress("com.aicheng.whale.pre.deepseek_dp_tp_test");
        prefillEndpoint.setProtocol("http");
        prefillEndpoint.setPath("/");
        prefillEndpoint.setGroup("ea119_PPU_ZW810E_16TP_decode64");
        assertEquals(prefillEndpoint, roleEndpoint.getPrefillEndpoint());
        Endpoint decodeEndpoint = new Endpoint();
        decodeEndpoint.setAddress("com.aicheng.whale.pre.test_pd_gang2.decode");
        decodeEndpoint.setProtocol("http");
        decodeEndpoint.setPath("/");
        decodeEndpoint.setGroup("ea119_PPU_ZW810E_16TP_decode64");
        assertEquals(decodeEndpoint, roleEndpoint.getDecodeEndpoint());
    }

    @Test
    void resolvesRoleEndpointsAcrossGroupsWithoutAModelIdentifier() {
        ServiceRoute route = ModelServiceConfigParser.parse("""
                {"role_endpoints":[
                  {"group":"g1","prefill_endpoint":{"address":"p1"},"decode_endpoint":{"address":"d1"}},
                  {"group":"g2","prefill_endpoint":{"address":"p2"}}
                ]}
                """);
        ModelMetaConfig config = new ModelMetaConfig(route);

        var prefill = config.endpointsWithGroup(RoleType.PREFILL);
        assertEquals(List.of("g1", "g2"), prefill.stream().map(pair -> pair.getLeft()).toList());
        assertEquals(List.of("p1", "p2"), prefill.stream().map(pair -> pair.getRight().getAddress()).toList());
        assertEquals(List.of(RoleType.DECODE, RoleType.PREFILL), config.requiredRoles());
        assertTrue(config.endpointsWithGroup(RoleType.ENCODER).isEmpty());
        assertTrue(config.endpointsWithGroup(null).isEmpty());
        var decode = config.endpointsWithGroup(RoleType.DECODE);
        assertEquals("d1", decode.getFirst().getRight().getAddress());
        assertNull(decode.getLast().getRight());

        prefill.clear();
        assertEquals(2, config.endpointsWithGroup(RoleType.PREFILL).size());
    }

    @Test
    void should_load_pd_fusion_and_vit_endpoints() throws Exception {
        String json = """
                {
                  "role_endpoints": [{
                    "group": "g1",
                    "pd_fusion_endpoint": {"address": "pd", "protocol": "http", "path": "/"},
                    "vit_endpoint": {"address": "vit", "protocol": "http", "path": "/"}
                  }]
                }
                """;

        ServiceRoute serviceRoute = objectMapper.readValue(json, ServiceRoute.class);

        assertTrue(serviceRoute.getAllRoleTypes().containsAll(List.of(RoleType.PDFUSION, RoleType.VIT)));
        assertEquals(1, serviceRoute.getRoleEndpoints(RoleType.PDFUSION).size());
        assertEquals(1, serviceRoute.getRoleEndpoints(RoleType.VIT).size());
    }

    @Test
    void should_load_encoder_endpoint_as_independent_role() throws Exception {
        String json = """
                {
                  "role_endpoints": [{
                    "group": "g1",
                    "encoder_endpoint": {"address": "encoder", "protocol": "http", "path": "/"},
                    "vit_endpoint": {"address": "vit", "protocol": "http", "path": "/"}
                  }]
                }
                """;

        ServiceRoute route = objectMapper.readValue(json, ServiceRoute.class);
        ModelMetaConfig config = new ModelMetaConfig(route);

        assertEquals(List.of(RoleType.VIT, RoleType.ENCODER), config.requiredRoles());
        assertEquals("encoder", route.getRoleEndpoints(RoleType.ENCODER).getFirst().getAddress());
        assertEquals("g1", route.getRoleEndpoints(RoleType.ENCODER).getFirst().getGroup());
        assertEquals("vit", route.getRoleEndpoints(RoleType.VIT).getFirst().getAddress());
        assertEquals(RoleType.ENCODER, RoleType.fromString("ENCODER"));
        assertEquals(StrategyErrorType.NO_ENCODER_WORKER, RoleType.ENCODER.getErrorType());
    }

    @Test
    void usesDefaultDashScopeBaseUrlWhenOmitted() throws Exception {
        String json = """
                {
                  "address": "v-test",
                  "discovery": {"type": "dashscope"}
                }
                """;

        Endpoint endpoint = objectMapper.readValue(json, Endpoint.class);

        assertEquals(DiscoveryConfig.DEFAULT_DASHSCOPE_BASE_URL, endpoint.getDiscovery().getBaseUrl());
    }

    @Test
    void loadsKvcmConfigurationWithCodeDefaults() throws Exception {
        String json = """
                {
                  "kvcm": {
                    "address": "v-kvcm",
                    "namespace": "vllm-test-0",
                    "discovery": {"type": "dashscope"}
                  },
                  "role_endpoints": []
                }
                """;

        ServiceRoute serviceRoute = objectMapper.readValue(json, ServiceRoute.class);

        assertNotNull(serviceRoute.getKvcm());
        assertEquals("v-kvcm", serviceRoute.getKvcm().getAddress());
        assertEquals("vllm-test-0", serviceRoute.getKvcm().getNamespace());
        assertEquals("grpc", serviceRoute.getKvcm().toEndpoint().getProtocol());
        assertEquals(KvcmConfig.DEFAULT_BOOTSTRAP_PORT, serviceRoute.getKvcm().getPort());
    }

    private void assertServiceRoute(ServiceRoute serviceRoute) {
        assertEquals(1, serviceRoute.getRoleEndpoints().size());
        assertEquals("ea119_PPU_ZW810E_16TP_decode64", serviceRoute.getRoleEndpoints().getFirst().getGroup());
        assertEquals(vipServerEndpoint("com.aicheng.whale.pre.deepseek_dp_tp_test"),
                serviceRoute.getRoleEndpoints().getFirst().getPrefillEndpoint());
        assertEquals(vipServerEndpoint("com.aicheng.whale.pre.test_pd_gang2.decode"),
                serviceRoute.getRoleEndpoints().getFirst().getDecodeEndpoint());
        assertEquals("ea119_PPU_ZW810E_16TP_decode64",
                serviceRoute.getRoleEndpoints().getFirst().getPrefillEndpoint().getGroup());
        assertEquals("ea119_PPU_ZW810E_16TP_decode64",
                serviceRoute.getRoleEndpoints().getFirst().getDecodeEndpoint().getGroup());
    }

    private Endpoint vipServerEndpoint(String address) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.VIPSERVER);
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol("http");
        endpoint.setPath("/");
        endpoint.setDiscovery(discovery);
        endpoint.setGroup("ea119_PPU_ZW810E_16TP_decode64");
        return endpoint;
    }
}
