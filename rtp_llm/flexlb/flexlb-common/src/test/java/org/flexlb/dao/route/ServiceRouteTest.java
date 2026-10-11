package org.flexlb.dao.route;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class ServiceRouteTest {
    private static final String ROUTE_JSON = """
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
    void testConfigLoaderList() {
        List<ServiceRoute> serviceRoutes = JsonUtils.toObject("[" + ROUTE_JSON + "]", new TypeReference<>() {
        });
        assertStandardRoute(serviceRoutes.getFirst());
    }

    private static void assertStandardRoute(ServiceRoute serviceRoute) {
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints().size());
        GroupRoleEndPoint roleEndpoint = serviceRoute.getRoleEndpoints().getFirst();
        Assertions.assertEquals("ea119_PPU_ZW810E_16TP_decode64", roleEndpoint.getGroup());
        Endpoint prefillEndpoint = new Endpoint();
        prefillEndpoint.setAddress("com.aicheng.whale.pre.deepseek_dp_tp_test");
        prefillEndpoint.setProtocol("http");
        prefillEndpoint.setPath("/");
        Assertions.assertEquals(prefillEndpoint, roleEndpoint.getPrefillEndpoint());
        Endpoint decodeEndpoint = new Endpoint();
        decodeEndpoint.setAddress("com.aicheng.whale.pre.test_pd_gang2.decode");
        decodeEndpoint.setProtocol("http");
        decodeEndpoint.setPath("/");
        Assertions.assertEquals(decodeEndpoint, roleEndpoint.getDecodeEndpoint());
    }

    @Test
    void should_load_pd_fusion_and_vit_endpoints() throws Exception {
        String json = """
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

        Assertions.assertTrue(serviceRoute.getAllRoleTypes().containsAll(
                List.of(RoleType.PDFUSION, RoleType.VIT)));
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints(RoleType.PDFUSION).size());
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints(RoleType.VIT).size());
    }
}
