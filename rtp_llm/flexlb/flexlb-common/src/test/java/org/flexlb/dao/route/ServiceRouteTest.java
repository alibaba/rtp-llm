package org.flexlb.dao.route;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class ServiceRouteTest {

    private static final String SERVICE_ROUTE_JSON = """
            {
              "service_id": "aigc.text-generation.generation.engine_service",
              "role_endpoints": [{
                "group": "ea119_PPU_ZW810E_16TP_decode64",
                "prefill_endpoint": {
                  "address": "com.aicheng.whale.pre.deepseek_dp_tp_test",
                  "protocol": "http",
                  "path": "/",
                  "discovery": {"type": "vipserver"}
                },
                "decode_endpoint": {
                  "address": "com.aicheng.whale.pre.test_pd_gang2.decode",
                  "protocol": "http",
                  "path": "/",
                  "discovery": {"type": "vipserver"}
                }
              }]
            }
            """;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void testConfigLoader() throws Exception {
        ServiceRoute serviceRoute = objectMapper.readValue(SERVICE_ROUTE_JSON, ServiceRoute.class);

        assertServiceRoute(serviceRoute);
    }

    @Test
    void testConfigLoaderList() {
        List<ServiceRoute> serviceRoutes = JsonUtils.toObject(
                "[" + SERVICE_ROUTE_JSON + "]",
                new TypeReference<>() {
                });

        assertServiceRoute(serviceRoutes.getFirst());
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

        Assertions.assertEquals(
                DiscoveryConfig.DEFAULT_DASHSCOPE_BASE_URL,
                endpoint.getDiscovery().getBaseUrl());
    }

    private void assertServiceRoute(ServiceRoute serviceRoute) {
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints().size());
        Assertions.assertEquals(
                "ea119_PPU_ZW810E_16TP_decode64",
                serviceRoute.getRoleEndpoints().getFirst().getGroup());
        Assertions.assertEquals(
                vipServerEndpoint("com.aicheng.whale.pre.deepseek_dp_tp_test"),
                serviceRoute.getRoleEndpoints().getFirst().getPrefillEndpoint());
        Assertions.assertEquals(
                vipServerEndpoint("com.aicheng.whale.pre.test_pd_gang2.decode"),
                serviceRoute.getRoleEndpoints().getFirst().getDecodeEndpoint());
    }

    private Endpoint vipServerEndpoint(String address) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.VIPSERVER);
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol("http");
        endpoint.setPath("/");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
