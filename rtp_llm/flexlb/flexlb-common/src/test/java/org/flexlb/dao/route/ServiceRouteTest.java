package org.flexlb.dao.route;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class ServiceRouteTest {
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void testConfigLoader() throws Exception {
        String TEST_JSON = "{\n" +
                "\t\"service_id\": \"aigc.text-generation.generation.engine_service\",\n" +
                "\t\"prefill_endpoint\": {\n" +
                "\t\t\"address\": \"com.aicheng.whale.pre.deepseek_dp_tp_test\",\n" +
                "\t\t\"protocol\": \"http\",\n" +
                "\t\t\"path\": \"/\"\n" +
                "\t},\n" +
                "\t\"decode_endpoint\": {\n" +
                "\t\t\"address\": \"com.aicheng.whale.pre.test_pd_gang2.decode\",\n" +
                "\t\t\"protocol\": \"http\",\n" +
                "\t\t\"path\": \"/\"\n" +
                "\t},\n" +
                "\t\"role_endpoints\": [{\n" +
                "\t\t\"group\": \"ea119_PPU_ZW810E_16TP_decode64\",\n" +
                "\t\t\"prefill_endpoint\": {\n" +
                "\t\t\t\"address\": \"com.aicheng.whale.pre.deepseek_dp_tp_test\",\n" +
                "\t\t\t\"protocol\": \"http\",\n" +
                "\t\t\t\"path\": \"/\"\n" +
                "\t\t},\n" +
                "\t\t\"decode_endpoint\": {\n" +
                "\t\t\t\"address\": \"com.aicheng.whale.pre.test_pd_gang2.decode\",\n" +
                "\t\t\t\"protocol\": \"http\",\n" +
                "\t\t\t\"path\": \"/\"\n" +
                "\t\t}\n" +
                "\t}]\n" +
                "}";
        ServiceRoute serviceRoute = objectMapper.readValue(TEST_JSON, ServiceRoute.class);
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints().size());
        Assertions.assertEquals("ea119_PPU_ZW810E_16TP_decode64", serviceRoute.getRoleEndpoints().get(0).getGroup());
        Endpoint prefillEndpoint = new Endpoint();
        prefillEndpoint.setAddress("com.aicheng.whale.pre.deepseek_dp_tp_test");
        prefillEndpoint.setProtocol("http");
        prefillEndpoint.setPath("/");
        Assertions.assertEquals(prefillEndpoint, serviceRoute.getRoleEndpoints().get(0).getPrefillEndpoint());
        Endpoint decodeEndpoint = new Endpoint();
        decodeEndpoint.setAddress("com.aicheng.whale.pre.test_pd_gang2.decode");
        decodeEndpoint.setProtocol("http");
        decodeEndpoint.setPath("/");
        Assertions.assertEquals(decodeEndpoint, serviceRoute.getRoleEndpoints().get(0).getDecodeEndpoint());
    }

    @Test
    void testConfigLoaderList() {
        String TEST_JSON = "[{\n" +
                "\t\"service_id\": \"aigc.text-generation.generation.engine_service\",\n" +
                "\t\"prefill_endpoint\": {\n" +
                "\t\t\"address\": \"com.aicheng.whale.pre.deepseek_dp_tp_test\",\n" +
                "\t\t\"protocol\": \"http\",\n" +
                "\t\t\"path\": \"/\"\n" +
                "\t},\n" +
                "\t\"decode_endpoint\": {\n" +
                "\t\t\"address\": \"com.aicheng.whale.pre.test_pd_gang2.decode\",\n" +
                "\t\t\"protocol\": \"http\",\n" +
                "\t\t\"path\": \"/\"\n" +
                "\t},\n" +
                "\t\"role_endpoints\": [{\n" +
                "\t\t\"group\": \"ea119_PPU_ZW810E_16TP_decode64\",\n" +
                "\t\t\"prefill_endpoint\": {\n" +
                "\t\t\t\"address\": \"com.aicheng.whale.pre.deepseek_dp_tp_test\",\n" +
                "\t\t\t\"protocol\": \"http\",\n" +
                "\t\t\t\"path\": \"/\"\n" +
                "\t\t},\n" +
                "\t\t\"decode_endpoint\": {\n" +
                "\t\t\t\"address\": \"com.aicheng.whale.pre.test_pd_gang2.decode\",\n" +
                "\t\t\t\"protocol\": \"http\",\n" +
                "\t\t\t\"path\": \"/\"\n" +
                "\t\t}\n" +
                "\t}]\n" +
                "}]";
        List<ServiceRoute> serviceRoutes = JsonUtils.toObject(TEST_JSON, new TypeReference<List<ServiceRoute>>() {
        });
        ServiceRoute serviceRoute = serviceRoutes.get(0);
        Assertions.assertEquals(1, serviceRoute.getRoleEndpoints().size());
        Assertions.assertEquals("ea119_PPU_ZW810E_16TP_decode64", serviceRoute.getRoleEndpoints().get(0).getGroup());
        Endpoint prefillEndpoint = new Endpoint();
        prefillEndpoint.setAddress("com.aicheng.whale.pre.deepseek_dp_tp_test");
        prefillEndpoint.setProtocol("http");
        prefillEndpoint.setPath("/");
        Assertions.assertEquals(prefillEndpoint, serviceRoute.getRoleEndpoints().get(0).getPrefillEndpoint());
        Endpoint decodeEndpoint = new Endpoint();
        decodeEndpoint.setAddress("com.aicheng.whale.pre.test_pd_gang2.decode");
        decodeEndpoint.setProtocol("http");
        decodeEndpoint.setPath("/");
        Assertions.assertEquals(decodeEndpoint, serviceRoute.getRoleEndpoints().get(0).getDecodeEndpoint());
    }
}