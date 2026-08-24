package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ServerStatusTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void serializesSelectedEngineIndexWithoutChangingPhysicalAddress() {
        ServerStatus status = new ServerStatus();
        status.setServerIp("10.0.0.8");
        status.setHttpPort(8080);
        status.setGrpcPort(8081);
        status.setSelectedEngineIndex(1, 2);

        var json = objectMapper.valueToTree(status);

        assertEquals("10.0.0.8", json.get("server_ip").asText());
        assertEquals(8080, json.get("http_port").asInt());
        assertEquals(8081, json.get("grpc_port").asInt());
        assertEquals(1, json.get("engine_index").asInt());
        assertFalse(json.has("routingEngineIndex"));
        assertFalse(json.has("logicalIpPort"));
        assertEquals("10.0.0.8:8080@1", status.getLogicalIpPort());
    }

    @Test
    void omitsSingleEngineIndexButKeepsIndexedRoutingIdentity() {
        ServerStatus status = new ServerStatus();
        status.setServerIp("10.0.0.8");
        status.setHttpPort(8080);
        status.setSelectedEngineIndex(0, 1);

        var json = objectMapper.valueToTree(status);

        assertFalse(json.has("engine_index"));
        assertFalse(json.has("routingEngineIndex"));
        assertFalse(json.has("logicalIpPort"));
        assertEquals("10.0.0.8:8080@0", status.getLogicalIpPort());
    }

    @Test
    void serializesIndexZeroWhenItBelongsToAMultiEngineWorker() {
        ServerStatus status = new ServerStatus();
        status.setServerIp("10.0.0.8");
        status.setHttpPort(8080);
        status.setSelectedEngineIndex(0, 2);

        var json = objectMapper.valueToTree(status);

        assertEquals(0, json.get("engine_index").asInt());
        assertFalse(json.has("logicalIpPort"));
        assertEquals("10.0.0.8:8080@0", status.getLogicalIpPort());
    }

    @ParameterizedTest
    @CsvSource({"-1,1", "1,1", "0,0", "0,-1"})
    void rejectsInvalidSelectedEngineIdentity(int engineIndex, int multiEngineNum) {
        ServerStatus status = new ServerStatus();

        assertThrows(
                IllegalArgumentException.class,
                () -> status.setSelectedEngineIndex(engineIndex, multiEngineNum));
    }
}
