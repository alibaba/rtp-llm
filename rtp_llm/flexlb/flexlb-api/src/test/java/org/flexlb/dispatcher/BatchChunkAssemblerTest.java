package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dispatcher.SubBatchSpec;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchChunkAssemblerTest {

    @Test
    void splitArrayDividesEvenly() {
        JSONArray arr = JSONArray.of("a", "b", "c", "d");
        List<JSONArray> chunks = BatchChunkAssembler.splitArray(arr, 2);
        assertEquals(2, chunks.size());
        assertEquals(JSONArray.of("a", "b"), chunks.get(0));
        assertEquals(JSONArray.of("c", "d"), chunks.get(1));
    }

    @Test
    void splitArrayLastChunkShorter() {
        JSONArray arr = JSONArray.of("a", "b", "c", "d", "e");
        List<JSONArray> chunks = BatchChunkAssembler.splitArray(arr, 2);
        assertEquals(3, chunks.size());
        assertEquals(1, chunks.get(2).size());
    }

    @Test
    void splitArrayEmptyReturnsEmptyList() {
        assertTrue(BatchChunkAssembler.splitArray(new JSONArray(), 5).isEmpty());
    }

    @Test
    void splitByCountFrontLoadsRemainder() {
        JSONArray arr = JSONArray.of(1, 2, 3, 4, 5, 6, 7);
        List<JSONArray> chunks = BatchChunkAssembler.splitByCount(arr, 3);
        assertEquals(3, chunks.size());
        assertEquals(3, chunks.get(0).size());
        assertEquals(2, chunks.get(1).size());
        assertEquals(2, chunks.get(2).size());
    }

    @Test
    void splitByCountClampsToTotalWhenRequestedExceeds() {
        JSONArray arr = JSONArray.of("a", "b");
        List<JSONArray> chunks = BatchChunkAssembler.splitByCount(arr, 5);
        assertEquals(2, chunks.size());
    }

    @Test
    void specAwareSplitRoutesBySizeAndCount() {
        JSONArray arr = JSONArray.of(1, 2, 3, 4);
        assertEquals(2, BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 2)).size());
        assertEquals(3, BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.COUNT, 3)).size());
    }

    @Test
    void buildChunkBodiesDeepClonesAndReplacesArray() {
        JSONObject envelope = new JSONObject();
        envelope.put("model", "m");
        envelope.put("prompt_batch", JSONArray.of("a", "b", "c"));
        JSONObject gc = new JSONObject();
        gc.put("temperature", 0.5);
        envelope.put("generate_config", gc);

        List<JSONArray> chunks = List.of(JSONArray.of("a"), JSONArray.of("b", "c"));
        List<JSONObject> bodies = BatchChunkAssembler.buildChunkBodies(envelope, chunks, "prompt_batch");

        assertEquals(2, bodies.size());
        assertEquals(JSONArray.of("a"), bodies.get(0).getJSONArray("prompt_batch"));
        assertEquals(JSONArray.of("b", "c"), bodies.get(1).getJSONArray("prompt_batch"));
        assertEquals("m", bodies.get(0).getString("model"));
        // Each chunk has its own generate_config so per-chunk mutations don't leak.
        assertNotSame(bodies.get(0).getJSONObject("generate_config"),
                bodies.get(1).getJSONObject("generate_config"));
        // Original envelope is untouched.
        assertEquals(3, envelope.getJSONArray("prompt_batch").size());
        assertFalse(envelope.getJSONObject("generate_config").containsKey("force_batch"));
    }

    @Test
    void injectForceBatchAddsWhenAbsent() {
        JSONObject body = new JSONObject();
        BatchChunkAssembler.injectForceBatch(body);
        assertEquals(true, body.getJSONObject("generate_config").getBoolean("force_batch"));
    }

    @Test
    void injectForceBatchPreservesUserFalse() {
        JSONObject body = new JSONObject();
        JSONObject gc = new JSONObject();
        gc.put("force_batch", false);
        body.put("generate_config", gc);
        BatchChunkAssembler.injectForceBatch(body);
        assertEquals(false, body.getJSONObject("generate_config").getBoolean("force_batch"));
    }

    @Test
    void stampPreAssignedBeAppendsRoleAddrs() {
        JSONObject body = new JSONObject();
        List<JSONObject> bodies = List.of(body);
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setRole(RoleType.PDFUSION);
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);

        BatchChunkAssembler.stampPreAssignedBe(bodies, List.of(target));

        JSONArray addrs = body.getJSONObject("generate_config").getJSONArray("role_addrs");
        assertEquals(1, addrs.size());
        JSONObject addr = addrs.getJSONObject(0);
        assertEquals("PDFUSION", addr.getString("role"));
        assertEquals("10.0.0.1", addr.getString("ip"));
        assertEquals(8088, addr.getIntValue("http_port"));
        assertEquals(50051, addr.getIntValue("grpc_port"));
    }

    @Test
    void stampPreAssignedBePreservesUserRoleAddrs() {
        JSONObject body = new JSONObject();
        JSONObject gc = new JSONObject();
        JSONArray userAddrs = new JSONArray();
        userAddrs.add(JSONObject.of("role", "PREFILL", "ip", "1.1.1.1", "http_port", 80, "grpc_port", 50));
        gc.put("role_addrs", userAddrs);
        body.put("generate_config", gc);

        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setRole(RoleType.PDFUSION);
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);

        BatchChunkAssembler.stampPreAssignedBe(List.of(body), List.of(target));

        JSONArray addrs = body.getJSONObject("generate_config").getJSONArray("role_addrs");
        assertEquals(2, addrs.size());
        assertEquals("PREFILL", addrs.getJSONObject(0).getString("role"));
        assertEquals("PDFUSION", addrs.getJSONObject(1).getString("role"));
    }

    @Test
    void stampPreAssignedBeNoOpOnEmptyTargets() {
        JSONObject body = new JSONObject();
        BatchChunkAssembler.stampPreAssignedBe(List.of(body), List.of());
        assertTrue(body.isEmpty());
    }

    @Test
    void stampPreAssignedBeToleratesShortTargetList() {
        JSONObject body0 = new JSONObject();
        JSONObject body1 = new JSONObject();
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setRole(RoleType.PDFUSION);
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);

        BatchChunkAssembler.stampPreAssignedBe(List.of(body0, body1), List.of(target));

        assertFalse(body0.isEmpty());
        assertTrue(body1.isEmpty());
    }
}
