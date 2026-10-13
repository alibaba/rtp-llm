package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ResponseCopyTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void copiesAllResponseDataWithoutSharingMutableObjects() throws Exception {
        Response source = mapper.readValue("""
                {
                  "server_status": [{
                    "role": "PREFILL", "server_ip": "127.0.0.1",
                    "http_port": 8000, "grpc_port": 8001, "dp_rank": 2,
                    "prefill_time": 3, "group": "group-a", "request_id": 42,
                    "success": true, "code": 201, "message": "accepted",
                    "debug_info": {
                      "running_batch_size": 4, "queue_size": 5, "waiting_time_ms": 6,
                      "available_kv_cache_len": 7, "estimate_ttft_ms": 8,
                      "estimate_tpot_ms": 9, "hit_cache_len": 10
                    }
                  }, null],
                  "success": true, "code": 202, "error_message": "detail",
                  "real_master_host": "master:9000", "queue_length": 11,
                  "enqueued_by_master": true, "ready": false,
                  "admission_reject_reason": "RESOURCE_EXHAUSTED",
                  "worker_summary": {
                    "PREFILL": {"discovered": 12, "alive": 13, "maxQueueTokens": 14},
                    "DECODE": null
                  }
                }
                """, Response.class);

        Response copy = Response.copyOf(source);
        assertEquals(mapper.valueToTree(source), mapper.valueToTree(copy));
        assertNotSame(source, copy);
        assertNotSame(source.getServerStatus(), copy.getServerStatus());
        assertNotSame(source.getServerStatus().get(0), copy.getServerStatus().get(0));
        assertNotSame(source.getServerStatus().get(0).getDebugInfo(),
                copy.getServerStatus().get(0).getDebugInfo());
        assertNotSame(source.getWorkerSummary(), copy.getWorkerSummary());
        assertNotSame(source.getWorkerSummary().get("PREFILL"), copy.getWorkerSummary().get("PREFILL"));

        copy.getServerStatus().get(0).getDebugInfo().setQueueSize(99);
        copy.getServerStatus().get(0).setServerIp("changed");
        copy.getServerStatus().clear();
        copy.getWorkerSummary().get("PREFILL").setAlive(99);
        copy.getWorkerSummary().clear();
        assertEquals(5, source.getServerStatus().get(0).getDebugInfo().getQueueSize());
        assertEquals("127.0.0.1", source.getServerStatus().get(0).getServerIp());
        assertEquals(2, source.getServerStatus().size());
        assertEquals(13, source.getWorkerSummary().get("PREFILL").getAlive());
        assertEquals(2, source.getWorkerSummary().size());
    }

    @Test
    void responseFactoriesPreserveSourceAndErrorDetails() {
        Response source = new Response();
        source.setCode(503);
        source.setReady(false);
        Response success = Response.buildSuccessResponse(source, true);
        assertTrue(success.isSuccess());
        assertEquals(200, success.getCode());
        assertTrue(success.isEnqueuedByMaster());
        assertFalse(success.isReady());
        assertFalse(source.isSuccess());
        assertEquals(503, source.getCode());
        assertFalse(source.isEnqueuedByMaster());
        assertFalse(Response.buildSuccessResponse(source, false).isEnqueuedByMaster());

        StrategyErrorType type = StrategyErrorType.DISPATCH_FAILED;
        Response failure = Response.buildErrorResponse(type, "delivery failed");
        assertFalse(failure.isSuccess());
        assertEquals(type.getErrorCode(), failure.getCode());
        assertEquals(type.buildErrorMessage("delivery failed"), failure.getErrorMessage());
        assertEquals(AdmissionRejectReason.UNSPECIFIED, failure.getAdmissionRejectReason());
        assertEquals(type.buildErrorMessage(null),
                Response.buildErrorResponse(type, null).getErrorMessage());
    }

    @Test
    void copiesSelectedLogicalEngineIdentity() {
        ServerStatus sourceStatus = new ServerStatus();
        sourceStatus.setServerIp("10.0.0.8");
        sourceStatus.setHttpPort(8080);
        sourceStatus.setSelectedEngineIndex(1, 2);
        Response source = new Response();
        source.setServerStatus(java.util.List.of(sourceStatus));

        ServerStatus copy = Response.copyOf(source).getServerStatus().getFirst();

        assertEquals(1, copy.getEngineIndex());
        assertEquals(2, copy.getRoutingMultiEngineNum());
        assertEquals("10.0.0.8:8080@1", copy.getLogicalIpPort());
    }

    @Test
    void preservesNullSourcesAndNestedValues() {
        assertNull(Response.copyOf(null));
        assertNull(ServerStatus.copyOf(null));
        assertNull(DebugInfo.copyOf(null));
        assertNull(Response.WorkerRoleSummary.copyOf(null));
        Response copy = Response.copyOf(new Response());
        assertNull(copy.getServerStatus());
        assertNull(copy.getWorkerSummary());
        assertNull(ServerStatus.copyOf(new ServerStatus()).getDebugInfo());
        assertTrue(copy.isReady());
    }
}
