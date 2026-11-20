package org.flexlb.balance.requestlog;

import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestPhase;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks request log data construction and serialized fields.
 */
class PvLogDataTest {

    @Test
    void includesPhaseToDistinguishTwoDecisionsWithTheSameRequestId() throws Exception {
        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        context.setPhase(RequestPhase.ENCODER);
        Request request = Request.builder().requestId("shared-id").build();
        context.setRequest(request);

        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(JsonUtils.toStringOrEmpty(pv(context)));

        assertEquals("shared-id", json.path("requestId").asText());
        assertEquals("ENCODER", json.path("phase").asText());
    }

    @Test
    void includesEncoderCacheHitLengthOnlyWhenClientProvidesIt() throws Exception {
        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        Request request = Request.builder().requestId("encoder-stage").build();
        context.setRequest(request);
        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();

        var omitted = mapper.readTree(JsonUtils.toStringOrEmpty(pv(context)));
        assertFalse(omitted.has("encoderCacheHitLen"));

        context.setRequest(request.toBuilder().encoderCacheHitLen(0L).build());
        var knownMiss = mapper.readTree(JsonUtils.toStringOrEmpty(pv(context)));
        assertTrue(knownMiss.has("encoderCacheHitLen"));
        assertEquals(0L, knownMiss.path("encoderCacheHitLen").asLong());
    }

    @Test
    void terminalResponseUsesFinalOutcomeWithoutMutatingRoutingResponse() throws Exception {
        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        Response routed =
                Response.error(org.flexlb.worker.endpoint.StrategyErrorType.REQUEST_CANCELLED);
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(routed));
        int deadlineCode =
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED
                        .getErrorCode();
        PvLogData data =
                new PvLogData(
                        context,
                        false,
                        "Schedule RPC deadline exceeded",
                        deadlineCode,
                        null,
                        "LOCAL_MASTER",
                        "REQUEST_STATE_TIMED_OUT",
                        null,
                        System.currentTimeMillis());
        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(JsonUtils.toStringOrEmpty(data));
        assertEquals(deadlineCode, json.path("code").asInt());
        assertFalse(json.path("response").has("code"));
        assertEquals("Schedule RPC deadline exceeded", json.path("error").asText());
        assertFalse(json.path("response").has("error_message"));
        assertFalse(json.path("response").has("success"));
        assertFalse(json.has("realMasterHost"));
        assertFalse(json.path("response").has("real_master_host"));
        assertFalse(json.has("admissionRejectReason"));
        assertFalse(json.path("response").has("admission_reject_reason"));
        assertEquals(
                org.flexlb.worker.endpoint.StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                routed.getCode());
    }

    @Test
    void pvKeepsDecodePreemptRequestIdsUnderServerStatusCompaction() throws Exception {
        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        Request request = Request.builder().requestId("incoming").build();
        context.setRequest(request);
        org.flexlb.worker.status.ServerStatus decode = new org.flexlb.worker.status.ServerStatus();
        decode.setRole(RoleType.DECODE);
        decode.setServerIp("10.0.0.3");
        decode.setRequestId("decode-internal");
        decode.setPrefillTime(7);
        decode.setDebugInfo(new org.flexlb.worker.status.DebugInfo());
        decode.setPreemptRequestIds(List.of("victim-a", "victim-b"));
        org.flexlb.worker.status.ServerStatus prefill = new org.flexlb.worker.status.ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        Response response = new Response();
        response.setServerStatus(List.of(decode, prefill));
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(response));

        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(JsonUtils.toStringOrEmpty(pv(context)));

        var statuses = json.path("response").path("server_status");
        var decodeIds = statuses.get(0).path("preempt_request_ids");
        assertEquals(2, decodeIds.size());
        assertEquals("victim-a", decodeIds.get(0).asText());
        assertEquals("victim-b", decodeIds.get(1).asText());
        assertEquals("DECODE", statuses.get(0).path("role").asText());
        assertEquals(0, statuses.get(1).path("preempt_request_ids").size());
        assertFalse(statuses.get(0).has("request_id"));
        assertFalse(statuses.get(0).has("prefill_time"));
        assertFalse(statuses.get(0).has("debug_info"));
        assertEquals(List.of("victim-a", "victim-b"), decode.getPreemptRequestIds());
    }

    @Test
    void omitsRoutingDecisionWhenSnapshotIsAbsent() {
        Request request = Request.builder().requestId("1001").build();

        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        context.setRequest(request);
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(new Response()));

        String json = JsonUtils.toStringOrEmpty(pv(context));

        assertFalse(json.contains("routingDecisions"));
    }

    @Test
    void includesSequenceLengthAndOmitsPayloadDiagnostics() {
        Request request = Request.builder().requestId("2").seqLen(999).build();

        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        context.setRequest(request);
        context.setRequestMessageBytes(1234L);

        PvLogData data = pv(context);

        assertEquals(Long.valueOf(999), data.getSeqLen());
        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"seqLen\":999"));
        assertFalse(json.contains("\"inputIdsCount\""));
        assertFalse(json.contains("\"requestMessageBytes\""));
        assertFalse(json.contains("\"requestBodyBytes\""));
    }

    @Test
    void keepsFailurePvWhenRequestBodyCannotBeDeserialized() {
        RequestContext context = new RequestContext(new org.flexlb.config.FlexlbConfig());

        PvLogData data = pv(context);

        assertNull(data.getRequestId());
        assertNull(data.getSeqLen());
        String json = JsonUtils.toStringOrEmpty(data);
        assertFalse(json.contains("\"requestId\""));
        assertFalse(json.contains("\"seqLen\""));
        assertFalse(json.contains("\"inputIdsCount\""));
        assertFalse(json.contains("\"requestBodyBytes\""));
        assertTrue(json.contains("\"success\":false"));
    }

    private static PvLogData pv(RequestContext data) {
        var response = data.getResponse();
        return new PvLogData(
                data,
                response != null && response.isSuccess(),
                response == null ? null : response.getErrorMessage(),
                response == null ? 0 : response.getCode(),
                null,
                "LOCAL_MASTER",
                null,
                null,
                System.currentTimeMillis());
    }
}
