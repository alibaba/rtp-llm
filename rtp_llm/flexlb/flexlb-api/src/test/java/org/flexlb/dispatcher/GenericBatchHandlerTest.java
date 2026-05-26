package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.EntityResponse;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GenericBatchHandlerTest {

    private final ObjectMapper mapper = new ObjectMapper();
    private final BatchEndpointSpec spec = new BatchEndpointSpec(
            "/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null);

    @Test
    void singleChunkFansOutOfOneAndReturns200() {
        FanoutService fanout = mock(FanoutService.class);
        ObjectNode feResp = mapper.createObjectNode();
        ArrayNode rb = feResp.putArray("response_batch");
        rb.addObject().put("response", "r0");
        rb.addObject().put("response", "r1");
        when(fanout.dispatchChunks(eq("/batch_infer"), anyList(), eq(spec)))
                .thenReturn(Mono.just(List.of(SubBatchResult.ok(feResp, 2, 0))));

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:5"));
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
                .verifyComplete();
        verify(fanout).dispatchChunks(eq("/batch_infer"), argThat(list -> list.size() == 1), eq(spec));
    }

    @Test
    void multiChunkSplitsAndMergesReturning200() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(eq("/batch_infer"), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:2"));
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
                .verifyComplete();
        verify(fanout).dispatchChunks(eq("/batch_infer"), argThat(list -> list.size() == 2), eq(spec));
    }

    @Test
    void allFailedReturns500() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenReturn(
                Mono.just(List.of(
                        SubBatchResult.failed(2, 0, "fe_down"),
                        SubBatchResult.failed(2, 2, "fe_down"))));

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:2"));
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c").add("d");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, resp.statusCode()))
                .verifyComplete();
    }

    @Test
    void emptyBatchReturnsShapedEmptyEnvelopeNotEmptyObject() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:5"));
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> {
                    assertEquals(HttpStatus.OK, resp.statusCode());
                    EntityResponse<?> entity = (EntityResponse<?>) resp;
                    ObjectNode out = (ObjectNode) entity.entity();
                    assertEquals(0, out.get("response_batch").size());
                    assertTrue(out.get("response_batch").isArray());
                })
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

    @Test
    void nonObjectBodyReturns400() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:2"));

        JsonNode arrayBody = mapper.createArrayNode().add("a").add("b");
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(arrayBody));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.BAD_REQUEST, resp.statusCode()))
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

    @Test
    void missingArrayFieldReturns400() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:2"));

        ObjectNode body = mapper.createObjectNode();
        body.put("unrelated", "data");
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.BAD_REQUEST, resp.statusCode()))
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

    @Test
    void nonArrayRequestFieldReturns400() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:2"));

        ObjectNode body = mapper.createObjectNode();
        body.put("prompt_batch", "not-an-array");
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.BAD_REQUEST, resp.statusCode()))
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

    @Test
    void injectsForceBatchTrueWhenGenerateConfigAbsent() {
        List<ObjectNode> captured = runSingleChunkAndCaptureChunkBodies(mapper.createObjectNode()
                .set("prompt_batch", mapper.createArrayNode().add("a")));

        assertEquals(1, captured.size());
        JsonNode gc = captured.get(0).get("generate_config");
        assertNotNull(gc, "generate_config must be created when absent");
        assertTrue(gc.isObject());
        assertTrue(gc.get("force_batch").asBoolean(),
                "dispatcher must inject force_batch=true when user supplied no generate_config");
    }

    @Test
    void injectsForceBatchTrueWhenGenerateConfigExistsButForceBatchAbsent() {
        ObjectNode body = mapper.createObjectNode();
        body.set("prompt_batch", mapper.createArrayNode().add("a"));
        ObjectNode userGc = body.putObject("generate_config");
        userGc.put("temperature", 0.7);
        userGc.put("max_new_tokens", 100);

        List<ObjectNode> captured = runSingleChunkAndCaptureChunkBodies(body);

        JsonNode gc = captured.get(0).get("generate_config");
        assertTrue(gc.get("force_batch").asBoolean(),
                "dispatcher must inject force_batch=true when generate_config exists without that field");
        assertEquals(0.7, gc.get("temperature").asDouble(), 1e-9,
                "user's other generate_config fields must be preserved");
        assertEquals(100, gc.get("max_new_tokens").asInt());
    }

    @Test
    void respectsUserSetForceBatchTrue() {
        ObjectNode body = mapper.createObjectNode();
        body.set("prompt_batch", mapper.createArrayNode().add("a"));
        body.putObject("generate_config").put("force_batch", true);

        List<ObjectNode> captured = runSingleChunkAndCaptureChunkBodies(body);

        assertTrue(captured.get(0).get("generate_config").get("force_batch").asBoolean());
    }

    @Test
    void respectsUserSetForceBatchFalse() {
        ObjectNode body = mapper.createObjectNode();
        body.set("prompt_batch", mapper.createArrayNode().add("a"));
        body.putObject("generate_config").put("force_batch", false);

        List<ObjectNode> captured = runSingleChunkAndCaptureChunkBodies(body);

        assertFalse(captured.get(0).get("generate_config").get("force_batch").asBoolean(),
                "user's explicit force_batch=false opt-out must be honored, not overwritten");
    }

    @Test
    void injectsForceBatchIntoEveryChunkOnSplit() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:2"));
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c").add("d").add("e");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .expectNextCount(1)
                .verifyComplete();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<ObjectNode>> captor = ArgumentCaptor.forClass(List.class);
        verify(fanout).dispatchChunks(eq("/batch_infer"), captor.capture(), eq(spec));
        List<ObjectNode> chunks = captor.getValue();
        assertEquals(3, chunks.size(), "5 prompts at K=2 → 3 chunks");
        for (ObjectNode chunk : chunks) {
            assertTrue(chunk.get("generate_config").get("force_batch").asBoolean(),
                    "every chunk must carry force_batch=true so the per-chunk FE schedules its prompts together");
        }
    }



    @Test
    void preAssignBeStampsTargetIntoGenerateConfigRoleAddrsInOrder() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });
        BatchScheduleClient batchClient = mock(BatchScheduleClient.class);
        List<org.flexlb.dao.loadbalance.BatchScheduleTarget> targets = List.of(
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION),
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.2", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION),
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.3", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION));
        when(batchClient.requestTargets(3)).thenReturn(Mono.just(targets));

        GenericBatchHandler handler = new GenericBatchHandler(
                fanout, mapper, SubBatchSpec.parse("size:2"), batchClient, true);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c").add("d").add("e");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .expectNextCount(1)
                .verifyComplete();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<ObjectNode>> captor = ArgumentCaptor.forClass(List.class);
        verify(fanout).dispatchChunks(eq("/batch_infer"), captor.capture(), eq(spec));
        List<ObjectNode> chunks = captor.getValue();
        assertEquals(3, chunks.size());
        for (int i = 0; i < 3; i++) {
            ObjectNode chunk = chunks.get(i);
            assertNull(chunk.get("pre_assigned_be"),
                    "stamping must use generate_config.role_addrs, never a top-level field "
                            + "(pydantic extra=ignore would drop it on FE side)");
            JsonNode roleAddrs = chunk.get("generate_config").get("role_addrs");
            assertNotNull(roleAddrs, "chunk " + i + " must have generate_config.role_addrs when preAssignBe=true");
            assertTrue(roleAddrs.isArray() && roleAddrs.size() == 1,
                    "exactly one role_addr per chunk for single-role batch_schedule");
            JsonNode addr = roleAddrs.get(0);
            assertEquals("PDFUSION", addr.get("role").asText(),
                    "role serializes as enum.name() to match Python RoleType enum");
            assertEquals("10.0.0." + (i + 1), addr.get("ip").asText(),
                    "chunk " + i + " must carry the i-th target");
            assertEquals(23840, addr.get("http_port").asInt());
            assertEquals(23841, addr.get("grpc_port").asInt());
        }
    }

    @Test
    void preAssignBePreservesUserSuppliedRoleAddrs() {
        // Defensive: if a user already set generate_config.role_addrs on the request body
        // (rare but legitimate, e.g. integration-testing a specific BE), the dispatcher
        // appends its target rather than wiping the user's entry. FE will see both addrs
        // and pick whichever role matches its routing pipeline.
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });
        BatchScheduleClient batchClient = mock(BatchScheduleClient.class);
        when(batchClient.requestTargets(1)).thenReturn(Mono.just(List.of(
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.42", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION))));

        GenericBatchHandler handler = new GenericBatchHandler(
                fanout, mapper, SubBatchSpec.parse("size:5"), batchClient, true);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a");
        ObjectNode userGc = body.putObject("generate_config");
        ArrayNode userRoleAddrs = userGc.putArray("role_addrs");
        ObjectNode userAddr = userRoleAddrs.addObject();
        userAddr.put("role", "DECODE");
        userAddr.put("ip", "192.168.1.99");
        userAddr.put("http_port", 9000);
        userAddr.put("grpc_port", 9001);

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .expectNextCount(1)
                .verifyComplete();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<ObjectNode>> captor = ArgumentCaptor.forClass(List.class);
        verify(fanout).dispatchChunks(eq("/batch_infer"), captor.capture(), eq(spec));
        JsonNode roleAddrs = captor.getValue().get(0).get("generate_config").get("role_addrs");
        assertEquals(2, roleAddrs.size(),
                "user-supplied role_addr must be preserved, dispatcher's appended");
        assertEquals("DECODE", roleAddrs.get(0).get("role").asText(),
                "user entry preserved at index 0 (insertion order)");
        assertEquals("PDFUSION", roleAddrs.get(1).get("role").asText());
        assertEquals("10.0.0.42", roleAddrs.get(1).get("ip").asText());
    }

    @Test
    void preAssignBeFailureLeavesChunksUnstampedAndStillFansOut() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });
        BatchScheduleClient batchClient = mock(BatchScheduleClient.class);
        // Coordinator unreachable / no BE — client collapses to empty list per its contract.
        when(batchClient.requestTargets(anyInt())).thenReturn(Mono.just(List.of()));

        GenericBatchHandler handler = new GenericBatchHandler(
                fanout, mapper, SubBatchSpec.parse("size:2"), batchClient, true);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
                .verifyComplete();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<ObjectNode>> captor = ArgumentCaptor.forClass(List.class);
        verify(fanout).dispatchChunks(eq("/batch_infer"), captor.capture(), eq(spec));
        for (ObjectNode chunk : captor.getValue()) {
            JsonNode gc = chunk.get("generate_config");
            assertNull(gc == null ? null : gc.get("role_addrs"),
                    "empty target list must leave generate_config.role_addrs unset — never half-stamp");
        }
    }

    @Test
    void preAssignBeOffSkipsBatchScheduleClientEntirely() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });
        BatchScheduleClient batchClient = mock(BatchScheduleClient.class);

        GenericBatchHandler handler = new GenericBatchHandler(
                fanout, mapper, SubBatchSpec.parse("size:2"), batchClient, false);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .expectNextCount(1)
                .verifyComplete();

        verify(batchClient, never()).requestTargets(anyInt());

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<ObjectNode>> captor = ArgumentCaptor.forClass(List.class);
        verify(fanout).dispatchChunks(eq("/batch_infer"), captor.capture(), eq(spec));
        for (ObjectNode chunk : captor.getValue()) {
            JsonNode gc = chunk.get("generate_config");
            assertNull(gc == null ? null : gc.get("role_addrs"),
                    "preAssignBe=false must leave generate_config.role_addrs unset — opt-in flag honored");
        }
    }

    private List<ObjectNode> runSingleChunkAndCaptureChunkBodies(ObjectNode body) {
        FanoutService fanout = mock(FanoutService.class);
        ObjectNode feResp = mapper.createObjectNode();
        feResp.putArray("response_batch").addObject().put("response", "r0");
        when(fanout.dispatchChunks(any(), anyList(), eq(spec)))
                .thenReturn(Mono.just(List.of(SubBatchResult.ok(feResp, 1, 0))));

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, SubBatchSpec.parse("size:5"));
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .expectNextCount(1)
                .verifyComplete();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<ObjectNode>> captor = ArgumentCaptor.forClass(List.class);
        verify(fanout).dispatchChunks(eq("/batch_infer"), captor.capture(), eq(spec));
        return captor.getValue();
    }

}
