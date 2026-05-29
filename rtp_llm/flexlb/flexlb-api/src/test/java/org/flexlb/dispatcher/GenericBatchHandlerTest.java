package org.flexlb.dispatcher;

import static org.flexlb.dispatcher.BatchEndpointSpec.FailedItemFactory;
import static org.flexlb.dispatcher.DispatcherTestSupport.genericBatchHandler;

import org.flexlb.dispatcher.FanoutService.SubBatchResult;

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
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

/**
 * Tests for {@link GenericBatchHandler}'s orchestration: request parsing, validation, empty-batch
 * short-circuit, fanout integration, merger integration, error mapping, and the pre-assign
 * config-vs-client gating contract.
 *
 * <p>Pure chunk-shape assertions (force_batch injection, role_addrs stamping, split distribution)
 * live in {@link BatchChunkBuilderTest} — testing them through the handler with
 * {@link ArgumentCaptor} adds Spring/reactive noise for no extra signal. This suite keeps the
 * two pre-assign cases that exercise the handler's own {@code resolvePreAssignedTargets} branch
 * (preAssignBe=false bypass, empty target list degradation), since those are handler-level
 * integration concerns rather than utility-level transforms.
 */
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

        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:5");
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

        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2");
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
    void allFailedReturns500WithDistinctReasons() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenReturn(
                Mono.just(List.of(
                        SubBatchResult.failed(2, 0, "ConnectException: Connection refused"),
                        SubBatchResult.failed(2, 2, "ConnectException: Connection refused"),
                        SubBatchResult.failed(2, 4, "ReadTimeoutException"))));

        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2");
        ObjectNode body = mapper.createObjectNode();
        ArrayNode batch = body.putArray("prompt_batch");
        for (int i = 0; i < 6; i++) {
            batch.add("p" + i);
        }

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> {
                    assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, resp.statusCode());
                    EntityResponse<?> entity = (EntityResponse<?>) resp;
                    ObjectNode out = (ObjectNode) entity.entity();
                    assertEquals("all_sub_batches_failed", out.get("error").asText());
                    assertEquals(6, out.get("failed_count").asInt());
                    assertEquals(3, out.get("total_chunks").asInt());
                    ArrayNode reasons = (ArrayNode) out.get("failed_reasons");
                    assertEquals(2, reasons.size());
                    assertEquals("ConnectException: Connection refused", reasons.get(0).asText());
                    assertEquals("ReadTimeoutException", reasons.get(1).asText());
                })
                .verifyComplete();
    }

    @Test
    void emptyBatchReturnsShapedEmptyEnvelopeNotEmptyObject() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:5");
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
        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2");

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
        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2");

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
        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2");

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
    void preAssignBeFailureLeavesChunksUnstampedAndStillFansOut() {
        // Handler-level contract: empty target list from BatchScheduleClient (its documented
        // failure-collapse semantics) must degrade to no-stamp without blocking fanout. The
        // stamp-correctness assertions for non-empty target lists live in BatchChunkBuilderTest.
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
        when(batchClient.requestTargets(anyInt())).thenReturn(Mono.just(List.of()));

        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2", batchClient, true);
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
        // Handler-level contract: preAssignBe=false must bypass the BatchScheduleClient call
        // entirely (not just degrade the stamp), so a misconfigured client can't even reach
        // master when the opt-in flag is off.
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

        GenericBatchHandler handler = genericBatchHandler(fanout, mapper, "size:2", batchClient, false);
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
}
