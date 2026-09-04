package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.server.EntityResponse;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Pins the batch handler's routing contract for registered paths:
 * <ul>
 *   <li>array field present as a JSON array → split / fanout / merge;</li>
 *   <li>JSON object without the array field (legacy {@code prompt}, OpenAI single-string
 *       {@code input}) → verbatim passthrough to one FE, per the registry contract;</li>
 *   <li>non-object body → 400, nothing reaches an FE;</li>
 *   <li>every sub-batch failed → 500 with {@code error/failed_count/total_chunks/failed_reasons}
 *       (reasons deduplicated).</li>
 * </ul>
 */
@Timeout(30)
@ExtendWith(MockitoExtension.class)
class BatchHandlerContractTest {

    @Mock
    private FanoutService fanoutService;
    @Mock
    private DispatchConfig cfg;
    @Mock
    private BatchScheduleClient batchScheduleClient;
    @Mock
    private PassthroughClient passthroughClient;
    @Mock
    private ServerRequest serverRequest;

    private final ObjectMapper mapper = new ObjectMapper();
    private BatchHandler handler;

    @BeforeEach
    void setUp() {
        lenient().when(cfg.getSubBatchSpec()).thenReturn(SubBatchSpec.parse("count:2"));
        lenient().when(cfg.isPreAssignBe()).thenReturn(false);
        // Master mode requests FE assignment for each splittable batch. The explicit dimensions
        // vary by endpoint and preAssignBe, so the generic fixture accepts either combination.
        lenient().when(batchScheduleClient.requestTargets(
                        org.mockito.ArgumentMatchers.anyInt(), anyBoolean(), anyBoolean()))
                .thenReturn(Mono.just(java.util.List.of()));
        // BatchHandler now relays the caller's end-to-end headers + query to each chunk.
        ServerRequest.Headers headers = mock(ServerRequest.Headers.class);
        lenient().when(headers.asHttpHeaders()).thenReturn(new org.springframework.http.HttpHeaders());
        lenient().when(serverRequest.headers()).thenReturn(headers);
        lenient().when(serverRequest.uri()).thenReturn(java.net.URI.create("http://master/dispatcher/batch_infer"));
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
    }

    private void stubBody(String json) {
        when(serverRequest.bodyToMono(byte[].class))
                .thenReturn(Mono.just(json.getBytes(StandardCharsets.UTF_8)));
    }

    private ServerResponse stubPassthroughResponse() {
        ServerResponse passthroughResponse = ServerResponse.ok().bodyValue("fe-says-hi").block();
        when(passthroughClient.forward(eq(serverRequest), any(byte[].class)))
                .thenReturn(Mono.just(passthroughResponse));
        return passthroughResponse;
    }

    @Test
    void objectBodyWithoutArrayFieldFallsThroughToPassthrough() {
        // Root path's registered field is prompt_batch; the historical `prompt` variant is a
        // legal FE request that must reach FE verbatim instead of dying with 400 here.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt\":[\"a\",\"b\"]}");
        ServerResponse passthroughResponse = stubPassthroughResponse();

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertSame(passthroughResponse, out,
                "non-batch-shaped body on a registered path must be passthrough-forwarded");
        verifyNoInteractions(fanoutService, batchScheduleClient);
    }

    @Test
    void singleStringEmbeddingsInputFallsThroughToPassthrough() {
        // OpenAI allows `input` as a plain string; the dispatcher only batches when it's a list.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        stubBody("{\"model\":\"m\",\"input\":\"hello world\"}");
        ServerResponse passthroughResponse = stubPassthroughResponse();

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertSame(passthroughResponse, out,
                "single-string input is a legal OpenAI embedding request and must reach FE");
        verifyNoInteractions(fanoutService, batchScheduleClient);
    }

    @Test
    void objectElementEmbeddingsInputFallsThroughToPassthrough() {
        // A single multimodal/chat embedding input is an array of ContentPart/ChatMessage
        // objects — one input, not a batch. Splitting it per element would fragment the
        // input into broken sub-requests.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        stubBody("{\"model\":\"m\",\"input\":["
                + "{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://x/y.png\"}},"
                + "{\"type\":\"text\",\"text\":\"describe\"}]}");
        ServerResponse passthroughResponse = stubPassthroughResponse();

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertSame(passthroughResponse, out,
                "object-element input is a single embedding input and must reach FE whole");
        verifyNoInteractions(fanoutService, batchScheduleClient);
    }

    @Test
    void promptBatchWithSampleAlignedImagesFallsThroughToPassthrough() {
        // Root `/` and /batch_infer accept top-level `images`/`urls` positionally aligned to
        // prompt_batch; the dispatcher does not slice them, so a split would mis-align every
        // chunk. Forward the intact body to one FE instead.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\"],"
                + "\"images\":[[\"http://x/0.png\"],[\"http://x/1.png\"]]}");
        ServerResponse passthroughResponse = stubPassthroughResponse();

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertSame(passthroughResponse, out,
                "a prompt_batch body carrying sample-aligned images must be forwarded whole");
        verifyNoInteractions(fanoutService, batchScheduleClient);
    }

    @Test
    void promptBatchWithListAdapterNameFallsThroughToPassthrough() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\"],"
                + "\"generate_config\":{\"adapter_name\":[\"lora0\",\"lora1\"]}}");
        ServerResponse passthroughResponse = stubPassthroughResponse();

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertSame(passthroughResponse, out,
                "a list-form adapter_name is aligned to prompt_batch and must be forwarded whole");
        verifyNoInteractions(fanoutService, batchScheduleClient);
    }

    @Test
    void stringListEmbeddingsInputStillSplits() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        stubBody("{\"model\":\"m\",\"input\":[\"a\",\"b\",\"c\"]}");
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(3, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        verifyNoInteractions(passthroughClient);
        org.mockito.Mockito.verify(fanoutService)
                .dispatchChunks(eq("/v1/embeddings"), anyList(), anyList(), eq(spec), any(), any());
    }

    @Test
    void emptyRerankerDocumentsReturnsSchemaCompleteEmptyResponse() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/reranker");
        stubBody("{\"query\":\"cape pants\",\"documents\":[]}");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.OK, out.statusCode());
        ObjectNode response = parseBody(out);
        assertEquals(0, response.get("results").size());
        assertEquals(0L, response.get("total_tokens").asLong());
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void rerankerRejectsInvalidRewrittenControlsBeforeFanout() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/reranker");
        stubBody("{\"query\":\"cape pants\",\"documents\":[\"a\",\"b\"],\"top_k\":1.5}");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.BAD_REQUEST, out.statusCode());
        assertEquals("invalid_batch_request", parseBody(out).get("error").asText());
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void nonPreAssignableEndpointRequestsFeOnlyAndDoesNotAdvanceBeCursor() {
        // This endpoint ignores generate_config, so master is asked for FE only. A stampable BE
        // fixture proves that BE data is neither requested nor accidentally written.
        org.mockito.Mockito.when(cfg.isPreAssignBe()).thenReturn(true);
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/batch/chat/completions");
        stubBody("{\"requests\":[{\"messages\":[]},{\"messages\":[]}]}");
        // A stampable BE target — so the skip is proven to come from the endpoint being
        // non-preAssignable, not from an unstampable (portless/roleless) target.
        org.flexlb.dao.loadbalance.BatchScheduleTarget beTarget =
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 8088, 50051,
                        org.flexlb.dao.route.RoleType.PDFUSION);
        beTarget.setFeUrl("http://fe-1");
        when(batchScheduleClient.requestTargets(anyInt(), eq(false), eq(true)))
                .thenReturn(Mono.just(List.of(beTarget)));
        @SuppressWarnings("rawtypes")
        org.mockito.ArgumentCaptor<List> chunkBodies = org.mockito.ArgumentCaptor.forClass(List.class);
        when(fanoutService.dispatchChunks(anyString(), chunkBodies.capture(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(2, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        verify(batchScheduleClient).requestTargets(anyInt(), eq(false), eq(true));
        // ...but no chunk carries a stamped BE role_addrs on a non-preAssignable endpoint.
        for (Object o : chunkBodies.getValue()) {
            JSONObject gc = ((JSONObject) o).getJSONObject("generate_config");
            assertTrue(gc == null || gc.getJSONArray("role_addrs") == null,
                    "a non-preAssignable endpoint must not stamp role_addrs");
        }
    }

    @Test
    void localFeModeWithoutBePreassignmentSkipsMasterEntirely() {
        when(cfg.getFeAllocation()).thenReturn("local");
        when(cfg.isPreAssignBe()).thenReturn(false);
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        stubBody("{\"model\":\"m\",\"input\":[\"a\",\"b\"]}");
        @SuppressWarnings("rawtypes")
        org.mockito.ArgumentCaptor<List> feAssignments =
                org.mockito.ArgumentCaptor.forClass(List.class);
        when(fanoutService.dispatchChunks(anyString(), anyList(), feAssignments.capture(),
                any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(2, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        verifyNoInteractions(batchScheduleClient);
        assertTrue(feAssignments.getValue().isEmpty(),
                "FanoutService must source local mode from FePool, not a stale master assignment");
    }

    @Test
    void chunkCountAboveMasterLimitIsRejectedBeforeScheduling() {
        when(cfg.getSubBatchSpec()).thenReturn(SubBatchSpec.parse("size:2"));
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics(), 2);
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        stubBody("{\"model\":\"m\",\"input\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, out.statusCode());
        ObjectNode body = parseBody(out);
        assertEquals("too_many_sub_batches", body.get("error").asText());
        assertTrue(body.get("message").asText().contains("maximum is 2"));
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void preAssignStillRunsForPromptBatchEndpoints() {
        org.mockito.Mockito.when(cfg.isPreAssignBe()).thenReturn(true);
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\"]}");
        when(batchScheduleClient.requestTargets(anyInt(), eq(true), eq(true)))
                .thenReturn(Mono.just(List.of()));
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(2, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        org.mockito.Mockito.verify(batchScheduleClient)
                .requestTargets(anyInt(), eq(true), eq(true));
    }

    @Test
    void nonObjectBodyIsRejectedWith400WithoutTouchingFe() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("[1,2,3]");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertNotNull(out);
        assertEquals(HttpStatus.BAD_REQUEST, out.statusCode());
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void nonObjectGenerateConfigIsRejectedWith400NotMaskedAs500() {
        // Chunk assembly requires an object; reject a deterministic caller error at the HTTP
        // boundary instead of letting a downstream conversion masquerade as a 500.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\"],\"generate_config\":\"oops\"}");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertNotNull(out);
        assertEquals(HttpStatus.BAD_REQUEST, out.statusCode());
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void callerSuppliedRoleAddrsIsRejectedBeforeScheduling() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\"],\"generate_config\":{"
                + "\"role_addrs\":[{\"role\":\"PDFUSION\",\"ip\":\"1.2.3.4\"}]}}");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.BAD_REQUEST, out.statusCode());
        assertTrue(parseBody(out).get("message").asText().contains("role_addrs"));
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void callerSuppliedRoleAddrsIsRejectedBeforeWholeBodyPassthrough() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\"],\"images\":[[\"https://example/image.png\"]],"
                + "\"generate_config\":{\"role_addrs\":[{\"role\":\"PDFUSION\","
                + "\"ip\":\"1.2.3.4\"}]}}");

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.BAD_REQUEST, out.statusCode());
        assertTrue(parseBody(out).get("message").asText().contains("role_addrs"));
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void aggregateFanoutLimitMapsToStable413() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\"]}");
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.error(new AggregateResponseTooLargeException(8)));

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, out.statusCode());
        assertEquals("batch_response_too_large", parseBody(out).get("error").asText());
    }

    @Test
    void oversizedBodyMapsTo413WithStableErrorCode() {
        // DataBufferLimitException (body over the codec's max-in-memory-size) is a deterministic
        // client error: it must surface as 413 with a stable error code instead of a 500 that
        // invites pointless retries. This class asserts the HTTP contract only; pv assertions
        // live in BatchHandlerPvTest.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        when(serverRequest.bodyToMono(byte[].class)).thenReturn(Mono.error(
                new DataBufferLimitException("Exceeded limit on max bytes to buffer : 16777216")));

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertNotNull(out);
        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, out.statusCode());
        ObjectNode body = parseBody(out);
        assertEquals("request_body_too_large", body.get("error").asText());
        verifyNoInteractions(fanoutService, batchScheduleClient, passthroughClient);
    }

    @Test
    void allChunksFailedReturns500WithDedupedReasons() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\",\"c\",\"d\"]}");
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(
                        SubBatchResult.failed(2, 0, "boom", 500),
                        SubBatchResult.failed(2, 2, "boom", 500))));

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertNotNull(out);
        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, out.statusCode(),
                "500 is reserved for the every-sub-batch-failed case");
        ObjectNode body = parseBody(out);
        assertEquals("all_sub_batches_failed", body.get("error").asText());
        assertEquals(4, body.get("failed_count").asInt(), "failed_count counts items");
        assertEquals(4, body.get("total_count").asInt(),
                "total_count is item units (matches _partial_failure); all items failed here");
        assertEquals(2, body.get("total_chunks").asInt());
        assertEquals(1, body.get("failed_reasons").size(),
                "identical reasons must be deduplicated: " + body.get("failed_reasons"));
        assertEquals("fe_server_error", body.get("failed_reasons").get(0).asText(),
                "deduped reason is the bounded public code derived from the FE status");
        verifyNoInteractions(passthroughClient);
    }

    @Test
    void allChunksFailedWithUniformFeClientErrorReturnsThatStatus() {
        // When every sub-batch failed with the SAME FE 4xx (a client error, not a transport
        // failure), the all-failed response surfaces that 4xx instead of masking it as 500 — a
        // client that sent a bad batch gets a 4xx, not a misleading server error.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\",\"c\",\"d\"]}");
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(
                        SubBatchResult.failed(2, 0, "fe_http_400", 400),
                        SubBatchResult.failed(2, 2, "fe_http_400", 400))));

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertNotNull(out);
        assertEquals(HttpStatus.BAD_REQUEST, out.statusCode(),
                "a uniform FE 4xx across all sub-batches surfaces as that 4xx, not 500");
        ObjectNode body = parseBody(out);
        assertEquals("all_sub_batches_failed", body.get("error").asText());
    }

    @Test
    void rerankerFailsClosedWhenOnlyOneChunkFails() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/reranker");
        stubBody("{\"query\":\"cape pants\",\"documents\":[\"d0\",\"d1\",\"d2\",\"d3\"]}");
        JSONObject okBody = new JSONObject();
        okBody.put("results", com.alibaba.fastjson2.JSONArray.of(
                JSONObject.of("index", 0, "relevance_score", 0.1),
                JSONObject.of("index", 1, "relevance_score", 0.2)));
        okBody.put("total_tokens", 8);
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(
                        SubBatchResult.ok(okBody, 2, 0),
                        SubBatchResult.failed(2, 2, "timeout"))));

        ServerResponse out = handler.handle(serverRequest, spec).block();

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, out.statusCode());
        ObjectNode response = parseBody(out);
        assertEquals("sub_batch_failed", response.get("error").asText());
        assertEquals(2, response.get("failed_count").asInt());
        assertEquals(4, response.get("total_count").asInt());
        assertEquals(2, response.get("total_chunks").asInt());
    }

    private ObjectNode parseBody(ServerResponse resp) {
        Object value = ((EntityResponse<?>) resp).entity();
        try {
            if (value instanceof byte[] bytes) {
                return (ObjectNode) mapper.readTree(bytes);
            }
            if (value instanceof JSONObject json) {
                return (ObjectNode) mapper.readTree(json.toJSONString());
            }
            throw new IllegalStateException("unexpected entity type: " + value.getClass());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
