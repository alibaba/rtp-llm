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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
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
        when(fanoutService.dispatchChunks(anyString(), anyList(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(3, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        verifyNoInteractions(passthroughClient);
        org.mockito.Mockito.verify(fanoutService)
                .dispatchChunks(eq("/v1/embeddings"), anyList(), eq(spec));
    }

    @Test
    void preAssignSkipsEndpointsWhoseFeModelIgnoresGenerateConfig() {
        // FE's BatchChatCompletionRequest (plain pydantic BaseModel) silently drops unknown
        // top-level fields, so stamping generate_config.role_addrs there is a dead write.
        // With preAssignBe on, the handler must not consume a /batch_schedule round-trip
        // (advancing master's RR cursor and emitting fake pre-assign metrics) for it.
        org.mockito.Mockito.when(cfg.isPreAssignBe()).thenReturn(true);
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/batch/chat/completions");
        stubBody("{\"requests\":[{\"messages\":[]},{\"messages\":[]}]}");
        when(fanoutService.dispatchChunks(anyString(), anyList(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(2, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        verifyNoInteractions(batchScheduleClient);
    }

    @Test
    void preAssignStillRunsForPromptBatchEndpoints() {
        org.mockito.Mockito.when(cfg.isPreAssignBe()).thenReturn(true);
        handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\"]}");
        when(batchScheduleClient.requestTargets(anyInt()))
                .thenReturn(Mono.just(List.of()));
        when(fanoutService.dispatchChunks(anyString(), anyList(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(2, 0, "fe_http_500"))));

        handler.handle(serverRequest, spec).block();

        org.mockito.Mockito.verify(batchScheduleClient).requestTargets(anyInt());
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
        when(fanoutService.dispatchChunks(anyString(), anyList(), any()))
                .thenReturn(Mono.just(List.of(
                        SubBatchResult.failed(2, 0, "fe_http_500"),
                        SubBatchResult.failed(2, 2, "fe_http_500"))));

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
        assertEquals("fe_http_500", body.get("failed_reasons").get(0).asText());
        verifyNoInteractions(passthroughClient);
    }

    @Test
    void allChunksFailedWithUniformFeClientErrorReturnsThatStatus() {
        // When every sub-batch failed with the SAME FE 4xx (a client error, not a transport
        // failure), the all-failed response surfaces that 4xx instead of masking it as 500 — a
        // client that sent a bad batch gets a 4xx, not a misleading server error.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        stubBody("{\"prompt_batch\":[\"a\",\"b\",\"c\",\"d\"]}");
        when(fanoutService.dispatchChunks(anyString(), anyList(), any()))
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
