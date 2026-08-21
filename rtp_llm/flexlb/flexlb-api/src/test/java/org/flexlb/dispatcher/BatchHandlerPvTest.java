package org.flexlb.dispatcher;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;
import org.springframework.web.reactive.function.server.ServerRequest;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class BatchHandlerPvTest {

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

    private ch.qos.logback.classic.Logger pvLogger;
    private ListAppender<ILoggingEvent> pvAppender;
    private Level originalPvLevel;

    @BeforeEach
    void setUp() {
        when(cfg.getSubBatchSpec()).thenReturn(SubBatchSpec.parse("count:2"));
        when(cfg.isPreAssignBe()).thenReturn(false);
        lenient().when(batchScheduleClient.requestTargets(org.mockito.ArgumentMatchers.anyInt()))
                .thenReturn(Mono.just(List.of()));
        // BatchHandler now relays the caller's end-to-end headers + query to each chunk.
        ServerRequest.Headers headers = mock(ServerRequest.Headers.class);
        lenient().when(headers.asHttpHeaders()).thenReturn(new org.springframework.http.HttpHeaders());
        lenient().when(serverRequest.headers()).thenReturn(headers);
        lenient().when(serverRequest.uri()).thenReturn(java.net.URI.create("http://master/dispatcher/batch_infer"));

        pvLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        originalPvLevel = pvLogger.getLevel();
        pvLogger.setLevel(Level.INFO);
        pvAppender = new ListAppender<>();
        pvAppender.start();
        pvLogger.addAppender(pvAppender);
    }

    @AfterEach
    void tearDown() {
        pvLogger.detachAppender(pvAppender);
        pvAppender.stop();
        pvLogger.setLevel(originalPvLevel);
    }

    @Test
    void pv_failed_chunks_counts_failed_chunks_not_failed_items() {
        // 5 items split into 2 chunks; chunk0 (3 items) succeeds, chunk1 (2 items) fails.
        // pv.failedChunks documents "chunks that returned a non-2xx or threw" — must be 1, not 2.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");

        JSONObject okBody = new JSONObject();
        JSONArray okArr = new JSONArray();
        okArr.add("r0");
        okArr.add("r1");
        okArr.add("r2");
        okBody.put(spec.getResponseArrayField(), okArr);

        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(
                        SubBatchResult.ok(okBody, 3, 0),
                        SubBatchResult.failed(2, 3, "fe_http_500"))));

        byte[] body = "{\"prompt_batch\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}".getBytes(StandardCharsets.UTF_8);
        when(serverRequest.bodyToMono(byte[].class)).thenReturn(Mono.just(body));

        BatchHandler handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        handler.handle(serverRequest, spec).block();

        assertEquals(1, pvAppender.list.size(), "exactly one pv record per request");
        String pvJson = pvAppender.list.get(0).getFormattedMessage();
        assertTrue(pvJson.contains("\"failedChunks\":1"),
                "failedChunks must count chunks (1), not failed items (2): " + pvJson);
        assertFalse(pvJson.contains("\"failedChunks\":2"), pvJson);
    }

    @Test
    void pv_client_cancel_before_response_is_recorded_as_499() {
        // A client that disconnects mid-fanout terminates the chain with CANCEL and no response
        // ever gets a status; the pv record must say 499 / "client cancelled" instead of the
        // 0 that would read as "request never finished" in pv.log.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        // Lenient: whether fanout is reached before the cancel propagates depends on timing (the
        // handler now always resolves targets from the master first). The assertion under test is
        // the 499 pv record, not that fanout was invoked, so this stub may go unused on a fast cancel.
        lenient().when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.never());
        byte[] body = "{\"prompt_batch\":[\"a\",\"b\"]}".getBytes(StandardCharsets.UTF_8);
        when(serverRequest.bodyToMono(byte[].class)).thenReturn(Mono.just(body));

        BatchHandler handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        reactor.test.StepVerifier.create(handler.handle(serverRequest, spec))
                .thenCancel()
                .verify();

        assertEquals(1, pvAppender.list.size(), "exactly one pv record per request, even on cancel");
        String pvJson = pvAppender.list.get(0).getFormattedMessage();
        assertTrue(pvJson.contains("\"httpStatus\":499"),
                "a cancelled request must be recorded with the 499 convention: " + pvJson);
        assertTrue(pvJson.contains("client cancelled"),
                "a cancelled request must carry the stable cancel reason: " + pvJson);
    }

    @Test
    void pv_identifies_endpoint_model_caller_and_chunk_shape_without_request_body() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/reranker");

        JSONObject firstBody = JSONObject.of(
                "results", JSONArray.of(
                        JSONObject.of("index", 0, "relevance_score", 0.2),
                        JSONObject.of("index", 1, "relevance_score", 0.8),
                        JSONObject.of("index", 2, "relevance_score", 0.1)),
                "total_tokens", 10);
        JSONObject secondBody = JSONObject.of(
                "results", JSONArray.of(
                        JSONObject.of("index", 0, "relevance_score", 0.9),
                        JSONObject.of("index", 1, "relevance_score", 0.7)),
                "total_tokens", 11);
        when(fanoutService.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(
                        SubBatchResult.ok(firstBody, 3, 0),
                        SubBatchResult.ok(secondBody, 2, 3))));
        byte[] body = ("{\"query\":\"q\",\"model\":\"bge_reranker_large\","
                + "\"__request_id__\":146280,"
                + "\"documents\":[\"d0\",\"d1\",\"d2\",\"d3\",\"d4\"]}")
                .getBytes(StandardCharsets.UTF_8);
        when(serverRequest.bodyToMono(byte[].class)).thenReturn(Mono.just(body));

        BatchHandler handler = new BatchHandler(fanoutService, cfg, batchScheduleClient, passthroughClient,
                DispatcherTestSupport.noopMetrics());
        handler.handle(serverRequest, spec).block();

        assertEquals(1, pvAppender.list.size(), "exactly one pv record per request");
        JSONObject pv = JSONObject.parseObject(pvAppender.list.get(0).getFormattedMessage());
        assertEquals("/v1/reranker", pv.getString("path"));
        assertEquals("bge_reranker_large", pv.getString("model"));
        assertEquals("146280", pv.getString("callerRequestId"));
        assertEquals("count:2", pv.getString("splitPolicy"));
        assertEquals(5, pv.getIntValue("totalItems"));
        assertEquals(2, pv.getIntValue("chunkCount"));
        assertEquals(2, pv.getIntValue("minChunkItems"));
        assertEquals(3, pv.getIntValue("maxChunkItems"));
        assertFalse(pv.toJSONString().contains("documents"),
                "dispatcher PV must describe shape without duplicating request contents");
    }
}
