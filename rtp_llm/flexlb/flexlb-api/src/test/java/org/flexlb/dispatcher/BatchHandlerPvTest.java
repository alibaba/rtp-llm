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

        when(fanoutService.dispatchChunks(anyString(), anyList(), any(), any(), any()))
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
        when(fanoutService.dispatchChunks(anyString(), anyList(), any(), any(), any()))
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
}
