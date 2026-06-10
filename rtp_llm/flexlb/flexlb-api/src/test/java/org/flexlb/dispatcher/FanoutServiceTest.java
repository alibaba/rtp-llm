package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FanoutServiceTest {

    private static final BatchEndpointSpec BATCH_INFER = BatchEndpointSpec.BY_PATH.get("/batch_infer");

    @Test
    void chunkFailureWarnsAreRateLimitedDuringOutage() {
        // During an FE outage at production QPS, per-chunk WARNs reach tens of thousands of
        // lines per second — enough to cost real throughput. Failures must still produce
        // SubBatchResult.failed per chunk, but the WARN stream must be rate-limited.
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a"));
        when(feClient.postBytes(anyString(), anyString(), any()))
                .thenReturn(Mono.error(new RuntimeException("connection refused")));
        FanoutService svc = new FanoutService(feClient, pool);

        ch.qos.logback.classic.Logger flexlbLogger =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        ch.qos.logback.core.read.ListAppender<ch.qos.logback.classic.spi.ILoggingEvent> appender =
                new ch.qos.logback.core.read.ListAppender<>();
        appender.start();
        flexlbLogger.addAppender(appender);
        try {
            java.util.List<JSONObject> chunks = new java.util.ArrayList<>();
            for (int i = 0; i < 200; i++) {
                chunks.add(chunk("p" + i));
            }
            List<SubBatchResult> subs =
                    svc.dispatchChunks("/batch_infer", chunks, BATCH_INFER).block();

            assertNotNull(subs);
            assertEquals(200, subs.size(), "every chunk must still resolve to a result");
            assertTrue(subs.stream().noneMatch(SubBatchResult::success));

            long warns = appender.list.stream()
                    .filter(e -> e.getFormattedMessage().contains("FE chunk failed"))
                    .count();
            assertTrue(warns >= 1, "outage must still be visible in the log");
            assertTrue(warns <= 5,
                    "chunk-failure WARNs must be rate-limited, got " + warns + " for 200 failures");
        } finally {
            flexlbLogger.detachAppender(appender);
            appender.stop();
        }
    }

    @Test
    void fansOutChunksAndPreservesOrder() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a", "http://b"));
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any()))
                .thenReturn(Mono.just(responseBatchBytes("r2")));

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER))
                .assertNext(subs -> {
                    assertEquals(2, subs.size());
                    SubBatchResult s0 = subs.get(0);
                    assertTrue(s0.success());
                    assertEquals(0, s0.startIndex());
                    assertEquals(2, s0.chunkSize());
                    assertEquals("r0",
                            s0.body().getJSONArray("response_batch").getJSONObject(0).getString("response"));
                    SubBatchResult s1 = subs.get(1);
                    assertTrue(s1.success());
                    assertEquals(2, s1.startIndex());
                    assertEquals(1, s1.chunkSize());
                    assertEquals("r2",
                            s1.body().getJSONArray("response_batch").getJSONObject(0).getString("response"));
                })
                .verifyComplete();
    }

    @Test
    void failedChunkBecomesFailedSubResultNotAnError() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a", "http://b"));
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any()))
                .thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER))
                .assertNext(subs -> {
                    assertEquals(2, subs.size());
                    assertTrue(subs.get(0).success());
                    assertFalse(subs.get(1).success());
                    assertEquals(2, subs.get(1).startIndex());
                    assertEquals(1, subs.get(1).chunkSize());
                    assertNotNull(subs.get(1).reason());
                    assertTrue(subs.get(1).reason().contains("RuntimeException"));
                })
                .verifyComplete();
    }

    @Test
    void emptyFePoolFailsChunksSoftly() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of());

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1")), BATCH_INFER))
                .assertNext(subs -> {
                    assertEquals(1, subs.size());
                    assertFalse(subs.get(0).success());
                    assertEquals(0, subs.get(0).startIndex());
                    assertEquals(2, subs.get(0).chunkSize());
                    assertTrue(subs.get(0).reason().contains("IllegalStateException"));
                })
                .verifyComplete();
    }

    private static FePool fePool(List<String> staticUrls) {
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(() -> staticUrls);
        FeHealthChecker hc = mock(FeHealthChecker.class);
        when(hc.isAlive(anyString())).thenReturn(true);
        return new FePool(refresher, hc);
    }

    private static JSONObject chunk(String... prompts) {
        JSONObject body = new JSONObject();
        JSONArray arr = new JSONArray();
        for (String p : prompts) {
            arr.add(p);
        }
        body.put("prompt_batch", arr);
        return body;
    }

    private static byte[] responseBatchBytes(String... responses) {
        StringBuilder sb = new StringBuilder("{\"response_batch\":[");
        for (int i = 0; i < responses.length; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("{\"response\":\"").append(responses[i]).append("\"}");
        }
        sb.append("]}");
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }
}
