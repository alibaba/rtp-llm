package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.HttpHeaders;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.flexlb.dispatcher.DispatcherTestSupport.fePool;
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
        when(feClient.postBytes(anyString(), anyString(), any(), any(), any()))
                .thenReturn(Mono.error(new RuntimeException("connection refused")));
        FanoutService svc = new FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());

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
                    svc.dispatchChunks("/batch_infer", chunks, BATCH_INFER, new HttpHeaders(), null).block();

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
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r2")));

        FanoutService svc = new FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER, new HttpHeaders(), null))
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
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER, new HttpHeaders(), null))
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
    void emptyFeBodyBecomesFailedPlaceholderNotAVanishedChunk() {
        // A 200 with no body completes bodyToMono empty. Without the switchIfEmpty guard the
        // chunk would silently disappear from collectList and the merged response array would
        // come back shorter than the request — breaking index correlation for the caller.
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a", "http://b"));
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.empty());

        FanoutService svc = new FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER, new HttpHeaders(), null))
                .assertNext(subs -> {
                    assertEquals(2, subs.size(), "the empty-body chunk must still produce a result");
                    assertTrue(subs.get(0).success());
                    assertFalse(subs.get(1).success());
                    assertEquals(2, subs.get(1).startIndex());
                    assertEquals(1, subs.get(1).chunkSize());
                    assertTrue(subs.get(1).reason().contains("empty"));
                })
                .verifyComplete();
    }

    @Test
    void garbageFeBodyCountsExactlyOnceAsFailedChunk() {
        // The FE body is parsed BEFORE the ok report: a 200 whose body is not JSON (proxy error
        // page, truncated write) must produce exactly one reportChunk with result "failed" —
        // never an "ok" for the 200 plus a "failed" for the parse — and still hold the chunk's
        // failed placeholder at the right index.
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a", "http://b"));
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just("not json".getBytes(StandardCharsets.UTF_8)));
        DispatcherTestSupport.RecordingMetrics metrics = DispatcherTestSupport.recordingMetrics();
        FanoutService svc = new FanoutService(feClient, pool, metrics);

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER, new HttpHeaders(), null))
                .assertNext(subs -> {
                    assertEquals(2, subs.size(), "the garbage-body chunk must still produce a result");
                    assertTrue(subs.get(0).success());
                    assertFalse(subs.get(1).success());
                    assertEquals(2, subs.get(1).startIndex());
                    assertEquals(1, subs.get(1).chunkSize());
                    assertNotNull(subs.get(1).reason());
                })
                .verifyComplete();

        assertEquals(2, metrics.chunkReports.size(), "exactly one report per chunk — no double count");
        assertEquals(1, metrics.chunkReports.stream()
                        .filter(r -> "ok".equals(r.result())).count(),
                "only the healthy chunk reports ok");
        List<DispatcherTestSupport.RecordingMetrics.ChunkReport> failed = metrics.chunkReports.stream()
                .filter(r -> "failed".equals(r.result())).toList();
        assertEquals(1, failed.size(), "the garbage-body chunk reports failed exactly once");
        assertEquals("transport", failed.get(0).reason(),
                "a parse failure carries no FE status, so it categorizes as transport");
    }

    @Test
    void wellFormed200MetersOkButWrongLengthArrayMetersMalformed() {
        // A valid-JSON 200 whose response array is the wrong length (or absent) is merged as a
        // failure, so it must meter as malformed_body, not ok — the metric follows the merge's own
        // wellFormed authority. chunk0 (size 2) gets a matching 2-element array → ok; chunk1
        // (size 1) gets a 2-element array → length mismatch → malformed.
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a", "http://b"));
        when(feClient.postBytes(eq("http://a"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0", "r1")));
        when(feClient.postBytes(eq("http://b"), eq("/batch_infer"), any(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("x0", "x1")));
        DispatcherTestSupport.RecordingMetrics metrics = DispatcherTestSupport.recordingMetrics();
        FanoutService svc = new FanoutService(feClient, pool, metrics);

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), BATCH_INFER, new HttpHeaders(), null))
                .assertNext(subs -> assertEquals(2, subs.size()))
                .verifyComplete();

        assertEquals(2, metrics.chunkReports.size(), "exactly one report per chunk");
        assertEquals(1, metrics.chunkReports.stream()
                        .filter(r -> "ok".equals(r.result())).count(),
                "only the well-formed chunk reports ok");
        List<DispatcherTestSupport.RecordingMetrics.ChunkReport> failed = metrics.chunkReports.stream()
                .filter(r -> "failed".equals(r.result())).toList();
        assertEquals(1, failed.size());
        assertEquals("malformed_body", failed.get(0).reason(),
                "a wrong-length response array is a malformed 200, distinct from a transport failure");
    }

    @Test
    void emptyFePoolFailsChunksSoftly() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of());

        FanoutService svc = new FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());

        StepVerifier.create(svc.dispatchChunks(
                        "/batch_infer", List.of(chunk("p0", "p1")), BATCH_INFER, new HttpHeaders(), null))
                .assertNext(subs -> {
                    assertEquals(1, subs.size());
                    assertFalse(subs.get(0).success());
                    assertEquals(0, subs.get(0).startIndex());
                    assertEquals(2, subs.get(0).chunkSize());
                    assertTrue(subs.get(0).reason().contains("IllegalStateException"));
                })
                .verifyComplete();
    }

    @Test
    void fanoutWriteNullsControlsNullPreservationInChunkPayload() {
        // /v1/embeddings sets fanoutWriteNulls=true so a user-supplied explicit null reaches FE
        // byte-for-byte; /batch_infer sets it false and drops the null (the common-wire-shape win).
        FeClient feClient = mock(FeClient.class);
        FePool pool = fePool(List.of("http://a"));
        ArgumentCaptor<byte[]> payload = ArgumentCaptor.forClass(byte[].class);
        when(feClient.postBytes(anyString(), anyString(), payload.capture(), any(), any()))
                .thenReturn(Mono.just(responseBatchBytes("r0")));
        FanoutService svc = new FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());

        JSONObject embeddingBody = new JSONObject();
        embeddingBody.put("input", JSONArray.of("a"));
        embeddingBody.put("user", null);
        svc.dispatchChunks("/v1/embeddings", List.of(embeddingBody),
                BatchEndpointSpec.BY_PATH.get("/v1/embeddings"), new HttpHeaders(), null).block();
        assertTrue(new String(payload.getValue(), StandardCharsets.UTF_8).contains("\"user\":null"),
                "fanoutWriteNulls=true must preserve the explicit null");

        JSONObject promptBody = new JSONObject();
        promptBody.put("prompt_batch", JSONArray.of("a"));
        promptBody.put("user", null);
        svc.dispatchChunks("/batch_infer", List.of(promptBody), BATCH_INFER, new HttpHeaders(), null).block();
        assertFalse(new String(payload.getValue(), StandardCharsets.UTF_8).contains("\"user\""),
                "fanoutWriteNulls=false must drop the explicit null");
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
