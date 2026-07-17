package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.http.HttpHeaders;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.resources.ConnectionProvider;
import reactor.test.StepVerifier;

@Timeout(30)
class FeClientTest {

    private MockWebServer server;
    private ConnectionProvider connectionProvider;

    @BeforeEach
    void start() throws Exception {
        server = new MockWebServer();
        server.start();
        connectionProvider = ConnectionProvider.builder("test").build();
    }

    @AfterEach
    void stop() throws Exception {
        server.shutdown();
        connectionProvider.disposeLater().block(java.time.Duration.ofSeconds(5));
    }

    @Test
    void postsToBatchInferAndReturnsResponseBytes() throws Exception {
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(
                WebClient.builder(), connectionProvider, cfg);

        JSONObject body = new JSONObject();
        body.put("prompt_batch", JSONArray.of("hi"));

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", JSON.toJSONBytes(body), new HttpHeaders(), null))
                .assertNext(bytes -> {
                    JSONObject parsed = JSON.parseObject(bytes);
                    Assertions.assertEquals("ok",
                            parsed.getJSONArray("response_batch").getJSONObject(0).getString("response"));
                })
                .verifyComplete();

        RecordedRequest rec = server.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
        Assertions.assertNotNull(rec, "FE never received the request within 5s");
        Assertions.assertEquals("/batch_infer", rec.getPath());
        Assertions.assertEquals("POST", rec.getMethod());
    }

    @Test
    void relaysEndToEndHeadersAndQueryStringToEveryChunkRequest() throws Exception {
        // The split path must not silently change auth/tenancy/tracing semantics relative to the
        // passthrough path: Authorization, custom end-to-end headers, and the original query
        // string all have to reach the FE-side chunk request.
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(WebClient.builder(), connectionProvider, cfg);

        HttpHeaders inbound = new HttpHeaders();
        inbound.add("Authorization", "Bearer x");
        inbound.add("X-Trace-Id", "trace-keep-me");

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", "{}".getBytes(), inbound, "role=PREFILL&verbose=1"))
                .expectNextCount(1)
                .verifyComplete();

        RecordedRequest rec = server.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
        Assertions.assertNotNull(rec, "FE never received the chunk request within 5s");
        Assertions.assertEquals("/batch_infer?role=PREFILL&verbose=1", rec.getPath(),
                "the caller's query string must survive the split path");
        Assertions.assertEquals("Bearer x", rec.getHeader("Authorization"),
                "Authorization must be relayed to every chunk request");
        Assertions.assertEquals("trace-keep-me", rec.getHeader("X-Trace-Id"),
                "custom end-to-end headers must be relayed to every chunk request");
    }

    @Test
    void dropsAcceptEncodingAndInboundContentTypeOnChunkRequests() throws Exception {
        // The fanout path parses each FE body as raw bytes and re-serializes each chunk body as
        // JSON, so an inbound accept-encoding (a gzipped FE body would break the parse) must be
        // dropped and an inbound content-type must not describe the re-serialized entity.
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(WebClient.builder(), connectionProvider, cfg);

        HttpHeaders inbound = new HttpHeaders();
        inbound.add("Accept-Encoding", "gzip");
        inbound.add("Content-Type", "text/plain");

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", "{}".getBytes(), inbound, null))
                .expectNextCount(1)
                .verifyComplete();

        RecordedRequest rec = server.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
        Assertions.assertNotNull(rec, "FE never received the chunk request within 5s");
        Assertions.assertNull(rec.getHeader("Accept-Encoding"),
                "accept-encoding must not be relayed — the fanout parses the FE body as plain bytes");
        String contentType = rec.getHeader("Content-Type");
        Assertions.assertNotNull(contentType);
        Assertions.assertTrue(contentType.startsWith("application/json"),
                "chunk bodies are re-serialized JSON; the inbound content-type must not win: " + contentType);
    }

    @Test
    void feNon2xxResponseErrorsWithExtractableStatus() {
        // .retrieve() turns a 5xx into a WebClientResponseException; the fanout path relies on
        // DispatcherResponses.httpStatusOf recovering the status so a chunk degrades to a failed
        // SubBatchResult carrying the real FE status (which the all-failed merge can then surface).
        server.enqueue(new MockResponse()
                .setResponseCode(500)
                .setHeader("Content-Type", "application/json")
                .setBody("{\"error\":\"backend boom\"}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(WebClient.builder(), connectionProvider, cfg);

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", "{}".getBytes(), new HttpHeaders(), null))
                .expectErrorSatisfies(e ->
                        Assertions.assertEquals(500, DispatcherResponses.httpStatusOf(e)))
                .verify(java.time.Duration.ofSeconds(5));
    }

    @Test
    void feSlowerThanBatchTimeoutFailsTheCall() {
        // batchTimeoutMs is the header-wait budget: an FE that has not started responding
        // within it must fail the sub-call (the chunk then degrades to SubBatchResult.failed
        // upstream) instead of hanging the whole fanout.
        server.enqueue(new MockResponse()
                .setHeadersDelay(2, java.util.concurrent.TimeUnit.SECONDS)
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(300);
        FeClient client = new FeClient(WebClient.builder(), connectionProvider, cfg);

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", "{}".getBytes(), new HttpHeaders(), null))
                .expectError()
                .verify(java.time.Duration.ofSeconds(5));
    }

    @Test
    void feStallingMidBodyFailsAtOverallTimeout() {
        // batchTimeoutMs (the responseTimeout) is reset by every body byte, so an FE that sends
        // headers and then trickles the body could otherwise pin the request and its pooled
        // connection forever. The whole-call cap must cut it at batchTimeoutMs + bodyReadMarginMs
        // and surface a transport failure for the chunk.
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("x".repeat(64))
                .throttleBody(1, 50, java.util.concurrent.TimeUnit.MILLISECONDS));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(600);
        cfg.setBodyReadMarginMs(200);
        FeClient client = new FeClient(WebClient.builder(), connectionProvider, cfg);

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", "{}".getBytes(), new HttpHeaders(), null))
                .expectErrorSatisfies(e -> {
                    Assertions.assertInstanceOf(java.util.concurrent.TimeoutException.class, e,
                            "the whole-call cap must fire — each 50ms trickle resets the per-read responseTimeout");
                    Assertions.assertEquals(0, DispatcherResponses.httpStatusOf(e),
                            "a timeout is a transport failure and carries no FE HTTP status");
                })
                .verify(java.time.Duration.ofSeconds(5));
    }
}
