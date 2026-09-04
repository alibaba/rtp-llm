package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutException;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okio.Buffer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.http.HttpMethod;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.http.codec.HttpMessageWriter;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.EntityResponse;
import org.springframework.web.reactive.function.server.HandlerStrategies;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.reactive.result.view.ViewResolver;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;
import reactor.test.StepVerifier;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Timeout(30)
class PassthroughClientTest {

    private MockWebServer server;

    @BeforeEach
    void start() throws Exception {
        server = new MockWebServer();
        server.start();
    }

    @AfterEach
    void stop() throws Exception {
        server.shutdown();
    }

    @Test
    void forwardsPathAndReturnsBodyVerbatim() throws Exception {
        server.enqueue(new MockResponse()
                .setBody("{\"status\":\"ok\"}")
                .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        Mono<ServerResponse> resp = client.forward(request);
        StepVerifier.create(resp)
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = takeRequestWithin(server);
        Assertions.assertEquals("/worker_status", rec.getPath());
    }

    @Test
    void bareDispatcherPathNormalizesToFeRoot() throws Exception {
        // POST /dispatcher (no trailing slash) misses the /dispatcher/ batch route and lands in
        // passthrough; forwarding the literal "/dispatcher" would 404 at FE with no hint the
        // caller merely dropped the slash. It must normalize to the FE root path instead.
        server.enqueue(new MockResponse()
                .setBody("{\"status\":\"ok\"}")
                .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("/dispatcher"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = takeRequestWithin(server);
        Assertions.assertEquals("/", rec.getPath());
    }

    @Test
    void forwardsFeNon2xxStatusVerbatim() throws Exception {
        // A reachable FE returning 404/500 is the FE's own answer, not a dispatcher failure — it
        // must pass through with the original status (502 is reserved for "could not reach any FE").
        server.enqueue(new MockResponse()
                .setResponseCode(404)
                .setBody("{\"error\":\"no such model\"}")
                .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/v1/models/missing"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> Assertions.assertEquals(404, r.statusCode().value(),
                        "FE 404 must pass through verbatim, not be remapped to 502"))
                .verifyComplete();

        RecordedRequest rec = takeRequestWithin(server);
        Assertions.assertEquals("/v1/models/missing", rec.getPath());
    }

    @Test
    void nonIanaFeStatusCodePropagatesWithoutException() throws Exception {
        // Intermediaries mint non-IANA codes (599 is a common proxy-timeout convention).
        // rawStatusCode() must carry it through verbatim; resolving the HttpStatus enum would
        // throw IllegalArgumentException and turn the FE's own answer into a dispatcher 502.
        server.enqueue(new MockResponse()
                .setResponseCode(599)
                .setBody("{\"error\":\"proxy timeout\"}")
                .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/v1/models"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> Assertions.assertEquals(599, r.rawStatusCode(),
                        "non-IANA FE status must pass through verbatim, not be remapped to 502"))
                .verifyComplete();

        takeRequestWithin(server);
    }

    @Test
    void feNeverSendingHeadersFailsAs502WithinHeadersTimeout() {
        // exchange() alone waits forever on an FE that accepted the connection but never sends
        // response headers (an OOM-wedged process keeps its port open); the headers-phase cap
        // must fail the forward at batchTimeoutMs and surface the shared 502 error envelope.
        server.enqueue(new MockResponse()
                .setHeadersDelay(2, TimeUnit.SECONDS)
                .setBody("{\"status\":\"ok\"}"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(300);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), cfg);

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> {
                    Assertions.assertEquals(502, r.statusCode().value());
                    String body = new String((byte[]) ((EntityResponse<?>) r).entity(), StandardCharsets.UTF_8);
                    Assertions.assertTrue(body.contains("passthrough_failed"),
                            "headers timeout must surface the shared passthrough error envelope");
                    // Anti-leak: the constant message must never regress to e.getMessage(), which
                    // embeds the FE address and the exception class.
                    Assertions.assertFalse(body.contains(server.getHostName()),
                            "502 body must not leak the FE host: " + body);
                    Assertions.assertFalse(body.contains(String.valueOf(server.getPort())),
                            "502 body must not leak the FE port: " + body);
                    Assertions.assertFalse(body.contains("Exception"),
                            "502 body must not leak exception class text: " + body);
                })
                .expectComplete()
                .verify(Duration.ofSeconds(5));
    }

    @Test
    void emptyFePoolBecomes502JsonError() {
        FePool pool = DispatcherTestSupport.fePool(List::of, url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> {
                    Assertions.assertEquals(502, r.statusCode().value());
                    Assertions.assertEquals(org.springframework.http.MediaType.APPLICATION_JSON,
                            r.headers().getContentType());
                    String body = new String((byte[]) ((EntityResponse<?>) r).entity(), StandardCharsets.UTF_8);
                    Assertions.assertTrue(body.contains("upstream request failed"),
                            "502 body must be the constant message: " + body);
                    // Anti-leak: a regression back to e.getMessage() would embed the pick
                    // failure's exception class text.
                    Assertions.assertFalse(body.contains("Exception"),
                            "502 body must not leak exception class text: " + body);
                })
                .verifyComplete();
    }

    @Test
    void assemblyFailureAfterHeadersReleasesFeBodyAndReturns502() {
        // exchange() defers FE connection release to body consumption at writeTo time. If building
        // the passthrough ServerResponse throws AFTER FE headers arrive (e.g. copyEndToEnd chokes on
        // a pathological header set), the body is never handed to a self-releasing consumer, so that
        // non-consuming exit must release it explicitly or the pooled connection leaks. Drive the
        // real catch branch by injecting a ClientResponse whose header copy throws, and assert both
        // the shared 502 envelope and that releaseBody() was actually subscribed.
        //
        // Note: that catch guards the response-assembly block (header copy, body wiring,
        // doOnCancel), not the PV/metrics emit — the emit runs in a separate best-effort try that
        // swallows its own failures (so it can never leak the body nor double-emit). asHttpHeaders()
        // is the earliest reachable throw in the assembly block and stands in for the rest.
        AtomicBoolean released = new AtomicBoolean(false);
        ClientResponse.Headers headers = mock(ClientResponse.Headers.class);
        when(headers.asHttpHeaders()).thenThrow(new RuntimeException("boom copying FE headers"));
        ClientResponse feResponse = mock(ClientResponse.class);
        when(feResponse.rawStatusCode()).thenReturn(200);
        when(feResponse.headers()).thenReturn(headers);
        when(feResponse.releaseBody()).thenReturn(Mono.<Void>fromRunnable(() -> released.set(true)));

        WebClient webClient = WebClient.builder()
                .exchangeFunction(req -> Mono.just(feResponse))
                .build();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of("http://fe-unused:8088"), url -> true);
        PassthroughClient client =
                new PassthroughClient(webClient, pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> {
                    Assertions.assertEquals(502, r.statusCode().value());
                    String body = new String((byte[]) ((EntityResponse<?>) r).entity(), StandardCharsets.UTF_8);
                    Assertions.assertTrue(body.contains("passthrough_failed"),
                            "assembly failure must surface the shared passthrough 502 envelope: " + body);
                })
                .verifyComplete();

        // Mutation guard: drop `clientResponse.releaseBody().subscribe()` from the assembly catch and
        // this fails — the FE body (its pooled connection) would leak on this non-consuming exit.
        Assertions.assertTrue(released.get(),
                "assembly failure must release the FE body so the pooled connection is not leaked");
    }

    @Test
    void preservesQueryStringOnForward() throws Exception {
        server.enqueue(new MockResponse()
                .setBody("{\"status\":\"ok\"}")
                .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status?role=PREFILL&verbose=1"))
                .body(Flux.empty());

        Mono<ServerResponse> resp = client.forward(request);
        StepVerifier.create(resp)
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = takeRequestWithin(server);
        Assertions.assertEquals("/worker_status?role=PREFILL&verbose=1", rec.getPath());
    }

    @Test
    void forwardsToFeStrippingDispatcherPrefix() throws Exception {
        server.enqueue(new MockResponse().setBody("ok").setResponseCode(200));
        FePool pool = DispatcherTestSupport.fePool(() -> List.of("http://" + server.getHostName() + ":" + server.getPort()), url -> true);
        PassthroughClient passthrough =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/dispatcher/worker_status?role=PREFILL"))
                .body(Flux.empty());

        StepVerifier.create(passthrough.forward(request))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest recorded = takeRequestWithin(server);
        Assertions.assertEquals("/worker_status?role=PREFILL", recorded.getPath());
    }

    @Test
    void metricPathTagCollapsesUnregisteredPathsToOther() {
        // The passthrough accepts arbitrary client URIs; the kmonitor path tag must stay bounded
        // to the registered spec paths so a scanner or typo'd path cannot mint unbounded tag
        // values. Unregistered paths collapse to "other"; registered ones keep their own tag.
        server.enqueue(new MockResponse().setBody("ok"));
        server.enqueue(new MockResponse().setBody("ok"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        DispatcherTestSupport.RecordingMetrics metrics = DispatcherTestSupport.recordingMetrics();
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, metrics, new DispatchConfig());

        MockServerRequest unregistered = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/dispatcher/worker_status"))
                .body(Flux.empty());
        StepVerifier.create(client.forward(unregistered))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        MockServerRequest registered = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("/dispatcher/batch_infer"))
                .body(Flux.empty());
        StepVerifier.create(client.forward(registered))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        Assertions.assertEquals(2, metrics.requestReports.size());
        Assertions.assertEquals("passthrough", metrics.requestReports.get(0).type());
        Assertions.assertEquals("other", metrics.requestReports.get(0).path(),
                "unregistered paths must collapse to the bounded \"other\" tag");
        Assertions.assertEquals("/batch_infer", metrics.requestReports.get(1).path(),
                "registered spec paths keep their own tag");
    }

    @Test
    void stripsHopByHopAndFramingHeadersFromOutboundRequest() throws Exception {
        server.enqueue(new MockResponse().setBody("ok").setResponseCode(200));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .header("Host", "phony.example:9999")
                .header("Connection", "close")
                .header("Upgrade", "websocket")
                .header("Transfer-Encoding", "chunked")
                .header("Proxy-Authorization", "Basic deadbeef")
                .header("TE", "trailers")
                .header("X-Trace-Id", "trace-keep-me")
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = takeRequestWithin(server);
        Assertions.assertNotEquals("phony.example:9999", rec.getHeader("Host"),
                "inbound Host must not leak — WebClient owns the outbound Host");
        Assertions.assertNull(rec.getHeader("Connection"));
        Assertions.assertNull(rec.getHeader("Upgrade"));
        Assertions.assertNull(rec.getHeader("Proxy-Authorization"));
        Assertions.assertNull(rec.getHeader("TE"));
        Assertions.assertEquals("trace-keep-me", rec.getHeader("X-Trace-Id"),
                "non-hop-by-hop headers must still pass through");
    }

    @Test
    void hopByHopFilterIsCaseInsensitive() throws Exception {
        server.enqueue(new MockResponse().setBody("ok").setResponseCode(200));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .header("PROXY-AUTHORIZATION", "Basic deadbeef")
                .header("connection", "close")
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = takeRequestWithin(server);
        Assertions.assertNull(rec.getHeader("Proxy-Authorization"));
        Assertions.assertNull(rec.getHeader("Connection"));
    }

    @Test
    void stripsHopByHopHeadersFromForwardedResponse() throws Exception {
        server.enqueue(new MockResponse()
                .setBody("{\"ok\":true}")
                .setHeader("Content-Type", "application/json")
                .setHeader("Connection", "close")
                .setHeader("Keep-Alive", "timeout=5")
                .setHeader("Proxy-Authenticate", "Basic realm=fe")
                .setHeader("X-Backend-Id", "fe-7"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> {
                    Assertions.assertEquals(200, r.statusCode().value());
                    Assertions.assertFalse(r.headers().containsKey("Connection"));
                    Assertions.assertFalse(r.headers().containsKey("Keep-Alive"));
                    Assertions.assertFalse(r.headers().containsKey("Proxy-Authenticate"));
                    Assertions.assertEquals("fe-7", r.headers().getFirst("X-Backend-Id"),
                            "end-to-end response headers must still pass through");
                })
                .verifyComplete();
    }

    @Test
    void cancellingForwardDoesNotThrowAndReleasesUpstream() throws Exception {
        server.enqueue(new MockResponse()
                .setBody("first")
                .setBodyDelay(2, TimeUnit.SECONDS));
        server.enqueue(new MockResponse().setBody("second"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        reactor.netty.resources.ConnectionProvider provider =
                reactor.netty.resources.ConnectionProvider.builder("passthrough-cancel-release")
                        .maxConnections(1)
                        .pendingAcquireTimeout(Duration.ofSeconds(3))
                        .build();
        try {
            WebClient webClient = WebClient.builder()
                    .clientConnector(new ReactorClientHttpConnector(HttpClient.create(provider)))
                    .build();
            PassthroughClient client = new PassthroughClient(
                    webClient, pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());

            MockServerRequest first = MockServerRequest.builder()
                    .method(HttpMethod.GET).uri(URI.create("/a")).body(Flux.empty());
            ServerResponse firstResponse = client.forward(first).block(Duration.ofSeconds(5));
            Assertions.assertNotNull(firstResponse, "first response headers were never received");
            Assertions.assertEquals("/a", takeRequestWithin(server).getPath());

            // Subscribe the delayed response body, then cancel that real upstream subscription.
            // Cancelling only client.forward() before it emits a ServerResponse never exercises
            // the body-release path and can leave the first queued response for the next request.
            MockServerWebExchange firstExchange = MockServerWebExchange.from(
                    MockServerHttpRequest.get("http://x/a"));
            Disposable firstBody = firstResponse.writeTo(firstExchange, responseContext()).subscribe();
            firstBody.dispose();

            // maxConnections=1 proves cancellation released the pool slot. Consume and assert the
            // second body so accidentally reading the first queued response cannot pass the test.
            MockServerRequest second = MockServerRequest.builder()
                    .method(HttpMethod.GET).uri(URI.create("/b")).body(Flux.empty());
            MockServerWebExchange secondExchange = MockServerWebExchange.from(
                    MockServerHttpRequest.get("http://x/b"));
            client.forward(second)
                    .flatMap(response -> response.writeTo(secondExchange, responseContext()))
                    .block(Duration.ofSeconds(5));
            Assertions.assertEquals("second",
                    secondExchange.getResponse().getBodyAsString().block(Duration.ofSeconds(5)));
            Assertions.assertEquals("/b", takeRequestWithin(server).getPath());
        } finally {
            provider.disposeLater().block(Duration.ofSeconds(5));
        }
    }

    /** Production wiring must not impose a response timeout on a valid quiet SSE stream. */
    @Test
    void streamingResponseWithLongBodyDelayIsNotCutOff() {
        server.enqueue(buildSseResponseWithDelayedBody());

        reactor.netty.resources.ConnectionProvider provider =
                reactor.netty.resources.ConnectionProvider.builder("sse-production-wiring").build();
        try {
            WebClient webClient = new DispatcherConfiguration()
                    .dispatcherPassthroughWebClient(WebClient.builder(), provider);
            PassthroughClient client = streamingPassthroughClient(webClient);

            StepVerifier.create(forwardForStreaming(client))
                    .expectComplete()
                    .verify(Duration.ofSeconds(10));
        } finally {
            provider.disposeLater().block(Duration.ofSeconds(5));
        }
    }

    /**
     * Regression guard: if anyone reintroduces {@code responseTimeout} on the passthrough
     * HttpClient, ReadTimeoutHandler fires during the body delay and the stream dies.
     * This test pins that behavior so the failure is loud the moment the wiring drifts.
     */
    @Test
    void addingResponseTimeoutWouldKillTheStream() {
        server.enqueue(buildSseResponseWithDelayedBody());

        HttpClient http = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 2000)
                .responseTimeout(Duration.ofMillis(500));
        WebClient webClient = WebClient.builder()
                .clientConnector(new ReactorClientHttpConnector(http))
                .build();
        PassthroughClient client = streamingPassthroughClient(webClient);

        StepVerifier.create(forwardForStreaming(client))
                .expectErrorMatches(ex -> hasCause(ex, ReadTimeoutException.class))
                .verify(Duration.ofSeconds(5));
    }

    private PassthroughClient streamingPassthroughClient(WebClient webClient) {
        FePool pool = DispatcherTestSupport.fePool(
                () -> List.of("http://" + server.getHostName() + ":" + server.getPort()),
                url -> true);
        return new PassthroughClient(webClient, pool, DispatcherTestSupport.noopMetrics(), new DispatchConfig());
    }

    private static RecordedRequest takeRequestWithin(MockWebServer server) throws InterruptedException {
        RecordedRequest rec = server.takeRequest(5, TimeUnit.SECONDS);
        Assertions.assertNotNull(rec, "FE never received the forwarded request within 5s");
        return rec;
    }

    /** Body delay must stay comfortably above the responseTimeout in the regression guard. */
    private static MockResponse buildSseResponseWithDelayedBody() {
        Buffer body = new Buffer();
        for (int i = 0; i < 6; i++) {
            body.writeUtf8("data: chunk" + i + "\n\n");
        }
        return new MockResponse()
                .setHeader("content-type", "text/event-stream")
                .setBody(body)
                .setBodyDelay(1500, TimeUnit.MILLISECONDS);
    }

    private static Mono<Void> forwardForStreaming(PassthroughClient client) {
        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/v1/chat/completions"))
                .body(Flux.empty());
        MockServerWebExchange exchange = MockServerWebExchange.from(
                MockServerHttpRequest.post("http://x/dispatcher/v1/chat/completions"));
        return client.forward(request).flatMap(resp -> resp.writeTo(exchange, responseContext()));
    }

    private static ServerResponse.Context responseContext() {
        return new ServerResponse.Context() {
            @Override
            public List<HttpMessageWriter<?>> messageWriters() {
                return HandlerStrategies.withDefaults().messageWriters();
            }

            @Override
            public List<ViewResolver> viewResolvers() {
                return Collections.emptyList();
            }
        };
    }

    private static boolean hasCause(Throwable t, Class<? extends Throwable> type) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (type.isInstance(c)) {
                return true;
            }
        }
        return false;
    }
}
