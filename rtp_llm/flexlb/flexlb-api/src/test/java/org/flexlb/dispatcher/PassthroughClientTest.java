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
import org.springframework.http.HttpMethod;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.http.codec.HttpMessageWriter;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.HandlerStrategies;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.reactive.result.view.ViewResolver;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;
import reactor.test.StepVerifier;

import java.net.URI;
import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

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
                new PassthroughClient(WebClient.builder().build(), pool);

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        Mono<ServerResponse> resp = client.forward(request);
        StepVerifier.create(resp)
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = server.takeRequest();
        Assertions.assertEquals("/worker_status", rec.getPath());
    }

    @Test
    void emptyFePoolBecomesErrorMono() {
        FePool pool = DispatcherTestSupport.fePool(List::of, url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool);

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .expectError(IllegalStateException.class)
                .verify();
    }

    @Test
    void preservesQueryStringOnForward() throws Exception {
        server.enqueue(new MockResponse()
                .setBody("{\"status\":\"ok\"}")
                .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool);

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status?role=PREFILL&verbose=1"))
                .body(Flux.empty());

        Mono<ServerResponse> resp = client.forward(request);
        StepVerifier.create(resp)
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = server.takeRequest();
        Assertions.assertEquals("/worker_status?role=PREFILL&verbose=1", rec.getPath());
    }

    @Test
    void forwardsToFeStrippingDispatcherPrefix() throws Exception {
        server.enqueue(new MockResponse().setBody("ok").setResponseCode(200));
        FePool pool = DispatcherTestSupport.fePool(() -> List.of("http://" + server.getHostName() + ":" + server.getPort()), url -> true);
        PassthroughClient passthrough =
                new PassthroughClient(WebClient.builder().build(), pool);

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/dispatcher/worker_status?role=PREFILL"))
                .body(Flux.empty());

        StepVerifier.create(passthrough.forward(request))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest recorded = server.takeRequest();
        Assertions.assertEquals("/worker_status?role=PREFILL", recorded.getPath());
    }

    @Test
    void stripsHopByHopAndFramingHeadersFromOutboundRequest() throws Exception {
        server.enqueue(new MockResponse().setBody("ok").setResponseCode(200));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool);

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

        RecordedRequest rec = server.takeRequest();
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
                new PassthroughClient(WebClient.builder().build(), pool);

        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("/worker_status"))
                .header("PROXY-AUTHORIZATION", "Basic deadbeef")
                .header("connection", "close")
                .body(Flux.empty());

        StepVerifier.create(client.forward(request))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        RecordedRequest rec = server.takeRequest();
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
                new PassthroughClient(WebClient.builder().build(), pool);

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
        server.enqueue(new MockResponse().setBody("first"));
        server.enqueue(new MockResponse().setBody("second"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = DispatcherTestSupport.fePool(() -> List.of(base), url -> true);
        PassthroughClient client =
                new PassthroughClient(WebClient.builder().build(), pool);

        MockServerRequest first = MockServerRequest.builder()
                .method(HttpMethod.GET).uri(URI.create("/a")).body(Flux.empty());
        // Subscribe and immediately cancel — doOnCancel must release the FE channel without throwing.
        StepVerifier.create(client.forward(first)).thenCancel().verify();

        // A follow-up request must still complete successfully through the same pool.
        MockServerRequest second = MockServerRequest.builder()
                .method(HttpMethod.GET).uri(URI.create("/b")).body(Flux.empty());
        StepVerifier.create(client.forward(second))
                .assertNext(r -> Assertions.assertEquals(200, r.statusCode().value()))
                .verifyComplete();
    }

    /**
     * Production-equivalent wiring: ChannelOption.CONNECT_TIMEOUT_MILLIS for dead-FE fast-fail,
     * but no {@code responseTimeout} — mid-stream silence is normal for SSE. A 6-second body
     * delay must not be cut off.
     */
    @Test
    void streamingResponseWithLongBodyDelayIsNotCutOff() {
        server.enqueue(buildSseResponseWith6sBodyDelay());

        HttpClient http = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 2000);
        WebClient webClient = WebClient.builder()
                .clientConnector(new ReactorClientHttpConnector(http))
                .build();
        PassthroughClient client = streamingPassthroughClient(webClient);

        StepVerifier.create(forwardForStreaming(client))
                .expectComplete()
                .verify(Duration.ofSeconds(20));
    }

    /**
     * Regression guard: if anyone reintroduces {@code responseTimeout} on the passthrough
     * HttpClient, ReadTimeoutHandler fires during the 6-second body delay and the stream dies.
     * This test pins that behavior so the failure is loud the moment the wiring drifts.
     */
    @Test
    void addingResponseTimeoutWouldKillTheStream() {
        server.enqueue(buildSseResponseWith6sBodyDelay());

        HttpClient http = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 2000)
                .responseTimeout(Duration.ofSeconds(2));
        WebClient webClient = WebClient.builder()
                .clientConnector(new ReactorClientHttpConnector(http))
                .build();
        PassthroughClient client = streamingPassthroughClient(webClient);

        StepVerifier.create(forwardForStreaming(client))
                .expectErrorMatches(ex -> hasCause(ex, ReadTimeoutException.class))
                .verify(Duration.ofSeconds(10));
    }

    private PassthroughClient streamingPassthroughClient(WebClient webClient) {
        FePool pool = DispatcherTestSupport.fePool(
                () -> List.of("http://" + server.getHostName() + ":" + server.getPort()),
                url -> true);
        return new PassthroughClient(webClient, pool);
    }

    private static MockResponse buildSseResponseWith6sBodyDelay() {
        Buffer body = new Buffer();
        for (int i = 0; i < 6; i++) {
            body.writeUtf8("data: chunk" + i + "\n\n");
        }
        return new MockResponse()
                .setHeader("content-type", "text/event-stream")
                .setBody(body)
                .setBodyDelay(6, TimeUnit.SECONDS);
    }

    private static Mono<Void> forwardForStreaming(PassthroughClient client) {
        MockServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/v1/chat/completions"))
                .body(Flux.empty());
        MockServerWebExchange exchange = MockServerWebExchange.from(
                MockServerHttpRequest.post("http://x/dispatcher/v1/chat/completions"));
        ServerResponse.Context context = new ServerResponse.Context() {
            @Override
            public List<HttpMessageWriter<?>> messageWriters() {
                return HandlerStrategies.withDefaults().messageWriters();
            }

            @Override
            public List<ViewResolver> viewResolvers() {
                return Collections.emptyList();
            }
        };
        return client.forward(request).flatMap(resp -> resp.writeTo(exchange, context));
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
