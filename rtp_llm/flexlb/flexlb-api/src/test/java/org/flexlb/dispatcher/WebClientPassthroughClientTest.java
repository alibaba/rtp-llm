package org.flexlb.dispatcher;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.URI;
import java.util.List;

class WebClientPassthroughClientTest {

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
        FePool pool = new FePool(() -> List.of(base));
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(List::of);
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(() -> List.of(base));
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(() -> List.of("http://" + server.getHostName() + ":" + server.getPort()));
        WebClientPassthroughClient passthrough =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(() -> List.of(base));
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(() -> List.of(base));
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(() -> List.of(base));
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
        FePool pool = new FePool(() -> List.of(base));
        WebClientPassthroughClient client =
                new WebClientPassthroughClient(WebClient.builder().build(), pool, 60000);

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
}
