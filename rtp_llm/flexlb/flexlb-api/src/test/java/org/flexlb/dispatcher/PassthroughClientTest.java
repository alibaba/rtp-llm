package org.flexlb.dispatcher;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okhttp3.mockwebserver.SocketPolicy;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.HandlerStrategies;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerRequest;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

@Timeout(15)
class PassthroughClientTest {
    private final MockWebServer fe = new MockWebServer();
    private final ConnectionProvider connections = ConnectionProvider.builder("passthrough-test")
            .maxConnections(1).pendingAcquireTimeout(Duration.ofSeconds(1)).build();
    private final DispatchConfig cfg = new DispatchConfig();
    private final DispatcherMetricsReporter metrics = mock(DispatcherMetricsReporter.class);

    @AfterEach
    void close() throws Exception {
        connections.disposeLater().block(Duration.ofSeconds(5));
        fe.shutdown();
    }

    private PassthroughClient proxy(List<String> hosts) {
        WebClient web = WebClient.builder().clientConnector(new ReactorClientHttpConnector(
                HttpClient.create(connections))).build();
        return new PassthroughClient(web, DispatcherTestSupport.fePool(hosts), metrics, cfg);
    }

    private WebTestClient http(PassthroughClient proxy) {
        return WebTestClient.bindToRouterFunction(RouterFunctions.route()
                .POST("/dispatcher/**", proxy::forward).build()).build();
    }

    @ParameterizedTest
    @ValueSource(ints = {200, 302, 400, 500, 599})
    void preservesWireBodyStatusQueryAndEndToEndHeaders(int status) throws Exception {
        fe.enqueue(new MockResponse().setResponseCode(status).setBody("exact 中文 body")
                .setHeader("Content-Type", "text/plain").setHeader("X-Trace", "response")
                .setHeader("Connection", "X-Private").setHeader("X-Private", "secret"));
        http(proxy(List.of(fe.url("/").toString().replaceAll("/$", "")))).post()
                .uri(java.net.URI.create("/dispatcher/?q=a%2Fb")).header("Authorization", "Bearer caller")
                .header("Connection", "x-PRIVATE").header("X-Private", "secret")
                .header(DispatcherHeaders.TRUSTED_ROUTING_HEADER, "forged")
                .bodyValue("unchanged request").exchange().expectStatus().isEqualTo(status)
                .expectHeader().valueEquals("X-Trace", "response")
                .expectHeader().doesNotExist("X-Private").expectBody(String.class).isEqualTo("exact 中文 body");
        RecordedRequest request = fe.takeRequest(2, TimeUnit.SECONDS);
        assertNotNull(request);
        assertEquals("/?q=a%2Fb", request.getPath());
        assertEquals("unchanged request", request.getBody().readUtf8());
        assertEquals("Bearer caller", request.getHeader("Authorization"));
        assertNull(request.getHeader("X-Private"));
        assertNull(request.getHeader(DispatcherHeaders.TRUSTED_ROUTING_HEADER));
        verify(metrics).reportRequest(eq("passthrough"), eq("/"), eq(status), anyLong());
    }

    @Test
    void emptyPoolAndMissingHeadersReturnRedacted502() {
        http(proxy(List.of())).post().uri("/dispatcher/").exchange().expectStatus().isEqualTo(502)
                .expectBody().jsonPath("$.message").isEqualTo("upstream request failed");
        cfg.setBatchTimeoutMs(100);
        fe.enqueue(new MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE));
        http(proxy(List.of(fe.url("/").toString()))).post().uri("/dispatcher/")
                .exchange().expectStatus().isEqualTo(502);
    }

    @Test
    void unwrittenResponseDoesNotAcquireConnectionAndBarePathReachesRoot() throws Exception {
        PassthroughClient proxy = proxy(List.of(fe.url("/").toString().replaceAll("/$", "")));
        MockServerWebExchange exchange = MockServerWebExchange.from(MockServerHttpRequest.post("/dispatcher"));
        assertNotNull(proxy.forward(ServerRequest.create(exchange, HandlerStrategies.withDefaults().messageReaders())).block());
        assertEquals(0, fe.getRequestCount());
        fe.enqueue(new MockResponse().setBody("ok"));
        http(proxy).post().uri("/dispatcher").exchange().expectBody(String.class).isEqualTo("ok");
        assertEquals("/", fe.takeRequest(2, TimeUnit.SECONDS).getPath());
    }

    @Test
    void cancellingBodyReleasesOnlyConnectionAndDelayedStreamOutlivesHeadersTimeout() throws Exception {
        cfg.setBatchTimeoutMs(1000);
        PassthroughClient proxy = proxy(List.of(fe.url("/").toString().replaceAll("/$", "")));
        fe.enqueue(new MockResponse().setBody("first\nsecond\n").throttleBody(6, 1, TimeUnit.SECONDS));
        MockServerWebExchange exchange = MockServerWebExchange.from(MockServerHttpRequest.post("/dispatcher/stream"));
        exchange.getResponse().setWriteHandler(body -> body.take(1).doOnNext(DataBufferUtils::release).then());
        RouterFunctions.toHttpHandler(RouterFunctions.route().POST("/dispatcher/**", proxy::forward).build())
                .handle(exchange.getRequest(), exchange.getResponse()).block(Duration.ofSeconds(5));
        fe.enqueue(new MockResponse().setHeader("Content-Type", "text/event-stream")
                .setBody("data: final\n\n").setBodyDelay(1200, TimeUnit.MILLISECONDS));
        http(proxy).post().uri("/dispatcher/stream").exchange().expectStatus().isOk()
                .expectBody(String.class).isEqualTo("data: final\n\n");
        assertEquals(2, fe.getRequestCount());
    }
}
