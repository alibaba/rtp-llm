package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutException;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okio.Buffer;
import org.junit.jupiter.api.AfterEach;
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
import reactor.netty.http.client.HttpClient;
import reactor.test.StepVerifier;

import java.net.URI;
import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

class StreamingPassthroughTest {

    private static final int CONNECT_TIMEOUT_MS = 2000;
    private static final int MAX_STREAM_DURATION_MS = 600_000;

    private MockWebServer fe;

    @BeforeEach
    void up() throws Exception {
        fe = new MockWebServer();
        fe.start();
    }

    @AfterEach
    void down() throws Exception {
        fe.shutdown();
    }

    /**
     * Production-equivalent wiring: ChannelOption.CONNECT_TIMEOUT_MILLIS for dead-FE fast-fail,
     * but no {@code responseTimeout} — mid-stream silence is normal for SSE. A 6-second body
     * delay must not be cut off.
     */
    @Test
    void streamingResponseWithLongBodyDelayIsNotCutOff() {
        fe.enqueue(buildSseResponseWith6sBodyDelay());

        WebClient webClient = passthroughWebClientMirroringProduction();
        WebClientPassthroughClient client = passthroughClient(webClient);

        StepVerifier.create(forward(client))
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
        fe.enqueue(buildSseResponseWith6sBodyDelay());

        HttpClient http = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, CONNECT_TIMEOUT_MS)
                .responseTimeout(Duration.ofSeconds(2));
        WebClient webClient = WebClient.builder()
                .clientConnector(new ReactorClientHttpConnector(http))
                .build();
        WebClientPassthroughClient client = passthroughClient(webClient);

        StepVerifier.create(forward(client))
                .expectErrorMatches(ex -> hasCause(ex, ReadTimeoutException.class))
                .verify(Duration.ofSeconds(10));
    }

    private static WebClient passthroughWebClientMirroringProduction() {
        HttpClient http = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, CONNECT_TIMEOUT_MS);
        return WebClient.builder()
                .clientConnector(new ReactorClientHttpConnector(http))
                .build();
    }

    private WebClientPassthroughClient passthroughClient(WebClient webClient) {
        FePool pool = new FePool(() -> List.of("http://" + fe.getHostName() + ":" + fe.getPort()), url -> true);
        return new WebClientPassthroughClient(webClient, pool);
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

    private static reactor.core.publisher.Mono<Void> forward(WebClientPassthroughClient client) {
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
