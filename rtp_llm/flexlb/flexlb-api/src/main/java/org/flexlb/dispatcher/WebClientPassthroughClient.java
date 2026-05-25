package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;

@RequiredArgsConstructor
public class WebClientPassthroughClient implements PassthroughClient {

    private final WebClient webClient;
    private final FePool fePool;
    private final int maxStreamDurationMs;

    /**
     * Uses the deprecated {@link WebClient.RequestHeadersSpec#exchange() exchange()} rather than
     * {@code exchangeToMono} because the upstream body {@link Flux} is handed to a
     * {@link ServerResponse} that subscribes lazily (at {@code writeTo} time);
     * {@code exchangeToMono} would release the upstream connection the instant its lambda
     * {@code Mono} completes — which is immediate, since {@code .body(BodyInserter)} is
     * synchronous — leaving nothing left to read by the time the downstream subscribes.
     * {@code exchange()} defers connection release until the body is consumed, which matches the
     * passthrough's intended lifetime.
     */
    @Override
    @SuppressWarnings("deprecation")
    public Mono<ServerResponse> forward(ServerRequest request) {
        return Mono.fromCallable(fePool::next).flatMap(feBaseUrl -> {
            URI src = request.uri();
            String fePath = src.getRawPath().startsWith("/dispatcher/")
                    ? src.getRawPath().substring("/dispatcher".length())
                    : src.getRawPath();
            String pathAndQuery = src.getRawQuery() == null ? fePath : fePath + "?" + src.getRawQuery();
            URI target = URI.create(feBaseUrl + pathAndQuery);
            Flux<DataBuffer> bodyStream = request.bodyToFlux(DataBuffer.class);
            return webClient.method(request.method())
                    .uri(target)
                    .headers(h -> h.addAll(request.headers().asHttpHeaders()))
                    .body(BodyInserters.fromDataBuffers(bodyStream))
                    .exchange()
                    .flatMap(clientResponse ->
                            ServerResponse.status(clientResponse.statusCode())
                                    .headers(h -> h.addAll(clientResponse.headers().asHttpHeaders()))
                                    .body(BodyInserters.fromDataBuffers(
                                            clientResponse.bodyToFlux(DataBuffer.class)
                                                    .timeout(Duration.ofMillis(maxStreamDurationMs)))));
        });
    }
}
