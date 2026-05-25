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
    private final int timeoutMs;

    @Override
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
                    .exchangeToMono(clientResponse ->
                            ServerResponse.status(clientResponse.statusCode())
                                    .headers(h -> h.addAll(clientResponse.headers().asHttpHeaders()))
                                    .body(clientResponse.bodyToFlux(DataBuffer.class), DataBuffer.class))
                    .timeout(Duration.ofMillis(timeoutMs));
        });
    }
}
