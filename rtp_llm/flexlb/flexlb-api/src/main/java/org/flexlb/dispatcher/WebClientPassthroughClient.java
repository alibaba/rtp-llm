package org.flexlb.dispatcher;

import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;

public class WebClientPassthroughClient implements PassthroughClient {

    private final WebClient webClient;
    private final FePool fePool;
    private final int timeoutMs;

    public WebClientPassthroughClient(WebClient webClient, FePool fePool, int timeoutMs) {
        this.webClient = webClient;
        this.fePool = fePool;
        this.timeoutMs = timeoutMs;
    }

    @Override
    public Mono<ServerResponse> forward(ServerRequest request) {
        String target = fePool.next() + request.path();
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
    }
}
