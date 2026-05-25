package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

@RequiredArgsConstructor
public class DispatchHandler {

    private final PassthroughClient passthroughClient;

    public Mono<ServerResponse> handlePassthrough(ServerRequest request) {
        return passthroughClient.forward(request);
    }
}
