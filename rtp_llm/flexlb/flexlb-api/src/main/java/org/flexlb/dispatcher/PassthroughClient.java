package org.flexlb.dispatcher;

import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

public interface PassthroughClient {

    /** Forward the request verbatim to one FE; complete the response onto the caller. */
    Mono<ServerResponse> forward(ServerRequest request);
}
