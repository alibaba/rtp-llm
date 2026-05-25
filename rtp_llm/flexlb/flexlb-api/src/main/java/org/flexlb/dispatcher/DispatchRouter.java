package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@RequiredArgsConstructor
public class DispatchRouter {

    private final DispatchHandler handler;

    public RouterFunction<ServerResponse> routes() {
        return route()
                .route(RequestPredicates.path("/dispatcher/**"), handler::handlePassthrough)
                .build();
    }
}
