package org.flexlb.dispatcher;

import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

public class DispatchRouter {

    private final DispatchHandler handler;

    public DispatchRouter(DispatchHandler handler) {
        this.handler = handler;
    }

    public RouterFunction<ServerResponse> routes() {
        return route()
                .POST("/batch_infer", handler::handleBatch)
                .route(RequestPredicates.all(), handler::handlePassthrough)
                .build();
    }
}
