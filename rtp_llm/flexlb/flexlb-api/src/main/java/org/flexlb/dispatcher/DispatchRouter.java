package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

@RequiredArgsConstructor
public class DispatchRouter {

    private final GenericBatchHandler batchHandler;
    private final DispatchHandler passthroughHandler;
    private final List<BatchEndpointSpec> specs;

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : specs) {
            b.POST("/dispatcher" + spec.getPath(), req -> batchHandler.handle(req, spec));
        }
        return b.route(RequestPredicates.path("/dispatcher/**"), passthroughHandler::handlePassthrough)
                .build();
    }
}
