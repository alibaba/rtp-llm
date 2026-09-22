package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

@Component
@RequiredArgsConstructor
@Lazy
public class DispatchRouter {

    private final BatchHandler batchHandler;
    private final PassthroughClient passthroughClient;

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            String path = "/dispatcher" + spec.getPath();
            b.POST("/dispatcher/_dryrun" + spec.getPath(), req -> batchHandler.handle(req, spec, true));
            b.POST(path, req -> batchHandler.handle(req, spec, false));
            // Both /dispatcher and /dispatcher/ must split root batches.
            if (path.endsWith("/")) {
                b.POST(path.substring(0, path.length() - 1), req -> batchHandler.handle(req, spec, false));
                b.POST("/dispatcher/_dryrun", req -> batchHandler.handle(req, spec, true));
            }
        }
        return b.route(RequestPredicates.path("/dispatcher/_dryrun/**"),
                        req -> DispatcherResponses.error(400, "invalid_batch_request", "unknown dry-run endpoint or method"))
                .route(RequestPredicates.path("/dispatcher/**"), passthroughClient::forward)
                .build();
    }
}
