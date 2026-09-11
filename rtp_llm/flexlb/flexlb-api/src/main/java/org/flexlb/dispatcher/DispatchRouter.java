package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

/** Dispatcher HTTP routes on the existing FlexLB listener. */
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatchRouter {

    private final BatchHandler batchHandler;
    private final PassthroughClient passthroughClient;

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            String path = "/dispatcher" + spec.getPath();
            b.POST(path, req -> batchHandler.handle(req, spec));
            // Both /dispatcher and /dispatcher/ must split root batches.
            if (path.endsWith("/")) {
                String bare = path.substring(0, path.length() - 1);
                b.POST(bare, req -> batchHandler.handle(req, spec));
            }
        }
        return b.route(RequestPredicates.path("/dispatcher/**"), passthroughClient::forward)
                .build();
    }
}
