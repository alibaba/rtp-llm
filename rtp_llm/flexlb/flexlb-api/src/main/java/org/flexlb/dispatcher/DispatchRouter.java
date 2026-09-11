package org.flexlb.dispatcher;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

/** Dispatcher HTTP routes on the existing FlexLB listener. */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatchRouter {

    private final BatchHandler batchHandler;
    private final PassthroughClient passthroughClient;
    private final DispatcherInspectionHandler inspectionHandler;

    public DispatchRouter(BatchHandler batchHandler,
                          PassthroughClient passthroughClient,
                          DispatcherInspectionHandler inspectionHandler) {
        this.batchHandler = batchHandler;
        this.passthroughClient = passthroughClient;
        this.inspectionHandler = inspectionHandler;
    }

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            String path = "/dispatcher" + spec.getPath();
            b.POST(path, req -> batchHandler.handle(req, spec));
            // The root spec's path is "/", so the loop above only registers "/dispatcher/". A
            // caller posting to "/dispatcher" would miss every batch route and be silently
            // passthrough-forwarded to a single FE unsplit — register the bare form too.
            if (path.endsWith("/")) {
                String bare = path.substring(0, path.length() - 1);
                b.POST(bare, req -> batchHandler.handle(req, spec));
            }
        }
        // Diagnostics are excluded from graceful drain; dry-run only allocates on explicit opt-in.
        b.GET("/dispatcher/_snapshot", inspectionHandler::snapshot);
        b.POST("/dispatcher/_dryrun/**", inspectionHandler::dryRun);
        return b.route(RequestPredicates.path("/dispatcher/**"), req -> passthroughClient.forward(req)
                        .doOnDiscard(ServerResponse.class, PassthroughClient::releaseIfUnwritten))
                .build();
    }

}
