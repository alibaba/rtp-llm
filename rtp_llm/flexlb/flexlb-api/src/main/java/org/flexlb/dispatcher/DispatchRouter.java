package org.flexlb.dispatcher;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

/**
 * Reactive {@code RouterFunction} for {@code /dispatcher/**}. Each registered batch endpoint
 * (see {@link org.flexlb.dispatcher.BatchEndpointSpec#SPECS}) gets a POST
 * route that delegates to {@link org.flexlb.dispatcher.BatchHandler};
 * {@code GET /dispatcher/_snapshot} and {@code POST /dispatcher/_dryrun/**} go to
 * {@link DispatcherInspectionHandler}; everything else under {@code /dispatcher/**} is
 * forwarded to one FE via {@link PassthroughClient#forward}.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatchRouter {

    private final BatchHandler batchHandler;
    private final PassthroughClient passthroughClient;
    private final DispatcherInspectionHandler inspectionHandler;
    private final List<BatchEndpointSpec> specs;

    public DispatchRouter(BatchHandler batchHandler,
                          PassthroughClient passthroughClient,
                          DispatcherInspectionHandler inspectionHandler,
                          List<BatchEndpointSpec> specs) {
        this.batchHandler = batchHandler;
        this.passthroughClient = passthroughClient;
        this.inspectionHandler = inspectionHandler;
        this.specs = specs;
    }

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : specs) {
            b.POST("/dispatcher" + spec.getPath(), req -> batchHandler.handle(req, spec));
        }
        b.GET("/dispatcher/_snapshot", inspectionHandler::snapshot);
        b.POST("/dispatcher/_dryrun/**", inspectionHandler::dryRun);
        return b.route(RequestPredicates.path("/dispatcher/**"), passthroughClient::forward)
                .build();
    }
}
